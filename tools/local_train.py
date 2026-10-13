"""ლოკალური (CPU) წვრთნა Kaggle-ის სესიებს შორის — როცა კვირის კვოტა არ ჰყოფნის.

სერვისი: aivoice-local.service (ჩართულია ჩატვირთვისას — დენის გათიშვის შემდეგ თავისით აგრძელებს).
მართვა: data/local_train.json  {"slot": "base_ka", "run": true|false, "reason": "..."} — წერს
        tools/kaggle_train.py watch (კვოტის მიხედვით) ან ხელით.
მდგომარეობა: training/runs/<slot>/local/state.json (პანელი კითხულობს).

წვრთნა გრძელდება უახლესი checkpoint-იდან: ლოკალური (local-e*-s*.ckpt, ყოველ ~30 წუთში, ეპოქის
შუაშიც) ან Kaggle-ის kaggle_out/final.ckpt — რომელსაც მეტი ნაბიჯი აქვს. დაზიანებული ფაილი (გათიშვა
ჩაწერისას) გამოტოვდება. Kaggle-ზე დაბრუნებისას kaggle_train.py უახლეს ლოკალურ checkpoint-ს იღებს.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import api_url, cfg, expand, path as cpath  # noqa: E402,F401
PY = ROOT / "training" / ".venv" / "bin" / "python"
CTRL = ROOT / "data" / "local_train.json"
CKPT_STEPS = 1400  # ≈30 წთ (2.56 წმ/ბატჩი, ბატჩზე 2 global_step)
BATCH, WORKERS, THREADS = 8, 1, 12  # 12 = CPUQuota 1200%; 1 worker — თითო ~2 GB RAM-ს იკავებს (3-ზე 9 GB-ს ზღვარს ასცდა)
DESIGNER_EVERY_H = 3
# ფონემების ქეში (~190 ათასი პატარა ფაილი): /mnt/d-ზე (drvfs) პატარა ფაილები ძალიან ნელია, Linux-ის დისკი კი
# ფიზიკურად C:-ზეა (ხურდება — მომხმარებლის მოთხოვნა 2026-10-08). ამიტომ ქეში RAM-შია (/dev/shm), D:-ზე კი
# ერთ tar ფაილად ინახება და გაშვებისას იქიდან იხსნება. checkpoint-ები და ლოგები D:-ზეა.
CACHE = Path("/dev/shm/aivoice")
OLD_CACHE = Path.home() / ".cache" / "aivoice"  # ძველი ადგილი (C:) — გადატანის შემდეგ იშლება

PROGRESS = re.compile(r"Epoch (\d+):\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+) \[([^<]+)<([^,]+),\s*([\d.]+)(s/it|it/s)")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def write_json(p: Path, data):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def ckpt_info(p: Path):
    """(epoch, global_step) ან None, თუ ფაილი დაზიანებულია."""
    import torch
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        return int(ck["epoch"]), int(ck["global_step"])
    except Exception:
        return None


def best_ckpt(slot: str):
    """უახლესი გამართული checkpoint: [(step, epoch, path)] — მეტი ნაბიჯი = უახლესი."""
    run = cpath("runs") / slot
    cands = list((run / "local").glob("lightning_logs/version_*/checkpoints/local-*.ckpt"))
    cands += [run / "kaggle_out" / "final.ckpt"]
    good = []
    for p in cands:
        if p.exists() and (i := ckpt_info(p)):
            good.append((i[1], i[0], p))
    return max(good) if good else None


class Local:
    def __init__(self, slot: str):
        self.slot = slot
        self.run = cpath("runs") / slot
        self.dir = self.run / "local"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_p = self.dir / "state.json"
        self.state = read_json(self.state_p, {}) or {}
        self.proc = None
        self.log = None
        self.buf = b""
        self.last_save = 0.0

    def save(self, **kw):
        self.state.update(kw, slot=self.slot, updated=now())
        write_json(self.state_p, self.state)

    def job(self):
        return read_json(ROOT / "data" / "kaggle_jobs" / f"{self.slot}.json", {})

    def metadata(self, j) -> Path:
        # Kaggle-ის kernel-ის მსგავსად: ბრჭყალები CSV-ს არღვევს
        src = Path(j["dataset_dir"]) / "metadata.csv"
        dst = self.dir / "metadata.csv"
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            dst.write_text(src.read_text(encoding="utf-8").replace('"', ""), encoding="utf-8")
        return dst

    def ensure_cache(self) -> Path:
        """ქეში RAM-ში; წყარო — D:-ზე შენახული tar (ან ძველი ქეში C:-ზე, რომელიც ერთხელ გადაიტანება)."""
        cache = CACHE / f"{self.slot}_local" / "cache"
        tar = self.dir / "phoneme_cache.tar"
        if cache.exists() and any(cache.iterdir()):
            return cache
        cache.parent.mkdir(parents=True, exist_ok=True)
        old = OLD_CACHE / f"{self.slot}_local"
        if not tar.exists() and (old / "cache").exists():
            subprocess.run(["tar", "-cf", str(tar) + ".tmp", "-C", str(old), "cache"], check=True)
            os.replace(str(tar) + ".tmp", tar)
            shutil.rmtree(old, ignore_errors=True)
        if tar.exists():
            subprocess.run(["tar", "-xf", str(tar), "-C", str(cache.parent)], check=True)
        return cache

    def save_cache(self):
        """ახლად აგებული ქეში (prepare) → D: — რომ გადატვირთვის შემდეგ თავიდან აგება არ დასჭირდეს."""
        tar = self.dir / "phoneme_cache.tar"
        cache = CACHE / f"{self.slot}_local" / "cache"
        if not tar.exists() and cache.exists():
            subprocess.run(["tar", "-cf", str(tar) + ".tmp", "-C", str(cache.parent), "cache"], check=True)
            os.replace(str(tar) + ".tmp", tar)

    def start(self):
        j = self.job()
        best = best_ckpt(self.slot)
        if not best:
            self.save(running=False, error="checkpoint ვერ ვიპოვე")
            return
        step, epoch, ckpt = best
        if epoch + 1 >= j["max_epochs"]:
            self.save(running=False, done=True, epoch=epoch, step=step)
            return
        meta = self.metadata(j)
        cache = self.ensure_cache()
        cmd = [str(PY), "-m", "piper.train", "fit",
               "--data.voice_name", self.slot,
               "--data.csv_path", str(meta),
               "--data.audio_dir", str(Path(j["dataset_dir"]) / "audio"),
               "--model.sample_rate", "22050",
               "--data.espeak_voice", "ka",
               "--data.cache_dir", str(cache),
               "--data.config_path", str(self.dir / "config.json"),
               "--data.batch_size", str(BATCH),
               "--data.num_workers", str(WORKERS),
               "--data.validation_split", "0.01",
               "--data.num_test_examples", "2",
               "--data.trim_silence", "true" if j.get("trim_silence", True) else "false",
               "--model.mos_metric", "none",
               "--model.num_speakers", str(j["num_speakers"]),
               "--trainer.accelerator", "cpu",
               "--trainer.devices", "1",
               "--trainer.max_epochs", str(j["max_epochs"]),
               "--trainer.default_root_dir", str(self.dir),
               "--ckpt_path", str(ckpt)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(THREADS),
               "MKL_NUM_THREADS": str(THREADS), "PIPER_CKPT_EVERY_STEPS": str(CKPT_STEPS)}
        if j.get("audio_from_source"):
            env["PIPER_AUDIO_FROM_SOURCE"] = "1"
        self.log = (self.dir / "train.log").open("a", encoding="utf-8")
        self.log.write(f"\n==== {now()} resume {ckpt} (epoch {epoch}, step {step})\n")
        self.log.flush()
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                     cwd=ROOT / "training", start_new_session=True)
        os.set_blocking(self.proc.stdout.fileno(), False)
        self.save(running=True, pid=self.proc.pid, started=now(), resumed_from=str(ckpt),
                  epoch=epoch, step=step, error=None, done=False)
        self.check_config_later = True

    def stop(self, why: str):
        if not self.proc:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()
        except ProcessLookupError:
            pass
        self.finish(f"გაჩერდა: {why}")

    def finish(self, note: str, error: str | None = None):
        if self.log:
            self.log.write(f"==== {now()} {note}\n")
            self.log.close()
        self.proc, self.log = None, None
        self.save(running=False, pid=None, stopped=now(), note=note, error=error)

    def poll(self):
        """გამოსავლის კითხვა (არაბლოკირებადი) + პროგრესი state.json-ში."""
        if not self.proc:
            return
        while True:
            try:
                chunk = self.proc.stdout.read(65536)
            except BlockingIOError:
                chunk = None
            if not chunk:
                break
            self.buf += chunk
        parts = re.split(rb"[\r\n]", self.buf)
        self.buf = parts.pop()
        for line in parts:
            s = line.decode("utf-8", "replace")
            m = PROGRESS.search(s)
            if m:
                ep, _, i, n, _, _, rate, unit = m.groups()
                if time.time() - self.last_save > 30:
                    sec = float(rate) if unit == "s/it" else 1 / max(float(rate), 1e-9)
                    self.save(epoch=int(ep), step_in_epoch=int(i), steps_per_epoch=int(n), sec_per_step=round(sec, 2),
                              epoch_hours=round(int(n) * sec / 3600, 1))
                    self.last_save = time.time()
            elif s.strip() and "it/s" not in s and "s/it" not in s:
                self.log.write(s + "\n")
                self.log.flush()
        if getattr(self, "check_config_later", False) and (self.dir / "config.json").exists():
            # დინამიკების ნუმერაცია Kaggle-ის მოდელს უნდა ემთხვეოდეს — სხვაგვარად ხმები აირევა
            self.check_config_later = False
            self.save_cache()  # config.json ჩნდება prepare-ის შემდეგ — ქეში უკვე სრულია
            a = (read_json(self.dir / "config.json", {}) or {}).get("speaker_id_map")
            b = (read_json(self.run / "kaggle_out" / "config.json", {}) or {}).get("speaker_id_map")
            if a != b:
                self.stop("speaker_id_map არ ემთხვევა Kaggle-ის მოდელს")
                self.save(error="speaker_id_map არ ემთხვევა — წვრთნა შეჩერდა", blocked=True)
                return
        rc = self.proc.poll()
        if rc is not None:
            self.poll_rest()
            if rc == 0:
                self.finish("დასრულდა (max_epochs)")
            else:
                self.finish(f"exit {rc}", error=f"exit code {rc} — იხ. local/train.log")

    def poll_rest(self):
        try:
            rest = self.proc.stdout.read() or b""
        except BlockingIOError:
            rest = b""
        if rest and self.log:
            self.log.write((self.buf + rest).decode("utf-8", "replace")[-4000:] + "\n")

    def maybe_export_designer(self):
        """ხმის შემქმნელი ახალ ლოკალურ checkpoint-ზე — ახალი ეპოქის შემდეგ, არაუმეტეს ყოველ 3 სთ-ში."""
        if time.time() - (self.state.get("designer_at_ts") or 0) < DESIGNER_EVERY_H * 3600:
            return
        cands = sorted(self.dir.glob("lightning_logs/version_*/checkpoints/local-*.ckpt"), key=lambda p: p.stat().st_mtime)
        if not cands:
            return
        info = ckpt_info(cands[-1])
        meta = read_json(cpath("designer") / "meta.json", {}) or {}
        if not info or (meta.get("epoch") is not None and info[0] <= meta["epoch"]):
            return
        r = subprocess.run([str(PY), str(ROOT / "tools" / "export_designer.py"), str(cands[-1]),
                            "--config", str(self.run / "kaggle_out" / "config.json"), "--source", "local"],
                           capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4"})
        self.save(designer_at_ts=time.time(), designer_epoch=info[0],
                  designer_result=(r.stdout.strip() or r.stderr.strip())[-300:])


def main():
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    cur: Local | None = None
    fails = 0
    retry_at = 0.0
    try:
        while True:
            ctrl = read_json(CTRL, {}) or {}
            want = ctrl.get("slot") if ctrl.get("run") else None
            if cur and cur.proc and cur.slot != want:
                cur.stop(ctrl.get("reason") or "მართვის ფაილით")
            if want and (not cur or cur.slot != want):
                cur = Local(want)
            if cur and want and not cur.proc and time.time() >= retry_at and not cur.state.get("done") \
                    and not cur.state.get("blocked"):
                cur.start()
            if cur and cur.proc:
                cur.poll()
                if not cur.proc and cur.state.get("error"):
                    fails += 1  # შეცდომაზე — 10, 20, 40… წუთში თავიდან (მაქს. 3 სთ)
                    retry_at = time.time() + min(3 * 3600, 600 * 2 ** (fails - 1))
                elif cur.proc and cur.state.get("step_in_epoch"):
                    fails = 0
                if cur.proc:
                    try:
                        cur.maybe_export_designer()
                    except Exception as e:  # ექსპორტის შეცდომამ წვრთნა არ უნდა გააჩეროს
                        cur.save(designer_result=str(e)[-300:])
            time.sleep(20)
    finally:
        if cur and cur.proc:
            cur.stop("სერვისი გაჩერდა")


if __name__ == "__main__":
    main()
