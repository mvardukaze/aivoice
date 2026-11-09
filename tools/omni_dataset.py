"""სინთეზური (OmniVoice) ქართული dataset საბაზო Piper მოდელისთვის: base_ka_v4_omni_nc — ⚠ მხოლოდ არაკომერციული.

OmniVoice-ის კოდი Apache-2.0-ია, მაგრამ წონები CC-BY-NC (Emilia), აუდიო ტოკენიზატორი — Boson Higgs Audio 2
Community License. ამიტომ აქ შექმნილი ჩანაწერები და მათზე ნაწვრთნი ყველა მოდელი არაკომერციულია (cc-by-nc-4.0):
საჯარო Hugging Face-ზე არ იტვირთება (tools/models.py public_files ამას ამოწმებს), სახელში ყოველთვის „omni_nc“.

    .venv/bin/python tools/omni_dataset.py refs-push [--seeds 3]   40 ხმა × კანდიდატები Kaggle-ზე (voice design)
    .venv/bin/python tools/omni_dataset.py status
    .venv/bin/python tools/omni_dataset.py refs-fetch               → {data_root}/omni_nc/refs_kernel/
    .venv/bin/python tools/omni_dataset.py pick                      საუკეთესო კანდიდატი თითო ხმაზე → data/omni_voices_nc.json
    .venv/bin/python tools/omni_dataset.py gen-push --part a|b       input dataset (ერთხელ) + გენერაციის kernel
    .venv/bin/python tools/omni_dataset.py gen-fetch --part a|b
    .venv/bin/python tools/omni_dataset.py watch                     ლოდინი → fetch → (საჭიროებისას) გაგრძელება → build
    .venv/bin/python tools/omni_dataset.py build [--tar]             ფილტრი + {datasets}/base_ka_v4_omni_nc/ (+ GCP არქივი)

ხმები: omni_f01…f20, omni_m01…m20 (არცერთი ნამდვილი ადამიანი — მხოლოდ აღწერით შექმნილი); shard i ↔ ხმა i.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from app.config import path as cpath  # noqa: E402
from kaggle_train import WORK, kaggle, upload_dataset, user, wait_dataset_ready  # noqa: E402

SLOT = "base_ka_v4_omni_nc"
LABEL = "v4-omni (არაკომერციული)"
LICENSE = {"license": "cc-by-nc-4.0", "commercial": False,
           "license_note": "OmniVoice-ის სინთეზური ჩანაწერები: წონები CC-BY-NC-4.0 (Emilia), აუდიო ტოკენიზატორი — "
                           "Boson Higgs Audio 2 Community License. მხოლოდ არაკომერციული გამოყენება; საჯაროდ არ ქვეყნდება."}
D = cpath("data_root") / "omni_nc"
CORPUS = cpath("data_root") / "omni_corpus"
VOICES_JSON = ROOT / "data" / "omni_voices_nc.json"
TEMPLATE = ROOT / "kaggle" / "kernel_omni_gen.py"
PIP = "omnivoice @ git+https://github.com/k2-fsa/OmniVoice.git@08be0b4ccbac3e13e374e86fbfead4b4cac343e2"
INPUT_SLUG = "aivoice-omni-nc-input"
NUM_STEP = 16
REF_SPEED = 0.9  # ნაგულისხმევი OmniVoice ქართულისთვის ცოტა სწრაფია; clone-ში ტემპი reference-იდან მოდის
REF_TEXT = ("დღეს ამინდი მშვიდი და თბილია, ამიტომ საღამოს მეგობრებთან ერთად ძველ უბანში გავისეირნებთ "
            "და ახალ წიგნის მაღაზიასაც ვნახავთ.")
TESTS = ["როდის გადაწყვიტეთ, რომ ამ პროფესიას გაჰყოლოდით?",
         "ყურადღება, ორას მეტრში მოუხვიეთ მარჯვნივ!",
         "მთის სოფლებში ზამთარი ადრე იწყება და გაზაფხულამდე გრძელდება."]

# სქესი × ასაკი × სიმაღლე (OmniVoice-ის voice design-ის ლექსიკონი: omnivoice/utils/voice_design.py)
_F = [("young adult", "moderate pitch"), ("young adult", "high pitch"), ("young adult", "low pitch"),
      ("young adult", "moderate pitch"), ("young adult", "high pitch"), ("young adult", "low pitch"),
      ("young adult", "moderate pitch"),
      ("middle-aged", "moderate pitch"), ("middle-aged", "low pitch"), ("middle-aged", "high pitch"),
      ("middle-aged", "moderate pitch"), ("middle-aged", "low pitch"), ("middle-aged", "high pitch"),
      ("elderly", "moderate pitch"), ("elderly", "low pitch"), ("elderly", "high pitch"), ("elderly", "moderate pitch"),
      ("teenager", "moderate pitch"), ("teenager", "high pitch"), ("teenager", "moderate pitch")]
_M = [("young adult", "moderate pitch"), ("young adult", "low pitch"), ("young adult", "high pitch"),
      ("young adult", "moderate pitch"), ("young adult", "low pitch"), ("young adult", "very low pitch"),
      ("young adult", "moderate pitch"),
      ("middle-aged", "low pitch"), ("middle-aged", "moderate pitch"), ("middle-aged", "very low pitch"),
      ("middle-aged", "low pitch"), ("middle-aged", "moderate pitch"), ("middle-aged", "high pitch"),
      ("elderly", "low pitch"), ("elderly", "moderate pitch"), ("elderly", "low pitch"), ("elderly", "high pitch"),
      ("teenager", "moderate pitch"), ("teenager", "high pitch"), ("teenager", "low pitch")]
VOICES = ([{"name": f"omni_f{i + 1:02d}", "shard": f"voice_{i + 1:02d}.jsonl", "instruct": f"female, {a}, {p}"}
           for i, (a, p) in enumerate(_F)] +
          [{"name": f"omni_m{i + 1:02d}", "shard": f"voice_{i + 21:02d}.jsonl", "instruct": f"male, {a}, {p}"}
           for i, (a, p) in enumerate(_M)])
# ნაწილები (ცალ-ცალკე Kaggle kernel-ები) — თითოში ორივე სქესი, რომ ერთი ნაწილიც დაბალანსებული იყოს
PARTS = {"a": [v["name"] for v in VOICES if int(v["name"][-2:]) <= 10],
         "b": [v["name"] for v in VOICES if int(v["name"][-2:]) > 10]}
GH_HYPHEN = re.compile(r"([სშცჩხქფთტპკწჭყ])-ღ")
STRIP = str.maketrans("", "", ".,!?;:„“\"'—–-()… ")


def rj(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def wj(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(p)


def kid(name: str) -> str:
    return f"{user()}/aivoice-omni-nc-{name}"


def push_kernel(name: str, cfg: dict, timeout_s: int, datasets: list[str]):
    k = WORK / f"kernel-omni-nc-{name}"
    k.mkdir(parents=True, exist_ok=True)
    (k / "gen.py").write_text(TEMPLATE.read_text(encoding="utf-8").replace(
        "__CONFIG__", json.dumps(cfg, ensure_ascii=False).replace("\\", "\\\\")), encoding="utf-8")
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": kid(name), "title": f"aivoice omni nc {name}", "code_file": "gen.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": datasets}))
    kaggle("kernels", "push", "-p", str(k), "-t", str(timeout_s))
    print("✓", f"https://www.kaggle.com/code/{kid(name)}")


def kstatus(name: str) -> str:
    r = kaggle("kernels", "status", kid(name), check=False)
    s = (r.stdout + r.stderr).lower()
    for w in ("complete", "error", "cancel", "running", "queued"):
        if w in s:
            return w
    return s.strip()[-80:] or "unknown"


def omni_text(r: dict) -> str:
    """ka_text-ის დეფისი („ოდეს-ღაც“) მხოლოდ espeak-ისთვისაა — OmniVoice-ს საწყისი დამწერლობა სჭირდება.
    დეფისი რჩება, თუ ის საწყის ტექსტშიც იყო („…-ღამის“ არა, მაგრამ შემთხვევით მსგავსი)."""
    out = []
    for w in r["text"].split(" "):
        core = w.strip(".,!?;:„“\"'—–()…")
        if GH_HYPHEN.search(core) and core not in r.get("raw", ""):
            w = GH_HYPHEN.sub(r"\1ღ", w)
        out.append(w)
    return " ".join(out)


# ---------------- refs ----------------
def cmd_refs_push(a):
    cfg = {"mode": "refs", "pip": PIP, "num_step": NUM_STEP, "ref_text": REF_TEXT, "ref_speed": REF_SPEED,
           "tests": TESTS, "test_speeds": [1.0, 0.9],
           "voices": [{"name": v["name"], "instruct": v["instruct"],
                       "seeds": [1000 + i * 37 + k * 7919 for k in range(a.seeds)]} for i, v in enumerate(VOICES)]}
    push_kernel("refs", cfg, 3 * 3600, [])


def cmd_status(_):
    for n in ("refs", "gen-a", "gen-b"):
        print(n, kstatus(n))


def fetch(name: str, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", kid(name), "-p", str(dst), "-o")


def cmd_refs_fetch(_):
    fetch("refs", D / "refs_kernel")
    print(json.dumps(rj(D / "refs_kernel" / "status.json"), ensure_ascii=False))


def cmd_pick(a):
    """თითო ხმაზე საუკეთესო კანდიდატი: გამართული (MMS CER ≤ 0.08, 6–13 წმ, clipping-ის გარეშე), F0 სქესს
    შეესაბამება (ქალი ≥ 150 Hz, კაცი ≤ 175 Hz), clone-ის ტესტებიც კარგად იკითხება; რჩეულიდან — დაბალი CER."""
    res = rj(D / "refs_kernel" / "refs.json", [])
    by = defaultdict(list)
    for r in res:
        if not r.get("error"):
            by[r["name"]].append(r)
    out, bad = [], []
    refs = D / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    for v in VOICES:
        cands = []
        for r in by.get(v["name"], []):
            tests = [t for t in r.get("tests", []) if t["speed"] == 1.0]
            tcer = sum(t["cer"] for t in tests) / max(1, len(tests))
            f0 = r.get("f0") or 0
            sex_ok = (f0 >= 150) if v["name"].startswith("omni_f") else (0 < f0 <= 175)
            ok = (r["cer"] <= 0.08 and 6 <= r["dur"] <= 13 and r.get("clip", 0) <= 0.001 and r.get("max_gap_s", 0) <= 1.0
                  and sex_ok and tcer <= 0.12)
            cands.append((not ok, round(r["cer"] + tcer, 3), r, tcer))
        if not cands:
            bad.append(v["name"])
            continue
        cands.sort(key=lambda c: (c[0], c[1]))
        notok, _, r, tcer = cands[0]
        if notok:
            bad.append(v["name"])
        shutil.copy2(D / "refs_kernel" / "refs" / r["file"], refs / f"{v['name']}.wav")
        t1 = [t for t in r.get("tests", []) if t["speed"] == 1.0]
        t9 = [t for t in r.get("tests", []) if t["speed"] == 0.9]
        cps1 = sum(t["cps"] for t in t1) / max(1, len(t1))
        cps9 = sum(t["cps"] for t in t9) / max(1, len(t9))
        rtf = sum(t["wall"] for t in t1) / max(1e-3, sum(t["dur"] for t in t1))
        out.append({**v, "seed": r["seed"], "ref": f"{v['name']}.wav", "ref_text": REF_TEXT, "ref_speed": REF_SPEED,
                    "num_step": NUM_STEP, "ref_dur": r["dur"], "ref_cer": r["cer"], "ref_cps": r["cps"], "f0": r.get("f0"),
                    "test_cer": round(tcer, 3), "clone_cps_x1": round(cps1, 2), "clone_cps_x09": round(cps9, 2),
                    "clone_rtf": round(rtf, 3), "ok": not notok,
                    "candidates": [{"seed": c[2]["seed"], "ok": not c[0], "cer": c[2]["cer"], "test_cer": round(c[3], 3),
                                    "f0": c[2].get("f0"), "dur": c[2]["dur"]} for c in cands]})
        print(f"{v['name']} {v['instruct']:38} seed {r['seed']:6} f0 {r.get('f0')!s:>6} ref {r['dur']:5.1f}s cer {r['cer']:.3f} "
              f"test {tcer:.3f} cps ref {r['cps']:.1f} x1 {cps1:.1f} x.9 {cps9:.1f} rtf {rtf:.3f} {'OK' if not notok else '✗'}")
    wj(VOICES_JSON, {"_note": f"{SLOT}: OmniVoice-ით აღწერით შექმნილი ხმები (არა ნამდვილი ადამიანები). reference-ები: "
                              f"{refs}. ⚠ არაკომერციული (cc-by-nc-4.0).", **LICENSE, "slot": SLOT,
                     "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "voices": out})
    print(f"\n{len(out) - len(bad)}/{len(VOICES)} OK; პრობლემური: {bad}")


# ---------------- gen ----------------
def gen_dir(part: str) -> Path:
    return D / "gen" / part


def tars(part: str | None = None) -> list[Path]:
    parts = [part] if part else list(PARTS)
    return sorted(p for x in parts for p in gen_dir(x).glob("r*/tars/*.tar"))


def rows_from_tars(paths) -> dict[str, dict]:
    """id → row (ბოლო ვერსია). tar-ში: <voice>/rows.*.jsonl + <voice>/<id>.flac."""
    out = {}
    for t in paths:
        with tarfile.open(t) as tf:
            for m in tf:
                if m.name.endswith(".jsonl"):
                    for l in tf.extractfile(m).read().decode("utf-8").splitlines():
                        if l.strip():
                            r = json.loads(l)
                            r["_tar"] = str(t)
                            out[r["id"]] = r
    return out


def ensure_input() -> str:
    """Kaggle input dataset: shards (omni_text-ით) + reference-ები + voices.json. ერთხელ."""
    mark = D / "input.uploaded.json"
    vj = rj(VOICES_JSON)
    if not vj:
        raise SystemExit("ჯერ: refs-fetch + pick")
    sig = {"voices": [(v["name"], v["seed"]) for v in vj["voices"]]}
    if (m := rj(mark)) and m.get("sig") == sig:
        return m["ref"]
    up = WORK / "omni-nc-input"
    shutil.rmtree(up, ignore_errors=True)
    (up / "shards").mkdir(parents=True)
    (up / "refs").mkdir()
    for v in vj["voices"]:
        rows = [json.loads(l) for l in (CORPUS / "shards" / v["shard"]).read_text(encoding="utf-8").splitlines() if l.strip()]
        with (up / "shards" / v["shard"]).open("w", encoding="utf-8") as f:
            for r in rows:
                o = {"id": r["id"], "text": r["text"]}
                ot = omni_text(r)
                if ot != r["text"]:
                    o["omni_text"] = ot
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        shutil.copy2(D / "refs" / v["ref"], up / "refs" / v["ref"])
    (up / "voices.json").write_text(json.dumps(vj, ensure_ascii=False, indent=1), encoding="utf-8")
    (up / "LICENSE.txt").write_text("NON-COMMERCIAL. " + LICENSE["license_note"] + "\n", encoding="utf-8")
    ref = upload_dataset(up, INPUT_SLUG, "aivoice omni nc input (non-commercial)")
    wait_dataset_ready(ref, timeout=3600)
    time.sleep(60)
    wj(mark, {"ref": ref, "sig": sig, "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return ref


def part_state(part: str) -> dict:
    return rj(gen_dir(part) / "state.json", {}) or {"runs": []}


def cmd_gen_push(a):
    ref = ensure_input()
    vj = {v["name"]: v for v in rj(VOICES_JSON)["voices"]}
    done = defaultdict(list)
    for r in rows_from_tars(tars(a.part)).values():
        if not r.get("error"):
            done[r["voice"]].append(r["id"])
    st = part_state(a.part)
    run = len(st["runs"]) + 1
    voices = []
    for name in PARTS[a.part]:
        v = vj[name]
        n = sum(1 for l in (CORPUS / "shards" / v["shard"]).open(encoding="utf-8") if l.strip())
        todo = n - len(done[name])
        if todo <= 0:
            continue
        voices.append({"name": name, "ref": v["ref"], "ref_text": v["ref_text"], "seed": v["seed"], "speed": a.speed,
                       "shard": v["shard"], "done": done[name], "todo": todo})
    if not voices:
        print(f"ნაწილი {a.part}: ყველაფერი გაკეთებულია")
        return
    budget = int(a.hours * 3600)
    cfg = {"mode": "gen", "pip": PIP, "num_step": NUM_STEP, "run": run, "attempts": 2, "chunk": 250,
           "budget_s": budget, "voices": voices}
    print(f"ნაწილი {a.part}, გაშვება r{run}: {len(voices)} ხმა, {sum(v['todo'] for v in voices)} წინადადება, ლიმიტი {a.hours} სთ")
    if a.dry_run:
        return
    push_kernel(f"gen-{a.part}", cfg, min(12 * 3600 - 300, budget + 5400), [ref])
    st["runs"].append({"run": run, "pushed": time.strftime("%Y-%m-%dT%H:%M:%S"), "voices": len(voices),
                       "todo": sum(v["todo"] for v in voices), "budget_h": a.hours, "speed": a.speed})
    wj(gen_dir(a.part) / "state.json", st)


def cmd_gen_fetch(a):
    st = part_state(a.part)
    if not st["runs"]:
        raise SystemExit("ჯერ არ გაშვებულა")
    run = st["runs"][-1]
    d = gen_dir(a.part) / f"r{run['run']}"
    fetch(f"gen-{a.part}", d)
    s = rj(d / "status.json", {})
    run.update(fetched=time.strftime("%Y-%m-%dT%H:%M:%S"), wall_s=s.get("wall_s"), codes=s.get("codes"))
    wj(gen_dir(a.part) / "state.json", st)
    for v in s.get("voices", []):
        print(json.dumps(v, ensure_ascii=False))


def remaining() -> dict:
    vj = {v["name"]: v for v in rj(VOICES_JSON)["voices"]}
    have = defaultdict(int)
    for r in rows_from_tars(tars()).values():
        if not r.get("error"):
            have[r["voice"]] += 1
    out = {}
    for name, v in vj.items():
        n = sum(1 for l in (CORPUS / "shards" / v["shard"]).open(encoding="utf-8") if l.strip())
        out[name] = n - have[name]
    return out


def cmd_watch(a):
    """გენერაციის ნაწილების ლოდინი: დასრულებულს ჩამოტვირთავს; დაუმთავრებელს (ლიმიტი/შეცდომა) თავიდან უშვებს
    (მაქს. 3-ჯერ); ყველაფრის ბოლოს — build --tar."""
    while True:
        pending = False
        for part in PARTS:
            st = part_state(part)
            if not st["runs"]:
                continue
            run = st["runs"][-1]
            if run.get("fetched"):
                rem = sum(remaining()[n] for n in PARTS[part])
                if rem > 0 and len(st["runs"]) < 4:
                    print(time.strftime("%H:%M"), part, "დარჩა", rem, "— ხელახლა", flush=True)
                    ns = argparse.Namespace(part=part, hours=a.hours, speed=run.get("speed", 1.0), dry_run=False)
                    cmd_gen_push(ns)
                    pending = True
                continue
            s = kstatus(f"gen-{part}")
            print(time.strftime("%H:%M"), part, s, flush=True)
            if s in ("complete", "error", "cancel"):
                cmd_gen_fetch(argparse.Namespace(part=part))
                pending = True
            else:
                pending = True
        if not pending:
            break
        time.sleep(600)
    rem = remaining()
    print("დარჩენილი:", {k: v for k, v in rem.items() if v}, flush=True)
    cmd_build(argparse.Namespace(tar=True, dry_run=False))


# ---------------- build ----------------
def keep(r: dict, med_cps: float) -> str | None:
    if r.get("error") or not r.get("file"):
        return "no_audio"
    lim = 0.10 if r["letters"] >= 15 else 0.05
    if r["cer"] > lim:
        return "cer"
    if (r.get("clip") or 0) > 0.001:
        return "clipping"
    if (r.get("snr") or 0) < 20:
        return "snr"
    if not 1.0 <= r["dur"] <= 15:
        return "duration"
    if (r.get("max_gap_s") or 0) > 1.2:
        return "pause"
    if not (8 <= r["cps"] <= 22 and 0.7 * med_cps <= r["cps"] <= 1.4 * med_cps):
        return "rate"
    return None


def cmd_build(a):
    vj = {v["name"]: v for v in rj(VOICES_JSON)["voices"]}
    rows = rows_from_tars(tars())
    by = defaultdict(list)
    for r in rows.values():
        by[r.get("voice")].append(r)
    med = {}
    for name, rs in by.items():
        c = sorted(r["cps"] for r in rs if r.get("cps"))
        med[name] = c[len(c) // 2] if c else 14
    kept, dropped = [], defaultdict(int)
    per = defaultdict(lambda: {"clips": 0, "sec": 0.0, "dropped": 0, "gen": 0, "retried": 0, "cer_sum": 0.0})
    for r in rows.values():
        p = per[r.get("voice")]
        p["gen"] += 1
        p["retried"] += (r.get("attempts") or 1) > 1
        why = keep(r, med.get(r.get("voice"), 14))
        if why:
            dropped[why] += 1
            p["dropped"] += 1
            continue
        kept.append(r)
        p["clips"] += 1
        p["sec"] += r["dur"]
        p["cer_sum"] += r["cer"]
    out = cpath("datasets") / SLOT
    print(f"{len(rows)} გენერირებული → {len(kept)} დარჩა; ამოღებული: {dict(dropped)}")
    for name in sorted(per):
        p = per[name]
        print(f"  {name}: {p['clips']:5} ({p['sec'] / 3600:.2f} სთ), ამოღებული {p['dropped']}, ხელახლა {p['retried']}")
    if a.dry_run:
        return
    (out / "audio").mkdir(parents=True, exist_ok=True)
    have = {p.name for p in (out / "audio").glob("*.flac")}
    want = {f"{r['voice']}_{r['id']}.flac": r for r in kept}
    by_tar = defaultdict(list)
    for fn, r in want.items():
        if fn not in have:
            by_tar[r["_tar"]].append(r)
    for i, (t, rs) in enumerate(sorted(by_tar.items()), 1):
        names = {f"{r['voice']}/{r['file']}": f"{r['voice']}_{r['id']}.flac" for r in rs}
        with tarfile.open(t) as tf:
            for m in tf:
                if m.name in names:
                    (out / "audio" / names[m.name]).write_bytes(tf.extractfile(m).read())
        if i % 20 == 0:
            print(f"  ამოღება {i}/{len(by_tar)}", flush=True)
    for fn in have - set(want):
        (out / "audio" / fn).unlink()
    # metadata: ხმები ფიქსირებული რიგით (f01…f20, m01…m20) — Piper-ი ID-ებს გამოჩენის რიგით ანიჭებს
    order = [v["name"] for v in VOICES if v["name"] in vj]
    rank = {n: i for i, n in enumerate(order)}
    kept.sort(key=lambda r: (rank.get(r["voice"], 99), r["id"]))
    lines = [f"{r['voice']}_{r['id']}.flac|{r['voice']}|{r['text']}" for r in kept]
    (out / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    speakers = {}
    for n in order:
        p, v = per[n], vj[n]
        speakers[n] = {"minutes": round(p["sec"] / 60, 1), "f0_median": v.get("f0"), "clips": p["clips"],
                       "instruct": v["instruct"], "seed": v["seed"], "synthetic": True}
    wj(out / "speakers.json", speakers)
    hours = round(sum(p["sec"] for p in per.values()) / 3600, 2)
    stats = {"source": "OmniVoice (k2-fsa/OmniVoice, num_step 16) — სინთეზური ხმები აღწერით (voice design → clone prompt); "
                       "ტექსტები: CC0 (Common Voice 27) + ჩვენი შაბლონები (tools/omni_corpus.py)",
             **LICENSE, "slot": SLOT, "speakers": len(speakers), "clips": len(lines), "hours": hours,
             "generated": len(rows), "dropped": dict(dropped),
             "mean_cer": round(sum(r["cer"] for r in kept) / max(1, len(kept)), 4),
             "rules": {"cer": "≤ 0.10 (≥ 15 ასო), ≤ 0.05 (მოკლე)", "clip": "≤ 0.1%", "snr": "≥ 20 dB", "dur": "1–15 წმ",
                       "pause": "≤ 1.2 წმ", "rate": "8–22 ასო/წმ და ხმის მედიანის 0.7–1.4×"},
             "status": "მზადაა", "built": time.strftime("%Y-%m-%dT%H:%M:%S")}
    wj(out / "stats.json", stats)
    (out / "LICENSE").write_text(
        "Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)\n"
        "https://creativecommons.org/licenses/by-nc/4.0/\n\n"
        "This dataset was generated with OmniVoice (k2-fsa/OmniVoice). The OmniVoice code is Apache-2.0, but its\n"
        "pretrained weights are CC-BY-NC-4.0 (trained on Emilia) and its audio tokenizer is under the Boson Higgs\n"
        "Audio 2 Community License. Therefore this audio and any model trained on it may be used for\n"
        "NON-COMMERCIAL purposes only. Do not publish it on public model hubs.\n\n"
        "Texts: Mozilla Common Voice 27 (ka) sentences (CC0) and project templates.\n", encoding="utf-8")
    (out / "README.md").write_text(
        f"# {SLOT} — სინთეზური ქართული მეტყველება (OmniVoice) ⚠ არაკომერციული\n\n"
        f"- ხმები: {len(speakers)} (omni_f01…f20, omni_m01…m20), შექმნილი მხოლოდ აღწერით (OmniVoice voice design),\n"
        "  არცერთი ნამდვილი ადამიანის ხმა არ არის კლონირებული.\n"
        f"- ჩანაწერები: {len(lines)}, {hours} სთ; 22050 Hz FLAC, სიჩუმე მოჭრილი, ხმამაღლობა გათანაბრებული.\n"
        f"- ფილტრი: MMS (facebook/mms-1b-all, kat) CER, clipping, SNR, პაუზა, ტემპი — stats.json.\n"
        "- ლიცენზია: **CC BY-NC 4.0** — OmniVoice-ის წონები CC-BY-NC-ია (Emilia), ტოკენიზატორი Boson Higgs Audio 2\n"
        "  Community License-ით. ეს dataset და მასზე ნაწვრთნი ყველა მოდელი მხოლოდ არაკომერციულია; საჯარო\n"
        "  Hugging Face-ზე არ იტვირთება.\n", encoding="utf-8")
    write_selection(kept, speakers, stats)
    print(json.dumps({k: stats[k] for k in ("speakers", "clips", "hours", "mean_cer", "dropped")}, ensure_ascii=False))
    if a.tar:
        tar = cpath("gcp") / f"{SLOT}.tar"
        tmp = tar.with_suffix(".part")
        with tarfile.open(tmp, "w") as t:
            for name in ("metadata.csv", "speakers.json", "stats.json", "LICENSE", "README.md"):
                t.add(out / name, arcname=f"{SLOT}/{name}")
            for l in lines:
                f = l.split("|", 1)[0]
                t.add(out / "audio" / f, arcname=f"{SLOT}/audio/{f}")
        tmp.replace(tar)
        print(f"არქივი: {tar} ({tar.stat().st_size / 1e9:.2f} GB)", flush=True)


def write_selection(kept, speakers, stats):
    """base_ctl.py-სთვის (start → mark_launched, models()): შერჩევის ფაილი იმავე ფორმით, რაც select_best.py-ს."""
    p = ROOT / "data" / "selections" / f"{SLOT}.json"
    old = rj(p, {}) or {}
    sel = {"name": SLOT, "created": old.get("created") or time.strftime("%Y-%m-%dT%H:%M:%S"), "base": None,
           "source": "omnivoice", **LICENSE,
           "rules": stats["rules"],
           "stats": {"hours": stats["hours"], "clips": stats["clips"], "speakers": stats["speakers"],
                     "mean_cer": stats["mean_cer"]},
           "speakers": {n: {"minutes": s["minutes"], "f0": s["f0_median"], "clips": s["clips"]} for n, s in speakers.items()},
           "clips": [f"{r['voice']}_{r['id']}.flac" for r in kept]}
    for k in ("launched", "model", "dataset", "speaker_order"):
        if k in old:
            sel[k] = old[k]
    wj(p, sel)


def main():
    ap = argparse.ArgumentParser(prog="tools/omni_dataset.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("refs-push").add_argument("--seeds", type=int, default=3)
    for c in ("status", "refs-fetch"):
        sub.add_parser(c)
    sub.add_parser("pick")
    p = sub.add_parser("gen-push")
    p.add_argument("--part", choices=list(PARTS), required=True)
    p.add_argument("--hours", type=float, default=10.0, help="kernel-ის შიდა ლიმიტი (Kaggle-ის 12 სთ-მდე)")
    p.add_argument("--speed", type=float, default=1.0, help="clone-ში (ტემპი უკვე reference-შია, ref_speed 0.9)")
    p.add_argument("--dry-run", action="store_true")
    sub.add_parser("gen-fetch").add_argument("--part", choices=list(PARTS), required=True)
    sub.add_parser("watch").add_argument("--hours", type=float, default=10.0)
    p = sub.add_parser("build")
    p.add_argument("--tar", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    {"refs-push": cmd_refs_push, "status": cmd_status, "refs-fetch": cmd_refs_fetch, "pick": cmd_pick,
     "gen-push": cmd_gen_push, "gen-fetch": cmd_gen_fetch, "watch": cmd_watch, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
