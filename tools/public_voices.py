"""სინთეზური ხმების „საჯარო“ ხმის შემქმნელი: ნამდვილი ხმების ნაცვლად — მხოლოდ ნარევები.

    ./aivoice public-voices [--model base_ka_v3] [--per-bin 10] [--seed 1]
შედეგი: <data_root>/designer_<model>_public/ — იგივე designer.onnx (ის ხმას g ვექტორით იღებს — ცხრილს არ შეიცავს),
        embeddings.npy მხოლოდ სინთეზური ხმებით, config.json (speaker_id_map: syn…), speakers.json (სიმაღლე), meta.json.
ხმის შემქმნელში ჩნდება ცალკე მოდელად („v3 საჯარო“). ნარევების შემადგენლობა (ნამდვილი ხმების ჰეშები) ინახება მხოლოდ
mixes.json-ში (არ ქვეყნდება).

ხმა = 4–6 ნამდვილი ხმის ნარევი სიმაღლის ზოლიდან (85–270 Hz, 8 ზოლი); წესები:
  • ყველაზე დიდი წილი ≤ 35%;  • ყველაზე ახლო ნამდვილ ხმასთან კოსინუსური მსგავსება < 0.8 (არცერთს არ ჰგავს);
  • სინთეზურ ხმებს შორის მსგავსება < 0.95 (ერთმანეთს არ იმეორებს);
  • ზოლებში < 150 Hz — მხოლოდ კაცის, > 170 Hz — მხოლოდ ქალის ტემბრის ხმები (voice_traits); შუაში — ორივე;
  • მხოლოდ კარგად მოლაპარაკეები (MMS რეიტინგის უკეთესი ნახევარი).
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import path as cpath  # noqa: E402

BINS = [95, 115, 135, 155, 175, 195, 220, 250]  # სამიზნე სიმაღლე (Hz)
MAX_W, MAX_REAL_SIM, MAX_SYN_SIM = 0.35, 0.8, 0.95


def main():
    ap = argparse.ArgumentParser(prog="./aivoice public-voices")
    ap.add_argument("--model", default="base_ka_v3")
    ap.add_argument("--per-bin", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    src = cpath("designer").parent / f"designer_{a.model}"
    out = cpath("designer").parent / f"designer_{a.model}_public"
    cfg = json.loads((src / "config.json").read_text())
    smap = cfg["speaker_id_map"]
    emb = np.load(src / "embeddings.npy")
    info = json.loads((cpath("datasets") / a.model / "speakers.json").read_text(encoding="utf-8"))
    traits = (json.loads((ROOT / "data" / f"voice_traits_{a.model}.json").read_text(encoding="utf-8"))
              if (ROOT / "data" / f"voice_traits_{a.model}.json").exists() else {}).get("speakers", {})
    rank_f = ROOT / "data" / f"mms_rank_{a.model}.json"
    rank = {v["name"]: v["cer"] for v in (json.loads(rank_f.read_text(encoding="utf-8")) if rank_f.exists() else {}).values()
            if v.get("kind") == "speaker"}

    names = [n for n in smap if (info.get(n) or {}).get("f0_median")]
    if rank:
        med = sorted(rank.get(n, 1) for n in names)[len(names) // 2]
        names = [n for n in names if rank.get(n, 1) <= med]
    E = {n: emb[smap[n]] for n in smap}
    U_real = np.stack([E[n] / np.linalg.norm(E[n]) for n in smap])
    f0 = {n: info[n]["f0_median"] for n in names}
    rng = np.random.default_rng(a.seed)

    mixes, U_syn = [], []
    for target in BINS:
        def gender_ok(n):
            g = traits.get(n, {}).get("gender")
            return g is None or target <= 150 and g == "male" or target >= 170 and g == "female" or 150 < target < 170
        made, tries, width = 0, 0, 12
        while made < a.per_bin and tries < 4000:
            tries += 1
            if tries % 400 == 0:
                width += 6  # ზოლში ხმები ცოტაა — ვაფართოებთ
            pool = [n for n in names if abs(f0[n] - target) <= width and gender_ok(n)]
            if len(pool) < 6:
                width += 4
                continue
            k = int(rng.integers(4, 7))
            pick = list(rng.choice(pool, k, replace=False))
            w = rng.dirichlet(np.ones(k) * 2.0)
            for _ in range(20):  # ყველაზე დიდი წილი ≤ 35%
                w = np.minimum(w, MAX_W)
                w = w / w.sum()
            if w.max() > MAX_W + 1e-6:
                continue
            g = sum(wi * E[n] for wi, n in zip(w, pick))
            u = g / np.linalg.norm(g)
            real_sim = float((U_real @ u).max())
            if real_sim >= MAX_REAL_SIM:
                continue
            if U_syn and float((np.stack(U_syn) @ u).max()) >= MAX_SYN_SIM:
                continue
            made += 1
            est = float(sum(wi * f0[n] for wi, n in zip(w, pick)))
            mixes.append({"name": f"syn{target:03d}_{made:02d}", "bin": target, "f0": round(est, 1), "vector": g,
                          "weights": {n: round(float(wi), 3) for n, wi in zip(pick, w)}, "nearest_real_sim": round(real_sim, 3)})
            U_syn.append(u)
        print(f"{target} Hz: {made} ხმა ({tries} ცდა, ზოლი ±{width})", flush=True)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(src / "designer.onnx", out / "designer.onnx")
    np.save(out / "embeddings.npy", np.stack([m["vector"] for m in mixes]).astype(np.float32))
    pub_cfg = {**cfg, "speaker_id_map": {m["name"]: i for i, m in enumerate(mixes)}, "num_speakers": len(mixes)}
    (out / "config.json").write_text(json.dumps(pub_cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "speakers.json").write_text(json.dumps({m["name"]: {"f0_median": m["f0"], "minutes": None} for m in mixes},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    meta = json.loads((src / "meta.json").read_text())
    meta.update(public=True, synthetic_voices=len(mixes), source_model=a.model, exported=time.strftime("%Y-%m-%dT%H:%M:%S"),
                speakers=len(mixes))
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "mixes.json").write_text(json.dumps([{k: v for k, v in m.items() if k != "vector"} for m in mixes],
                                               ensure_ascii=False, indent=1), encoding="utf-8")
    sims = [m["nearest_real_sim"] for m in mixes]
    print(json.dumps({"voices": len(mixes), "out": str(out), "nearest_real_sim_max": max(sims),
                      "nearest_real_sim_mean": round(sum(sims) / len(sims), 3)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
