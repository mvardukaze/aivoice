"""ხმის თვისებები ტემბრით: სქესი (ქალი/კაცი) და ასაკი (ახალგაზრდა/ზრდასრული) — Common Voice-ის მითითებებს არ ვენდობით.

    ./aivoice traits [--model base_ka_v3]
შედეგი: data/voice_traits_<model>.json — {spk: {gender, p_female, age, p_young, f0, label_gender, label_age, mismatch}}
(ხმის შემქმნელი: ფილტრი რეიტინგში, რუკაზე, მზა ნარევებში)

როგორ: თითო ხმას საბაზო მოდელში თავისი „ანაბეჭდი“ აქვს (ტემბრი, სიმაღლე, მანერა — embeddings.npy). ლოგისტიკური
რეგრესია ისწავლება მათზე, ვინც სქესი/ასაკი მიუთითა, და თითო ხმა ფასდება „გამოტოვებით“ (cross-validation) —
ანუ მისი საკუთარი მითითება მის შეფასებაში არ მონაწილეობს. სადაც შეფასება მითითებას ეწინააღმდეგება — mismatch.
სქესისთვის ანაბეჭდს ემატება ხმის სიმაღლე (F0, ნამდვილი ჩანაწერებიდან).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import path as cpath  # noqa: E402

YOUNG, ADULT = {"teens", "twenties"}, {"thirties", "fourties", "fifties", "sixties", "seventies", "eighties", "nineties"}


def labels(names: set) -> dict:
    csv.field_size_limit(10 ** 9)
    out = {}
    for kind in ("validated", "other"):
        with open(cpath("sources") / "cv27_ka" / f"{kind}.tsv", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                n = "cv_" + r["client_id"][:10]
                if n in names and n not in out and (r.get("gender") or r.get("age")):
                    out[n] = (r.get("gender") or "", r.get("age") or "")
    return out


def oof(X, y, mask):
    """mask-ით მონიშნულებზე — cross-validation-ის პროგნოზი; დანარჩენებზე — ყველაზე ნასწავლი მოდელი."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced"))
    p = np.zeros(len(X))
    idx = np.where(mask)[0]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    p[idx] = cross_val_predict(clf, X[idx], y[idx], cv=cv, method="predict_proba")[:, 1]
    clf.fit(X[idx], y[idx])
    rest = np.where(~mask)[0]
    if len(rest):
        p[rest] = clf.predict_proba(X[rest])[:, 1]
    acc = float(((p[idx] > 0.5) == y[idx]).mean())
    return p, acc


def main():
    ap = argparse.ArgumentParser(prog="./aivoice traits")
    ap.add_argument("--model", default="base_ka_v3")
    a = ap.parse_args()
    d = cpath("designer") if a.model == "base_ka" else cpath("designer").parent / f"designer_{a.model}"
    smap = json.loads((d / "config.json").read_text())["speaker_id_map"]
    emb = np.load(d / "embeddings.npy")
    names = sorted(smap, key=smap.get)
    ds = "base_ka_v1" if a.model == "base_ka" else a.model
    info = json.loads((cpath("datasets") / ds / "speakers.json").read_text(encoding="utf-8"))
    f0 = np.array([info.get(n, {}).get("f0_median") or np.nan for n in names], dtype=float)
    f0 = np.where(np.isnan(f0), np.nanmedian(f0), f0)
    E = emb[[smap[n] for n in names]]
    lab = labels(set(names))

    # სქესი: ანაბეჭდი + log(F0)
    Xg = np.hstack([E, np.log(f0)[:, None] * 10])
    yg = np.array([1 if lab.get(n, ("", ""))[0].startswith("female") else 0 for n in names])
    mg = np.array([lab.get(n, ("", ""))[0].startswith(("female", "male")) for n in names])
    pf, acc_g = oof(Xg, yg, mg)
    # ასაკი: ახალგაზრდა (თინეიჯერი/20-იანი) ↔ ზრდასრული (30+)
    ages = [lab.get(n, ("", ""))[1] for n in names]
    ya = np.array([1 if x in YOUNG else 0 for x in ages])
    ma = np.array([x in YOUNG | ADULT for x in ages])
    py, acc_a = oof(Xg, ya, ma)

    out = {}
    for i, n in enumerate(names):
        g, age = lab.get(n, ("", ""))
        gender = "female" if pf[i] >= 0.5 else "male"
        agecls = "young" if py[i] >= 0.5 else "adult"
        mism = []
        if g.startswith(("female", "male")) and not g.startswith(gender):
            mism.append("gender")
        if age in YOUNG | ADULT and ((age in YOUNG) != (agecls == "young")):
            mism.append("age")
        out[n] = {"gender": gender, "p_female": round(float(pf[i]), 3), "age": agecls, "p_young": round(float(py[i]), 3),
                  "f0": round(float(f0[i]), 1), "label_gender": g, "label_age": age, "mismatch": mism}
    res = {"model": a.model, "accuracy": {"gender": round(acc_g, 3), "age": round(acc_a, 3)},
           "labeled": {"gender": int(mg.sum()), "age": int(ma.sum())}, "speakers": out}
    p = ROOT / "data" / f"voice_traits_{a.model}.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    print(json.dumps({"accuracy": res["accuracy"], "labeled": res["labeled"],
                      "groups": Counter(f"{v['gender']}/{v['age']}" for v in out.values()),
                      "mismatch": Counter(m for v in out.values() for m in v["mismatch"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
