"""„ღ“-ს გამოთქმის შეფასება (GOP) ბრმა ტესტის ჩანაწერებზე — MMS (kat) CTC-ით.

ამომცნობი თავისით „ღ“-ს არ წერს (tools/gh_test.py, 2026-10-12: 0 / 20), ამიტომ ვადარებთ, რამდენად ალბათურია
ჩანაწერი სხვადასხვა ტექსტით: „ოდესღაც ...“ / „ოდესხაც ...“ / „ოდესაც ...“ / „ოდესრაც ...“.
ქულა = logP(ღ) − max(logP(დანარჩენი)); > 0 — მოდელის აზრით „ღ“ ისმის.
შედეგი ემატება data/ghtest/results.json-ს (mms_gop, mms_gh).
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GHT = ROOT / "data" / "ghtest"
REST = " ძალიან დიდი ხნის წინ"
VARIANTS = {"ღ": "ოდესღაც", "ხ": "ოდესხაც", "∅": "ოდესაც", "რ": "ოდესრაც"}


def main():
    import librosa
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    torch.set_num_threads(8)
    res = json.loads((GHT / "results.json").read_text(encoding="utf-8"))
    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="kat")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="kat", ignore_mismatched_sizes=True).eval()
    tok = proc.tokenizer
    targets = {k: tok(w + REST).input_ids for k, w in VARIANTS.items()}
    for k, ids in targets.items():
        assert tok.unk_token_id not in ids, f"{k}: უცნობი სიმბოლო ლექსიკონში"
    for r in res["rows"]:
        a, _ = librosa.load(str(GHT / "clips" / f"{r['id']:02d}.wav"), sr=16000)
        with torch.no_grad():
            logits = model(**proc(a, sampling_rate=16000, return_tensors="pt")).logits
        lp = torch.log_softmax(logits, -1).transpose(0, 1)  # T, 1, V
        T = lp.shape[0]
        scores = {}
        for k, ids in targets.items():
            loss = torch.nn.functional.ctc_loss(lp, torch.tensor([ids]), torch.tensor([T]), torch.tensor([len(ids)]),
                                                blank=tok.pad_token_id, reduction="sum", zero_infinity=True)
            scores[k] = -float(loss)
        r["mms_variants"] = {k: round(v, 2) for k, v in scores.items()}
        r["mms_gop"] = round(scores["ღ"] - max(v for k, v in scores.items() if k != "ღ"), 2)
        r["mms_gh"] = r["mms_gop"] > 0
        print(r["id"], r["speaker"], r["mms_gop"], r["mms_variants"], flush=True)
    (GHT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
