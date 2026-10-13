import json, urllib.request, librosa, torch
from transformers import AutoProcessor, Wav2Vec2ForCTC
API = "http://127.0.0.1:8200/api/designer"
designs = json.loads(urllib.request.urlopen(API + "/designs").read())
D = next(d for d in designs if d["name"].startswith("ნეიტრალური D"))
A = next(d for d in designs if d["name"].startswith("ნეიტრალური A"))
names = list(dict.fromkeys(list(D["weights"]) + list(A["weights"])))
TEXTS = ["გამარჯობა! დღეს მშვენიერი ამინდია, არა?", "გამარჯობათ, როგორ ბრძანდებით?"]
torch.set_num_threads(8)
proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="kat")
m = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="kat", ignore_mismatched_sizes=True).eval()
def hear(weights, text):
    req = urllib.request.Request(API + "/synth", method="POST", headers={"Content-Type": "application/json"},
                                 data=json.dumps({"weights": weights, "text": text}).encode())
    url = json.loads(urllib.request.urlopen(req, timeout=180).read())["url"]
    a, _ = librosa.load("data/cache/designer/" + url.rsplit("/", 1)[-1], sr=16000)
    with torch.no_grad():
        return proc.batch_decode(m(**proc(a, sampling_rate=16000, return_tensors="pt")).logits.argmax(-1))[0]
for label, w in [("mix A", A["weights"]), ("mix D", D["weights"])] + [(n, {n: 1}) for n in names]:
    hs = [hear(w, t) for t in TEXTS]
    print(label, "| in A" if label in A["weights"] else "", "| in D" if label in D["weights"] else "",
          "| რჯ:", sum("რჯ" in h for h in hs), "/2 |", " / ".join(h[:30] for h in hs), flush=True)
