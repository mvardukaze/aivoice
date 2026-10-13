# ai_voices — Piper-ის ქართული ხმების ბიბლიოთეკა

მიზანი: წინასწარ გაწვრთნილი, **ფიქსირებული** ქართული ხმების ნაკრები Piper-ისთვის
(კაცი / ქალი / მოხუცი / ბავშვი). დუბლაჟისას თითო ადამიანს ერთი ხმა მიეკუთვნება და
მხოლოდ ემოცია იცვლება — ტემბრი არასდროს (videoka-ს DSP-ტემბრის პრობლემის გამო).

ვებ-პანელი: **https://aivoice.nikora.dev**.

## სტრუქტურა

```
app/            FastAPI (main.py) + static/ (index.html, app.js, style.css)
voices/         catalog.json — ხმების ცნობარი; <voice>/ — .onnx + .onnx.json
datasets/       <name>/metadata.csv + wav/ (+ stats.json)
training/       .venv (torch CUDA + lightning), checkpoints/, runs/<name>/status.json
tools/          dataset-ის და წვრთნის სკრიპტები
data/           roadmap.json (ეტაპები პანელზე), cache/ (სინთეზის wav-ები)
```

## გაშვება

| რა | სად |
|---|---|
| სერვისი | `systemd: aivoice.service` → uvicorn `127.0.0.1:8200` |
| venv | `.venv` (Python 3.12, fastapi, piper-tts) |
| ვებ-სერვერი | OpenLiteSpeed, vhost `aivoice` → `conf/vhosts/aivoice/vhconf.conf` |
| ლოგები | `journalctl -u aivoice`, `/usr/local/lsws/logs/aivoice.*.log` |

```bash
sudo systemctl restart aivoice      # კოდის ცვლილების შემდეგ
```

## ახალი ხმის დამატება

1. `.onnx` და `.onnx.json` → `voices/<dir>/`
2. `voices/catalog.json`-ში ჩანაწერს `status: "ready"` და `model: "<dir>/<file>.onnx"`
3. `sudo systemctl restart aivoice`

## ემოციები

`app/main.py` → `EMOTIONS` (პრესეტები) და `EMOTION_LIMITS` (დაშვებული დიაპაზონი).
იცვლება მხოლოდ Piper-ის `length_scale`, `noise_scale`, `noise_w_scale` და `volume`.
