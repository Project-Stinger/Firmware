# ML Personalization Web UI (MVP)

This is a **local** prototype of the end-user website flow:

- Pull `ml_log.bin` from the blaster over **Web Serial** (no extra apps).
- Run the training pipeline locally on your machine (FastAPI backend).
- Show per-shot prediction plots for **both** LR + MLP.
- Upload both model files back to the blaster over Web Serial (**no UF2**).

## Requirements

- Chrome / Edge (Web Serial). Safari/Firefox won’t work for Web Serial.
- Python venv with deps installed.

## Run

```bash
cd /Users/stan/Documents/GitHub/Firmware
source .venv/bin/activate
python -m pip install -r python/requirements.txt
python -m pip install -r ml_web/server/requirements.txt

python -m ml_web.server.app
```

Then open:

- `http://127.0.0.1:8000/`

## Notes

- The UI assumes the firmware supports `MLDUMP` and `MLMODEL_PUT_LR/MLMODEL_PUT_MLP`.
- The blaster enters a safe USB service mode when the CDC port is opened (DTR asserted) and exits when disconnected.
- The backend writes temporary job artifacts under `ml_web/_jobs/` (gitignored).
