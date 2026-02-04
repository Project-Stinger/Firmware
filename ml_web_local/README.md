# `ml_web_local/` (browser-only trainer)

This is an **experimental** “record → train → upload weights” flow that runs fully in your browser:

- Pulls `MLDUMP` over Web Serial
- Trains **both** LR + MLP locally (no Python / no server)
- Uploads weights back via `MLMODEL_PUT_LR` / `MLMODEL_PUT_MLP` (no UF2)

## Run

Web Serial requires a secure context (`https://` or `http://localhost`). Serve this folder locally:

```sh
cd /Users/stan/Documents/GitHub/Firmware
python3 -m http.server 8001 --directory ml_web_local
```

Open in Chrome/Edge:

- `http://127.0.0.1:8001/`

## Notes

- This page only talks to the blaster you select in the Web Serial picker.
- Use “Download debug trace” when debugging serial/protocol issues.

