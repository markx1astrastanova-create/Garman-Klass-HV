# Deploy GKHV to Streamlit Community Cloud

The existing React interface runs as a Streamlit custom component. Python calls
the existing backend functions directly. No Railway, Vercel, public API server,
or TradingView account is required. Anonymous TradingView access can be limited
or blocked upstream; hosting on Streamlit does not guarantee market-data access.

## Community Cloud

1. Sign in at https://share.streamlit.io/ with the GitHub account that owns this repository.
2. Create an app from `markx1astrastanova-create/Garman-Klass-HV`, branch `main`.
3. Set the main file path to `streamlit_app.py`.
4. Select Python 3.13 in Advanced settings, matching the tested local runtime.
5. Deploy. No secrets are required.

Root `requirements.txt` installs Python dependencies. The committed
`FRONTEND/streamlit_dist/` bundle means Community Cloud does not need Node.js.
Streamlit is not exposing the old `/api/*` REST endpoints: React sends requests
through Streamlit's component bridge. Existing standalone React/FastAPI use
still works with `VITE_API_URL`.

## Local run

```powershell
uv venv .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## After changing React

Rebuild and commit the generated bundle with the source changes:

```powershell
cd FRONTEND
npm ci
npm run build:streamlit
```

## Focused checks

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The frontend production build is `npm run build`. The Streamlit-specific build
is `npm run build:streamlit`, both run from `FRONTEND/`.
