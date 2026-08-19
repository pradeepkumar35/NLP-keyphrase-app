# Keyphrase Extraction — Streamlit App

A free, CPU-only keyphrase extraction web app. Deploys the extractive pipeline
(KeyBERT + YAKE + TextRank + MultipartiteRank) with contextual expansion
(all-mpnet-base-v2). No GPU required.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (free, no credit card)
1. Push this repo to GitHub (already done for `pradeepkumar35/NLP-keyphrase-app`).
2. Go to https://share.streamlit.io → **Sign in with GitHub**.
3. **New app** → select repo `NLP-keyphrase-app` → branch `main` → main file `app.py` → **Deploy**.
4. Open the generated URL.

Notes:
- Free tier: 1 GB RAM, apps sleep after ~30–60 min idle and cold-start on the next visit (~1–3 min). That's expected.
- `packages.txt` installs `git` so `pke` can be pip-installed from GitHub.
- If the app runs out of memory with mpnet, switch `EMBEDDING_MODEL` in `app.py` to
  `sentence-transformers/all-MiniLM-L6-v2` (smaller, slightly lower quality).
