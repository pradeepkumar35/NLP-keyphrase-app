"""Keyphrase Extraction — Streamlit web app (CPU-only, deployable for free).

Deploys the CPU-only extractive pipeline (KeyBERT + YAKE + TextRank +
MultipartiteRank) + contextual expansion. No GPU required.
"""
import html
import os
import sys
import time

import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# One-time environment setup: ensure NLTK data is present.
# The spaCy model (en_core_web_sm) is installed via requirements.txt at build
# time, because Streamlit's runtime site-packages are read-only.
# ---------------------------------------------------------------------------
def _ensure_deps():
    import nltk
    for pkg in ["stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4",
                "averaged_perceptron_tagger", "universal_tagset"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    import spacy
    spacy.load("en_core_web_sm")


# ---------------------------------------------------------------------------
# Pipeline (identical logic to before; only the UI changed).
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

_EXTRACTIVE_DEFAULTS = dict(
    model_name=EMBEDDING_MODEL,
    use_gpu=False,
    top_n=10,
    redundancy_threshold=0.82,
    diversity_penalty=0.65,
    prioritize_named_entities=False,
    ngram_range=(1, 3),
    clean_boundaries=True,
    use_noun_chunks=True,
    boost_exact_matches=True,
    use_position_weight=True,
    use_tfidf_weight=True,
    use_ensemble=True,
    use_lemmatization=True,
    use_partial_matching=True,
    use_semantic_matching=True,
    use_enhanced_pos_filtering=True,
    use_title_lead_boost=True,
    method_weights={
        "keybert": 0.38,
        "multipartiterank": 0.27,
        "yake": 0.18,
        "textrank": 0.17,
    },
)


@st.cache_resource(show_spinner="Loading models (one-time, may take a minute)...")
def load_pipeline():
    import pipeline as P
    _ensure_deps()
    extractor = P.HybridExtractiveKeyphraseExtractor(**_EXTRACTIVE_DEFAULTS)
    expander = P.ContextualKeyphraseExpander(
        use_gpu=False,
        similarity_threshold=0.55,
        max_suggestions=5,
        use_phrase_quality_check=True,
        use_collocations=True,
        use_pos_patterns=True,
        use_keybert=True,
        keybert_diversity=0.7,
        model_name=EMBEDDING_MODEL,
    )
    return extractor, expander


# ---------------------------------------------------------------------------
# Sample articles
# ---------------------------------------------------------------------------
SAMPLES = {
    "AI in Healthcare": """Artificial intelligence continues to transform healthcare through numerous applications. Machine learning algorithms can predict patient outcomes by analyzing electronic health records. Computer vision systems examine medical images to detect early signs of disease. Natural language processing extracts valuable information from clinical notes. These AI-powered diagnostic tools assist physicians in making more accurate diagnoses and treatment recommendations.""",
    "Electric Vehicles": """Electric vehicles are transforming the automotive industry with rapid technological advancements. Battery technology continues to improve, extending driving ranges while reducing charging times. Regenerative braking systems recover energy during deceleration. Charging infrastructure is expanding globally, with fast-charging networks enabling long-distance travel. The transition to electric mobility represents the most significant shift in automotive technology in over a century.""",
    "Cybersecurity Threats": """Ransomware remains a dominant cyber threat, but attack methodologies are evolving. Supply chain attacks have increased as threat actors compromise trusted software distribution channels. Phishing campaigns use artificial intelligence to generate personalized content that evades security filters. Zero-day vulnerability exploitation continues to accelerate. These evolving threats require comprehensive, defense-in-depth security strategies.""",
}


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
DOMAINS = ["auto"] + [
    "Artificial Intelligence", "Automotive", "Cybersecurity", "Food",
    "Environment", "Real Estate", "Entertainment",
]

CSS = """
<style>
:root {
  --ink: #1e293b;
  --muted: #64748b;
  --primary: #6d28d9;
  --primary-2: #8b5cf6;
  --card-bg: #ffffff;
  --border: #e2e8f0;
  --chip-bg: #ede9fe;
  --chip-ink: #5b21b6;
}

.stApp {
  background: linear-gradient(180deg, #f5f3ff 0%, #f8fafc 40%, #ffffff 100%);
  font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
}

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 860px; }

.hero {
  background: linear-gradient(135deg, #6d28d9 0%, #7c3aed 45%, #2563eb 100%);
  color: #fff;
  border-radius: 20px;
  padding: 2.2rem 2.4rem;
  margin-bottom: 1.4rem;
  box-shadow: 0 14px 34px rgba(109,40,217,.25);
}
.hero h1 { font-size: 2.1rem; font-weight: 800; margin: 0 0 .3rem; letter-spacing: -.5px; }
.hero p  { font-size: 1rem; margin: 0; opacity: .92; line-height: 1.5; }
.hero .kbd {
  display:inline-block; background:rgba(255,255,255,.18); padding:.15rem .55rem;
  border-radius:8px; font-size:.78rem; margin-top:.5rem;
}

.section-label { font-weight: 700; color: var(--ink); margin: 1.2rem 0 .4rem; font-size: .95rem; letter-spacing: .3px; }

.kp-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: .9rem 1.1rem;
  margin-bottom: .7rem;
  box-shadow: 0 2px 8px rgba(30,41,59,.05);
}
.kp-top { display:flex; align-items:baseline; justify-content:space-between; gap:.6rem; margin-bottom:.45rem; }
.kp-phrase { font-size: 1.12rem; font-weight: 700; color: var(--ink); }
.kp-score { font-size:.8rem; color: var(--muted); font-weight:600; white-space:nowrap; }
.kp-bar { height: 6px; background:#eef2ff; border-radius:999px; overflow:hidden; margin-bottom:.55rem; }
.kp-bar > div { height:100%; border-radius:999px; background:linear-gradient(90deg,var(--primary-2),var(--primary)); }
.chips { display:flex; flex-wrap:wrap; gap:.35rem; }
.chip {
  background: var(--chip-bg); color: var(--chip-ink); border-radius:999px;
  padding:.22rem .6rem; font-size:.78rem; font-weight:600;
}
.chip-label { font-size:.72rem; color:var(--muted); font-weight:700; margin-right:.25rem; align-self:center; }

.metric-card {
  background: var(--card-bg); border:1px solid var(--border); border-radius:14px;
  padding: .8rem 1rem; text-align:center; box-shadow:0 2px 8px rgba(30,41,59,.05);
}
.metric-card .v { font-size:1.5rem; font-weight:800; color:var(--primary); line-height:1.1; }
.metric-card .l { font-size:.72rem; color:var(--muted); font-weight:700; text-transform:uppercase; letter-spacing:.4px; }

.empty-state { text-align:center; color:var(--muted); padding:2rem 1rem; }
.empty-state .big { font-size:2.4rem; }
</style>
"""


def hero():
    st.markdown(
        f"""
        <div class="hero">
          <h1>🔑 Keyphrase Extractor</h1>
          <p>Paste a news article and instantly get its most important keyphrases —
             extracted with a hybrid of KeyBERT, YAKE, TextRank &amp; MultipartiteRank,
             then expanded with related concepts.</p>
          <span class="kbd">free · CPU-only · no signup</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, extra=""):
    st.markdown(
        f'<div class="metric-card"><div class="v">{value}</div>'
        f'<div class="l">{label}</div>{extra}</div>',
        unsafe_allow_html=True,
    )


def render_results(kps, expanded, detected_domain, dt):
    st.markdown('<div class="section-label">RESULTS</div>', unsafe_allow_html=True)

    c = st.columns(3)
    with c[0]:
        metric_card("Domain", detected_domain.capitalize() if detected_domain else "—")
    with c[1]:
        metric_card("Keyphrases", len(kps))
    with c[2]:
        metric_card("Time", f"{dt:.1f}s")

    st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)

    if not kps:
        st.markdown(
            '<div class="empty-state"><div class="big">🤷</div>No keyphrases found. '
            "Try a longer article.</div>",
            unsafe_allow_html=True,
        )
        return

    for kp, score in kps:
        phrase = html.escape(str(kp))
        score_pct = max(2.0, min(100.0, score * 100.0 / 5.0))
        score_txt = f"{float(score):.2f}"
        sugs = expanded.get(kp, [])
        chips_html = ""
        if sugs:
            chips = "".join(
                f'<span class="chip">{html.escape(str(s))}</span>' for s, _ in sugs
            )
            chips_html = (
                f'<div class="chips"><span class="chip-label">related</span>{chips}</div>'
            )
        st.markdown(
            f"""
            <div class="kp-card">
              <div class="kp-top"><span class="kp-phrase">{phrase}</span>
                <span class="kp-score">score {score_txt}</span></div>
              <div class="kp-bar"><div style="width:{score_pct:.0f}%"></div></div>
              {chips_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Keyphrase Extractor", page_icon="🔑", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)
hero()

# Input area
st.markdown('<div class="section-label">INPUT</div>', unsafe_allow_html=True)

text = st.text_area(
    "Article text",
    height=220,
    placeholder="Paste a news article here (350–500 words works best)...",
    label_visibility="collapsed",
)

if SAMPLES:
    sample_name = st.pills(
        "Try a sample", options=list(SAMPLES.keys()), label_visibility="collapsed"
    )
    if sample_name:
        text = SAMPLES[sample_name]

col1, col2 = st.columns([1, 2])
with col1:
    domain = st.selectbox("Domain", DOMAINS, index=0)
with col2:
    run = st.button("✨ Extract Keyphrases", type="primary", use_container_width=True)

# Cached results across widget interactions
if "result" not in st.session_state:
    st.session_state.result = None

if run:
    if not text or not text.strip():
        st.warning("Please paste some text first.")
        st.session_state.result = None
    else:
        with st.spinner("Extracting keyphrases... (first run loads the model)"):
            t0 = time.time()
            extractor, expander = load_pipeline()
            domain_clean = None if domain == "auto" else domain
            kps = extractor.extract_keyphrases_with_scores(text)

            detected_domain = domain_clean
            if detected_domain is None:
                try:
                    import pipeline as P
                    detected_domain = P._detect_domain_from_text(text)
                except Exception:
                    detected_domain = "general"

            expanded = expander.expand_keyphrases(
                kps, text, domain=detected_domain,
                min_quality_score=0.68, num_suggestions=5, use_curated=True,
            )
            dt = time.time() - t0
        st.session_state.result = (kps, expanded, detected_domain, dt)

if st.session_state.result is not None:
    kps, expanded, detected_domain, dt = st.session_state.result
    render_results(kps, expanded, detected_domain, dt)

st.markdown("---")
st.caption("CPU-only hybrid extraction · free hosted · models cached after first run")
