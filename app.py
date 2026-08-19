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
# Pipeline
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

DOMAINS = ["auto"] + [
    "Artificial Intelligence", "Automotive", "Cybersecurity", "Food",
    "Environment", "Real Estate", "Entertainment",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def score_norm(score, lo=0.0, hi=5.0):
    return max(0.0, min(1.0, (score - lo) / (hi - lo)))


def chips_markdown(suggestions):
    if not suggestions:
        return None
    parts = "".join(
        f'<span class="chip">{html.escape(str(s))}</span>' for s, _ in suggestions
    )
    return f'<div class="chips"><span class="chip-hint">related</span>{parts}</div>'


LIGHT_CSS = """
<style>
#MainMenu {visibility: hidden;}
.chips {display:flex; flex-wrap:wrap; gap:.35rem; margin-top:.35rem;}
.chip {background:#ede9fe; color:#5b21b6; border-radius:999px; padding:.2rem .65rem;
       font-size:.78rem; font-weight:600;}
.chip-hint {font-size:.72rem; color:#94a3b8; font-weight:700; align-self:center;}
.score-wrap {display:flex; align-items:baseline; justify-content:space-between; margin-bottom:.25rem;}
</style>
"""


def render_results(kps, expanded, detected_domain, dt):
    st.markdown("### Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Domain", (detected_domain or "—").capitalize())
    c2.metric("Keyphrases", len(kps))
    c3.metric("Time", f"{dt:.1f}s")

    if not kps:
        st.info("No keyphrases found. Try a longer article.")
        return

    for kp, score in kps:
        with st.container(border=True):
            st.markdown(
                f'<div class="score-wrap"><span style="font-size:1.05rem;font-weight:700">'
                f'{html.escape(str(kp))}</span>'
                f'<span style="color:#64748b;font-size:.85rem">{float(score):.2f}</span></div>',
                unsafe_allow_html=True,
            )
            st.progress(score_norm(score))
            sugs = expanded.get(kp, [])
            cm = chips_markdown(sugs)
            if cm:
                st.markdown(cm, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Keyphrase Extractor", page_icon="🔑", layout="centered")
st.markdown(LIGHT_CSS, unsafe_allow_html=True)

st.title("🔑 Keyphrase Extractor")
st.caption("Hybrid extractive keyphrase extraction with contextual expansion — free, CPU-only.")

st.divider()

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# NOTE: "input_text" is a plain session-state variable, NOT a widget key —
# this lets us load a sample below without hitting Streamlit's "cannot modify
# a widget after it is instantiated" error.
typed = st.text_area(
    "Article text",
    height=220,
    placeholder="Paste a news article here (350–500 words works best)...",
    value=st.session_state.input_text,
)
st.session_state.input_text = typed

if SAMPLES:
    st.markdown("**Or load a sample:**")
    scols = st.columns(len(SAMPLES))
    for col, (sname, stext) in zip(scols, SAMPLES.items()):
        # Buttons don't persist selection, so this cannot re-trigger on rerun.
        if col.button(sname, use_container_width=True):
            st.session_state.input_text = stext
            st.rerun()

ctrl1, ctrl2 = st.columns([1, 2])
with ctrl1:
    domain = st.selectbox("Domain", DOMAINS, index=0)
with ctrl2:
    run = st.button("Extract Keyphrases", type="primary", use_container_width=True)

result_key = None
if run:
    text = st.session_state.input_text
    if not text or not text.strip():
        st.warning("Please paste some text first.")
        result_key = None
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
        result_key = True

if "result" in st.session_state and st.session_state.result:
    kps, expanded, detected_domain, dt = st.session_state.result
    render_results(kps, expanded, detected_domain, dt)

st.divider()
st.caption("CPU-only hybrid extraction · models cached after first run")
