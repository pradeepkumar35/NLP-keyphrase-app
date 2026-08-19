"""Keyphrase Extraction — Streamlit web app (CPU-only, deployable for free).

Deploys the CPU-only extractive pipeline (KeyBERT + YAKE + TextRank +
MultipartiteRank) + contextual expansion. No GPU required.
"""
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
    # Model is pre-installed; loading verifies it. Raises a clear error if not.
    import spacy
    spacy.load("en_core_web_sm")


# ---------------------------------------------------------------------------
# Load pipeline once (cached across sessions).
# EMBEDDING_MODEL: switch to "sentence-transformers/all-MiniLM-L6-v2" if the
# Streamlit free tier (1 GB RAM) runs out of memory with mpnet.
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
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Keyphrase Extraction", page_icon="🔑", layout="centered")

st.title("🔑 Keyphrase Extraction")
st.caption("Hybrid extractive keyphrase extraction (KeyBERT + YAKE + TextRank + MultipartiteRank) with contextual expansion.")

DOMAINS = ["auto"] + [
    "Artificial Intelligence", "Automotive", "Cybersecurity", "Food",
    "Environment", "Real Estate", "Entertainment",
]

text = st.text_area(
    "Paste a news article (350–500 words works best)",
    height=260,
    placeholder="Paste your article here...",
)

col1, col2 = st.columns([1, 3])
with col1:
    domain = st.selectbox("Domain", DOMAINS, index=0)
with col2:
    run = st.button("Extract Keyphrases", type="primary", use_container_width=True)

if run:
    if not text or not text.strip():
        st.warning("Please paste some text first.")
    else:
        with st.spinner("Extracting keyphrases... (first run loads models)"):
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

        st.success(f"Detected domain: **{detected_domain}**  ·  {len(kps)} keyphrases in {dt:.1f}s")

        for kp, score in kps:
            st.markdown(f"**{kp}**  ·  `{score:.3f}`")
            sugs = expanded.get(kp, [])
            if sugs:
                sugs_str = ", ".join([f"*{s}*" for s, _ in sugs])
                st.caption(f"↳ related: {sugs_str}")
            st.divider()

st.markdown("---")
st.caption("CPU-only · free hosted · models cached on first run")
