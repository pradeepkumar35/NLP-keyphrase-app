import torch
import numpy as np
import spacy
import pke
import nltk
import re
import string
import math
import os
import pickle
from collections import Counter
from typing import List, Dict, Tuple, Set, Optional, Union, Any
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('stopwords')
    nltk.data.find('punkt')
    nltk.data.find('wordnet')
    nltk.data.find('omw-1.4')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

try:
    import nltk.corpus.reader.wordnet
    from nltk.corpus import wordnet as wn

    if not hasattr(wn, '_morphy') or not hasattr(wn, 'morphy'):
        nltk.download('wordnet')
        nltk.download('omw-1.4')

        if not hasattr(wn, '_morphy') or not hasattr(wn, 'morphy'):
            from nltk.stem.wordnet import WordNetLemmatizer
            original_init = WordNetLemmatizer.__init__

            def patched_init(self):
                self.lemmatize = lambda word, pos='n': word

            WordNetLemmatizer.__init__ = patched_init
            print("Applied NLTK 3.9 WordNet lemmatizer patch")
except Exception as e:
    print(f"Warning: NLTK WordNet initialization error: {e}")
    print("Trying alternative approach...")
    try:
        import sys
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk==3.9b1", "--force-reinstall"])
        print("Installed NLTK 3.9b1 as a workaround")

        import importlib
        importlib.reload(nltk)
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    except Exception as e2:
        print(f"Warning: Could not fix NLTK WordNet issue: {e2}")
        print("Lemmatization will be disabled")

class HybridExtractiveKeyphraseExtractor:
    

    DOMAIN_STOPWORDS = {
        'general': ['etc', 'e.g', 'i.e'],
        'tech': ['technology', 'technologies'],
        'business': ['business', 'company'],
        'science': ['study', 'studies'],
        'news': ['report', 'reported', 'according']
    }

    BOUNDARY_WORDS = {
        'the', 'a', 'an', 'this', 'that', 'these', 'those', 'their', 'our', 'your',
        'its', 'his', 'her', 'they', 'them', 'which', 'what', 'who', 'whom', 'whose'
    }

    def __init__(
    self,
    model_name: str = "all-mpnet-base-v2",
    use_gpu: bool = True,
    top_n: int = 20,
    language: str = "en",
    alpha: float = 1.1,
    threshold: float = 0.74,
    method: str = 'mmr',
    min_df: int = 1,
    redundancy_threshold: float = 0.80,
    diversity_penalty: float = 0.5,
    prioritize_named_entities: bool = False,
    ngram_range: Tuple[int, int] = (1, 3),
    clean_boundaries: bool = True,
    use_noun_chunks: bool = True,
    boost_exact_matches: bool = True,
    use_position_weight: bool = True,
    use_tfidf_weight: bool = True,
    use_ensemble: bool = True,
    use_lemmatization: bool = True,
    use_partial_matching: bool = True,
    use_semantic_matching: bool = True,
    use_enhanced_pos_filtering: bool = True,
    use_title_lead_boost: bool = True,
    method_weights: Dict[str, float] = None,
    idf_corpus_path: str = None
    ):
        
        self.top_n = top_n
        self.language = language
        self.alpha = alpha
        self.threshold = threshold
        self.method = method
        self.min_df = min_df
        self.redundancy_threshold = redundancy_threshold
        self.diversity_penalty = diversity_penalty
        self.model_name = model_name
        self.prioritize_named_entities = prioritize_named_entities
        self.ngram_range = ngram_range
        self.clean_boundaries = clean_boundaries
        self.use_noun_chunks = use_noun_chunks
        self.boost_exact_matches = boost_exact_matches
        self.use_position_weight = use_position_weight
        self.use_tfidf_weight = use_tfidf_weight
        self.use_ensemble = use_ensemble
        self.use_enhanced_pos_filtering = use_enhanced_pos_filtering
        self.use_title_lead_boost = use_title_lead_boost
        self.use_lemmatization = use_lemmatization
        self.use_partial_matching = use_partial_matching
        self.use_semantic_matching = use_semantic_matching

        self.lemmatizer = WordNetLemmatizer()

        if method_weights is None:
            self.method_weights = {
                'keybert': 0.50,
                'multipartiterank': 0.25,
                'textrank': 0.15,
                'yake': 0.10
            }
        else:
            self.method_weights = method_weights
        self.tfidf_vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        stop_words='english',
        max_features=10000,
        min_df=1,
        norm='l2',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
        )
        self.initialize_idf_corpus(idf_corpus_path)

        self.word_tfidf_cache = {}

        try:
            try:
                from nltk.corpus import stopwords
                self.all_stopwords = set(stopwords.words('english'))
            except (ImportError, ModuleNotFoundError):
                nltk.download('stopwords')
                self.all_stopwords = set([
                    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
                    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
                    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
                    'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
                    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
                    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                    's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
                ])
        except Exception as e:
            print(f"Warning: Error loading stopwords: {e}")
            self.all_stopwords = set(['a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'when', 'where', 'how', 'why', 'who'])

        for domain, words in self.DOMAIN_STOPWORDS.items():
            self.all_stopwords.update(words)

        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        print(f"Using device: {self.device}")

        print(f"Loading SentenceTransformer model: {model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer(model_name)
            self.sentence_model.to(self.device)
        except Exception as e:
            print(f"Error loading SentenceTransformer: {str(e)}")
            raise

        print("Loading KeyBERT")
        try:
            from keybert import KeyBERT
            self.kw_model = KeyBERT(model=self.sentence_model)
        except Exception as e:
            print(f"Error loading KeyBERT: {str(e)}")
            raise

        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Downloading spaCy model...")
                import subprocess
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")
        except ImportError:
            print("spaCy not available, using simplified document classification")
            self.nlp = None

        print("Advanced Hybrid Extractive Keyphrase Extractor initialized with ensemble methods")

    def post_process_keyphrases(self, keyphrases: List[Tuple[str, float]], text: str) -> List[Tuple[str, float]]:
        
        processed_keyphrases = []

        prefixes_to_remove = ['the ', 'a ', 'an ', 'this ', 'that ', 'these ', 'those ']

        for kp, score in keyphrases:
            if self.clean_boundaries:
                kp = self.clean_phrase_boundaries(kp)

            for prefix in prefixes_to_remove:
                if kp.lower().startswith(prefix):
                    kp = kp[len(prefix):]
                    break

            text_lower = text.lower()
            kp_lower = kp.lower()

            if kp_lower in text_lower:
                start_idx = text_lower.find(kp_lower)
                exact_form = text[start_idx:start_idx+len(kp_lower)]

                if exact_form[0].isupper():
                    kp = exact_form

            if kp and len(kp.split()) <= self.ngram_range[1]:
                processed_keyphrases.append((kp, score))

        if self.use_enhanced_pos_filtering:
            processed_keyphrases = self.filter_by_pos_patterns(processed_keyphrases)

        return processed_keyphrases

    def filter_by_pos_patterns(self, keyphrases: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        
        if not self.use_enhanced_pos_filtering or not self.nlp:
            return keyphrases

        filtered_keyphrases = []

        acceptable_patterns = [
            (['NOUN'], 0.95),
            (['PROPN'], 1.0),

            (['ADJ', 'NOUN'], 1.1),
            (['NOUN', 'NOUN'], 1.05),
            (['PROPN', 'PROPN'], 1.1),
            (['PROPN', 'NOUN'], 1.05),
            (['NOUN', 'PROPN'], 1.05),

            (['ADJ', 'NOUN', 'NOUN'], 1.15),
            (['ADJ', 'ADJ', 'NOUN'], 1.1),
            (['NOUN', 'NOUN', 'NOUN'], 1.05),
            (['PROPN', 'PROPN', 'PROPN'], 1.1),
            (['NOUN', 'PROPN', 'PROPN'], 1.05),
            (['PROPN', 'NOUN', 'NOUN'], 1.05),

            (['ADJ', 'NOUN', 'NOUN', 'NOUN'], 1.1),
            (['ADJ', 'ADJ', 'NOUN', 'NOUN'], 1.1),
            (['NOUN', 'PROPN', 'PROPN', 'PROPN'], 1.05),

            (['NOUN', 'ADP', 'NOUN'], 0.9),
            (['PROPN', 'ADP', 'NOUN'], 0.9),
            (['PROPN', 'ADP', 'PROPN'], 0.9),
            (['NOUN', 'ADP', 'PROPN'], 0.9),
            (['ADJ', 'NOUN', 'ADP', 'NOUN'], 0.95),

            (['VERB', 'NOUN'], 0.85),
            (['VERB', 'PROPN'], 0.85),
            (['VERB', 'ADJ', 'NOUN'], 0.9),
        ]

        low_quality_patterns = [
            (['DET'], 0.5),
            (['ADP'], 0.5),
            (['ADJ'], 0.7),
            (['VERB'], 0.7),
            (['ADV'], 0.6),
            (['DET', 'NOUN'], 0.8),
            (['ADP', 'NOUN'], 0.7),
            (['ADV', 'ADJ'], 0.7),
            (['VERB', 'DET'], 0.6),
        ]

        for kp, score in keyphrases:
            if not kp.strip():
                continue

            doc = self.nlp(kp)

            pos_tags = [token.pos_ for token in doc]

            matched = False
            score_multiplier = 1.0

            if len(pos_tags) <= 5:
                for pattern, multiplier in acceptable_patterns:
                    if pos_tags == pattern:
                        matched = True
                        score_multiplier = multiplier
                        break

                if not matched:
                    for pattern, multiplier in acceptable_patterns:
                        if len(pattern) <= len(pos_tags) and pos_tags[:len(pattern)] == pattern:
                            matched = True
                            score_multiplier = multiplier * 0.95
                            break

                if not matched:
                    for pattern, multiplier in low_quality_patterns:
                        if pos_tags == pattern or (len(pattern) <= len(pos_tags) and pos_tags[:len(pattern)] == pattern):
                            matched = True
                            score_multiplier = multiplier
                            break
            else:
                for pattern, multiplier in acceptable_patterns:
                    if len(pattern) <= len(pos_tags) and pos_tags[:len(pattern)] == pattern:
                        matched = True
                        score_multiplier = multiplier * 0.9
                        break

                if not matched and pos_tags and pos_tags[0] in ['NOUN', 'PROPN']:
                    matched = True
                    score_multiplier = 0.9

            if matched:
                adjusted_score = min(score * score_multiplier, 1.0)
                filtered_keyphrases.append((kp, adjusted_score))
            elif len(pos_tags) > 0 and pos_tags[0] in ['NOUN', 'PROPN', 'ADJ']:
                adjusted_score = min(score * 0.85, 1.0)
                filtered_keyphrases.append((kp, adjusted_score))

        if len(filtered_keyphrases) < len(keyphrases) * 0.3:
            print(f"Warning: POS filtering removed too many keyphrases ({len(keyphrases)} -> {len(filtered_keyphrases)}). Using original list.")
            return keyphrases

        return filtered_keyphrases

    def initialize_idf_corpus(self, corpus_path: str = None):
        
        if corpus_path and os.path.exists(corpus_path):
            try:
                with open(corpus_path, 'rb') as f:
                    self.idf_values = pickle.load(f)
                print(f"Loaded IDF corpus with {len(self.idf_values)} terms")
                return
            except Exception as e:
                print(f"Error loading IDF corpus: {e}")

        print("Creating comprehensive news-focused IDF corpus")

        news_corpus = [
            "The President signed an executive order addressing climate change and environmental protection measures.",
            "Congress passed a new bill aimed at infrastructure development with bipartisan support.",
            "The Supreme Court ruled on a landmark case regarding privacy rights in the digital age.",
            "Election polls show a tight race between the incumbent and challenger in key swing states.",
            "Diplomatic tensions escalated after the ambassador was recalled following controversial statements.",
            "The Senate committee held hearings on proposed legislation for healthcare reform.",
            "Local government officials announced new initiatives to address homelessness in urban areas.",
            "International leaders gathered at the summit to discuss global trade agreements.",
            "The administration faced criticism over its handling of the border security situation.",
            "Voters expressed concerns about economic policies ahead of the upcoming election.",

            "The stock market reached record highs following positive economic indicators and strong earnings reports.",
            "The Federal Reserve announced changes to interest rates, impacting mortgage and loan markets.",
            "The tech giant unveiled plans for a major expansion, creating thousands of new jobs.",
            "Inflation rates rose to their highest level in a decade, affecting consumer purchasing power.",
            "The company reported quarterly earnings that exceeded analysts' expectations by 15 percent.",
            "Supply chain disruptions continue to affect manufacturing sectors across multiple industries.",
            "The startup secured $50 million in venture capital funding for its innovative platform.",
            "Unemployment figures dropped to 4.5 percent, signaling continued economic recovery.",
            "The retail sector saw significant growth in online sales while physical stores struggled.",
            "The cryptocurrency market experienced volatility following regulatory announcements.",

            "Artificial intelligence advancements are transforming industries from healthcare to finance.",
            "The new smartphone features a revolutionary camera system and faster processor technology.",
            "Cybersecurity experts warned of increasing ransomware threats targeting critical infrastructure.",
            "Cloud computing services continue to grow as businesses move away from on-premises solutions.",
            "The social media platform introduced new features to address privacy concerns and misinformation.",
            "Quantum computing researchers achieved a breakthrough in error correction techniques.",
            "The electric vehicle manufacturer announced a new battery technology with extended range.",
            "Virtual reality applications are expanding beyond gaming into education and healthcare.",
            "The semiconductor shortage has impacted production across automotive and electronics industries.",
            "5G network deployment accelerated, promising faster connectivity and new applications.",

            "Researchers published promising results from clinical trials of the new cancer treatment.",
            "The vaccine showed 95 percent efficacy in preventing the disease in large-scale studies.",
            "Scientists discovered a new species in the remote rainforest region during the expedition.",
            "The space telescope captured unprecedented images of distant galaxies, revealing new data.",
            "Medical experts issued updated guidelines for managing chronic conditions and preventive care.",
            "The genomic study identified key genetic factors associated with longevity and disease resistance.",
            "Climate scientists reported accelerating ice melt in polar regions exceeding previous models.",
            "The public health department launched initiatives to address mental health awareness.",
            "Researchers developed a new diagnostic tool using artificial intelligence algorithms.",
            "The pharmaceutical company received approval for its breakthrough treatment for rare diseases.",

            "Record-breaking temperatures were recorded across multiple regions during the summer months.",
            "Renewable energy installations surpassed fossil fuel capacity additions for the first time.",
            "Conservation efforts successfully increased the population of the endangered species.",
            "The environmental impact assessment revealed concerns about the proposed development project.",
            "Extreme weather events caused significant damage to coastal communities and infrastructure.",
            "Sustainable agriculture practices gained traction as concerns about food security increased.",
            "The international agreement established new targets for reducing carbon emissions by 2030.",
            "Ocean plastic pollution reached alarming levels according to the latest marine research.",
            "Urban planning initiatives focused on green infrastructure and reducing carbon footprints.",
            "The drought conditions affected agricultural production across multiple growing regions.",

            "The university announced a major expansion of its online degree programs and digital learning.",
            "School districts implemented new curriculum standards focusing on STEM education.",
            "Research findings highlighted the impact of early childhood education on long-term outcomes.",
            "The education department allocated additional funding for underserved communities.",
            "Students demonstrated improved performance following the implementation of the new teaching methods.",
            "The college admissions process underwent significant changes to address equity concerns.",
            "Educational technology startups developed innovative tools for personalized learning experiences.",
            "Teachers adapted to hybrid learning models combining in-person and virtual instruction.",
            "The study examined the effectiveness of different approaches to literacy development.",
            "Higher education institutions faced challenges addressing student mental health needs.",

            "The film won multiple awards at the festival, receiving critical acclaim for its direction.",
            "The streaming service announced a slate of original content productions for the coming year.",
            "The musician's latest album debuted at number one on the charts with record-breaking streams.",
            "The art exhibition featured works exploring themes of identity and social justice.",
            "The bestselling novel will be adapted into a television series by the acclaimed director.",
            "Cultural institutions implemented digital initiatives to reach audiences during closures.",
            "The celebrity announced their involvement in humanitarian efforts addressing global issues.",
            "The video game release broke sales records with millions of copies sold in the first week.",
            "The theater production received standing ovations for its innovative staging and performances.",
            "Social media influencers partnered with brands on campaigns targeting younger demographics.",

            "The team secured the championship title after a dramatic overtime victory in the final game.",
            "The athlete broke the world record that had stood for over a decade in the competition.",
            "The league announced new safety protocols following concerns about player injuries.",
            "The international tournament drew record viewership across global broadcasting platforms.",
            "The coach implemented strategic changes that transformed the team's performance this season.",
            "The player signed a record-breaking contract extension with the franchise.",
            "The Olympic committee finalized preparations for the upcoming summer games.",
            "The sports technology company introduced advanced analytics tools for performance tracking.",
            "The stadium renovation project will increase capacity and enhance the fan experience.",
            "The investigation into alleged rule violations resulted in sanctions against the organization.",

            "The investigation led to multiple arrests in connection with the organized crime operation.",
            "The court ruling established a precedent for future cases involving digital privacy rights.",
            "Law enforcement agencies implemented new training programs focused on community relations.",
            "The jury reached a verdict after deliberating for three days in the high-profile case.",
            "The legislation introduced stricter penalties for specific categories of financial crimes.",
            "The forensic evidence played a crucial role in identifying the suspect in the investigation.",
            "The prison reform initiative aimed to reduce recidivism through education and rehabilitation.",
            "The legal challenge questioned the constitutionality of the recently passed legislation.",
            "The settlement agreement included significant compensation for affected individuals.",
            "The regulatory agency imposed fines following violations of consumer protection standards.",

            "The housing initiative aimed to address affordability challenges in metropolitan areas.",
            "Community organizations partnered to provide resources for vulnerable populations.",
            "The study documented disparities in access to essential services across different demographics.",
            "Activists organized peaceful demonstrations advocating for policy changes and social justice.",
            "The report highlighted progress and ongoing challenges in workplace diversity and inclusion.",
            "Urban development projects focused on revitalizing neighborhoods while preventing displacement.",
            "The survey revealed changing attitudes toward social issues among younger generations.",
            "Nonprofit organizations expanded programs addressing food insecurity in underserved communities.",
            "The panel discussion explored intersections between technology and social equity concerns.",
            "The initiative provided support services for veterans transitioning to civilian life."
        ]

        specialized_terminology = [
            "Legislation is being debated in parliament regarding constitutional amendments and electoral reform.",
            "Bipartisan support is needed for the bill to pass through both chambers of congress.",
            "The filibuster prevented the vote on the controversial legislation despite majority support.",
            "Gerrymandering has affected district boundaries, potentially impacting election outcomes.",
            "The caucus meeting determined the party's position on key policy initiatives.",

            "Quantitative easing measures were implemented by the central bank to stimulate economic growth.",
            "The yield curve inversion raised concerns about potential recession indicators.",
            "Fiscal policy adjustments were recommended to address the growing budget deficit.",
            "Market volatility increased following uncertainty about monetary policy directions.",
            "The merger and acquisition activity accelerated in the technology sector this quarter.",

            "The neural network architecture demonstrated superior performance in natural language processing tasks.",
            "Blockchain technology applications expanded beyond cryptocurrencies into supply chain verification.",
            "The API integration enabled seamless connectivity between multiple software platforms.",
            "Machine learning algorithms identified patterns that human analysts had overlooked.",
            "The Internet of Things devices created a comprehensive monitoring system for the facility.",

            "The genome sequencing revealed previously unknown genetic markers associated with the condition.",
            "Quantum entanglement experiments demonstrated non-local correlations between particles.",
            "The clinical trial entered phase three testing after promising preliminary results.",
            "Neuroplasticity research suggested new approaches to rehabilitation after brain injury.",
            "The particle accelerator experiments provided data supporting theoretical predictions.",

            "Biodiversity conservation efforts focused on protecting critical habitat corridors.",
            "Carbon sequestration technologies were evaluated for large-scale implementation potential.",
            "Renewable energy integration presented challenges for existing power grid infrastructure.",
            "Ecosystem services valuation informed policy decisions regarding natural resource management.",
            "Climate mitigation strategies included both technological solutions and behavioral changes."
        ]

        comprehensive_corpus = news_corpus + specialized_terminology

        corpus_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words='english',
            max_features=20000,
            min_df=2,
            max_df=0.9,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )

        corpus_vectorizer.fit(comprehensive_corpus)

        try:
            feature_names = corpus_vectorizer.get_feature_names_out()
        except AttributeError:
            feature_names = corpus_vectorizer.get_feature_names()

        self.idf_values = {}
        for term, idf_idx in corpus_vectorizer.vocabulary_.items():
            self.idf_values[term] = corpus_vectorizer.idf_[idf_idx]

        print(f"Created comprehensive IDF corpus with {len(self.idf_values)} terms")

        additional_terms = {
            "breaking news": 2.0,
            "exclusive report": 3.0,
            "sources say": 1.8,
            "according to officials": 1.5,
            "press conference": 2.2,
            "official statement": 2.3,
            "developing story": 2.1,
            "anonymous source": 2.8,
            "public opinion": 2.5,
            "latest update": 1.9,

            "united nations": 4.0,
            "world health organization": 4.2,
            "european union": 3.8,
            "federal reserve": 4.1,
            "supreme court": 3.9,
            "white house": 3.7,
            "wall street": 3.6,
            "silicon valley": 4.3,
            "climate change": 3.5,
            "artificial intelligence": 4.5
        }

        for term, idf in additional_terms.items():
            self.idf_values[term] = idf

        print(f"Added {len(additional_terms)} additional news-specific terms")
        print(f"Final IDF corpus contains {len(self.idf_values)} terms")

        if corpus_path:
            try:
                os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
                with open(corpus_path, 'wb') as f:
                    pickle.dump(self.idf_values, f)
                print(f"Saved IDF corpus to {corpus_path}")
            except Exception as e:
                print(f"Error saving IDF corpus: {e}")

    def normalize_text(self, text: str) -> str:
        
        text = re.sub(r'\s+', ' ', text)

        text = re.sub(r'[^\w\s\-\']', ' ', text)

        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def clean_phrase_boundaries(self, phrase: str) -> str:
        
        if not self.clean_boundaries:
            return phrase

        words = phrase.split()

        while words and words[0].lower() in self.BOUNDARY_WORDS:
            words.pop(0)

        while words and words[-1].lower() in self.BOUNDARY_WORDS:
            words.pop()

        cleaned_phrase = ' '.join(words)

        return cleaned_phrase

    def preprocess_text(self, text: str) -> str:
        
        text = self.normalize_text(text)

        doc = self.nlp(text)

        processed_tokens = []

        entities_spans = [(ent.start, ent.end, ent.text) for ent in doc.ents]
        acronyms = [token.text for token in doc if token.text.isupper() and len(token.text) > 1]

        compound_terms = []
        technical_prefixes = ['pre-', 'post-', 'multi-', 'inter-', 'intra-', 'micro-', 'macro-', 'nano-', 'cyber-']

        for i, token in enumerate(doc):
            if '-' in token.text and not token.is_punct:
                compound_terms.append((token.i, token.i + 1, token.text))

            if i < len(doc) - 1:
                for prefix in technical_prefixes:
                    if token.text.lower().endswith(prefix[:-1]) and not doc[i+1].is_punct:
                        compound_terms.append((token.i, token.i + 2, f"{token.text}{doc[i+1].text}"))

        i = 0
        while i < len(doc):
            token = doc[i]

            if token.is_space:
                i += 1
                continue

            is_compound_start = any(start == token.i for start, _, _ in compound_terms)
            is_compound_part = any(start < token.i < end for start, end, _ in compound_terms)

            is_entity = any(start <= token.i < end for start, end, _ in entities_spans)

            if is_compound_start:
                for start, end, text in compound_terms:
                    if start == token.i:
                        processed_tokens.append(text)
                        i = end
                        break
            elif is_compound_part:
                i += 1
            elif (is_entity or token.text in acronyms or token.pos_ == "PROPN" or
                (token.text[0].isupper() and i > 0 and not doc[i-1].text.endswith('.'))):
                processed_tokens.append(token.text)
                i += 1
            else:
                processed_tokens.append(token.text.lower())
                i += 1

        processed_text = " ".join(processed_tokens)

        technical_patterns = [
            (r'(\w+)-of-(\w+)', r'\1_of_\2'),
            (r'(\w+)-to-(\w+)', r'\1_to_\2'),
            (r'(\w+)-(\w+)-(\w+)', r'\1_\2_\3'),
        ]

        for pattern, replacement in technical_patterns:
            processed_text = re.sub(pattern, replacement, processed_text)

        return processed_text
    def get_named_entities(self, text: str) -> List[Tuple[str, str]]:
        
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    def get_noun_chunks(self, text: str) -> List[str]:
        
        doc = self.nlp(text)
        chunks = [chunk.text.lower() for chunk in doc.noun_chunks]

        if self.clean_boundaries:
            chunks = [self.clean_phrase_boundaries(chunk) for chunk in chunks]

        chunks = [chunk for chunk in chunks if chunk and chunk not in self.all_stopwords]

        return chunks

    def get_custom_stopwords(self, text: str) -> Set[str]:
        
        return self.all_stopwords

    def extract_title_and_lead(self, text: str) -> Tuple[str, str, str]:
        
        lines = text.split('\n')

        title = ""
        for line in lines:
            if line.strip():
                title = line.strip()
                break

        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        lead_paragraph = ""
        if len(paragraphs) > 1 and title in paragraphs[0]:
            lead_paragraph = paragraphs[1]
        elif paragraphs:
            lead_paragraph = paragraphs[0]
            if lead_paragraph == title and len(paragraphs) > 1:
                lead_paragraph = paragraphs[1]

        first_paragraphs = ""
        if len(paragraphs) > 1:
            if title in paragraphs[0] and paragraphs[0] == title:
                first_paragraphs = " ".join(paragraphs[1:min(4, len(paragraphs))])
            else:
                first_paragraphs = " ".join(paragraphs[:min(3, len(paragraphs))])
        else:
            first_paragraphs = lead_paragraph

        return title, lead_paragraph, first_paragraphs

    def extract_with_keybert(self, text: str, nr_candidates: int = 20) -> List[Tuple[str, float]]:
        
        try:
            custom_stopwords = self.get_custom_stopwords(text)

            if self.method == 'mmr':
                keyphrases = self.kw_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=self.ngram_range,
                    stop_words=custom_stopwords,
                    use_mmr=True,
                    diversity=self.diversity_penalty,
                    top_n=nr_candidates
                )
            else:
                keyphrases = self.kw_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=self.ngram_range,
                    stop_words=custom_stopwords,
                    use_mmr=False,
                    top_n=nr_candidates
                )

            formatted_keyphrases = []
            for kp_item in keyphrases:
                if isinstance(kp_item, tuple) and len(kp_item) == 2:
                    if isinstance(kp_item[0], str):
                        kp, score = kp_item
                    else:
                        score, kp = kp_item

                    if self.clean_boundaries:
                        kp = self.clean_phrase_boundaries(kp)

                    if kp:
                        formatted_keyphrases.append((kp, score))
                else:
                    print(f"Skipping invalid keyphrase item: {kp_item}")

            if self.use_position_weight and formatted_keyphrases:
                position_weights = {}
                for kp, _ in formatted_keyphrases:
                    weight = self.calculate_position_weight(text, kp)
                    position_weights[kp] = weight
                    print(f"DEBUG: Position weight for '{kp}': {weight:.2f}")

                formatted_keyphrases = [(kp, score * position_weights[kp]) for kp, score in formatted_keyphrases]

            if self.use_tfidf_weight and formatted_keyphrases:
                tfidf_weights = self.calculate_tfidf_weights(text, [kp for kp, _ in formatted_keyphrases])

                for kp, weight in tfidf_weights.items():
                    print(f"DEBUG: TF-IDF weight for '{kp}': {weight:.2f}")

                formatted_keyphrases = [(kp, score * tfidf_weights[kp]) for kp, score in formatted_keyphrases]

            if self.prioritize_named_entities:
                entities = self.get_named_entities(text)
                for entity, entity_type in entities:
                    if entity_type in ['ORG', 'PRODUCT', 'PERSON', 'GPE', 'WORK_OF_ART', 'EVENT']:
                        entity_lower = entity.lower()
                        exists = False
                        for i, (kp, score) in enumerate(formatted_keyphrases):
                            if entity_lower == kp.lower():
                                formatted_keyphrases[i] = (kp, min(1.0, score * 1.1))
                                exists = True
                                break

                        if not exists and len(entity.split()) <= self.ngram_range[1]:
                            formatted_keyphrases.append((entity, 0.6))

            if self.use_noun_chunks:
                chunks = self.get_noun_chunks(text)
                for chunk in chunks:
                    if len(chunk.split()) <= self.ngram_range[1]:
                        chunk_lower = chunk.lower()
                        exists = False
                        for kp, _ in formatted_keyphrases:
                            if chunk_lower == kp.lower():
                                exists = True
                                break

                        if not exists:
                            formatted_keyphrases.append((chunk, 0.5))

            if self.boost_exact_matches:
                important_phrases = []

                quoted_phrases = re.findall(r'"([^"]+)"', text)
                important_phrases.extend(quoted_phrases)

                capitalized_phrases = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
                important_phrases.extend(capitalized_phrases)

                for phrase in important_phrases:
                    phrase_lower = phrase.lower()
                    exists = False
                    for i, (kp, score) in enumerate(formatted_keyphrases):
                        if phrase_lower == kp.lower():
                            formatted_keyphrases[i] = (kp, min(1.0, score * 1.2))
                            exists = True
                            break

                    if not exists and len(phrase.split()) <= self.ngram_range[1]:
                        formatted_keyphrases.append((phrase, 0.7))

            return formatted_keyphrases
        except Exception as e:
            print(f"Error in KeyBERT extraction: {str(e)}")
            return []

    def lemmatize_text(self, text: str) -> str:
        
        if not self.use_lemmatization:
            return text

        try:
            doc = self.nlp(text)

            lemmatized_tokens = []
            for token in doc:
                if token.pos_ == 'NOUN':
                    pos = 'n'
                elif token.pos_ == 'VERB':
                    pos = 'v'
                elif token.pos_ == 'ADJ':
                    pos = 'a'
                elif token.pos_ == 'ADV':
                    pos = 'r'
                else:
                    lemmatized_tokens.append(token.text)
                    continue

                try:
                    lemmatized_tokens.append(self.lemmatizer.lemmatize(token.text.lower(), pos))
                except Exception:
                    lemmatized_tokens.append(token.text.lower())

            lemmatized_text = ' '.join(lemmatized_tokens).strip()

            return lemmatized_text
        except Exception as e:
            print(f"Warning: Lemmatization failed: {e}")
            return text

    def calculate_position_weight(self, text: str, keyphrase: str) -> float:
        
        if not self.use_position_weight:
            return 1.0

        keyphrase_lower = keyphrase.lower()
        text_lower = text.lower()

        position = text_lower.find(keyphrase_lower)

        if position == -1:
            return 0.5

        if self.use_title_lead_boost:
            title, lead_paragraph, first_paragraphs = self.extract_title_and_lead(text)
            title_lower = title.lower()
            lead_lower = lead_paragraph.lower()
            first_paragraphs_lower = first_paragraphs.lower()

            title_exact_match = keyphrase_lower in title_lower
            lead_exact_match = keyphrase_lower in lead_lower
            first_paragraphs_exact_match = keyphrase_lower in first_paragraphs_lower and not title_exact_match and not lead_exact_match

            kp_words = set(keyphrase_lower.split())
            title_words = set(title_lower.split())
            lead_words = set(lead_lower.split())

            title_word_overlap = len(kp_words.intersection(title_words)) / len(kp_words) if kp_words else 0
            lead_word_overlap = len(kp_words.intersection(lead_words)) / len(kp_words) if kp_words else 0

            if title_exact_match:
                return 2.0
            elif lead_exact_match:
                return 1.5
            elif first_paragraphs_exact_match:
                return 1.3
            elif title_word_overlap > 0.75:
                return 1.4
            elif lead_word_overlap > 0.75:
                return 1.3
            elif title_word_overlap > 0.5 or lead_word_overlap > 0.5:
                return 1.2

        relative_position = position / len(text)

        if relative_position < 0.01:
            weight = 2.0
        else:
            weight = 0.5 + 1.5 * (math.log(1 + 10 * (1 - relative_position)) / math.log(11))

        normalized_weight = min(2.0, max(0.5, weight))

        return normalized_weight

    def calculate_tfidf_weights(self, text: str, keyphrases: List[str]) -> Dict[str, float]:
        
        if not keyphrases:
            return {}

        try:
            preprocessed_text = self.preprocess_text(text)

            tokens = preprocessed_text.lower().split()

            term_freq = {}
            for token in tokens:
                if token not in term_freq:
                    term_freq[token] = 0
                term_freq[token] += 1

            for token in term_freq:
                term_freq[token] = 1 + math.log(term_freq[token])

            word_tfidf = {}
            for token, tf in term_freq.items():
                if token in self.idf_values:
                    idf = self.idf_values[token]
                else:
                    idf = math.log(30 + 1)

                    self.idf_values[token] = idf

                word_tfidf[token] = tf * idf

            self.word_tfidf_cache = word_tfidf

            keyphrase_weights = {}
            for keyphrase in keyphrases:
                words = keyphrase.lower().split()

                word_scores = []
                for word in words:
                    if word in word_tfidf:
                        word_scores.append(word_tfidf[word])
                    elif word in self.idf_values:
                        word_scores.append(0.5 * self.idf_values[word])
                    else:
                        word_scores.append(0.5)

                if word_scores:
                    total_weight = 0
                    total_score = 0
                    for i, score in enumerate(word_scores):
                        if len(word_scores) > 1:
                            if i == 0 or i == len(word_scores) - 1:
                                pos_weight = 1.5
                            else:
                                pos_weight = 1.0
                        else:
                            pos_weight = 1.0

                        total_weight += pos_weight
                        total_score += score * pos_weight

                    avg_score = total_score / total_weight

                    if len(words) > 1:
                        length_bonus = 1.0 + 0.1 * (len(words) - 1)
                        keyphrase_weights[keyphrase] = avg_score * length_bonus
                    else:
                        keyphrase_weights[keyphrase] = avg_score
                else:
                    keyphrase_weights[keyphrase] = 0.5

            if keyphrase_weights:
                max_weight = max(keyphrase_weights.values())
                min_weight = min(keyphrase_weights.values())

                if max_weight > min_weight:
                    for kp in keyphrase_weights:
                        keyphrase_weights[kp] = 0.5 + ((keyphrase_weights[kp] - min_weight) /
                                                    (max_weight - min_weight)) * 1.5
                else:
                    for kp in keyphrase_weights:
                        keyphrase_weights[kp] = 1.0

            return keyphrase_weights

        except Exception as e:
            print(f"Error calculating TF-IDF weights: {str(e)}")
            import traceback
            traceback.print_exc()
            return {kp: 1.0 for kp in keyphrases}

    def update_idf_corpus(self, documents: List[str], save_path: str = None):
        
        if not documents:
            return

        try:
            update_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words='english',
                max_features=10000,
                min_df=1,
                norm='l2',
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True
            )

            update_vectorizer.fit(documents)

            try:
                feature_names = update_vectorizer.get_feature_names_out()
            except AttributeError:
                feature_names = update_vectorizer.get_feature_names()

            for term, idf_idx in update_vectorizer.vocabulary_.items():
                if term in self.idf_values:
                    self.idf_values[term] = 0.7 * self.idf_values[term] + 0.3 * update_vectorizer.idf_[idf_idx]
                else:
                    self.idf_values[term] = update_vectorizer.idf_[idf_idx]

            print(f"Updated IDF corpus, now contains {len(self.idf_values)} terms")

            if save_path:
                try:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, 'wb') as f:
                        pickle.dump(self.idf_values, f)
                    print(f"Saved updated IDF corpus to {save_path}")
                except Exception as e:
                    print(f"Error saving updated IDF corpus: {e}")

        except Exception as e:
            print(f"Error updating IDF corpus: {e}")
            import traceback
            traceback.print_exc()

    def extract_with_multipartiterank(self, text: str, nr_candidates: int = 20) -> List[Tuple[str, float]]:
        
        extractor = pke.unsupervised.MultipartiteRank()

        custom_stopwords = self.get_custom_stopwords(text)

        try:
            extractor.load_document(input=text, language=self.language)

            extractor.stoplist = list(custom_stopwords)

            extractor.candidate_selection(
                pos={'NOUN', 'PROPN', 'ADJ'}
            )

            extractor.candidate_weighting(
                alpha=self.alpha,
                threshold=self.threshold,
                method='average'
            )

            keyphrases = extractor.get_n_best(n=nr_candidates)

            keyphrases = self.post_process_keyphrases(keyphrases, text)

            if self.use_position_weight:
                keyphrases = [(self.clean_phrase_boundaries(kp), score) for kp, score in keyphrases]

            keyphrases = [(kp, score) for kp, score in keyphrases if kp]

            if self.use_position_weight:
                position_weights = {}
                for kp, _ in keyphrases:
                    weight = self.calculate_position_weight(text, kp)
                    position_weights[kp] = weight
                    print(f"DEBUG: MultipartiteRank position weight for '{kp}': {weight:.2f}")

                keyphrases = [(kp, score * position_weights[kp]) for kp, score in keyphrases]

            if self.use_tfidf_weight:
                tfidf_weights = self.calculate_tfidf_weights(text, [kp for kp, _ in keyphrases])

                for kp, weight in tfidf_weights.items():
                    print(f"DEBUG: MultipartiteRank TF-IDF weight for '{kp}': {weight:.2f}")

                keyphrases = [(kp, score * tfidf_weights[kp]) for kp, score in keyphrases]

        except Exception as e:
            print(f"Error in MultipartiteRank: {str(e)}")
            import traceback
            traceback.print_exc()
            keyphrases = []

        return keyphrases

    def extract_with_yake(self, text: str, nr_candidates: int = 20) -> List[Tuple[str, float]]:
        
        extractor = pke.unsupervised.YAKE()

        try:
            extractor.load_document(input=text, language=self.language, normalization=None)

            extractor.candidate_selection(n=self.ngram_range[1])

            extractor.candidate_weighting(window=3, use_stems=False)

            raw_keyphrases = extractor.get_n_best(n=nr_candidates)

            if raw_keyphrases:
                max_score = max([score for _, score in raw_keyphrases])
                keyphrases = [(kp, 1.0 - (score / max_score)) for kp, score in raw_keyphrases]
            else:
                keyphrases = []

            if self.clean_boundaries:
                keyphrases = [(self.clean_phrase_boundaries(kp), score) for kp, score in keyphrases]

            keyphrases = [(kp, score) for kp, score in keyphrases if kp]

            if self.use_position_weight:
                position_weights = {}
                for kp, _ in keyphrases:
                    weight = self.calculate_position_weight(text, kp)
                    position_weights[kp] = weight
                    print(f"DEBUG: YAKE position weight for '{kp}': {weight:.2f}")

                keyphrases = [(kp, score * position_weights[kp]) for kp, score in keyphrases]

            if self.use_tfidf_weight:
                tfidf_weights = self.calculate_tfidf_weights(text, [kp for kp, _ in keyphrases])

                for kp, weight in tfidf_weights.items():
                    print(f"DEBUG: YAKE TF-IDF weight for '{kp}': {weight:.2f}")

                keyphrases = [(kp, score * tfidf_weights[kp]) for kp, score in keyphrases]

        except Exception as e:
            print(f"Error in YAKE: {str(e)}")
            keyphrases = []

        return keyphrases

    def extract_with_textrank(self, text: str, nr_candidates: int = 20) -> List[Tuple[str, float]]:
        
        extractor = pke.unsupervised.TextRank()

        custom_stopwords = self.get_custom_stopwords(text)

        try:
            extractor.load_document(input=text, language=self.language)

            extractor.stoplist = list(custom_stopwords)

            extractor.candidate_selection(
                pos={'NOUN', 'PROPN', 'ADJ', 'VERB'}
            )

            extractor.candidate_weighting(
                window=5,
                pos={'NOUN', 'PROPN', 'ADJ', 'VERB'},
                top_percent=0.33
            )

            keyphrases = extractor.get_n_best(n=nr_candidates)

            if self.clean_boundaries:
                keyphrases = [(self.clean_phrase_boundaries(kp), score) for kp, score in keyphrases]

            keyphrases = [(kp, score) for kp, score in keyphrases if kp]

            if self.use_position_weight:
                position_weights = {}
                for kp, _ in keyphrases:
                    weight = self.calculate_position_weight(text, kp)
                    position_weights[kp] = weight
                    print(f"DEBUG: TextRank position weight for '{kp}': {weight:.2f}")

                keyphrases = [(kp, score * position_weights[kp]) for kp, score in keyphrases]

            if self.use_tfidf_weight:
                tfidf_weights = self.calculate_tfidf_weights(text, [kp for kp, _ in keyphrases])

                for kp, weight in tfidf_weights.items():
                    print(f"DEBUG: TextRank TF-IDF weight for '{kp}': {weight:.2f}")

                keyphrases = [(kp, score * tfidf_weights[kp]) for kp, score in keyphrases]

        except Exception as e:
            print(f"Error in TextRank: {str(e)}")
            import traceback
            traceback.print_exc()
            keyphrases = []

        return keyphrases

    def classify_document_type(self, text: str) -> Dict[str, float]:
        
        doc_characteristics = {
            'length': len(text),
            'avg_sentence_length': 0,
            'technical_density': 0,
            'domain_scores': {
                'technology': 0.0,
                'business': 0.0,
                'health': 0.0,
                'science': 0.0,
                'news': 0.0,
                'academic': 0.0,
                'politics': 0.0,
                'environment': 0.0,
                'entertainment': 0.0,
                'sports': 0.0
            },
            'structure_scores': {
                'formal': 0.0,
                'informal': 0.0,
                'narrative': 0.0,
                'descriptive': 0.0,
                'technical': 0.0
            }
        }

        doc = self.nlp(text[:10000])

        sentences = list(doc.sents)
        if sentences:
            doc_characteristics['avg_sentence_length'] = sum(len(sent) for sent in sentences) / len(sentences)

        technical_terms = 0
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN'] and
                token.is_alpha and
                not token.is_stop and
                len(token.text) > 3):
                if token.text.lower() in self.idf_values and self.idf_values[token.text.lower()] > 5.0:
                    technical_terms += 1

        if len(doc) > 0:
            doc_characteristics['technical_density'] = technical_terms / len(doc)

        domain_keywords = {
            'technology': [
                'technology', 'digital', 'software', 'hardware', 'data', 'internet', 'online',
                'app', 'application', 'computer', 'computing', 'network', 'system', 'platform',

                'ai', 'artificial intelligence', 'machine learning', 'deep learning', 'algorithm',
                'cloud', 'cybersecurity', 'security', 'privacy', 'encryption', 'blockchain',
                'automation', 'robot', 'robotics', 'iot', 'internet of things', 'virtual reality', 'vr',
                'augmented reality', 'ar', 'analytics', 'big data', 'database', 'programming',

                'google', 'microsoft', 'apple', 'amazon', 'facebook', 'meta', 'tesla', 'ibm',
                'intel', 'nvidia', 'samsung', 'oracle', 'cisco', 'twitter', 'linkedin', 'tiktok',

                'smartphone', 'mobile', 'website', 'browser', 'search engine', 'social media',
                'email', 'e-commerce', 'streaming', 'wifi', 'bluetooth', 'server', 'chip',
                'processor', 'interface', 'api', 'code', 'coding', 'developer', 'startup'
            ],

            'business': [
                'business', 'company', 'corporation', 'enterprise', 'industry', 'market', 'firm',
                'commercial', 'corporate', 'trade', 'commerce', 'sales', 'retail', 'wholesale',

                'finance', 'financial', 'investment', 'investor', 'stock', 'share', 'shareholder',
                'profit', 'revenue', 'income', 'earnings', 'loss', 'budget', 'funding', 'venture capital',
                'capital', 'asset', 'liability', 'equity', 'dividend', 'portfolio', 'merger', 'acquisition',

                'management', 'executive', 'ceo', 'cfo', 'coo', 'board', 'director', 'leadership',
                'strategy', 'strategic', 'operation', 'operational', 'performance', 'productivity',

                'marketing', 'advertising', 'brand', 'consumer', 'customer', 'client', 'product',
                'service', 'market share', 'competition', 'competitive', 'pricing', 'promotion',

                'economy', 'economic', 'gdp', 'growth', 'recession', 'inflation', 'deflation',
                'unemployment', 'employment', 'labor', 'workforce', 'supply', 'demand', 'sector'
            ],

            'health': [
                'health', 'healthcare', 'medical', 'medicine', 'clinical', 'hospital', 'doctor',
                'physician', 'nurse', 'patient', 'treatment', 'therapy', 'diagnosis', 'prognosis',

                'disease', 'disorder', 'condition', 'syndrome', 'infection', 'virus', 'bacterial',
                'chronic', 'acute', 'symptom', 'pain', 'inflammation', 'cancer', 'diabetes',
                'heart disease', 'stroke', 'alzheimer', 'dementia', 'obesity', 'hypertension',

                'public health', 'epidemic', 'pandemic', 'outbreak', 'vaccination', 'vaccine',
                'immunization', 'prevention', 'screening', 'mortality', 'morbidity', 'life expectancy',

                'insurance', 'medicare', 'medicaid', 'pharmacy', 'pharmaceutical', 'drug', 'medication',
                'prescription', 'clinic', 'emergency', 'surgery', 'surgical', 'specialist', 'primary care',

                'wellness', 'fitness', 'nutrition', 'diet', 'exercise', 'mental health', 'psychology',
                'psychiatry', 'therapy', 'counseling', 'wellbeing', 'lifestyle', 'stress', 'anxiety',
                'depression'
            ],

            'science': [
                'science', 'scientific', 'research', 'researcher', 'scientist', 'laboratory', 'lab',
                'experiment', 'experimental', 'theory', 'theoretical', 'hypothesis', 'evidence',
                'discovery', 'innovation', 'breakthrough',

                'physics', 'physical', 'particle', 'quantum', 'relativity', 'gravity', 'energy',
                'matter', 'atom', 'nuclear', 'electron', 'proton', 'neutron', 'radiation',

                'chemistry', 'chemical', 'molecule', 'molecular', 'compound', 'element', 'reaction',
                'catalyst', 'acid', 'base', 'organic', 'inorganic', 'polymer',

                'biology', 'biological', 'cell', 'cellular', 'gene', 'genetic', 'dna', 'rna',
                'protein', 'enzyme', 'organism', 'species', 'evolution', 'evolutionary',
                'ecology', 'ecosystem', 'biodiversity',

                'astronomy', 'astronomical', 'space', 'planet', 'planetary', 'star', 'stellar',
                'galaxy', 'cosmic', 'universe', 'solar system', 'nasa', 'telescope', 'satellite',
                'rocket', 'spacecraft', 'mission', 'orbit', 'mars', 'moon', 'jupiter'
            ],

            'news': [
                'news', 'report', 'reported', 'reporting', 'journalist', 'journalism', 'media',
                'press', 'broadcast', 'coverage', 'correspondent', 'reporter', 'editor',

                'today', 'yesterday', 'this week', 'this month', 'this year', 'breaking',
                'latest', 'update', 'developing', 'recent', 'current', 'ongoing',

                'according to', 'sources', 'officials', 'authorities', 'spokesperson',
                'statement', 'announced', 'confirmed', 'denied', 'claimed', 'alleged',

                'incident', 'event', 'situation', 'development', 'crisis', 'scandal',
                'controversy', 'investigation', 'probe', 'inquiry', 'hearing', 'testimony',

                'exclusive', 'special report', 'analysis', 'opinion', 'editorial', 'feature',
                'interview', 'profile', 'survey', 'poll', 'briefing', 'recap', 'highlights'
            ],

            'academic': [
                'research', 'study', 'analysis', 'investigation', 'experiment', 'observation',
                'survey', 'review', 'meta-analysis', 'literature review', 'case study',

                'paper', 'article', 'publication', 'journal', 'thesis', 'dissertation',
                'monograph', 'proceedings', 'abstract', 'introduction', 'methodology',
                'results', 'discussion', 'conclusion', 'references', 'bibliography',

                'theory', 'framework', 'model', 'paradigm', 'concept', 'hypothesis',
                'variable', 'correlation', 'causation', 'significance', 'validity',
                'reliability', 'replication', 'peer review', 'citation', 'impact factor',

                'university', 'college', 'institution', 'department', 'faculty', 'school',
                'academy', 'institute', 'laboratory', 'center', 'professor', 'researcher',
                'scholar', 'academic', 'student', 'undergraduate', 'graduate', 'postgraduate',
                'doctoral', 'phd', 'postdoc'
            ],

            'politics': [
                'government', 'administration', 'president', 'prime minister', 'congress',
                'parliament', 'senate', 'house', 'cabinet', 'minister', 'secretary', 'governor',
                'mayor', 'official', 'authority', 'agency', 'federal', 'state', 'local',

                'election', 'campaign', 'vote', 'voter', 'ballot', 'poll', 'polling',
                'candidate', 'incumbent', 'challenger', 'primary', 'caucus', 'debate',

                'democrat', 'republican', 'liberal', 'conservative', 'progressive',
                'left-wing', 'right-wing', 'centrist', 'moderate', 'radical', 'party',
                'partisan', 'bipartisan', 'coalition',

                'policy', 'legislation', 'law', 'bill', 'act', 'amendment', 'regulation',
                'deregulation', 'reform', 'initiative', 'proposal', 'budget', 'tax',
                'spending', 'deficit',

                'foreign policy', 'international', 'diplomatic', 'diplomacy', 'treaty',
                'agreement', 'alliance', 'summit', 'sanction', 'conflict', 'crisis',
                'security', 'defense', 'military', 'war', 'peace', 'negotiation'
            ],

            'environment': [
                'climate', 'climate change', 'global warming', 'greenhouse gas', 'carbon',
                'emission', 'temperature', 'weather', 'extreme weather', 'drought', 'flood',
                'hurricane', 'storm', 'wildfire', 'heatwave',

                'environment', 'environmental', 'ecosystem', 'biodiversity', 'habitat',
                'species', 'endangered', 'extinction', 'conservation', 'preservation',
                'wildlife', 'forest', 'deforestation', 'reforestation', 'wetland',

                'pollution', 'pollutant', 'contamination', 'waste', 'plastic', 'recycling',
                'landfill', 'toxic', 'hazardous', 'oil spill', 'air quality', 'water quality',

                'energy', 'renewable', 'solar', 'wind', 'hydroelectric', 'geothermal',
                'fossil fuel', 'coal', 'oil', 'natural gas', 'nuclear', 'power plant',

                'sustainable', 'sustainability', 'green', 'eco-friendly', 'carbon footprint',
                'carbon neutral', 'net zero', 'clean energy', 'organic', 'natural resource'
            ],

            'entertainment': [
                'movie', 'film', 'cinema', 'television', 'tv', 'show', 'series', 'episode',
                'documentary', 'drama', 'comedy', 'thriller', 'horror', 'action', 'sci-fi',
                'director', 'producer', 'actor', 'actress', 'cast', 'character', 'screenplay',
                'box office', 'streaming', 'netflix', 'disney', 'hbo', 'amazon prime',

                'music', 'song', 'album', 'single', 'artist', 'band', 'musician', 'singer',
                'rapper', 'concert', 'tour', 'performance', 'grammy', 'billboard', 'chart',
                'spotify', 'itunes', 'vinyl', 'genre', 'pop', 'rock', 'hip-hop', 'rap', 'jazz',

                'celebrity', 'star', 'famous', 'award', 'red carpet', 'premiere', 'interview',
                'paparazzi', 'gossip', 'scandal', 'viral', 'trending', 'social media',
                'instagram', 'twitter', 'tiktok', 'youtube', 'influencer',

                'game', 'gaming', 'video game', 'esports', 'book', 'novel', 'author',
                'bestseller', 'theater', 'broadway', 'art', 'exhibition', 'festival',
                'fashion', 'design', 'model', 'runway', 'collection'
            ],

            'sports': [
                'sports', 'sport', 'football', 'soccer', 'basketball', 'baseball', 'hockey',
                'tennis', 'golf', 'cricket', 'rugby', 'boxing', 'mma', 'wrestling', 'racing',
                'formula 1', 'nascar', 'swimming', 'track', 'field', 'gymnastics',

                'game', 'match', 'tournament', 'championship', 'league', 'season', 'playoff',
                'final', 'semifinal', 'quarterfinal', 'competition', 'event', 'olympic',
                'world cup', 'grand slam', 'title', 'trophy', 'medal', 'record',

                'team', 'player', 'athlete', 'coach', 'manager', 'referee', 'umpire',
                'captain', 'rookie', 'veteran', 'draft', 'trade', 'free agent', 'contract',
                'roster', 'lineup', 'bench', 'starter',

                'nfl', 'nba', 'mlb', 'nhl', 'fifa', 'uefa', 'ioc', 'ncaa', 'espn',

                'score', 'win', 'loss', 'victory', 'defeat', 'tie', 'draw', 'overtime',
                'penalty', 'foul', 'injury', 'performance', 'stat', 'statistic', 'ranking',
                'standing', 'point', 'goal', 'touchdown', 'homerun', 'basket'
            ]
        }

        text_lower = text.lower()
        domain_scores = {domain: 0.0 for domain in domain_keywords}

        for domain, keywords in domain_keywords.items():
            domain_score = 0.0

            high_importance = keywords[:int(len(keywords) * 0.3)]
            medium_importance = keywords[int(len(keywords) * 0.3):int(len(keywords) * 0.7)]
            low_importance = keywords[int(len(keywords) * 0.7):]

            for keyword in high_importance:
                if keyword in text_lower:
                    domain_score += 1.5
                elif len(keyword.split()) > 1 and all(word in text_lower for word in keyword.split()):
                    domain_score += 0.75

            for keyword in medium_importance:
                count = text_lower.count(' ' + keyword + ' ') + text_lower.count(keyword + ' ') + text_lower.count(' ' + keyword)
                if count > 0:
                    domain_score += count * 2.0

            for keyword in low_importance:
                count = text_lower.count(' ' + keyword + ' ') + text_lower.count(keyword + ' ') + text_lower.count(' ' + keyword)
                if count > 0:
                    domain_score += count * 1.0

            domain_scores[domain] = domain_score
        print(f"DEBUG: Raw domain score for {domain}: {domain_score:.2f}")

        total_score = sum(domain_scores.values())
        if total_score > 0:
            domain_scores = {domain: score / total_score for domain, score in domain_scores.items()}
        else:
            domain_scores = {domain: 0.0 for domain in domain_scores}

        doc_characteristics['domain_scores'] = domain_scores
        structure_features = {
            'formal': 0,
            'informal': 0,
            'narrative': 0,
            'descriptive': 0,
            'technical': 0
        }

        for token in doc:
            if len(token.text) > 8 and token.is_alpha:
                structure_features['technical'] += 1

            if token.pos_ == 'ADJ':
                structure_features['descriptive'] += 1

            if token.pos_ == 'VERB' and token.tag_ in ['VBD', 'VBN']:
                structure_features['narrative'] += 1

            if token.text.lower() in ['i', 'we', 'you', 'they', 'he', 'she']:
                structure_features['informal'] += 1
                structure_features['narrative'] += 0.5

            if len(token.text) > 6 and token.is_alpha and token.pos_ not in ['PRON']:
                structure_features['formal'] += 1

        total_features = sum(structure_features.values())
        if total_features > 0:
            for structure, count in structure_features.items():
                doc_characteristics['structure_scores'][structure] = count / total_features

        top_domains = sorted(doc_characteristics['domain_scores'].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"DEBUG: Top domains detected: {', '.join([f'{d}: {s:.4f}' for d, s in top_domains])}")
        print(f"DEBUG: All domain scores: {doc_characteristics['domain_scores']}")
        return doc_characteristics

    def estimate_method_confidence(
    self,
    method_name: str,
    keyphrases: List[Tuple[str, float]],
    doc_characteristics: Dict[str, Any]
    ) -> float:
        
        if not keyphrases:
            return 0.0

        base_confidence = self.method_weights.get(method_name, 0.25)

        scores = [score for _, score in keyphrases]

        avg_score = sum(scores) / len(scores) if scores else 0
        score_variance = sum((s - avg_score) ** 2 for s in scores) / len(scores) if len(scores) > 1 else 0
        score_range = max(scores) - min(scores) if scores else 0

        score_confidence = 1.0 - min(1.0, (score_variance * 5 + score_range * 0.5))

        coherence = 0.0
        if len(keyphrases) > 1:
            top_keyphrases = [kp for kp, _ in keyphrases[:5]]

            try:
                embeddings = self.get_embeddings(top_keyphrases, convert_to_tensor=False)

                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        sim = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0][0]
                        similarities.append(sim)

                coherence = sum(similarities) / len(similarities) if similarities else 0.0
            except Exception as e:
                print(f"Error calculating coherence: {e}")
                coherence = 0.5

        method_domain_fit = 0.5

        domain_scores = doc_characteristics['domain_scores']
        structure_scores = doc_characteristics['structure_scores']

        if method_name == 'keybert':
            method_domain_fit = (
                domain_scores.get('technology', 0) * 0.8 +
                domain_scores.get('science', 0) * 0.8 +
                domain_scores.get('academic', 0) * 0.7 +
                domain_scores.get('business', 0) * 0.6 +
                domain_scores.get('health', 0) * 0.6 +
                structure_scores.get('technical', 0) * 0.8 +
                structure_scores.get('formal', 0) * 0.7
            ) / 5.0

        elif method_name == 'multipartiterank':
            method_domain_fit = (
                domain_scores.get('academic', 0) * 0.8 +
                domain_scores.get('science', 0) * 0.7 +
                domain_scores.get('health', 0) * 0.6 +
                domain_scores.get('politics', 0) * 0.6 +
                domain_scores.get('business', 0) * 0.5 +
                structure_scores.get('formal', 0) * 0.8 +
                structure_scores.get('technical', 0) * 0.7 +
                (1.0 - structure_scores.get('narrative', 0)) * 0.5
            ) / 5.0

        elif method_name == 'yake':
            method_domain_fit = (
                domain_scores.get('news', 0) * 0.9 +
                domain_scores.get('entertainment', 0) * 0.8 +
                domain_scores.get('sports', 0) * 0.8 +
                domain_scores.get('politics', 0) * 0.7 +
                domain_scores.get('business', 0) * 0.6 +
                structure_scores.get('informal', 0) * 0.7 +
                (1.0 - structure_scores.get('technical', 0)) * 0.6
            ) / 5.0

        elif method_name == 'textrank':
            method_domain_fit = (
                domain_scores.get('news', 0) * 0.8 +
                domain_scores.get('entertainment', 0) * 0.7 +
                domain_scores.get('environment', 0) * 0.7 +
                domain_scores.get('sports', 0) * 0.6 +
                domain_scores.get('health', 0) * 0.6 +
                structure_scores.get('narrative', 0) * 0.8 +
                structure_scores.get('descriptive', 0) * 0.7
            ) / 5.0

        confidence = (
            base_confidence * 0.4 +
            score_confidence * 0.3 +
            coherence * 0.1 +
            method_domain_fit * 0.2
        )

        confidence = max(0.1, min(1.0, confidence))

        return confidence

    def normalize_keyphrase(self, keyphrase: str) -> str:
        
        doc = self.nlp(keyphrase)

        tokens = [token.text for token in doc if token.pos_ != 'DET' and token.pos_ != 'ADP']

        normalized = ' '.join(tokens).strip()

        if self.clean_boundaries:
            normalized = self.clean_phrase_boundaries(normalized)

        if self.use_lemmatization:
            normalized = self.lemmatize_text(normalized)

        return normalized

    def get_embeddings(self, phrases: List[str], convert_to_tensor: bool = True) -> np.ndarray:
        try:
            embeddings = self.sentence_model.encode(
                phrases,
                convert_to_tensor=convert_to_tensor,
                show_progress_bar=False
            )

            if convert_to_tensor and isinstance(embeddings, torch.Tensor):
                return embeddings.cpu().numpy()

            return embeddings
        except Exception as e:
            print(f"Error getting embeddings: {str(e)}")
            return np.array([])

    def is_subphrase(self, phrase1: str, phrase2: str) -> bool:
        
        phrase1_lower = phrase1.lower()
        phrase2_lower = phrase2.lower()

        return phrase1_lower in phrase2_lower or phrase2_lower in phrase1_lower

    def remove_redundant_keyphrases(self, keyphrases: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        
        if not keyphrases:
            return []

        sorted_keyphrases = sorted(keyphrases, key=lambda x: x[1], reverse=True)

        phrases = [kp[0] for kp in sorted_keyphrases]
        scores = [kp[1] for kp in sorted_keyphrases]

        embeddings = self.get_embeddings(phrases, convert_to_tensor=False)

        if len(embeddings) == 0:
            return sorted_keyphrases

        similarity_matrix = cosine_similarity(embeddings)

        to_remove = set()

        merged_phrases = {}

        for i in range(len(phrases)):
            if i in to_remove:
                continue

            phrase_i = phrases[i].lower()
            words_i = set(phrase_i.split())

            lemma_i = self.lemmatize_text(phrase_i) if self.use_lemmatization else phrase_i

            for j in range(len(phrases)):
                if i == j or j in to_remove:
                    continue

                phrase_j = phrases[j].lower()
                words_j = set(phrase_j.split())

                lemma_j = self.lemmatize_text(phrase_j) if self.use_lemmatization else phrase_j

                is_subphrase = (phrase_i in phrase_j or phrase_j in phrase_i or
                            lemma_i in lemma_j or lemma_j in lemma_i)

                if not is_subphrase:
                    hyphen_i = phrase_i.replace(' ', '-')
                    hyphen_j = phrase_j.replace(' ', '-')
                    dehyphen_i = phrase_i.replace('-', ' ')
                    dehyphen_j = phrase_j.replace('-', ' ')

                    is_subphrase = (hyphen_i == phrase_j or hyphen_j == phrase_i or
                                dehyphen_i == phrase_j or dehyphen_j == phrase_i)

                if is_subphrase:
                    if len(words_i) > len(words_j):
                        if scores[i] >= 0.8 * scores[j]:
                            to_remove.add(j)
                            merged_phrases[j] = i
                    elif len(words_j) > len(words_i):
                        if scores[j] >= 0.8 * scores[i]:
                            to_remove.add(i)
                            merged_phrases[i] = j
                            break
                    else:
                        if scores[i] >= scores[j]:
                            to_remove.add(j)
                            merged_phrases[j] = i
                        else:
                            to_remove.add(i)
                            merged_phrases[i] = j
                            break

        for i in range(len(phrases)):
            if i in to_remove:
                continue

            phrase_i = phrases[i]
            len_i = len(phrase_i.split())

            words_i = set(phrase_i.lower().split())

            for j in range(i + 1, len(phrases)):
                if j in to_remove:
                    continue

                phrase_j = phrases[j]
                len_j = len(phrase_j.split())

                words_j = set(phrase_j.lower().split())

                if len(words_i) > 0 and len(words_j) > 0:
                    overlap = len(words_i.intersection(words_j))
                    overlap_ratio_i = overlap / len(words_i)
                    overlap_ratio_j = overlap / len(words_j)
                    max_overlap_ratio = max(overlap_ratio_i, overlap_ratio_j)
                else:
                    max_overlap_ratio = 0.0

                base_threshold = 0.82

                max_len = max(len_i, len_j)
                length_factor = max_len / 3
                length_adjustment = length_factor * 0.03

                overlap_adjustment = max_overlap_ratio * 0.05

                score_ratio = scores[i] / scores[j] if scores[j] > 0 else float('inf')
                score_adjustment = 0.0
                if score_ratio > 2.0 or score_ratio < 0.5:
                    score_adjustment = 0.03

                threshold = max(0.68, base_threshold - length_adjustment + overlap_adjustment + score_adjustment)

                is_similar = similarity_matrix[i, j] > threshold
                has_high_overlap = max_overlap_ratio > 0.8

                if is_similar or has_high_overlap:

                    score_factor = 1.0 if score_ratio > 1.2 else (-1.0 if score_ratio < 0.8 else 0.0)

                    length_factor = 0.5 if len_i > len_j else (-0.5 if len_j > len_i else 0.0)

                    specificity_factor = 0.3 if len_i > len_j else (-0.3 if len_j > len_i else 0.0)

                    decision_score = (score_factor * 0.6) + (length_factor * 0.3) + (specificity_factor * 0.1)

                    if decision_score > 0 or (decision_score == 0 and scores[i] >= scores[j]):
                        to_remove.add(j)
                        merged_phrases[j] = i
                    else:
                        to_remove.add(i)
                        merged_phrases[i] = j
                        break

        absorbed_counts = {}
        for removed, kept in merged_phrases.items():
            absorbed_counts[kept] = absorbed_counts.get(kept, 0) + 1

        boosted_scores = scores.copy()
        for i in range(len(phrases)):
            if i in absorbed_counts:
                boost_factor = 1.0 + (0.1 * math.log(1 + absorbed_counts[i]))
                boosted_scores[i] *= boost_factor

        filtered_keyphrases = [
            (phrases[i], boosted_scores[i]) for i in range(len(sorted_keyphrases)) if i not in to_remove
        ]

        filtered_keyphrases.sort(key=lambda x: x[1], reverse=True)

        return filtered_keyphrases
    def combine_keyphrases_ensemble(
    self,
    method_keyphrases: Dict[str, List[Tuple[str, float]]],
    text: str = None
    ) -> List[Tuple[str, float]]:
        
        combined_dict = {}

        if text is None:
            return self._simple_ensemble(method_keyphrases)

        doc_characteristics = self.classify_document_type(text)

        adaptive_weights = {}
        confidence_sum = 0.0

        for method_name, keyphrases in method_keyphrases.items():
            if not keyphrases:
                adaptive_weights[method_name] = 0.0
                continue

            confidence = self.estimate_method_confidence(method_name, keyphrases, doc_characteristics)
            adaptive_weights[method_name] = confidence
            confidence_sum += confidence

        if confidence_sum > 0:
            for method_name in adaptive_weights:
                adaptive_weights[method_name] /= confidence_sum
        else:
            adaptive_weights = {name: weight for name, weight in self.method_weights.items()}

        print(f"DEBUG: Adaptive method weights:")
        for method_name, weight in adaptive_weights.items():
            print(f"DEBUG:   - {method_name}: {weight:.4f}")

        for method_name, keyphrases in method_keyphrases.items():
            if not keyphrases:
                continue

            method_weight = adaptive_weights.get(method_name, 0.0)

            if method_weight <= 0.0:
                continue

            normalized_keyphrases = [(self.normalize_keyphrase(kp), score) for kp, score in keyphrases]

            normalized_keyphrases = [(kp, score) for kp, score in normalized_keyphrases if kp]

            for kp, score in normalized_keyphrases:
                kp_lower = kp.lower()
                weighted_score = score * method_weight

                if self.use_position_weight:
                    position_weight = self.calculate_position_weight(text, kp)
                    weighted_score *= position_weight

                if self.use_tfidf_weight and kp in self.word_tfidf_cache:
                    tfidf_weight = self.word_tfidf_cache.get(kp, 1.0)
                    weighted_score *= tfidf_weight

                if kp_lower in combined_dict:
                    existing_kp, existing_score, existing_votes, methods = combined_dict[kp_lower]

                    new_score = existing_score + weighted_score
                    new_votes = existing_votes + 1
                    methods.add(method_name)

                    combined_dict[kp_lower] = (existing_kp, new_score, new_votes, methods)
                else:
                    combined_dict[kp_lower] = (kp, weighted_score, 1, {method_name})

        combined_dict = self._enhance_diversity(combined_dict, text, doc_characteristics)

        normalized_dict = {}
        for kp_lower, (kp, score, votes, methods) in combined_dict.items():
            method_diversity_boost = min(2.0, 1.0 + (len(methods) - 1) * 0.6)

            num_words = len(kp.split())
            length_bonus = 1.0 + 0.15 * (num_words - 1)
            length_bonus = min(1.6, length_bonus)

            normalized_score = (score / votes) * method_diversity_boost * length_bonus

            if num_words > 1:
                print(f"DEBUG: Length bonus for '{kp}' ({num_words} words): {length_bonus:.2f}x")

            primary_domain = max(doc_characteristics['domain_scores'].items(), key=lambda x: x[1])[0]
            primary_domain_score = doc_characteristics['domain_scores'][primary_domain]

            domain_boost = 1.0
            if primary_domain_score > 0.3:
                domain_important_terms = {
                    'technology': [
                        'artificial intelligence', 'machine learning', 'deep learning', 'algorithm',
                        'data science', 'neural network', 'cloud computing', 'cybersecurity',
                        'blockchain', 'software development', 'programming', 'automation',
                        'digital transformation', 'big data', 'internet of things', 'computer vision',
                        'robotics', 'virtual reality', 'augmented reality', 'api', 'mobile app'
                    ],
                    'business': [
                        'market analysis', 'business strategy', 'investment', 'financial',
                        'economic', 'revenue', 'profit margin', 'supply chain', 'customer acquisition',
                        'marketing strategy', 'competitive advantage', 'business model', 'startup',
                        'venture capital', 'merger', 'acquisition', 'stock market', 'shareholders',
                        'corporate', 'management', 'leadership', 'ceo', 'executive'
                    ],
                    'health': [
                        'medical research', 'clinical trial', 'patient care', 'treatment',
                        'healthcare system', 'diagnosis', 'therapy', 'pharmaceutical',
                        'disease prevention', 'public health', 'mental health', 'wellness',
                        'chronic condition', 'medical technology', 'healthcare policy',
                        'hospital', 'doctor', 'nurse', 'vaccine', 'medication', 'surgery'
                    ],
                    'science': [
                        'scientific research', 'experiment', 'laboratory', 'hypothesis',
                        'theory', 'discovery', 'innovation', 'quantum', 'molecular',
                        'genetic', 'evolutionary', 'physics', 'chemistry', 'biology',
                        'astronomy', 'neuroscience', 'particle', 'atom', 'cell', 'dna',
                        'ecosystem', 'species', 'climate science'
                    ],
                    'news': [
                        'breaking news', 'latest development', 'report', 'announcement',
                        'press release', 'statement', 'interview', 'investigation',
                        'coverage', 'media', 'journalist', 'correspondent', 'source',
                        'official', 'spokesperson', 'exclusive', 'update', 'developing story'
                    ],
                    'academic': [
                        'research paper', 'academic study', 'scholarly article', 'publication',
                        'peer review', 'methodology', 'literature review', 'theoretical framework',
                        'empirical evidence', 'data analysis', 'findings', 'conclusion',
                        'contribution', 'citation', 'academic journal', 'university',
                        'professor', 'student', 'thesis', 'dissertation', 'faculty'
                    ],
                    'politics': [
                        'government policy', 'legislation', 'election', 'political party',
                        'campaign', 'voter', 'democracy', 'administration', 'congress',
                        'parliament', 'diplomatic', 'international relations', 'national security',
                        'public policy', 'constitutional', 'president', 'prime minister',
                        'senator', 'representative', 'bill', 'law', 'regulation'
                    ],
                    'environment': [
                        'climate change', 'global warming', 'sustainability', 'renewable energy',
                        'carbon emissions', 'conservation', 'biodiversity', 'ecosystem',
                        'environmental protection', 'pollution', 'green technology',
                        'natural resources', 'wildlife', 'environmental policy',
                        'recycling', 'solar power', 'wind energy', 'fossil fuels'
                    ],
                    'entertainment': [
                        'movie premiere', 'television series', 'music album', 'celebrity',
                        'box office', 'streaming service', 'award show', 'performance',
                        'director', 'actor', 'actress', 'artist', 'concert', 'festival',
                        'entertainment industry', 'film', 'tv show', 'song', 'album',
                        'hollywood', 'broadway', 'bestseller', 'video game'
                    ],
                    'sports': [
                        'championship', 'tournament', 'athlete', 'team', 'competition',
                        'olympic', 'world cup', 'league', 'season', 'player', 'coach',
                        'record', 'performance', 'sports event', 'victory', 'defeat',
                        'football', 'soccer', 'basketball', 'baseball', 'tennis',
                        'golf', 'racing', 'medal', 'stadium', 'fans'
                    ]
                }

                important_terms = domain_important_terms.get(primary_domain, [])

                if any(term.lower() in kp_lower or kp_lower in term.lower() for term in important_terms):
                    domain_boost = 1.5

            normalized_score *= domain_boost
            normalized_dict[kp_lower] = (kp, normalized_score)

        combined_list = [(kp, score) for kp_lower, (kp, score) in normalized_dict.items()]
        combined_list.sort(key=lambda x: x[1], reverse=True)

        return combined_list
    def _simple_ensemble(self, method_keyphrases: Dict[str, List[Tuple[str, float]]]) -> List[Tuple[str, float]]:
        
        combined_dict = {}

        for method_name, keyphrases in method_keyphrases.items():
            if not keyphrases:
                continue

            method_weight = self.method_weights.get(method_name, 0.25)

            normalized_keyphrases = [(self.normalize_keyphrase(kp), score) for kp, score in keyphrases]

            normalized_keyphrases = [(kp, score) for kp, score in normalized_keyphrases if kp]

            for kp, score in normalized_keyphrases:
                kp_lower = kp.lower()
                weighted_score = score * method_weight

                if kp_lower in combined_dict:
                    existing_kp, existing_score, existing_votes = combined_dict[kp_lower]

                    new_score = existing_score + weighted_score
                    new_votes = existing_votes + 1

                    combined_dict[kp_lower] = (existing_kp, new_score, new_votes)
                else:
                    combined_dict[kp_lower] = (kp, weighted_score, 1)

        normalized_dict = {}
        for kp_lower, (kp, score, votes) in combined_dict.items():
            vote_boost = min(2.0, 1.0 + (votes - 1) * 0.3)
            normalized_score = (score / votes) * vote_boost
            normalized_dict[kp_lower] = (kp, normalized_score)

        combined_list = [(kp, score) for kp_lower, (kp, score) in normalized_dict.items()]
        combined_list.sort(key=lambda x: x[1], reverse=True)

        return combined_list

    def _enhance_diversity(
    self,
    combined_dict: Dict[str, Tuple[str, float, int, Set[str]]],
    text: str,
    doc_characteristics: Dict[str, Any]
    ) -> Dict[str, Tuple[str, float, int, Set[str]]]:
        
        if len(combined_dict) <= 5:
            return combined_dict

        position_data = []
        for kp_lower, (kp, score, votes, methods) in combined_dict.items():
            position = text.lower().find(kp_lower)
            if position != -1:
                relative_position = position / len(text)
                position_data.append({
                    'keyphrase': kp,
                    'keyphrase_lower': kp_lower,
                    'score': score,
                    'votes': votes,
                    'methods': methods,
                    'position': position,
                    'relative_position': relative_position
                })

        quartiles = {
            'first_quarter': [],
            'second_quarter': [],
            'third_quarter': [],
            'fourth_quarter': []
        }

        for item in position_data:
            rel_pos = item['relative_position']
            if rel_pos < 0.25:
                quartiles['first_quarter'].append(item)
            elif rel_pos < 0.5:
                quartiles['second_quarter'].append(item)
            elif rel_pos < 0.75:
                quartiles['third_quarter'].append(item)
            else:
                quartiles['fourth_quarter'].append(item)

        enhanced_dict = {}

        total_keyphrases = len(combined_dict)
        target_counts = {
            'first_quarter': int(total_keyphrases * 0.5),
            'second_quarter': int(total_keyphrases * 0.3),
            'third_quarter': int(total_keyphrases * 0.15),
            'fourth_quarter': int(total_keyphrases * 0.05)
        }

        for quartile, items in quartiles.items():
            if items:
                target_counts[quartile] = max(1, target_counts[quartile])

        for quartile in quartiles:
            quartiles[quartile].sort(key=lambda x: x['score'], reverse=True)

        for quartile, items in quartiles.items():
            count = min(len(items), target_counts[quartile])
            for i in range(count):
                item = items[i]
                kp_lower = item['keyphrase_lower']
                kp = item['keyphrase']
                score = item['score']
                votes = item['votes']
                methods = item['methods']

                diversity_boost = 1.0
                if quartile == 'second_quarter':
                    diversity_boost = 1.05
                elif quartile == 'third_quarter':
                    diversity_boost = 1.1
                elif quartile == 'fourth_quarter':
                    diversity_boost = 1.15

                enhanced_dict[kp_lower] = (kp, score * diversity_boost, votes, methods)

        for kp_lower, (kp, score, votes, methods) in combined_dict.items():
            if kp_lower not in enhanced_dict:
                enhanced_dict[kp_lower] = (kp, score, votes, methods)

        return enhanced_dict

    def combine_keyphrases(
        self,
        keybert_keyphrases: List[Tuple[str, float]],
        multipartite_keyphrases: List[Tuple[str, float]],
        yake_keyphrases: List[Tuple[str, float]] = None,
        textrank_keyphrases: List[Tuple[str, float]] = None
    ) -> List[Tuple[str, float]]:
        
        if self.use_ensemble and (yake_keyphrases is not None or textrank_keyphrases is not None):
            method_keyphrases = {
                'keybert': keybert_keyphrases,
                'multipartiterank': multipartite_keyphrases
            }

            if yake_keyphrases is not None:
                method_keyphrases['yake'] = yake_keyphrases

            if textrank_keyphrases is not None:
                method_keyphrases['textrank'] = textrank_keyphrases

            return self.combine_keyphrases_ensemble(method_keyphrases)

        normalized_keybert = [(self.normalize_keyphrase(kp), score) for kp, score in keybert_keyphrases]
        normalized_multipartite = [(self.normalize_keyphrase(kp), score) for kp, score in multipartite_keyphrases]

        normalized_keybert = [(kp, score) for kp, score in normalized_keybert if kp]
        normalized_multipartite = [(kp, score) for kp, score in normalized_multipartite if kp]

        combined_dict = {}

        for kp, score in normalized_keybert:
            combined_dict[kp.lower()] = (kp, score)

        for kp, score in normalized_multipartite:
            kp_lower = kp.lower()
            if kp_lower in combined_dict:
                existing_kp, existing_score = combined_dict[kp_lower]
                new_score = max(existing_score, score) * 1.1
                combined_dict[kp_lower] = (existing_kp, min(1.0, new_score))
            else:
                combined_dict[kp_lower] = (kp, score)

        keys = list(combined_dict.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                key_i = keys[i]
                key_j = keys[j]

                if key_i not in combined_dict or key_j not in combined_dict:
                    continue

                if self.is_subphrase(key_i, key_j):
                    score_i = combined_dict[key_i][1]
                    score_j = combined_dict[key_j][1]

                    if score_i >= score_j:
                        del combined_dict[key_j]
                    else:
                        del combined_dict[key_i]
                        break

        combined_keyphrases = list(combined_dict.values())
        combined_keyphrases.sort(key=lambda x: x[1], reverse=True)

        return combined_keyphrases

    def extract_keyphrases(self, text: str) -> List[str]:
        
        keyphrases_with_scores = self.extract_keyphrases_with_scores(text)

        return [kp for kp, _ in keyphrases_with_scores]

    def extract_keyphrases_with_scores(self, text: str) -> List[Tuple[str, float]]:
        
        if not text or len(text.strip()) < 10:
            return []

        text_with_expansions = self.expand_technical_abbreviations(text)

        preprocessed_text = self.preprocess_text(text_with_expansions)

        print(f"DEBUG: Extracting keyphrases with use_position_weight={self.use_position_weight}, use_tfidf_weight={self.use_tfidf_weight}, use_ensemble={self.use_ensemble}")

        keybert_keyphrases = self.extract_with_keybert(preprocessed_text, nr_candidates=self.top_n)
        print(f"DEBUG: KeyBERT extracted {len(keybert_keyphrases)} keyphrases")

        multipartite_keyphrases = self.extract_with_multipartiterank(preprocessed_text, nr_candidates=self.top_n)
        print(f"DEBUG: MultipartiteRank extracted {len(multipartite_keyphrases)} keyphrases")

        yake_keyphrases = None
        textrank_keyphrases = None

        if self.use_ensemble:
            yake_keyphrases = self.extract_with_yake(preprocessed_text, nr_candidates=self.top_n)
            print(f"DEBUG: YAKE extracted {len(yake_keyphrases) if yake_keyphrases else 0} keyphrases")

            textrank_keyphrases = self.extract_with_textrank(preprocessed_text, nr_candidates=self.top_n)
            print(f"DEBUG: TextRank extracted {len(textrank_keyphrases) if textrank_keyphrases else 0} keyphrases")

            method_keyphrases = {
                'keybert': keybert_keyphrases,
                'multipartiterank': multipartite_keyphrases,
                'yake': yake_keyphrases,
                'textrank': textrank_keyphrases
            }
            combined_keyphrases = self.combine_keyphrases_ensemble(method_keyphrases, text=preprocessed_text)
            print(f"DEBUG: Ensemble combined into {len(combined_keyphrases)} keyphrases")
        else:
            combined_keyphrases = self.combine_keyphrases(
                keybert_keyphrases,
                multipartite_keyphrases,
                None,
                None
            )
            print(f"DEBUG: Simple combination produced {len(combined_keyphrases)} keyphrases")

        filtered_keyphrases = self.remove_redundant_keyphrases(combined_keyphrases)
        print(f"DEBUG: After redundancy removal: {len(filtered_keyphrases)} keyphrases")

        final_keyphrases = []
        multi_word_phrases = [kp.lower() for kp, _ in filtered_keyphrases if ' ' in kp]

        print(f"DEBUG: Before post-filtering: {len(filtered_keyphrases)} keyphrases")
        print(f"DEBUG: Multi-word phrases: {len(multi_word_phrases)}")

        for kp, score in filtered_keyphrases:
            if ' ' in kp:
                final_keyphrases.append((kp, score))
                continue

            kp_lower = kp.lower()
            is_subpart = any(kp_lower in phrase for phrase in multi_word_phrases)

            if not is_subpart:
                final_keyphrases.append((kp, score))
            else:
                print(f"DEBUG: Filtered out single word '{kp}' as it's part of multi-word phrase(s)")

        final_keyphrases.sort(key=lambda x: x[1], reverse=True)
        print(f"DEBUG: After post-filtering: {len(final_keyphrases)} keyphrases")

        top_keyphrases = final_keyphrases[:self.top_n]
        print(f"DEBUG: Final top {len(top_keyphrases)} keyphrases: {[kp for kp, _ in top_keyphrases]}")

        return top_keyphrases

    def expand_technical_abbreviations(self, text: str) -> str:
        
        tech_abbr = {
            'AI': 'artificial intelligence',
            'ML': 'machine learning',
            'NLP': 'natural language processing',
            'IoT': 'internet of things',
            'API': 'application programming interface',
            'UI': 'user interface',
            'UX': 'user experience',
            'OS': 'operating system',
            'CPU': 'central processing unit',
            'GPU': 'graphics processing unit',
            'RAM': 'random access memory',

            'HTML': 'hypertext markup language',
            'CSS': 'cascading style sheets',
            'JS': 'javascript',
            'SQL': 'structured query language',
            'HTTP': 'hypertext transfer protocol',
            'URL': 'uniform resource locator',
            'JSON': 'javascript object notation',
            'REST': 'representational state transfer',
            'SDK': 'software development kit',
            'IDE': 'integrated development environment',

            'ROI': 'return on investment',
            'CRM': 'customer relationship management',
            'ERP': 'enterprise resource planning',
            'SaaS': 'software as a service',
            'B2B': 'business to business',
            'B2C': 'business to consumer',
            'KPI': 'key performance indicator',

            'LLM': 'large language model',
            'CNN': 'convolutional neural network',
            'RNN': 'recurrent neural network',
            'LSTM': 'long short-term memory',
            'NER': 'named entity recognition',
            'TF-IDF': 'term frequency-inverse document frequency',
            'BERT': 'bidirectional encoder representations from transformers',
            'GPT': 'generative pre-trained transformer',

            'AWS': 'amazon web services',
            'GCP': 'google cloud platform',
            'CI/CD': 'continuous integration and continuous deployment',
            'DevOps': 'development operations',
            'VM': 'virtual machine',
            'VPC': 'virtual private cloud',

            'SSL': 'secure sockets layer',
            'VPN': 'virtual private network',
            'MFA': 'multi-factor authentication',
            'SSO': 'single sign-on',

            'AR': 'augmented reality',
            'VR': 'virtual reality',
            'IoMT': 'internet of medical things',
            'UAV': 'unmanned aerial vehicle',
            'EV': 'electric vehicle',

            'FAQ': 'frequently asked questions',
            'CEO': 'chief executive officer',
            'CTO': 'chief technology officer',
            'CFO': 'chief financial officer',
            'COO': 'chief operating officer',
            'CIO': 'chief information officer',
            'HR': 'human resources',
            'PR': 'public relations',
            'R&D': 'research and development'
        }

        pattern = r'\b(' + '|'.join(re.escape(abbr) for abbr in tech_abbr.keys()) + r')\b'

        def expand_match(match):
            abbr = match.group(0)
            return f"{abbr} ({tech_abbr[abbr]})"

        expanded_text = re.sub(pattern, expand_match, text)

        return expanded_text

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 * magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def batch_extract_keyphrases(self, texts: List[str], batch_size: int = 8) -> List[List[str]]:
        
        all_keyphrases = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting keyphrases"):
            batch_texts = texts[i:i + batch_size]
            batch_results = []

            for text in batch_texts:
                keyphrases = self.extract_keyphrases(text)
                batch_results.append(keyphrases)

            all_keyphrases.extend(batch_results)

        return all_keyphrases

    def evaluate_with_partial_matching(self, texts: List[str], ground_truth_keyphrases: List[List[str]]) -> Dict[str, float]:
        
        total_precision = 0
        total_recall = 0
        total_f1 = 0

        for i, (text, true_keyphrases) in enumerate(zip(texts, ground_truth_keyphrases)):
            predicted_keyphrases = self.extract_keyphrases(text)

            if self.use_lemmatization:
                predicted_lower = [self.lemmatize_text(kp.lower()) for kp in predicted_keyphrases]
                true_lower = [self.lemmatize_text(kp.lower()) for kp in true_keyphrases]
            else:
                predicted_lower = [kp.lower() for kp in predicted_keyphrases]
                true_lower = [kp.lower() for kp in true_keyphrases]

            exact_matches = set(predicted_lower) & set(true_lower)

            partial_matches = set()
            for pred in predicted_lower:
                if pred in exact_matches:
                    continue
                for true in true_lower:
                    if true in exact_matches:
                        continue
                    if pred in true or true in pred:
                        partial_matches.add((pred, true))

            tp = len(exact_matches) + 0.7 * len(partial_matches)
            fp = len(predicted_lower) - len(exact_matches) - len(set(p for p, _ in partial_matches))
            fn = len(true_lower) - len(exact_matches) - len(set(t for _, t in partial_matches))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            total_precision += precision
            total_recall += recall
            total_f1 += f1

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(texts)} texts")

        avg_precision = total_precision / len(texts)
        avg_recall = total_recall / len(texts)
        avg_f1 = total_f1 / len(texts)

        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1
        }

    def evaluate(self, texts: List[str], ground_truth_keyphrases: List[List[str]]) -> Dict[str, float]:
        
        print(f"DEBUG: Evaluating with use_partial_matching={self.use_partial_matching}, use_semantic_matching={self.use_semantic_matching}, use_lemmatization={self.use_lemmatization}")

        total_precision = 0
        total_recall = 0
        total_f1 = 0

        for i, (text, true_keyphrases) in enumerate(zip(texts, ground_truth_keyphrases)):
            predicted_keyphrases = self.extract_keyphrases(text)

            print(f"DEBUG: Evaluating text {i+1}: {len(predicted_keyphrases)} predicted vs {len(true_keyphrases)} true keyphrases")

            if self.use_lemmatization:
                predicted_lower = [self.lemmatize_text(kp.lower()) for kp in predicted_keyphrases]
                true_lower = [self.lemmatize_text(kp.lower()) for kp in true_keyphrases]
                print(f"DEBUG: Applied lemmatization")
            else:
                predicted_lower = [kp.lower() for kp in predicted_keyphrases]
                true_lower = [kp.lower() for kp in true_keyphrases]

            exact_matches = set(predicted_lower) & set(true_lower)
            print(f"DEBUG: Found {len(exact_matches)} exact matches")

            partial_matches = set()
            semantic_matches = []

            if self.use_partial_matching:
                for pred in predicted_lower:
                    if pred in exact_matches:
                        continue
                    for true in true_lower:
                        if true in exact_matches:
                            continue
                        if pred in true or true in pred:
                            partial_matches.add((pred, true))
                print(f"DEBUG: Found {len(partial_matches)} partial matches")

            if self.use_semantic_matching and (set(predicted_lower) - exact_matches) and (set(true_lower) - exact_matches):
                remaining_pred = [p for p in predicted_lower if p not in exact_matches and not any(p in t or t in p for _, t in partial_matches)]
                remaining_true = [t for t in true_lower if t not in exact_matches and not any(p in t or t in p for p, _ in partial_matches)]

                if remaining_pred and remaining_true:
                    pred_embeddings = self.get_embeddings(remaining_pred)
                    true_embeddings = self.get_embeddings(remaining_true)

                    similarity_matrix = cosine_similarity(pred_embeddings, true_embeddings)

                    for p_idx, p in enumerate(remaining_pred):
                        for t_idx, t in enumerate(remaining_true):
                            similarity = similarity_matrix[p_idx, t_idx]
                            if similarity > 0.6:
                                semantic_matches.append((p, t, similarity))
                print(f"DEBUG: Found {len(semantic_matches)} semantic matches")

            if self.use_partial_matching or self.use_semantic_matching:
                tp = len(exact_matches) + 0.7 * len(partial_matches) + 0.8 * len(semantic_matches)
                fp = len(predicted_lower) - len(exact_matches) - len(set(p for p, _ in partial_matches)) - len(set(p for p, _, _ in semantic_matches))
                fn = len(true_lower) - len(exact_matches) - len(set(t for _, t in partial_matches)) - len(set(t for _, t, _ in semantic_matches))
            else:
                tp = len(exact_matches)
                fp = len(set(predicted_lower) - exact_matches)
                fn = len(set(true_lower) - exact_matches)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"DEBUG: Metrics - P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")

            total_precision += precision
            total_recall += recall
            total_f1 += f1

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(texts)} texts")

        avg_precision = total_precision / len(texts)
        avg_recall = total_recall / len(texts)
        avg_f1 = total_f1 / len(texts)

        print(f"DEBUG: Final metrics - P={avg_precision:.4f}, R={avg_recall:.4f}, F1={avg_f1:.4f}")

        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1
        }

    def evaluate_with_semantic_matching(self, texts: List[str], ground_truth_keyphrases: List[List[str]]) -> Dict[str, float]:
        
        total_precision = 0
        total_recall = 0
        total_f1 = 0

        for i, (text, true_keyphrases) in enumerate(zip(texts, ground_truth_keyphrases)):
            predicted_keyphrases = self.extract_keyphrases(text)

            if self.use_lemmatization:
                predicted_processed = [self.lemmatize_text(kp.lower()) for kp in predicted_keyphrases]
                true_processed = [self.lemmatize_text(kp.lower()) for kp in true_keyphrases]
            else:
                predicted_processed = [kp.lower() for kp in predicted_keyphrases]
                true_processed = [kp.lower() for kp in true_keyphrases]

            exact_matches = set(predicted_processed) & set(true_processed)

            remaining_pred = [p for p in predicted_processed if p not in exact_matches]
            remaining_true = [t for t in true_processed if t not in exact_matches]

            if not remaining_pred or not remaining_true:
                semantic_matches = []
            else:
                pred_embeddings = self.get_embeddings(remaining_pred)
                true_embeddings = self.get_embeddings(remaining_true)

                similarity_matrix = cosine_similarity(pred_embeddings, true_embeddings)

                semantic_matches = []
                for p_idx, p in enumerate(remaining_pred):
                    for t_idx, t in enumerate(remaining_true):
                        similarity = similarity_matrix[p_idx, t_idx]
                        if similarity > 0.6:
                            semantic_matches.append((p, t, similarity))

            tp = len(exact_matches) + len(semantic_matches)
            fp = len(predicted_processed) - len(exact_matches) - len(set(p for p, _, _ in semantic_matches))
            fn = len(true_processed) - len(exact_matches) - len(set(t for _, t, _ in semantic_matches))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            total_precision += precision
            total_recall += recall
            total_f1 += f1

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(texts)} texts")

        avg_precision = total_precision / len(texts)
        avg_recall = total_recall / len(texts)
        avg_f1 = total_f1 / len(texts)

        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1
        }

    def evaluate_with_details(self, texts: List[str], ground_truth_keyphrases: List[List[str]]) -> Dict:
        
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        per_article_results = []

        for i, (text, true_keyphrases) in enumerate(zip(texts, ground_truth_keyphrases)):
            predicted_keyphrases = self.extract_keyphrases(text)

            if self.use_lemmatization:
                predicted_processed = [self.lemmatize_text(kp.lower()) for kp in predicted_keyphrases]
                true_processed = [self.lemmatize_text(kp.lower()) for kp in true_keyphrases]
            else:
                predicted_processed = [kp.lower() for kp in predicted_keyphrases]
                true_processed = [kp.lower() for kp in true_keyphrases]

            exact_matches = set(predicted_processed) & set(true_processed)

            partial_matches = set()
            if self.use_partial_matching:
                for pred in predicted_processed:
                    if pred in exact_matches:
                        continue
                    for true in true_processed:
                        if true in exact_matches:
                            continue
                        if pred in true or true in pred:
                            partial_matches.add((pred, true))

            semantic_matches = []
            if self.use_semantic_matching and (set(predicted_processed) - exact_matches) and (set(true_processed) - exact_matches):
                remaining_pred = [p for p in predicted_processed if p not in exact_matches and not any(p in t or t in p for _, t in partial_matches)]
                remaining_true = [t for t in true_processed if t not in exact_matches and not any(p in t or t in p for p, _ in partial_matches)]

                if remaining_pred and remaining_true:
                    pred_embeddings = self.get_embeddings(remaining_pred)
                    true_embeddings = self.get_embeddings(remaining_true)

                    similarity_matrix = cosine_similarity(pred_embeddings, true_embeddings)

                    for p_idx, p in enumerate(remaining_pred):
                        for t_idx, t in enumerate(remaining_true):
                            similarity = similarity_matrix[p_idx, t_idx]
                            if similarity > 0.6:
                                semantic_matches.append((p, t, similarity_matrix[p_idx, t_idx]))

            if self.use_partial_matching or self.use_semantic_matching:
                tp = len(exact_matches) + 0.7 * len(partial_matches) + 0.8 * len(semantic_matches)
                fp = len(predicted_processed) - len(exact_matches) - len(set(p for p, _ in partial_matches)) - len(set(p for p, _, _ in semantic_matches))
                fn = len(true_processed) - len(exact_matches) - len(set(t for _, t in partial_matches)) - len(set(t for _, t, _ in semantic_matches))

            else:
                tp = len(exact_matches)
                fp = len(set(predicted_processed) - exact_matches)
                fn = len(set(true_processed) - exact_matches)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            total_precision += precision
            total_recall += recall
            total_f1 += f1

            text_snippet = text[:100] + "..." if len(text) > 100 else text
            per_article_results.append({
                "text_snippet": text_snippet,
                "predicted_keyphrases": predicted_keyphrases,
                "true_keyphrases": true_keyphrases,
                "exact_matches": list(exact_matches),
                "partial_matches": [(p, t) for p, t in partial_matches] if self.use_partial_matching else [],
                "semantic_matches": [(p, t) for p, t, _ in semantic_matches] if self.use_semantic_matching else [],
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(texts)} texts")

        avg_precision = total_precision / len(texts)
        avg_recall = total_recall / len(texts)
        avg_f1 = total_f1 / len(texts)

        all_false_positives = []
        all_false_negatives = []

        for result in per_article_results:
            all_matches_pred = set(result["exact_matches"])
            all_matches_pred.update([p for p, _ in result["partial_matches"]])
            all_matches_pred.update([p for p, _ in result["semantic_matches"]])

            all_matches_true = set(result["exact_matches"])
            all_matches_true.update([t for _, t in result["partial_matches"]])
            all_matches_true.update([t for _, t in result["semantic_matches"]])

            if self.use_lemmatization:
                predicted_processed = [self.lemmatize_text(kp.lower()) for kp in result["predicted_keyphrases"]]
                true_processed = [self.lemmatize_text(kp.lower()) for kp in result["true_keyphrases"]]
            else:
                predicted_processed = [kp.lower() for kp in result["predicted_keyphrases"]]
                true_processed = [kp.lower() for kp in result["true_keyphrases"]]

            false_positives = [p for p in predicted_processed if p not in all_matches_pred]
            false_negatives = [t for t in true_processed if t not in all_matches_true]

            all_false_positives.extend(false_positives)
            all_false_negatives.extend(false_negatives)

        fp_counts = {}
        fn_counts = {}

        for fp in all_false_positives:
            fp_counts[fp] = fp_counts.get(fp, 0) + 1

        for fn in all_false_negatives:
            fn_counts[fn] = fn_counts.get(fn, 0) + 1

        common_fps = sorted(fp_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        common_fns = sorted(fn_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1": avg_f1,
            "per_article_results": per_article_results,
            "common_false_positives": [fp for fp, _ in common_fps],
            "common_false_negatives": [fn for fn, _ in common_fns]
        }
