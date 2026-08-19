import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet as wn
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
import spacy
import time
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List, Dict, Tuple, Union, Optional, Set
from collections import defaultdict

try:
    nltk.data.find('wordnet')
    nltk.data.find('stopwords')
    nltk.data.find('punkt')
except LookupError:
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('punkt')

class ContextualKeyphraseExpander:
    

    def __init__(
        self,
        model_name="all-mpnet-base-v2",
        use_gpu=True,
        similarity_threshold=0.5,
        max_suggestions=3,
        use_wordnet=True,
        use_spacy=True,
        use_collocations=True,
        use_pos_patterns=False,
        pos_pattern="<J.*>*<N.*>+",
        use_keybert=False,
        keybert_diversity=0.5,
        keybert_model=None,
        sentence_model=None,
        use_phrase_quality_check=True
    ):
        
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.similarity_threshold = similarity_threshold
        self.max_suggestions = max_suggestions
        self.use_wordnet = use_wordnet
        self.use_spacy = use_spacy
        self.use_collocations = use_collocations
        self.use_pos_patterns = use_pos_patterns
        self.pos_pattern = pos_pattern
        self.use_keybert = use_keybert
        self.keybert_diversity = keybert_diversity
        self.keybert_model = keybert_model
        self.use_phrase_quality_check = use_phrase_quality_check
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stopwords = set(stopwords.words('english'))
        except Exception as e:
            print(f"Warning: Could not initialize NLTK resources: {str(e)}")
            print("Using fallback stopwords list")
            self.lemmatizer = None
            self.stopwords = set(['a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                                 'while', 'of', 'to', 'in', 'for', 'on', 'by', 'with', 'about', 'against',
                                 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                                 'from', 'up', 'down', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
                                 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'i', 'me', 'my',
                                 'mine', 'myself', 'you', 'your', 'yours', 'yourself', 'he', 'him', 'his',
                                 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'we', 'us',
                                 'our', 'ours', 'ourselves', 'they', 'them', 'their', 'theirs', 'themselves',
                                 'this', 'that', 'these', 'those', 'which', 'who', 'whom', 'whose', 'when',
                                 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
                                 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                                 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])

        if sentence_model is not None:
            print("Using provided sentence transformer model")
            self.sentence_model = sentence_model
        else:
            print(f"Loading sentence transformer model: {model_name}")
            try:
                try:
                    from transformers import AutoModel, AutoTokenizer
                    print("Using HuggingFace Transformers for advanced embeddings")
                    if '/' in model_name and not model_name.startswith('sentence-transformers/'):
                        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                        self._model = AutoModel.from_pretrained(model_name)
                        self._using_hf_transformers = True
                        self.sentence_model = type('', (), {'encode': self._encode_with_transformers})()
                    else:
                        self.sentence_model = SentenceTransformer(model_name)
                        self._using_hf_transformers = False
                except ImportError:
                    self.sentence_model = SentenceTransformer(model_name)
                    self._using_hf_transformers = False
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print("Falling back to default SentenceTransformer model")
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                self._using_hf_transformers = False

        if self.use_spacy:
            try:
                print("Loading spaCy model")
                self.nlp = spacy.load("en_core_web_sm")
                print("spaCy model loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load spaCy model: {str(e)}")
                print("Disabling spaCy functionality")
                self.use_spacy = False

        self.keybert_instance = None
        if self.use_keybert:
            try:
                if self.keybert_model is not None:
                    self.keybert_instance = self.keybert_model
                    print("Using provided KeyBERT model")
                else:
                    print("KeyBERT support enabled. To use it, uncomment the KeyBERT import and initialization.")
            except ImportError:
                print("Could not initialize KeyBERT. Please install it with: pip install keybert")
                self.use_keybert = False

        self._using_hf_transformers = getattr(self, '_using_hf_transformers', False)
        self._tokenizer = getattr(self, '_tokenizer', None)
        self._model = getattr(self, '_model', None)

        self._init_curated_expansions()

        print("Contextual Keyphrase Expander initialized")

    def _init_curated_expansions(self):
        
        self.curated_expansions = {
            "agriculture": [
                ("sustainable agriculture", 0.95),
                ("precision agriculture", 0.93),
                ("agricultural technology", 0.92),
                ("farming practices", 0.90),
                ("agricultural innovation", 0.89)
            ],
            "precision breeding act": [
                ("genetic improvement legislation", 0.95),
                ("breeding regulation", 0.93),
                ("agricultural genetics law", 0.92),
                ("crop improvement legislation", 0.91),
                ("plant breeding regulation", 0.90)
            ],
            "food security": [
                ("food supply chain", 0.95),
                ("food availability", 0.93),
                ("food access", 0.92),
                ("food stability", 0.91),
                ("food utilization", 0.90)
            ],
            "precision breeding": [
                ("genetic improvement", 0.95),
                ("targeted breeding", 0.93),
                ("selective breeding", 0.92),
                ("precision genetics", 0.91),
                ("advanced breeding techniques", 0.90)
            ],
            "climate change": [
                ("global warming", 0.95),
                ("climate crisis", 0.93),
                ("climate action", 0.92),
                ("climate adaptation", 0.91),
                ("climate mitigation", 0.90)
            ],
            "ecologically sound": [
                ("environmentally sustainable", 0.95),
                ("eco-friendly", 0.93),
                ("environmentally responsible", 0.92),
                ("ecologically sustainable", 0.91),
                ("environmentally sound", 0.90)
            ],

            "ransomware": [
                ("ransomware attacks", 0.95),
                ("ransomware protection", 0.93),
                ("ransomware prevention", 0.92),
                ("ransomware defense", 0.91),
                ("ransomware detection", 0.90)
            ],
            "ransomware activity": [
                ("ransomware attack campaigns", 0.95),
                ("ransomware threat operations", 0.93),
                ("ransomware criminal activity", 0.92),
                ("ransomware attack patterns", 0.91),
                ("ransomware infection vectors", 0.90)
            ],
            "cybercriminals": [
                ("threat actors", 0.95),
                ("malicious actors", 0.93),
                ("cyber attackers", 0.92),
                ("hacker groups", 0.91),
                ("criminal hackers", 0.90)
            ],
            "cyber threat": [
                ("cyber risk", 0.95),
                ("cyber attack vector", 0.93),
                ("cyber vulnerability", 0.92),
                ("cyber threat intelligence", 0.91),
                ("cyber threat landscape", 0.90)
            ],
            "lockbit": [
                ("lockbit ransomware", 0.95),
                ("lockbit ransomware group", 0.93),
                ("lockbit threat actor", 0.92),
                ("lockbit malware", 0.91),
                ("lockbit attack campaign", 0.90)
            ],
            "ddos": [
                ("distributed denial of service", 0.95),
                ("ddos attack", 0.93),
                ("ddos protection", 0.92),
                ("ddos mitigation", 0.91),
                ("network flooding", 0.90)
            ],
            "enterprise defense": [
                ("corporate cybersecurity", 0.95),
                ("enterprise security", 0.93),
                ("business cyber protection", 0.92),
                ("organizational security posture", 0.91),
                ("corporate security strategy", 0.90)
            ],

            "artificial intelligence": [
                ("machine learning", 0.95),
                ("deep learning", 0.93),
                ("neural networks", 0.92),
                ("ai algorithms", 0.91),
                ("ai systems", 0.90)
            ],
            "machine learning": [
                ("supervised learning", 0.95),
                ("unsupervised learning", 0.93),
                ("reinforcement learning", 0.92),
                ("predictive models", 0.91),
                ("ml algorithms", 0.90)
            ],

            "electric vehicles": [
                ("ev charging", 0.95),
                ("battery electric vehicles", 0.93),
                ("ev infrastructure", 0.92),
                ("zero-emission vehicles", 0.91),
                ("electric mobility", 0.90)
            ],
            "autonomous driving": [
                ("self-driving cars", 0.95),
                ("autonomous vehicles", 0.93),
                ("driver assistance systems", 0.92),
                ("autonomous navigation", 0.91),
                ("vehicle automation", 0.90)
            ],

            "renewable energy": [
                ("solar power", 0.95),
                ("wind energy", 0.93),
                ("clean energy", 0.92),
                ("sustainable energy", 0.91),
                ("green power", 0.90)
            ],
            "carbon emissions": [
                ("greenhouse gas emissions", 0.95),
                ("carbon footprint", 0.93),
                ("emission reduction", 0.92),
                ("carbon neutrality", 0.91),
                ("decarbonization", 0.90)
            ],

            "real estate market": [
                ("housing market", 0.95),
                ("property values", 0.93),
                ("real estate trends", 0.92),
                ("market conditions", 0.91),
                ("property investment", 0.90)
            ],
            "property development": [
                ("real estate development", 0.95),
                ("construction projects", 0.93),
                ("land development", 0.92),
                ("urban development", 0.91),
                ("property construction", 0.90)
            ],

            "streaming services": [
                ("video streaming", 0.95),
                ("content platforms", 0.93),
                ("subscription services", 0.92),
                ("on-demand content", 0.91),
                ("digital entertainment", 0.90)
            ],
            "film industry": [
                ("movie production", 0.95),
                ("cinema business", 0.93),
                ("filmmaking", 0.92),
                ("movie studios", 0.91),
                ("film distribution", 0.90)
            ]
        }

    def _encode_with_transformers(self, sentences, show_progress_bar=False, convert_to_tensor=True):
        
        import torch
        import numpy as np

        if isinstance(sentences, str):
            sentences = [sentences]

        encoded_input = self._tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        if self.use_gpu and torch.cuda.is_available():
            encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
            self._model = self._model.cuda()

        with torch.no_grad():
            model_output = self._model(**encoded_input)

        if hasattr(model_output, 'pooler_output'):
            embeddings = model_output.pooler_output
        else:
            attention_mask = encoded_input['attention_mask']
            last_hidden = model_output.last_hidden_state

            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            embeddings = sum_embeddings / sum_mask

        if embeddings.is_cuda:
            embeddings = embeddings.cpu()

        return embeddings.numpy()

    def _check_phrase_quality(self, phrase):
        
        if not self.use_phrase_quality_check:
            return True

        if len(phrase.split()) < 2:
            return True

        if self.stopwords:
            words = phrase.lower().split()
            stopword_ratio = sum(1 for word in words if word in self.stopwords) / len(words)
            if stopword_ratio > 0.5:
                return False

        if len(phrase.split()) > 5:
            return False

        if self.stopwords:
            words = phrase.lower().split()
            if words[0] in self.stopwords or words[-1] in self.stopwords:
                return False

        if self.use_spacy:
            doc = self.nlp(phrase)
            pos_tags = [token.pos_ for token in doc]

            unusual_patterns = [
                ['ADJ', 'ADJ', 'ADJ'],
                ['VERB', 'VERB'],
                ['ADV', 'ADV'],
                ['DET', 'VERB'],
                ['VERB', 'DET', 'VERB'],
                ['ADP', 'ADP'],
                ['VERB', 'PUNCT'],
                ['ADP', 'PUNCT']
            ]

            for pattern in unusual_patterns:
                if len(pos_tags) >= len(pattern):
                    for i in range(len(pos_tags) - len(pattern) + 1):
                        if pos_tags[i:i+len(pattern)] == pattern:
                            return False

            if pos_tags and pos_tags[-1] in ['ADP', 'DET', 'CCONJ', 'SCONJ']:
                return False

            if len(pos_tags) > 1 and 'NOUN' not in pos_tags and 'PROPN' not in pos_tags:
                return False

            noun_count = sum(1 for tag in pos_tags if tag in ['NOUN', 'PROPN'])
            if len(pos_tags) > 2 and noun_count / len(pos_tags) < 0.25:
                return False

            if len(pos_tags) > 2 and all(tag in ['DET', 'PROPN'] for tag in pos_tags):
                return True

        return True

    def expand_keyphrases(self, keyphrases, text, domain=None, num_suggestions=None, min_quality_score=0.68, use_curated=True):
        
        start_time = time.time()
        print(f"Expanding {len(keyphrases)} keyphrases...")

        if num_suggestions is None:
            num_suggestions = self.max_suggestions

        doc_embedding = self.sentence_model.encode([text], show_progress_bar=False)[0]

        keyphrase_texts = [kp for kp, _ in keyphrases]
        keyphrase_embeddings = self.sentence_model.encode(keyphrase_texts, show_progress_bar=False)

        expanded_keyphrases = {}

        no_suggestions_keyphrases = []

        for i, (keyphrase, original_score) in enumerate(keyphrases):
            print(f"Expanding keyphrase: {keyphrase}")

            if original_score < 0.05 and len(keyphrases) > 5:
                expanded_keyphrases[keyphrase] = []
                continue

            if use_curated and keyphrase.lower() in self.curated_expansions:
                print(f"Using curated expansions for '{keyphrase}'")
                expanded_keyphrases[keyphrase] = self.curated_expansions[keyphrase.lower()][:num_suggestions]
                continue

            candidates = self._get_candidate_expansions(keyphrase, text, domain)

            if not candidates:
                no_suggestions_keyphrases.append((keyphrase, i))
                expanded_keyphrases[keyphrase] = []
                continue

            if self.use_phrase_quality_check:
                candidates = [c for c in candidates if self._check_phrase_quality(c)]
                if not candidates:
                    no_suggestions_keyphrases.append((keyphrase, i))
                    expanded_keyphrases[keyphrase] = []
                    continue

            candidate_embeddings = self.sentence_model.encode(candidates, show_progress_bar=False)

            keyphrase_similarities = cosine_similarity(
                [keyphrase_embeddings[i]],
                candidate_embeddings
            )[0]

            doc_similarities = cosine_similarity(
                [doc_embedding],
                candidate_embeddings
            )[0]

            enhanced_contextual_scores = self._get_enhanced_contextual_relevance_scores(candidates, text, keyphrase=keyphrase)
            sentence_context_scores = self._get_sentence_context_scores(candidates, text, keyphrase=keyphrase)
            partial_match_scores = self._get_partial_match_scores(candidates, keyphrase)

            domain_relevance = np.zeros_like(enhanced_contextual_scores)
            if domain:
                domain_relevance = self._calculate_domain_relevance(candidates, domain)

            keyphrase_length = len(keyphrase.split())
            keyphrase_weight = max(0.25, min(0.4, 0.25 + 0.05 * keyphrase_length))
            doc_weight = 0.15

            length_adjustments = np.zeros_like(enhanced_contextual_scores)
            for i, candidate in enumerate(candidates):
                words = candidate.split()
                if len(words) == 2 or len(words) == 3:
                    length_adjustments[i] = 0.1
                elif len(words) == 4:
                    length_adjustments[i] = 0.05
                elif len(words) > 5:
                    length_adjustments[i] = -0.05
                elif len(words) == 1 and (domain_relevance[i] < 0.7 if domain else True):
                    length_adjustments[i] = -0.1

            quality_adjustments = np.zeros_like(enhanced_contextual_scores)
            for i, candidate in enumerate(candidates):
                candidate_lower = candidate.lower()

                if candidate_lower == keyphrase.lower() or \
                   (len(candidate_lower) > 3 and (candidate_lower in keyphrase.lower() or keyphrase.lower() in candidate_lower)):
                    quality_adjustments[i] = -0.15

                common_words = ["thing", "stuff", "item", "part", "piece", "aspect", "element", "component"]
                if any(word in candidate_lower.split() for word in common_words) and (domain_relevance[i] < 0.7 if domain else True):
                    quality_adjustments[i] = -0.1

            combined_scores = (keyphrase_weight * keyphrase_similarities +
                              doc_weight * doc_similarities +
                              0.15 * enhanced_contextual_scores +
                              0.10 * sentence_context_scores +
                              0.10 * partial_match_scores +
                              0.15 * domain_relevance +
                              length_adjustments +
                              quality_adjustments)

            combined_scores = np.clip(combined_scores, 0, 1)

            adaptive_threshold = max(min_quality_score, self._get_adaptive_threshold(keyphrase, domain))

            valid_indices = [idx for idx, score in enumerate(combined_scores) if score >= adaptive_threshold]

            if not valid_indices:
                no_suggestions_keyphrases.append((keyphrase, i))
                expanded_keyphrases[keyphrase] = []
                continue

            sorted_indices = sorted(valid_indices, key=lambda idx: combined_scores[idx], reverse=True)

            suggestions = []
            for idx in sorted_indices[:num_suggestions * 2]:
                score = float(combined_scores[idx])
                if score >= adaptive_threshold:
                    suggestions.append((candidates[idx], score))

            filtered_suggestions = self._filter_duplicate_suggestions(suggestions)

            expanded_keyphrases[keyphrase] = filtered_suggestions[:num_suggestions]

        important_keyphrases = [(kp, i) for kp, i in no_suggestions_keyphrases
                               if kp in [k for k, _ in keyphrases[:5]]]

        if important_keyphrases and domain:
            print(f"Applying fallback methods for {len(important_keyphrases)} important keyphrases with no suggestions...")
            for keyphrase, _ in important_keyphrases:
                fallback_suggestions = self._get_fallback_suggestions(keyphrase, expanded_keyphrases, domain)

                good_fallbacks = [s for s in fallback_suggestions if s[1] >= min_quality_score]
                if good_fallbacks:
                    expanded_keyphrases[keyphrase] = good_fallbacks
                    print(f"Found {len(good_fallbacks)} quality fallback suggestions for '{keyphrase}'")

        processing_time = time.time() - start_time
        print(f"Expansion completed in {processing_time:.2f}s")

        return expanded_keyphrases

    def _get_candidate_expansions(self, keyphrase, text, domain=None):
        
        candidates = set()

        if domain:
            domain_candidates = self._get_domain_expansions(keyphrase, domain)
            candidates.update(domain_candidates)

            if domain_candidates and len(domain.split()) == 1:
                related_domains = {
                    'cybersecurity': ['technology', 'artificial intelligence'],
                    'technology': ['cybersecurity', 'artificial intelligence'],
                    'artificial intelligence': ['technology', 'cybersecurity'],
                    'ai': ['artificial intelligence', 'technology', 'cybersecurity'],
                    'virtual reality': ['entertainment', 'technology', 'gaming'],
                    'vr': ['virtual reality', 'entertainment', 'technology', 'gaming'],
                    'entertainment': ['virtual reality', 'technology', 'gaming'],
                    'real estate': ['economics', 'business', 'housing'],
                    'automotive': ['technology', 'environment', 'transportation'],
                    'environment': ['climate', 'energy', 'automotive'],
                    'food': ['health', 'agriculture', 'nutrition'],
                    'health': ['food', 'medicine', 'nutrition']
                }

                domain_lower = domain.lower()
                if domain_lower in related_domains:
                    for related_domain in related_domains[domain_lower]:
                        related_expansions = self._get_domain_expansions(keyphrase, related_domain)
                        candidates.update(related_expansions)

        if self.use_spacy:
            spacy_candidates = self._get_spacy_expansions(keyphrase, text)
            candidates.update(spacy_candidates)

        if self.use_collocations:
            collocation_candidates = self._get_collocation_expansions(keyphrase, text)
            candidates.update(collocation_candidates)

        if self.use_pos_patterns:
            pos_pattern_candidates = self._get_pos_pattern_expansions(keyphrase, text)
            candidates.update(pos_pattern_candidates)

        if self.use_keybert and self.keybert_instance is not None:
            keybert_candidates = self._get_keybert_expansions(keyphrase, text)
            candidates.update(keybert_candidates)

        if self.use_wordnet:
            context_terms = self._extract_context_terms(text, keyphrase) if text else []
            wordnet_candidates = self._get_wordnet_expansions(keyphrase, context_terms=context_terms)
            candidates.update(wordnet_candidates)

        if keyphrase in candidates:
            candidates.remove(keyphrase)

        filtered_candidates = self._filter_trivial_variations(keyphrase, candidates)

        if domain and filtered_candidates:
            filtered_candidates = self._apply_domain_quality_filter(list(filtered_candidates), domain, keyphrase)

        return list(filtered_candidates)

    def _extract_context_terms(self, text, keyphrase, window_size=100, max_terms=20):
        
        if not text or not keyphrase:
            return []

        text_lower = text.lower()
        keyphrase_lower = keyphrase.lower()

        context_terms = set()
        start_pos = 0

        while start_pos < len(text_lower):
            pos = text_lower.find(keyphrase_lower, start_pos)
            if pos == -1:
                break

            context_start = max(0, pos - window_size)
            context_end = min(len(text_lower), pos + len(keyphrase_lower) + window_size)
            context = text_lower[context_start:context_end]

            if self.nlp is not None:
                doc = self.nlp(context)
                for token in doc:
                    if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and len(token.text) > 2:
                        if token.text.lower() not in self.stopwords:
                            context_terms.add(token.text.lower())
            else:
                tokens = context.split()
                for token in tokens:
                    if len(token) > 2 and token not in self.stopwords:
                        context_terms.add(token)

            start_pos = pos + 1

            if len(context_terms) >= max_terms:
                break

        return list(context_terms)

    def _apply_domain_quality_filter(self, candidates, domain, keyphrase):
        
        if not candidates or not domain:
            return candidates

        domain_bad_words = {
            'cybersecurity': ['somebody', 'someone', 'individual', 'person', 'human'],
            'artificial intelligence': ['somebody', 'someone', 'individual', 'person', 'human'],
            'ai': ['somebody', 'someone', 'individual', 'person', 'human'],
            'virtual reality': ['realism', 'realness', 'actuality', 'telephone', 'receiver', 'game of chance', 'bet on', 'double up', 'gambling'],
            'vr': ['realism', 'realness', 'actuality', 'telephone', 'receiver', 'game of chance', 'bet on', 'double up', 'gambling'],
            'gaming': ['game of chance', 'bet on', 'double up', 'gambling', 'wager'],
            'real estate': ['biological process', 'organic process'],
            'environment': ['biological process', 'organic process'],
            'food': ['biological process', 'organic process']
        }

        domain_good_words = {
            'cybersecurity': ['cyber', 'security', 'threat', 'attack', 'malware', 'phishing', 'ransomware', 'vulnerability', 'breach', 'hack', 'protection', 'defense'],
            'artificial intelligence': ['ai', 'machine learning', 'deep learning', 'neural', 'algorithm', 'model', 'data', 'training', 'prediction'],
            'ai': ['artificial intelligence', 'machine learning', 'deep learning', 'neural', 'algorithm', 'model', 'data', 'training', 'prediction'],
            'virtual reality': ['vr', 'headset', 'immersive', 'experience', '3d', 'virtual', 'simulation', 'gaming'],
            'vr': ['virtual reality', 'headset', 'immersive', 'experience', '3d', 'virtual', 'simulation', 'gaming'],
            'gaming': ['game', 'player', 'console', 'virtual', 'interactive', 'entertainment'],
            'real estate': ['housing', 'property', 'mortgage', 'market', 'home', 'building', 'rent', 'sale', 'price'],
            'environment': ['climate', 'pollution', 'sustainable', 'green', 'energy', 'conservation', 'ecosystem'],
            'food': ['nutrition', 'ingredient', 'recipe', 'cooking', 'meal', 'diet', 'restaurant']
        }

        bad_words = set()
        domain_lower = domain.lower()

        if domain_lower in domain_bad_words:
            bad_words.update(domain_bad_words[domain_lower])

        for d in domain_bad_words:
            if d in domain_lower or domain_lower in d:
                bad_words.update(domain_bad_words[d])

        good_words = set()

        if domain_lower in domain_good_words:
            good_words.update(domain_good_words[domain_lower])

        for d in domain_good_words:
            if d in domain_lower or domain_lower in d:
                good_words.update(domain_good_words[d])

        filtered_candidates = []
        for candidate in candidates:
            candidate_lower = candidate.lower()

            if any(bad_word in candidate_lower.split() for bad_word in bad_words):
                continue

            contains_good_word = any(good_word in candidate_lower for good_word in good_words)

            is_named_entity = False
            if any(word[0].isupper() for word in candidate.split() if len(word) > 1):
                is_named_entity = True

            is_technical_term = bool(re.search(r'[0-9\-_]', candidate))

            if contains_good_word or is_named_entity or is_technical_term or len(candidate.split()) > 1:
                filtered_candidates.append(candidate)

        if len(filtered_candidates) < min(3, len(candidates)):
            sorted_candidates = sorted(candidates, key=lambda x: len(x), reverse=True)
            filtered_candidates.extend(sorted_candidates[:min(3, len(candidates))])
            filtered_candidates = list(dict.fromkeys(filtered_candidates))

        return filtered_candidates

    def _get_wordnet_expansions(self, keyphrase, context_terms=None):
        
        expansions = set()

        words = keyphrase.split()

        if len(words) == 1:
            word = words[0].lower()

            synsets = wn.synsets(word)

            if context_terms and synsets:
                synset_scores = []
                for synset in synsets:
                    related_words = set()
                    for lemma in synset.lemmas():
                        related_words.add(lemma.name().lower().replace('_', ' '))
                    if synset.examples():
                        for example in synset.examples():
                            related_words.update(example.lower().split())
                    if synset.definition():
                        related_words.update(synset.definition().lower().split())

                    overlap = sum(1 for term in context_terms if term in related_words)
                    synset_scores.append((synset, overlap))

                synset_scores.sort(key=lambda x: x[1], reverse=True)
                synsets = [s[0] for s in synset_scores[:5]]
            else:
                synsets = synsets[:3]

            for synset in synsets:
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != word and len(synonym) > 2:
                        expansions.add(synonym)

                for hypernym in synset.hypernyms()[:2]:
                    for lemma in hypernym.lemmas():
                        hypernym_term = lemma.name().replace('_', ' ')
                        if hypernym_term != word and len(hypernym_term) > 2:
                            expansions.add(hypernym_term)

                for hyponym in synset.hyponyms()[:3]:
                    for lemma in hyponym.lemmas():
                        hyponym_term = lemma.name().replace('_', ' ')
                        if hyponym_term != word and len(hyponym_term) > 2:
                            expansions.add(hyponym_term)

                for hypernym in synset.hypernyms()[:1]:
                    for sister in hypernym.hyponyms()[:3]:
                        if sister != synset:
                            for lemma in sister.lemmas():
                                sister_term = lemma.name().replace('_', ' ')
                                if sister_term != word and len(sister_term) > 2:
                                    expansions.add(sister_term)

                for lemma in synset.lemmas():
                    for related_lemma in lemma.derivationally_related_forms():
                        related_term = related_lemma.name().replace('_', ' ')
                        if related_term != word and len(related_term) > 2:
                            expansions.add(related_term)

        else:
            last_word = words[-1].lower()
            last_word_expansions = set()

            for synset in wn.synsets(last_word)[:2]:
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != last_word and len(synonym) > 2:
                        last_word_expansions.add(synonym)

                for hypernym in synset.hypernyms()[:1]:
                    for lemma in hypernym.lemmas():
                        hypernym_term = lemma.name().replace('_', ' ')
                        if hypernym_term != last_word and len(hypernym_term) > 2:
                            last_word_expansions.add(hypernym_term)

            for expansion in last_word_expansions:
                new_keyphrase = ' '.join(words[:-1] + [expansion])
                expansions.add(new_keyphrase)

            if len(words) > 1:
                first_word = words[0].lower()
                first_word_expansions = set()

                for synset in wn.synsets(first_word)[:2]:
                    for lemma in synset.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        if synonym != first_word and len(synonym) > 2:
                            first_word_expansions.add(synonym)

                for expansion in first_word_expansions:
                    new_keyphrase = ' '.join([expansion] + words[1:])
                    expansions.add(new_keyphrase)

        return expansions

    def _get_adaptive_threshold(self, keyphrase, domain=None):
        
        threshold = max(self.similarity_threshold, 0.60)

        words = keyphrase.split()
        word_count = len(words)

        if word_count == 1:
            threshold += 0.15

            if len(keyphrase) <= 3:
                threshold += 0.10

            common_words = ['work', 'like', 'make', 'time', 'year', 'data', 'use', 'get', 'take', 'see',
                           'know', 'find', 'give', 'think', 'come', 'look', 'want', 'need', 'feel', 'tell',
                           'say', 'call', 'try', 'ask', 'offer', 'show', 'keep', 'hold', 'turn', 'follow',
                           'change', 'play', 'move', 'live', 'believe', 'bring', 'happen', 'write', 'provide',
                           'sit', 'stand', 'lose', 'pay', 'meet', 'run', 'learn', 'type', 'process', 'style',
                           'system', 'point', 'fact', 'help', 'world', 'case', 'day', 'place', 'party', 'plan']

            if keyphrase.lower() in self.stopwords or keyphrase.lower() in common_words:
                threshold += 0.15

        elif word_count == 2:
            threshold += 0.10
        elif word_count == 3:
            threshold += 0.05
        elif word_count >= 4:
            threshold -= 0.03

        has_proper_noun = False
        for word in words:
            if len(word) > 1 and word[0].isupper() and word[1:].islower():
                has_proper_noun = True
                break

        if has_proper_noun:
            threshold += 0.08

        if any(char.isdigit() for char in keyphrase):
            threshold += 0.12

        if domain:
            domain_lower = domain.lower()

            if domain_lower in ['artificial intelligence', 'ai', 'machine learning', 'deep learning', 'neural networks']:
                threshold += 0.08

            elif domain_lower in ['cybersecurity', 'cyber security', 'information security', 'network security', 'data security']:
                threshold += 0.10

            elif domain_lower in ['automotive', 'cars', 'vehicles', 'transportation', 'mobility']:
                threshold += 0.07

            elif domain_lower in ['food', 'cuisine', 'cooking', 'nutrition', 'gastronomy', 'culinary']:
                threshold += 0.05

            elif domain_lower in ['environment', 'climate', 'sustainability', 'ecology', 'conservation']:
                threshold += 0.07

            elif domain_lower in ['real estate', 'property', 'housing', 'real property', 'realty']:
                threshold += 0.09

            elif domain_lower in ['entertainment', 'media', 'film', 'music', 'television', 'gaming']:
                threshold += 0.04

            elif domain_lower in ['technology', 'tech', 'software', 'hardware', 'electronics', 'digital']:
                threshold += 0.08

            elif domain_lower in ['business', 'finance', 'economics', 'investment', 'banking', 'commerce']:
                threshold += 0.09

            elif domain_lower in ['science', 'medicine', 'healthcare', 'biology', 'physics', 'chemistry']:
                threshold += 0.10

        threshold = max(0.68, min(0.90, threshold))

        return threshold

    def _calculate_domain_relevance(self, candidates, domain):
        
        scores = np.zeros(len(candidates))

        domain_lower = domain.lower() if domain else "general"

        domain_vocabularies = self._get_domain_vocabularies()

        matching_domains = []

        if domain_lower in domain_vocabularies:
            matching_domains.append((domain_lower, 1.0))

        for domain_key in domain_vocabularies.keys():
            if domain_key == domain_lower:
                continue

            if domain_key in domain_lower or domain_lower in domain_key:
                overlap_len = len(set(domain_key).intersection(set(domain_lower)))
                match_score = overlap_len / max(len(domain_key), len(domain_lower))
                matching_domains.append((domain_key, match_score))

        if domain_lower != "general" and "general" not in [d[0] for d in matching_domains]:
            matching_domains.append(("general", 0.5))

        matching_domains.sort(key=lambda x: x[1], reverse=True)

        for i, candidate in enumerate(candidates):
            candidate_lower = candidate.lower()
            max_domain_score = 0.0

            for matching_domain, domain_match_score in matching_domains:
                if matching_domain in domain_vocabularies:
                    domain_terms = domain_vocabularies[matching_domain]

                    if candidate_lower in [term.lower() for term in domain_terms]:
                        term_score = 1.0 * domain_match_score
                        max_domain_score = max(max_domain_score, term_score)
                        continue

                    candidate_words = candidate_lower.split()
                    for term in domain_terms:
                        term_lower = term.lower()
                        term_words = term_lower.split()

                        common_words = set(candidate_words).intersection(set(term_words))
                        if common_words:
                            overlap_score = len(common_words) / min(len(candidate_words), len(term_words))
                            term_score = overlap_score * domain_match_score
                            max_domain_score = max(max_domain_score, term_score)

                        elif (candidate_lower in term_lower or term_lower in candidate_lower):
                            substring_score = min(len(candidate_lower), len(term_lower)) / max(len(candidate_lower), len(term_lower))
                            term_score = substring_score * domain_match_score * 0.8
                            max_domain_score = max(max_domain_score, term_score)

            scores[i] = max_domain_score

        return scores

    def _get_enhanced_contextual_relevance_scores(self, candidates, text, keyphrase=None):
        
        scores = np.zeros(len(candidates))

        text_lower = text.lower()

        sentences = text_lower.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        for i, candidate in enumerate(candidates):
            candidate_lower = candidate.lower()
            candidate_words = candidate_lower.split()

            term_freq = text_lower.count(candidate_lower)

            partial_matches = 0
            if term_freq == 0 and len(candidate_words) > 1:
                for sentence in sentences:
                    if all(word in sentence for word in candidate_words):
                        partial_matches += 1

            position_score = 0
            if term_freq > 0:
                first_pos = text_lower.find(candidate_lower) / len(text_lower)
                position_score = 1.0 - (0.5 * np.log1p(9 * first_pos + 1))
            elif partial_matches > 0:
                for idx, sentence in enumerate(sentences):
                    if all(word in sentence for word in candidate_words):
                        first_pos = idx / len(sentences)
                        position_score = (1.0 - (0.5 * np.log1p(9 * first_pos + 1))) * 0.8
                        break

            length_score = min(1.0, 0.3 * len(candidate_words))

            context_score = 0.0
            if keyphrase:
                keyphrase_lower = keyphrase.lower()
                keyphrase_words = keyphrase_lower.split()

                coherent_sentences = 0
                for sentence in sentences:
                    if keyphrase_lower in sentence and candidate_lower in sentence:
                        coherent_sentences += 1.5
                    elif all(word in sentence for word in keyphrase_words) and all(word in sentence for word in candidate_words):
                        coherent_sentences += 1.0
                    elif any(word in sentence for word in keyphrase_words) and all(word in sentence for word in candidate_words):
                        coherent_sentences += 0.5

                if sentences:
                    context_score = min(1.0, coherent_sentences / min(4, len(sentences)))

            surrounding_context_score = 0.0
            if term_freq > 0 or partial_matches > 0:
                context_windows = []

                if term_freq > 0:
                    start_pos = 0
                    while start_pos < len(text_lower):
                        pos = text_lower.find(candidate_lower, start_pos)
                        if pos == -1:
                            break

                        context_start = max(0, pos - 80)
                        context_end = min(len(text_lower), pos + len(candidate_lower) + 80)
                        context_window = text_lower[context_start:context_end]
                        context_windows.append(context_window)

                        start_pos = pos + 1

                elif partial_matches > 0:
                    for sentence in sentences:
                        if all(word in sentence for word in candidate_words):
                            context_windows.append(sentence)

                if context_windows:
                    context_words = set()
                    for window in context_windows:
                        words = window.split()
                        content_words = [w for w in words if w not in self.stopwords and len(w) > 2]
                        context_words.update(content_words)

                    surrounding_context_score = min(1.0, len(context_words) / 15)

                    if keyphrase:
                        keyphrase_words = keyphrase.lower().split()
                        keyphrase_in_context = sum(1 for word in keyphrase_words if any(word in window for window in context_windows))
                        if keyphrase_in_context > 0:
                            boost = keyphrase_in_context / len(keyphrase_words)
                            surrounding_context_score = min(1.0, surrounding_context_score + 0.2 * boost)

            tfidf_score = 0.0
            if term_freq > 0:
                tf = term_freq / max(1, len(text_lower.split()))
                idf = 1.0 / (0.1 + tf)
                tfidf_score = min(1.0, tf * idf / 10)

            term_freq_score = min(1.0, 0.3 * (term_freq + 0.7 * partial_matches))

            scores[i] = (0.25 * term_freq_score +
                         0.15 * position_score +
                         0.15 * length_score +
                         0.20 * context_score +
                         0.15 * surrounding_context_score +
                         0.10 * tfidf_score)

        return scores

    def _get_sentence_context_scores(self, candidates, text, keyphrase=None):
        
        scores = np.zeros(len(candidates))

        if not keyphrase or not text:
            return scores

        text_lower = text.lower()
        keyphrase_lower = keyphrase.lower()
        keyphrase_words = keyphrase_lower.split()

        sentences = [s.strip() for s in re.split(r'[.!?]', text_lower) if s.strip()]

        keyphrase_sentences = []
        for sentence in sentences:
            if keyphrase_lower in sentence or all(word in sentence for word in keyphrase_words):
                keyphrase_sentences.append(sentence)

        if not keyphrase_sentences:
            return scores

        for i, candidate in enumerate(candidates):
            candidate_lower = candidate.lower()
            candidate_words = candidate_lower.split()

            exact_matches = sum(1 for sentence in keyphrase_sentences if candidate_lower in sentence)
            partial_matches = sum(1 for sentence in keyphrase_sentences
                               if all(word in sentence for word in candidate_words)
                               and candidate_lower not in sentence)

            if keyphrase_sentences:
                exact_ratio = exact_matches / len(keyphrase_sentences)
                partial_ratio = partial_matches / len(keyphrase_sentences)
                scores[i] = min(1.0, exact_ratio + 0.5 * partial_ratio)

                if exact_matches + partial_matches > 1:
                    scores[i] = min(1.0, scores[i] + 0.1 * (exact_matches + partial_matches - 1))

        return scores

    def _get_partial_match_scores(self, candidates, keyphrase):
        
        scores = np.zeros(len(candidates))

        if not keyphrase:
            return scores

        keyphrase_lower = keyphrase.lower()
        keyphrase_words = keyphrase_lower.split()

        if len(keyphrase_words) == 1:
            for i, candidate in enumerate(candidates):
                candidate_lower = candidate.lower()
                candidate_words = candidate_lower.split()

                if len(candidate_words) == 1:
                    common_chars = set(keyphrase_lower).intersection(set(candidate_lower))
                    if common_chars:
                        char_overlap = len(common_chars) / max(len(keyphrase_lower), len(candidate_lower))
                        scores[i] = min(0.7, char_overlap)

                    if keyphrase_lower in candidate_lower or candidate_lower in keyphrase_lower:
                        substring_score = min(len(keyphrase_lower), len(candidate_lower)) / max(len(keyphrase_lower), len(candidate_lower))
                        scores[i] = max(scores[i], min(0.8, substring_score + 0.2))
                else:
                    for word in candidate_words:
                        if keyphrase_lower in word:
                            scores[i] = 0.6
                            break
        else:
            for i, candidate in enumerate(candidates):
                candidate_lower = candidate.lower()
                candidate_words = candidate_lower.split()

                common_words = set(keyphrase_words).intersection(set(candidate_words))
                if common_words:
                    word_overlap = len(common_words) / len(set(keyphrase_words).union(set(candidate_words)))
                    scores[i] = min(0.9, word_overlap + 0.1)

                    if len(common_words) > 1:
                        keyphrase_indices = [keyphrase_words.index(word) for word in common_words if word in keyphrase_words]
                        candidate_indices = [candidate_words.index(word) for word in common_words if word in candidate_words]

                        if keyphrase_indices == sorted(keyphrase_indices) and candidate_indices == sorted(candidate_indices):
                            scores[i] = min(1.0, scores[i] + 0.1)

        return scores

    def _calculate_contextual_relevance(self, candidates, text, keyphrase=None):
        
        scores = np.zeros(len(candidates))

        text_lower = text.lower()

        sentences = text_lower.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        for i, candidate in enumerate(candidates):
            candidate_lower = candidate.lower()
            candidate_words = candidate_lower.split()

            term_freq = text_lower.count(candidate_lower)

            partial_matches = 0
            if term_freq == 0 and len(candidate_words) > 1:
                for sentence in sentences:
                    if all(word in sentence for word in candidate_words):
                        partial_matches += 1

            position_score = 0
            if term_freq > 0:
                first_pos = text_lower.find(candidate_lower) / len(text_lower)
                position_score = 1.0 - first_pos
            elif partial_matches > 0:
                for idx, sentence in enumerate(sentences):
                    if all(word in sentence for word in candidate_words):
                        first_pos = idx / len(sentences)
                        position_score = (1.0 - first_pos) * 0.8
                        break

            length_score = min(1.0, 0.25 * len(candidate_words))

            context_score = 0.0
            if keyphrase:
                keyphrase_lower = keyphrase.lower()
                keyphrase_words = keyphrase_lower.split()

                coherent_sentences = 0
                for sentence in sentences:
                    if keyphrase_lower in sentence and candidate_lower in sentence:
                        coherent_sentences += 1
                    elif all(word in sentence for word in keyphrase_words) and all(word in sentence for word in candidate_words):
                        coherent_sentences += 0.5

                if sentences:
                    context_score = min(1.0, coherent_sentences / min(5, len(sentences)))

            surrounding_context_score = 0.0
            if term_freq > 0:
                start_pos = 0
                context_windows = []
                while start_pos < len(text_lower):
                    pos = text_lower.find(candidate_lower, start_pos)
                    if pos == -1:
                        break

                    context_start = max(0, pos - 50)
                    context_end = min(len(text_lower), pos + len(candidate_lower) + 50)
                    context_window = text_lower[context_start:context_end]
                    context_windows.append(context_window)

                    start_pos = pos + 1

                if context_windows:
                    context_words = set()
                    for window in context_windows:
                        words = window.split()
                        content_words = [w for w in words if w not in self.stopwords and len(w) > 2]
                        context_words.update(content_words)

                    surrounding_context_score = min(1.0, len(context_words) / 20)

            term_freq_score = min(1.0, 0.25 * (term_freq + 0.5 * partial_matches))

            scores[i] = (0.35 * term_freq_score +
                         0.20 * position_score +
                         0.15 * length_score +
                         0.15 * context_score +
                         0.15 * surrounding_context_score)

        return scores

    def _get_fallback_suggestions(self, keyphrase, expanded_keyphrases, domain):
        
        fallback_suggestions = []
        keyphrase_lower = keyphrase.lower()
        keyphrase_words = keyphrase_lower.split()

        domain_suggestions = self._get_domain_expansions(keyphrase, domain)
        if domain_suggestions:

            scored_domain_suggestions = []
            for suggestion in domain_suggestions:

                suggestion_words = suggestion.lower().split()
                common_words = set(keyphrase_words).intersection(set(suggestion_words))
                overlap_score = len(common_words) / max(len(keyphrase_words), len(suggestion_words))

                substring_match = False
                if not common_words and len(keyphrase_words) == 1 and len(keyphrase_words[0]) > 3:
                    for sw in suggestion_words:
                        if len(sw) > 3 and (keyphrase_words[0] in sw or sw in keyphrase_words[0]):
                            substring_match = True
                            overlap_score = 0.3
                            break

                if common_words or substring_match:

                    final_score = min(0.5 + overlap_score * 0.4, 0.8)
                    scored_domain_suggestions.append((suggestion, final_score))

            scored_domain_suggestions.sort(key=lambda x: x[1], reverse=True)
            fallback_suggestions.extend(scored_domain_suggestions[:self.max_suggestions])

        if len(fallback_suggestions) < self.max_suggestions and self.use_wordnet:

            wordnet_expansions = self._get_wordnet_expansions(keyphrase)

            if wordnet_expansions:

                scored_wordnet_suggestions = []
                for suggestion in wordnet_expansions:

                    scored_wordnet_suggestions.append((suggestion, 0.65))

                scored_wordnet_suggestions.sort(key=lambda x: len(x[0].split()), reverse=True)
                fallback_suggestions.extend(scored_wordnet_suggestions[:self.max_suggestions])

        if len(fallback_suggestions) < self.max_suggestions:

            for other_keyphrase, suggestions in expanded_keyphrases.items():
                if other_keyphrase == keyphrase or not suggestions:
                    continue

                other_words = other_keyphrase.lower().split()
                common_words = set(keyphrase_words).intersection(set(other_words))

                similarity_score = 0
                if common_words:
                    similarity_score = len(common_words) / min(len(keyphrase_words), len(other_words))

                if not common_words and len(keyphrase_words) == 1 and len(other_words) == 1:
                    kw = keyphrase_words[0]
                    ow = other_words[0]
                    if len(kw) > 3 and len(ow) > 3 and (kw in ow or ow in kw):
                        similarity_score = 0.3

                if similarity_score > 0:

                    for suggestion, score in suggestions:
                        if not any(suggestion.lower() == s[0].lower() for s in fallback_suggestions):
                            adjusted_score = score * (0.5 + 0.5 * similarity_score)
                            fallback_suggestions.append((suggestion, adjusted_score))

                        if len(fallback_suggestions) >= self.max_suggestions * 2:
                            break

                    if len(fallback_suggestions) >= self.max_suggestions * 2:
                        break

        if len(fallback_suggestions) < self.max_suggestions:
            if len(keyphrase_words) == 1 and len(keyphrase_words[0]) > 3:
                common_modifiers = ["advanced", "modern", "innovative", "effective", "efficient",
                                   "improved", "enhanced", "optimized", "specialized", "professional"]

                for modifier in common_modifiers:
                    suggestion = f"{modifier} {keyphrase_lower}"
                    if not any(suggestion.lower() == s[0].lower() for s in fallback_suggestions):
                        fallback_suggestions.append((suggestion, 0.6))

                        if len(fallback_suggestions) >= self.max_suggestions:
                            break

        fallback_suggestions.sort(key=lambda x: x[1], reverse=True)
        return fallback_suggestions[:self.max_suggestions]

    def _get_keybert_expansions(self, keyphrase, text):
        
        expansions = set()

        if not self.use_keybert or self.keybert_instance is None:
            return expansions

        try:

            keyword_candidates = []

            if text and len(text) > 0:
                keyphrase_lower = keyphrase.lower()
                keyphrase_words = keyphrase_lower.split()

                sentences = text.split('.')
                relevant_sentences = []

                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    sentence_lower = sentence.lower()
                    if keyphrase_lower in sentence_lower or any(word in sentence_lower for word in keyphrase_words):
                        relevant_sentences.append(sentence)

                if self.use_spacy and self.nlp is not None and relevant_sentences:
                    for sentence in relevant_sentences:
                        doc = self.nlp(sentence)
                        for chunk in doc.noun_chunks:
                            if len(chunk.text.split()) <= 3:
                                keyword_candidates.append(chunk.text.lower())

            keyphrase_words = keyphrase.lower().split()

            for candidate in keyword_candidates:
                if candidate.lower() == keyphrase.lower():
                    continue

                if any(word in candidate.lower().split() for word in keyphrase_words) or \
                   any(word in keyphrase_words for word in candidate.lower().split()):
                    expansions.add(candidate)

        except Exception as e:
            print(f"Error in KeyBERT expansion: {str(e)}")

        return expansions

    def _get_pos_pattern_expansions(self, keyphrase, text):
        
        expansions = set()

        if not hasattr(self, 'nlp') or self.nlp is None:
            return expansions

        doc = self.nlp(text)

        keyphrase_words = keyphrase.lower().split()

        candidate_phrases = []

        for sent in doc.sents:
            i = 0
            while i < len(sent):
                if sent[i].pos_ not in ['ADJ', 'NOUN']:
                    i += 1
                    continue

                start_idx = i
                has_noun = sent[i].pos_ == 'NOUN'

                i += 1
                while i < len(sent) and sent[i].pos_ in ['ADJ', 'NOUN']:
                    if sent[i].pos_ == 'NOUN':
                        has_noun = True
                    i += 1

                if has_noun and i > start_idx:
                    phrase = sent[start_idx:i].text.lower()
                    candidate_phrases.append(phrase)

        for phrase in candidate_phrases:
            if phrase == keyphrase.lower():
                continue

            if any(word in phrase.split() for word in keyphrase_words) or \
               any(word in keyphrase_words for word in phrase.split()):
                expansions.add(phrase)

        return expansions

    def _get_collocation_expansions(self, keyphrase, text):
        
        expansions = set()

        tokens = word_tokenize(text.lower())

        if self.stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]

        keyphrase_words = keyphrase.lower().split()

        bigram_finder = BigramCollocationFinder.from_words(tokens)
        bigram_finder.apply_freq_filter(2)
        bigram_scored = bigram_finder.score_ngrams(BigramAssocMeasures.pmi)

        trigram_finder = TrigramCollocationFinder.from_words(tokens)
        trigram_finder.apply_freq_filter(2)
        trigram_scored = trigram_finder.score_ngrams(TrigramAssocMeasures.pmi)

        for bigram, _ in bigram_scored:
            if any(word in bigram for word in keyphrase_words):
                expansions.add(' '.join(bigram))

        for trigram, _ in trigram_scored:
            if any(word in trigram for word in keyphrase_words):
                expansions.add(' '.join(trigram))

        return expansions

    def _get_spacy_expansions(self, keyphrase, text):
        
        expansions = set()

        doc = self.nlp(text)

        noun_chunks = set([chunk.text.lower() for chunk in doc.noun_chunks])

        keyphrase_lower = keyphrase.lower()
        words = keyphrase_lower.split()

        for chunk in noun_chunks:
            if chunk == keyphrase_lower:
                continue

            if any(word in chunk for word in words):
                expansions.add(chunk)

            chunk_words = chunk.split()
            if any(word in words for word in chunk_words):
                expansions.add(chunk)

        return expansions

    def _filter_duplicate_suggestions(self, suggestions, similarity_threshold=0.85):
        
        if not suggestions:
            return []

        filtered_by_quality = self._filter_low_quality_terms(suggestions)

        if not filtered_by_quality and suggestions:
            return [suggestions[0]]
        elif not filtered_by_quality:
            return []

        sorted_suggestions = sorted(filtered_by_quality, key=lambda x: x[1], reverse=True)

        filtered_suggestions = [sorted_suggestions[0]]
        filtered_texts = [sorted_suggestions[0][0].lower()]

        for suggestion, score in sorted_suggestions[1:]:
            suggestion_lower = suggestion.lower()

            if suggestion_lower in filtered_texts:
                continue

            substring_match = False
            for existing_text in filtered_texts:
                if suggestion_lower in existing_text or existing_text in suggestion_lower:
                    if len(suggestion_lower) > len(existing_text):
                        idx = filtered_texts.index(existing_text)
                        if score > filtered_suggestions[idx][1] * 0.9:
                            filtered_suggestions[idx] = (suggestion, score)
                            filtered_texts[idx] = suggestion_lower
                    substring_match = True
                    break

            if substring_match:
                continue

            suggestion_words = set(suggestion_lower.split())
            word_overlap_match = False

            for i, existing_text in enumerate(filtered_texts):
                existing_words = set(existing_text.split())

                if not suggestion_words or not existing_words:
                    continue

                intersection = suggestion_words.intersection(existing_words)
                union = suggestion_words.union(existing_words)

                if union:
                    jaccard_sim = len(intersection) / len(union)

                    if jaccard_sim > 0.7:
                        if score > filtered_suggestions[i][1]:
                            filtered_suggestions[i] = (suggestion, score)
                            filtered_texts[i] = suggestion_lower
                        word_overlap_match = True
                        break

            if word_overlap_match:
                continue

            filtered_suggestions.append((suggestion, score))
            filtered_texts.append(suggestion_lower)

        return sorted(filtered_suggestions, key=lambda x: x[1], reverse=True)

    def _filter_low_quality_terms(self, suggestions):
        
        if not suggestions:
            return []

        uncommon_terms = [
            "hurly burly", "flutter", "kerfuffle", "egression", "egress", "nursery gas",
            "building block", "chemical chain", "unit", "series", "stuff", "material",
            "across the nation", "across the country", "gift", "part", "attempt",
            "swear", "believe", "spherical", "gap", "rely", "relies", "believe", "swear",
            "epoch", "holocene", "interruption", "human action", "human activity", "deed",
            "act", "action", "auditory sensation", "sense impression", "sense datum",
            "sense experience", "sound property", "legal document", "legal instrument",
            "official document", "natural event", "relation", "occurrent", "clime", "alteration",
            "danger", "menace", "warning", "field of operation", "solid food", "nutrient", "assets",
            "activeness", "modification", "mood change", "mood", "navy department", "navy",
            "housing and urban development", "urban development", "housing", "urban",
            "department of housing", "department of transportation", "transportation",
            "department of agriculture", "department of housing and urban development",
            "advanced", "modern", "innovative", "effective", "efficient", "improved", "enhanced",
            "optimized", "specialized", "professional", "protection", "datum", "property",

            "entity", "object", "thing", "item", "element", "component", "piece", "segment",
            "section", "portion", "fragment", "bit", "chunk", "part", "division", "unit",
            "module", "constituent", "ingredient", "factor", "aspect", "feature", "attribute",
            "characteristic", "quality", "property", "trait", "nature", "character", "essence",
            "substance", "matter", "material", "stuff", "content", "subject", "topic", "theme",
            "concept", "notion", "idea", "thought", "impression", "perception", "conception",
            "understanding", "interpretation", "construal", "construction", "reading", "view",
            "opinion", "belief", "judgment", "assessment", "evaluation", "appraisal", "estimation",

            "Department of Health and Human Services", "Department of the Interior", "Health and Human Services",

            "auditory", "visual", "tactile", "olfactory", "gustatory", "sensation", "perception",
            "sense", "feeling", "impression", "experience", "awareness", "consciousness", "cognition",

            "action", "activity", "behavior", "conduct", "deed", "act", "move", "movement",
            "motion", "operation", "performance", "execution", "implementation", "accomplishment",
            "achievement", "attainment", "realization", "fulfillment", "completion", "conclusion",
            "termination", "cessation", "discontinuation", "interruption", "suspension", "pause",
            "break", "halt", "stop", "end", "finish", "close", "culmination", "climax", "finale",

            "good", "bad", "high", "low", "big", "small", "large", "little", "great", "minor",
            "major", "significant", "insignificant", "important", "unimportant", "relevant",
            "irrelevant", "pertinent", "impertinent", "applicable", "inapplicable", "appropriate",
            "inappropriate", "suitable", "unsuitable", "fitting", "unfitting", "proper", "improper",
            "correct", "incorrect", "right", "wrong", "true", "false", "accurate", "inaccurate",
            "precise", "imprecise", "exact", "inexact", "definite", "indefinite", "specific", "general"
        ]

        paris_agreement_terms = [
            "paris agreement framework", "paris climate agreement", "paris accord",
            "climate agreement", "international climate agreement", "global climate agreement",
            "climate treaty", "climate pact", "climate commitment", "climate protocol"
        ]

        nationally_determined_terms = [
            "nationally determined contributions", "ndcs", "national climate pledges",
            "national climate commitments", "national climate targets", "national climate plans",
            "national climate strategies", "national emission reduction targets",
            "national climate action plans", "country climate pledges"
        ]

        domain_specific_terms = {
            "climate": ["climate change", "climate policy", "climate action", "climate resilience",
                       "climate adaptation", "climate mitigation", "climate crisis", "climate emergency",
                       "global warming", "greenhouse gas", "carbon emissions", "carbon footprint",
                       "paris agreement", "emissions reduction", "net zero", "carbon neutral",
                       "renewable energy", "sustainable development", "climate science"],
            "food": ["food supply chain", "food security", "food safety", "food production",
                    "food distribution", "food logistics", "food traceability", "food system",
                    "sustainable food", "food technology", "food processing", "food waste",
                    "food industry", "agricultural production", "food transportation"],
            "agriculture": ["sustainable agriculture", "precision agriculture", "agricultural technology",
                         "agricultural innovation", "farming practices", "crop production", "livestock farming",
                         "agricultural systems", "agricultural research", "agricultural policy", "agricultural development",
                         "agricultural economics", "agricultural science", "agricultural engineering", "agricultural education",
                         "agricultural extension", "agricultural marketing", "agricultural finance", "agricultural trade",
                         "agricultural sustainability", "agricultural productivity", "agricultural diversification",
                         "agricultural intensification", "agricultural mechanization", "agricultural biotechnology",
                         "agricultural genetics", "agricultural breeding", "agricultural ecology", "agricultural biodiversity",
                         "agricultural conservation", "agricultural water management", "agricultural soil management",
                         "agricultural pest management", "agricultural disease management", "agricultural waste management",
                         "agricultural energy management", "agricultural climate adaptation", "agricultural resilience",
                         "agricultural value chain", "agricultural supply chain", "agricultural market chain",
                         "agricultural input supply", "agricultural output marketing", "agricultural processing",
                         "agricultural storage", "agricultural transportation", "agricultural distribution",
                         "agricultural consumption", "agricultural nutrition", "agricultural food security",
                         "agricultural food safety", "agricultural food quality", "agricultural food sovereignty",
                         "agricultural food systems", "agricultural food policy", "agricultural food governance",
                         "agricultural food regulation", "agricultural food standards", "agricultural food certification",
                         "agricultural food labeling", "agricultural food traceability", "agricultural food transparency"],
            "blockchain": ["blockchain technology", "distributed ledger", "smart contracts",
                          "cryptocurrency", "decentralized", "tokenization", "blockchain platform",
                          "blockchain solution", "blockchain application", "blockchain network",
                          "blockchain security", "blockchain verification", "blockchain transparency"],
            "iot": ["internet of things", "connected devices", "smart sensors", "iot devices",
                   "iot platform", "iot ecosystem", "iot security", "iot technology",
                   "iot applications", "iot solutions", "iot infrastructure", "iot connectivity",
                   "smart technology", "sensor networks", "device connectivity"],
            "automation": ["process automation", "robotic automation", "industrial automation",
                         "automation technology", "automated systems", "intelligent automation",
                         "automation solutions", "digital automation", "automation platform",
                         "business process automation", "workflow automation", "robotic process automation"],
            "environment": ["environmental protection", "sustainability", "conservation", "biodiversity",
                          "ecosystem", "natural resources", "environmental impact", "green technology",
                          "renewable resources", "environmental policy", "sustainable practices"],
            "transportation": ["logistics", "supply chain", "transportation network", "freight",
                             "shipping", "delivery", "transportation system", "transportation infrastructure",
                             "transportation technology", "sustainable transportation", "smart logistics"],
            "cybersecurity": ["cyber attacks", "cyber threats", "cyber defense", "cyber security", "cybersecurity",
                           "network security", "information security", "data protection", "data security",
                           "ransomware", "ransomware attacks", "ransomware threats", "ransomware protection", "ransomware defense",
                           "ransomware prevention", "ransomware detection", "ransomware response", "ransomware recovery",
                           "malware", "malware attacks", "malware threats", "malware protection", "malware defense",
                           "malware prevention", "malware detection", "malware response", "malware recovery",
                           "phishing", "phishing attacks", "phishing threats", "phishing protection", "phishing defense",
                           "phishing prevention", "phishing detection", "phishing response", "phishing recovery",
                           "ddos", "ddos attacks", "ddos threats", "ddos protection", "ddos defense",
                           "ddos prevention", "ddos detection", "ddos response", "ddos recovery",
                           "cyber threat intelligence", "cyber threat hunting", "cyber threat detection", "cyber threat response",
                           "cyber incident response", "cyber security incident", "cyber security breach", "cyber data breach",
                           "cyber extortion", "cyber fraud", "cyber scam", "cyber crime", "cyber criminal", "cyber criminals",
                           "cyber security awareness", "cyber security training", "cyber security policy", "cyber security compliance",
                           "cyber security audit", "cyber security assessment", "cyber penetration testing", "cyber pen testing",
                           "cyber security operations", "cyber security operations center", "cyber security information",
                           "cyber endpoint protection", "cyber endpoint security", "cyber endpoint detection",
                           "cyber security orchestration", "cyber security automation", "cyber security response",
                           "cyber security as a service", "managed cyber security service", "cyber security provider",
                           "cloud cyber security", "cloud cyber security posture", "cloud cyber security access",
                           "zero trust", "zero trust architecture", "zero trust network", "zero trust access",
                           "secure web gateway", "web application firewall", "api security", "container security",
                           "devsecops", "secure software development", "secure coding", "code review",
                           "cyber supply chain security", "cyber third-party risk", "cyber vendor risk",
                           "cyber risk assessment", "cyber risk management", "cyber compliance", "cyber regulatory compliance",
                           "cyber privacy", "cyber data privacy", "cyber privacy by design", "cyber privacy impact",
                           "cyber insurance", "cyber liability insurance", "cyber risk insurance", "cyber policy",
                           "cyber resilience", "cyber recovery", "cyber disaster recovery", "cyber business continuity",
                           "cyber security monitoring", "cyber security analytics", "cyber security intelligence", "cyber security metrics",
                           "cyber security dashboard", "cyber security reporting", "cyber security kpi", "cyber security roi",
                           "cyber security budget", "cyber security investment", "cyber security strategy", "cyber security roadmap",
                           "cyber security program", "cyber security governance", "cyber security leadership", "cyber security culture",
                           "computer security", "IT security", "digital security", "internet security",
                           "malware", "ransomware", "phishing", "spyware", "trojan", "virus", "worm",
                           "DDoS", "distributed denial of service", "denial of service", "DoS",
                           "intrusion detection", "intrusion prevention", "firewall", "antivirus",
                           "encryption", "decryption", "cryptography", "authentication", "authorization",
                           "access control", "identity management", "password management", "multi-factor authentication",
                           "two-factor authentication", "biometric authentication", "single sign-on", "SSO",
                           "vulnerability", "exploit", "patch", "security update", "security patch",
                           "zero-day", "zero-day exploit", "zero-day vulnerability", "zero-day attack",
                           "threat intelligence", "threat hunting", "threat detection", "threat response",
                           "incident response", "security incident", "security breach", "data breach",
                           "data leak", "data loss", "data theft", "identity theft", "fraud", "scam",
                           "social engineering", "spear phishing", "whaling", "vishing", "smishing",
                           "security awareness", "security training", "security policy", "security compliance",
                           "security audit", "security assessment", "penetration testing", "pen testing",
                           "red team", "blue team", "purple team", "white hat", "black hat", "gray hat",
                           "hacker", "ethical hacker", "security researcher", "security analyst",
                           "security engineer", "security architect", "security consultant", "security advisor",
                           "CISO", "chief information security officer", "CSO", "chief security officer",
                           "security operations center", "SOC", "security information and event management", "SIEM",
                           "endpoint protection", "endpoint security", "endpoint detection and response", "EDR",
                           "extended detection and response", "XDR", "managed detection and response", "MDR",
                           "security orchestration automation and response", "SOAR", "security as a service", "SECaaS",
                           "managed security service provider", "MSSP", "cloud security", "cloud security posture management",
                           "CSPM", "cloud access security broker", "CASB", "secure access service edge", "SASE",
                           "zero trust", "zero trust architecture", "zero trust network access", "ZTNA",
                           "secure web gateway", "SWG", "web application firewall", "WAF", "API security",
                           "container security", "kubernetes security", "DevSecOps", "secure software development",
                           "secure coding", "code review", "static application security testing", "SAST",
                           "dynamic application security testing", "DAST", "interactive application security testing", "IAST",
                           "runtime application self-protection", "RASP", "software composition analysis", "SCA",
                           "supply chain security", "third-party risk management", "vendor risk management",
                           "risk assessment", "risk management", "compliance", "regulatory compliance",
                           "GDPR", "CCPA", "HIPAA", "PCI DSS", "ISO 27001", "NIST", "CIS", "SOC 2",
                           "security framework", "security standard", "security regulation", "security law",
                           "privacy", "data privacy", "privacy by design", "privacy impact assessment", "PIA",
                           "data protection impact assessment", "DPIA", "data protection officer", "DPO",
                           "cyber insurance", "cyber liability insurance", "cyber risk insurance", "cyber policy",
                           "cyber resilience", "cyber recovery", "disaster recovery", "business continuity",
                           "backup", "restore", "recovery point objective", "RPO", "recovery time objective", "RTO",
                           "security monitoring", "security analytics", "security intelligence", "security metrics",
                           "security dashboard", "security reporting", "security KPI", "security ROI",
                           "security budget", "security investment", "security strategy", "security roadmap",
                           "security program", "security governance", "security leadership", "security culture"]
        }

        domain_matches = []

        filtered_suggestions = []
        for suggestion, score in suggestions:
            suggestion_lower = suggestion.lower()

            if any(term in suggestion_lower for term in uncommon_terms):
                continue

            words = suggestion_lower.split()
            if len(words) == 1 and len(words[0]) < 4:
                continue

            filtered_suggestions.append((suggestion, score))

        if not filtered_suggestions:
            return []

        for suggestion, score in filtered_suggestions:
            suggestion_lower = suggestion.lower()

            if "paris" in suggestion_lower or "agreement" in suggestion_lower:
                if any(term in suggestion_lower for term in paris_agreement_terms):
                    domain_matches.append((suggestion, min(1.0, score * 1.2)))
                    continue

            if "nationally" in suggestion_lower or "determined" in suggestion_lower or "ndcs" in suggestion_lower:
                if any(term in suggestion_lower for term in nationally_determined_terms):
                    domain_matches.append((suggestion, min(1.0, score * 1.2)))
                    continue

            if "climate" in suggestion_lower or "carbon" in suggestion_lower or "emission" in suggestion_lower:
                for term in domain_specific_terms["climate"]:
                    if term in suggestion_lower:
                        domain_matches.append((suggestion, min(1.0, score * 1.2)))
                        break
                continue

            if "food" in suggestion_lower or "agriculture" in suggestion_lower or "nutrition" in suggestion_lower:
                for term in domain_specific_terms["food"]:
                    if term in suggestion_lower:
                        domain_matches.append((suggestion, min(1.0, score * 1.2)))
                        break
                continue

            if "cyber" in suggestion_lower or "security" in suggestion_lower or "threat" in suggestion_lower:
                for term in domain_specific_terms.get("cybersecurity", []):
                    if term in suggestion_lower:
                        domain_matches.append((suggestion, min(1.0, score * 1.2)))
                        break
                continue

            if "blockchain" in suggestion_lower or "crypto" in suggestion_lower:
                for term in domain_specific_terms["blockchain"]:
                    if term in suggestion_lower:
                        domain_matches.append((suggestion, min(1.0, score * 1.2)))
                        break
                continue

            if "iot" in suggestion_lower or "internet of things" in suggestion_lower or "device" in suggestion_lower:
                for term in domain_specific_terms["iot"]:
                    if term in suggestion_lower:
                        domain_matches.append((suggestion, min(1.0, score * 1.2)))
                        break
                continue

            if "automation" in suggestion_lower or "automated" in suggestion_lower or "process" in suggestion_lower:
                for term in domain_specific_terms["automation"]:
                    if term in suggestion_lower:
                        domain_matches.append((suggestion, min(1.0, score * 1.2)))
                        break
                continue

            if "environment" in suggestion_lower or "sustainable" in suggestion_lower or "conservation" in suggestion_lower:
                for term in domain_specific_terms["environment"]:
                    if term in suggestion_lower:
                        domain_matches.append((suggestion, min(1.0, score * 1.2)))
                        break
                continue

            if "transport" in suggestion_lower or "logistics" in suggestion_lower or "supply chain" in suggestion_lower:
                for term in domain_specific_terms["transportation"]:
                    if term in suggestion_lower:
                        domain_matches.append((suggestion, min(1.0, score * 1.2)))
                        break
                continue

            for domain, terms in domain_specific_terms.items():
                if any(term in suggestion_lower for term in terms):
                    domain_matches.append((suggestion, score))
                    break

        if domain_matches:
            return sorted(domain_matches, key=lambda x: x[1], reverse=True)

        filtered = []
        for suggestion, score in suggestions:
            suggestion_lower = suggestion.lower()
            if not any(term in suggestion_lower for term in uncommon_terms):
                filtered.append((suggestion, score))

        return filtered

    def _filter_trivial_variations(self, keyphrase, candidates):
        
        filtered_candidates = set()
        keyphrase_lower = keyphrase.lower()
        keyphrase_words = keyphrase_lower.split()
        keyphrase_lemmas = set()

        if self.use_phrase_quality_check and self.lemmatizer:
            keyphrase_lemmas = {self.lemmatizer.lemmatize(word) for word in keyphrase_words}

        keyphrase_char_ngrams = set()
        for i in range(len(keyphrase_lower) - 2):
            keyphrase_char_ngrams.add(keyphrase_lower[i:i+3])

        for candidate in candidates:
            candidate_lower = candidate.lower()
            candidate_words = candidate_lower.split()

            if candidate_lower == keyphrase_lower:
                continue

            if any([
                candidate_lower + 's' == keyphrase_lower,
                candidate_lower == keyphrase_lower + 's',
                candidate_lower + 'es' == keyphrase_lower,
                candidate_lower == keyphrase_lower + 'es',
                candidate_lower[:-1] + 'ies' == keyphrase_lower and candidate_lower.endswith('y'),
                keyphrase_lower[:-1] + 'ies' == candidate_lower and keyphrase_lower.endswith('y'),
                candidate_lower[:-3] + 'ves' == keyphrase_lower and candidate_lower.endswith('fe'),
                keyphrase_lower[:-3] + 'ves' == candidate_lower and keyphrase_lower.endswith('fe'),
                candidate_lower[:-1] + 'ves' == keyphrase_lower and candidate_lower.endswith('f'),
                keyphrase_lower[:-1] + 'ves' == candidate_lower and keyphrase_lower.endswith('f')
            ]):
                continue

            if set(candidate_words) == set(keyphrase_words):
                continue

            if re.match(r'^(a|an|the)\s+' + re.escape(keyphrase_lower) + r'$', candidate_lower) or \
               re.match(r'^' + re.escape(keyphrase_lower) + r'\s+(a|an|the)$', candidate_lower):
                continue

            if re.match(r'^' + re.escape(keyphrase_lower) + r'\s+(of|in|for|with|on|at|by|to)\s+\w+$', candidate_lower) or \
               re.match(r'^\w+\s+(of|in|for|with|on|at|by|to)\s+' + re.escape(keyphrase_lower) + r'$', candidate_lower):
                if self.stopwords:
                    if candidate_lower.startswith(keyphrase_lower):
                        last_word = candidate_words[-1]
                        if last_word in self.stopwords:
                            continue
                    elif candidate_lower.endswith(keyphrase_lower):
                        first_word = candidate_words[0]
                        if first_word in self.stopwords:
                            continue

            if self.use_phrase_quality_check and self.lemmatizer:
                candidate_lemmas = {self.lemmatizer.lemmatize(word) for word in candidate_words}
                if candidate_lemmas == keyphrase_lemmas:
                    continue

            if len(candidate_lower) > 3 and len(keyphrase_lower) > 3:
                candidate_char_ngrams = set()
                for i in range(len(candidate_lower) - 2):
                    candidate_char_ngrams.add(candidate_lower[i:i+3])

                if len(keyphrase_char_ngrams) > 0 and len(candidate_char_ngrams) > 0:
                    overlap = len(keyphrase_char_ngrams.intersection(candidate_char_ngrams))
                    jaccard = overlap / len(keyphrase_char_ngrams.union(candidate_char_ngrams))

                    if jaccard > 0.8 and jaccard < 1.0:
                        continue

            if candidate_lower in keyphrase_lower or keyphrase_lower in candidate_lower:
                if len(candidate_words) > len(keyphrase_words):
                    if self.stopwords:
                        new_words = [w for w in candidate_words if w not in keyphrase_words]
                        meaningful_new_words = [w for w in new_words if w not in self.stopwords]
                        if len(meaningful_new_words) < 1:
                            continue
                else:
                    continue

            filtered_candidates.add(candidate)

        return filtered_candidates

    def _get_domain_vocabularies(self):
        
        return {
            "artificial intelligence": [
                "machine learning", "deep learning", "neural networks", "natural language processing",
                "computer vision", "AI ethics", "reinforcement learning", "generative AI",
                "large language models", "AI systems", "computational intelligence", "AI algorithms",
                "transformer models", "GPT", "BERT", "supervised learning", "unsupervised learning",
                "semi-supervised learning", "transfer learning", "federated learning", "edge AI",
                "AI assistants", "chatbots", "conversational AI", "AI agents", "autonomous systems",
                "computer reasoning", "knowledge representation", "expert systems", "decision support",
                "predictive analytics", "data mining", "pattern recognition", "image recognition",
                "speech recognition", "natural language understanding", "sentiment analysis",
                "recommendation systems", "personalization algorithms", "AI alignment", "AI safety",
                "responsible AI", "explainable AI", "AI transparency", "AI bias", "AI fairness",
                "AI governance", "AI regulation", "AI policy", "AI research", "AI development",
                "AI deployment", "AI adoption", "AI implementation", "AI integration", "AI solutions",
                "AI applications", "AI use cases", "AI benefits", "AI challenges", "AI limitations",
                "AI potential", "AI future", "AI trends", "AI innovations", "AI breakthroughs",
                "multimodal AI", "multimodal learning", "vision-language models", "VLMs", "text-to-image",
                "text-to-video", "text-to-3D", "text-to-audio", "diffusion models", "stable diffusion",
                "DALL-E", "Midjourney", "generative adversarial networks", "GANs", "variational autoencoders", "VAEs",
                "AI agents", "autonomous agents", "multi-agent systems", "agent-based modeling", "emergent behavior",
                "AI assistants", "AI copilots", "AI companions", "AI tutors", "AI coaches", "AI mentors",
                "retrieval-augmented generation", "RAG", "vector databases", "semantic search", "embedding models",
                "AI hallucinations", "factuality", "grounding", "knowledge graphs", "ontologies", "semantic networks",
                "AI reasoning", "chain-of-thought", "tree-of-thought", "graph-of-thought", "reasoning frameworks",
                "AI for science", "scientific discovery", "drug discovery", "protein folding", "AlphaFold",
                "AI for climate", "AI for healthcare", "AI for education", "AI for sustainability", "AI for social good",
                "AI chips", "neural processors", "TPUs", "NPUs", "AI accelerators", "neuromorphic computing",
                "quantum machine learning", "quantum neural networks", "quantum computing for AI",
                "AI benchmarks", "AI evaluation", "AI testing", "AI verification", "AI validation", "AI certification",
                "AI observability", "AI monitoring", "AI debugging", "AI interpretability", "AI explainability",
                "AI privacy", "federated learning", "differential privacy", "privacy-preserving AI", "secure AI",
                "AI security", "adversarial attacks", "adversarial examples", "model poisoning", "model stealing",
                "AI watermarking", "AI content detection", "synthetic content detection", "deepfake detection",
                "AI copyright", "AI intellectual property", "AI licensing", "AI patents", "AI trademarks",
                "AI standards", "AI best practices", "AI frameworks", "AI guidelines", "AI principles",
                "AI ethics boards", "AI ethics committees", "AI ethics councils", "AI ethics frameworks",
                "AI risk management", "AI safety research", "AI alignment research", "AI control problem",
                "AI consciousness", "artificial general intelligence", "AGI", "artificial superintelligence", "ASI",
                "human-AI collaboration", "human-in-the-loop", "human-centered AI", "augmented intelligence",
                "AI democratization", "AI accessibility", "AI inclusivity", "AI for everyone", "AI literacy"
            ],
            "cybersecurity": [
                "network security", "data protection", "encryption", "cyber threats",
                "vulnerability assessment", "security breach", "malware", "phishing",
                "ransomware", "zero-day exploit", "security protocols", "cyber defense",
                "information security", "cybersecurity framework", "security operations center",
                "threat intelligence", "incident response", "penetration testing", "ethical hacking",
                "security audit", "risk assessment", "security compliance", "data privacy",
                "identity management", "access control", "authentication", "authorization",
                "multi-factor authentication", "biometric security", "endpoint security",
                "cloud security", "application security", "mobile security", "IoT security",
                "cryptography", "public key infrastructure", "digital certificates", "secure coding",
                "security by design", "defense in depth", "security awareness", "security training",
                "social engineering", "spear phishing", "whaling", "vishing", "smishing",
                "DDoS attacks", "man-in-the-middle attacks", "SQL injection", "cross-site scripting",
                "buffer overflow", "privilege escalation", "backdoor", "rootkit", "keylogger",
                "spyware", "adware", "trojan horse", "worm", "virus", "botnet", "command and control",
                "data breach", "data leakage", "data loss prevention", "security monitoring",
                "security analytics", "security automation", "security orchestration", "SIEM",
                "intrusion detection", "intrusion prevention", "firewall", "web application firewall",
                "VPN", "secure gateway", "secure tunnel", "secure communications", "secure messaging"
            ],
            "automotive": [
                "electric vehicles", "autonomous driving", "vehicle safety", "automotive industry",
                "car manufacturing", "fuel efficiency", "connected cars", "vehicle emissions",
                "automotive technology", "car design", "automotive engineering", "vehicle performance",
                "hybrid vehicles", "plug-in hybrids", "battery electric vehicles", "hydrogen fuel cells",
                "EV charging infrastructure", "range anxiety", "battery technology", "battery management",
                "regenerative braking", "self-driving cars", "driver assistance systems", "ADAS",
                "lidar", "radar", "ultrasonic sensors", "camera systems", "vehicle-to-vehicle",
                "vehicle-to-infrastructure", "smart roads", "traffic management", "mobility as a service",
                "ride-sharing", "car-sharing", "micro-mobility", "last-mile transportation",
                "urban mobility", "sustainable transportation", "carbon-neutral vehicles",
                "emission standards", "catalytic converters", "particulate filters", "NOx reduction",
                "lightweight materials", "aerodynamics", "drag coefficient", "rolling resistance",
                "powertrain efficiency", "internal combustion engine", "transmission systems",
                "drivetrain", "suspension", "chassis", "body structure", "crash testing",
                "passive safety", "active safety", "collision avoidance", "emergency braking",
                "lane keeping", "adaptive cruise control", "blind spot detection", "parking assistance",
                "solid-state batteries", "battery swapping", "ultra-fast charging", "wireless charging",
                "vehicle-to-grid", "V2G", "bidirectional charging", "smart charging", "battery recycling",
                "battery second life", "circular economy", "sustainable materials", "recycled materials",
                "biobased materials", "carbon fiber", "composite materials", "additive manufacturing",
                "3D printing", "digital manufacturing", "industry 4.0", "smart factories", "cobots",
                "predictive maintenance", "over-the-air updates", "OTA", "software-defined vehicles",
                "vehicle operating systems", "automotive cybersecurity", "vehicle data security",
                "automotive data privacy", "vehicle connectivity", "5G automotive", "C-V2X", "DSRC",
                "intelligent transportation systems", "ITS", "smart traffic signals", "congestion management",
                "traffic optimization", "road pricing", "congestion charging", "low emission zones",
                "zero emission zones", "car-free zones", "pedestrianization", "walkable cities",
                "15-minute cities", "transit-oriented development", "multimodal transportation",
                "mobility hubs", "mobility platforms", "mobility apps", "mobility subscriptions",
                "robotaxis", "autonomous shuttles", "autonomous delivery", "drone delivery",
                "last-mile delivery", "urban air mobility", "eVTOL", "flying cars", "air taxis",
                "hyperloop", "maglev trains", "high-speed rail", "sustainable aviation", "electric aircraft",
                "synthetic fuels", "e-fuels", "biofuels", "renewable diesel", "green hydrogen",
                "fuel cell electric vehicles", "FCEV", "hydrogen infrastructure", "hydrogen refueling",
                "battery thermal management", "battery management systems", "BMS", "cell balancing",
                "battery chemistry", "lithium-ion", "LFP", "NMC", "solid electrolyte", "silicon anodes",
                "level 2 autonomy", "level 3 autonomy", "level 4 autonomy", "level 5 autonomy",
                "autonomous vehicle testing", "AV validation", "simulation testing", "digital twins",
                "high-definition mapping", "HD maps", "real-time mapping", "sensor fusion", "computer vision",
                "deep learning for automotive", "reinforcement learning for autonomous driving",
                "edge computing in vehicles", "automotive AI", "automotive cloud services"
            ],
            "environment": [
                "climate change", "renewable energy", "sustainability", "carbon emissions",
                "environmental protection", "green technology", "conservation", "biodiversity",
                "pollution control", "ecosystem", "environmental impact", "natural resources",
                "global warming", "greenhouse gases", "carbon footprint", "carbon neutrality",
                "carbon sequestration", "carbon capture", "carbon trading", "carbon tax",
                "climate action", "climate policy", "climate resilience", "climate adaptation",
                "climate mitigation", "Paris Agreement", "IPCC", "net zero emissions",
                "solar energy", "wind energy", "hydroelectric power", "geothermal energy",
                "biomass energy", "tidal energy", "wave energy", "energy storage",
                "smart grid", "energy efficiency", "green building", "LEED certification",
                "circular economy", "waste reduction", "recycling", "upcycling", "composting",
                "zero waste", "plastic pollution", "microplastics", "ocean acidification",
                "coral bleaching", "deforestation", "reforestation", "afforestation",
                "habitat restoration", "wildlife conservation", "endangered species",
                "protected areas", "national parks", "marine reserves", "conservation biology",
                "ecological footprint", "carrying capacity", "sustainable development",
                "environmental justice", "environmental health", "air quality", "water quality",
                "soil health", "land degradation", "desertification", "drought management",
                "flood control", "watershed management", "integrated water resources management",
                "climate crisis", "climate emergency", "climate tipping points", "climate feedback loops",
                "climate science", "climate modeling", "climate scenarios", "climate projections",
                "climate attribution", "extreme weather events", "heat waves", "droughts", "floods",
                "wildfires", "hurricanes", "typhoons", "cyclones", "sea level rise", "coastal erosion",
                "climate refugees", "climate migration", "climate justice", "climate equity",
                "intergenerational equity", "common but differentiated responsibilities", "CBDR",
                "nationally determined contributions", "NDCs", "global stocktake", "climate finance",
                "green climate fund", "loss and damage", "adaptation fund", "climate technology",
                "carbon budget", "carbon pricing", "carbon markets", "emissions trading system", "ETS",
                "carbon offsets", "carbon credits", "voluntary carbon market", "compliance carbon market",
                "carbon border adjustment mechanism", "CBAM", "carbon leakage", "carbon accounting",
                "science-based targets", "SBTi", "scope 1 emissions", "scope 2 emissions", "scope 3 emissions",
                "value chain emissions", "life cycle assessment", "LCA", "environmental product declarations",
                "EPD", "product carbon footprint", "PCF", "corporate sustainability reporting", "CSRD",
                "task force on climate-related financial disclosures", "TCFD", "climate risk",
                "physical climate risk", "transition climate risk", "climate stress testing", "climate scenarios",
                "direct air capture", "DAC", "bioenergy with carbon capture and storage", "BECCS",
                "carbon capture utilization and storage", "CCUS", "enhanced weathering", "ocean fertilization",
                "nature-based solutions", "NbS", "blue carbon", "mangrove restoration", "peatland restoration",
                "wetland conservation", "soil carbon sequestration", "regenerative agriculture",
                "agroforestry", "silvopasture", "conservation agriculture", "sustainable intensification",
                "precision agriculture", "climate-smart agriculture", "sustainable food systems",
                "food security", "water security", "energy security", "nexus approach", "systems thinking",
                "planetary boundaries", "doughnut economics", "green economy", "blue economy",
                "just transition", "green jobs", "green skills", "green recovery", "build back better",
                "sustainable consumption", "sustainable production", "sustainable lifestyles",
                "sustainable cities", "smart cities", "urban sustainability", "green infrastructure",
                "nature-positive", "biodiversity net gain", "ecological restoration", "rewilding",
                "ecosystem services", "natural capital", "biodiversity finance", "conservation finance"
            ],
            "food": [
                "nutrition", "culinary arts", "food safety", "dietary guidelines", "food production",
                "gastronomy", "food industry", "cooking techniques", "food science", "cuisine",
                "food culture", "food processing", "sustainable food systems", "organic farming",
                "regenerative agriculture", "precision agriculture", "vertical farming",
                "hydroponics", "aquaponics", "aeroponics", "urban farming", "community gardens",
                "farm-to-table", "local food movement", "slow food movement", "food sovereignty",
                "food security", "food access", "food deserts", "food waste", "food recovery",
                "food preservation", "fermentation", "canning", "dehydration", "freezing",
                "food additives", "food coloring", "food flavoring", "food texturizers",
                "food stabilizers", "food emulsifiers", "food fortification", "functional foods",
                "nutraceuticals", "probiotics", "prebiotics", "dietary supplements",
                "macronutrients", "micronutrients", "proteins", "carbohydrates", "fats",
                "vitamins", "minerals", "antioxidants", "phytonutrients", "dietary fiber",
                "food allergies", "food intolerances", "gluten-free", "dairy-free", "vegan",
                "vegetarian", "pescatarian", "flexitarian", "ketogenic diet", "paleo diet",
                "Mediterranean diet", "DASH diet", "intermittent fasting", "caloric restriction",
                "meal planning", "meal prep", "batch cooking", "food pairing", "flavor profiles",
                "culinary techniques", "knife skills", "mise en place", "food presentation",
                "molecular gastronomy", "modernist cuisine", "sous vide", "food technology", "food innovation",
                "alternative proteins", "plant-based proteins", "cultured meat", "lab-grown meat", "cellular agriculture",
                "insect protein", "algae protein", "mycoprotein", "protein isolates", "protein concentrates",
                "meat alternatives", "dairy alternatives", "egg alternatives", "seafood alternatives",
                "food tech startups", "food innovation hubs", "food incubators", "food accelerators",
                "food biotechnology", "food bioengineering", "food genomics", "food metabolomics",
                "precision nutrition", "personalized nutrition", "nutrigenomics", "nutrigenetics",
                "microbiome diet", "gut health", "gut-brain axis", "microbiota", "microbiome testing",
                "food as medicine", "medicinal foods", "adaptogens", "superfoods", "ancient grains",
                "heirloom varieties", "heritage breeds", "biodiversity conservation", "seed saving",
                "food biodiversity", "agrobiodiversity", "indigenous food systems", "traditional food knowledge",
                "culinary heritage", "food traditions", "food history", "food anthropology", "food sociology",
                "food psychology", "sensory science", "taste perception", "flavor science", "food neuroscience",
                "food design", "food architecture", "food styling", "food photography", "food media",
                "food tourism", "culinary tourism", "food experiences", "food festivals", "food events",
                "food education", "culinary education", "nutrition education", "cooking classes", "food literacy",
                "food policy", "food regulations", "food standards", "food labeling", "food certification",
                "food traceability", "food transparency", "blockchain in food", "food supply chain", "food logistics",
                "food distribution", "food retail", "food service", "food delivery", "ghost kitchens",
                "dark kitchens", "cloud kitchens", "virtual restaurants", "meal kits", "subscription food",
                "direct-to-consumer food", "community supported agriculture", "CSA", "farmers markets",
                "food cooperatives", "food hubs", "food commons", "food justice", "food equity", "food democracy"
            ],
            "real estate": [
                "property market", "housing", "commercial real estate", "real estate investment",
                "property development", "real estate agents", "property valuation",
                "real estate market", "property management", "real estate transactions",
                "real estate financing", "property ownership", "mortgage rates", "housing affordability",
                "rental market", "home prices", "real estate bubble", "housing demand", "housing supply",
                "property values", "real estate listings", "housing market volatility", "interest rates",
                "home buying", "property taxes", "real estate trends", "housing crisis", "home equity",
                "real estate development", "housing inventory", "real estate economics", "housing policy",
                "residential real estate", "single-family homes", "multi-family properties",
                "condominiums", "townhouses", "apartments", "co-ops", "vacation properties",
                "investment properties", "rental properties", "fix-and-flip", "buy-and-hold",
                "real estate appreciation", "real estate depreciation", "capital gains",
                "real estate taxes", "property tax assessment", "tax deductions", "1031 exchange",
                "real estate crowdfunding", "REITs", "real estate syndication", "private equity",
                "real estate portfolio", "diversification", "asset allocation", "risk management",
                "cash flow", "cap rate", "ROI", "NOI", "gross rent multiplier", "debt service coverage ratio",
                "loan-to-value ratio", "amortization", "fixed-rate mortgage", "adjustable-rate mortgage",
                "FHA loans", "VA loans", "conventional loans", "jumbo loans", "reverse mortgages",
                "mortgage insurance", "closing costs", "escrow", "title insurance", "home inspection",
                "appraisal", "zoning", "land use", "building codes", "permits", "easements",
                "encroachments", "liens", "foreclosure", "short sale", "real estate owned"
            ],
            "entertainment": [
                "film industry", "music business", "television production", "digital entertainment",
                "streaming services", "gaming industry", "entertainment media", "performing arts",
                "content creation", "entertainment technology", "media production", "creative arts",
                "movie studios", "film production", "cinematography", "film directing",
                "screenwriting", "film editing", "visual effects", "sound design", "film distribution",
                "box office", "film festivals", "award shows", "film criticism", "film genres",
                "blockbusters", "independent films", "documentaries", "animation", "short films",
                "music production", "record labels", "music publishing", "music licensing",
                "music streaming", "concert tours", "live performances", "music festivals",
                "music genres", "music composition", "songwriting", "music recording",
                "music mixing", "music mastering", "music distribution", "music marketing",
                "TV networks", "TV channels", "TV programming", "TV shows", "TV series",
                "TV episodes", "TV seasons", "TV pilots", "TV ratings", "TV advertising",
                "TV syndication", "TV streaming", "TV production companies", "showrunners",
                "video game development", "game design", "game programming", "game art",
                "game audio", "game testing", "game publishing", "game platforms",
                "console gaming", "PC gaming", "mobile gaming", "cloud gaming", "esports",
                "competitive gaming", "game streaming", "game communities", "game monetization",
                "theater", "broadway", "off-broadway", "regional theater", "community theater",
                "acting", "directing", "playwriting", "stage design", "costume design",
                "lighting design", "sound design", "choreography", "dance", "ballet",
                "contemporary dance", "hip-hop dance", "ballroom dance", "folk dance",
                "streaming platforms", "subscription video on demand", "SVOD", "ad-supported video on demand", "AVOD",
                "transactional video on demand", "TVOD", "over-the-top media", "OTT", "direct-to-consumer", "DTC",
                "content libraries", "original content", "exclusive content", "licensed content", "content acquisition",
                "content discovery", "recommendation algorithms", "personalized recommendations", "user engagement",
                "viewer retention", "churn rate", "subscriber growth", "streaming wars", "platform competition",
                "binge-watching", "binge-worthy content", "episodic content", "serialized storytelling", "anthology series",
                "limited series", "miniseries", "docuseries", "reality TV", "competition shows", "talent shows",
                "talk shows", "late-night shows", "variety shows", "sketch comedy", "stand-up comedy", "comedy specials",
                "premium cable", "basic cable", "broadcast television", "network television", "public broadcasting",
                "international distribution", "global rights", "territorial rights", "content localization", "dubbing",
                "subtitling", "closed captioning", "accessibility features", "audio description", "content ratings",
                "parental controls", "content moderation", "user-generated content", "UGC", "creator economy",
                "influencer marketing", "social media stars", "digital celebrities", "content creators", "YouTubers",
                "streamers", "Twitch streamers", "live streaming", "interactive streaming", "real-time engagement",
                "virtual events", "virtual concerts", "virtual fan experiences", "digital meet and greets",
                "virtual production", "LED volumes", "real-time rendering", "in-camera VFX", "motion capture",
                "performance capture", "digital humans", "digital doubles", "deepfakes", "synthetic media",
                "AI-generated content", "procedural content generation", "virtual influencers", "virtual celebrities",
                "virtual idols", "vtubers", "digital avatars", "digital fashion", "virtual merchandise", "digital collectibles",
                "NFTs", "blockchain entertainment", "web3 entertainment", "decentralized content", "fan ownership",
                "fan communities", "fandom", "fan engagement", "fan theories", "fan fiction", "fan art", "cosplay",
                "conventions", "comic-cons", "entertainment expos", "fan events", "meet and greets", "autograph signings",
                "merchandise", "licensed products", "consumer products", "brand extensions", "franchise development",
                "transmedia storytelling", "expanded universe", "cinematic universe", "shared universe", "crossovers",
                "spin-offs", "prequels", "sequels", "reboots", "remakes", "adaptations", "IP development", "IP acquisition"
            ],
            "virtual reality": [
                "augmented reality", "mixed reality", "immersive technology", "VR headsets",
                "virtual environments", "3D visualization", "spatial computing", "interactive experiences",
                "VR gaming", "virtual worlds", "immersive media", "VR applications",
                "extended reality", "XR", "AR glasses", "smart glasses", "holographic displays",
                "volumetric capture", "motion tracking", "hand tracking", "eye tracking",
                "haptic feedback", "force feedback", "tactile feedback", "spatial audio",
                "3D audio", "binaural audio", "ambisonics", "virtual presence", "telepresence",
                "social VR", "collaborative VR", "multi-user VR", "VR chat", "virtual meetings",
                "virtual conferences", "virtual events", "virtual tourism", "virtual travel",
                "virtual real estate", "virtual property", "metaverse", "digital twins",
                "virtual prototyping", "VR simulation", "VR training", "VR education",
                "VR therapy", "exposure therapy", "pain management", "rehabilitation",
                "VR fitness", "VR exercise", "VR meditation", "VR relaxation",
                "VR entertainment", "VR experiences", "VR storytelling", "VR filmmaking",
                "360-degree video", "stereoscopic 3D", "VR photography", "photogrammetry",
                "3D modeling", "3D scanning", "procedural generation", "real-time rendering",
                "VR development", "VR platforms", "VR frameworks", "VR SDKs", "WebXR"
            ],
            "general": [
                "technology", "innovation", "digital transformation", "business", "economy",
                "society", "culture", "education", "health", "science", "research",
                "development", "policy", "governance", "management", "leadership",
                "communication", "collaboration", "productivity", "efficiency",
                "sustainability", "resilience", "adaptation", "growth", "progress",
                "future trends", "emerging technologies", "disruptive innovation",
                "strategic planning", "problem solving", "critical thinking",
                "creative thinking", "decision making", "risk management",
                "performance optimization", "continuous improvement", "best practices",
                "knowledge sharing", "information management", "data analysis",
                "insights generation", "evidence-based approaches", "systems thinking",
                "holistic perspective", "interdisciplinary collaboration", "cross-functional teams"
            ]
        }

    def _get_domain_expansions(self, keyphrase, domain):
        
        domain_vocabularies = {
            "artificial intelligence": [
                "machine learning", "deep learning", "neural networks", "natural language processing",
                "computer vision", "AI ethics", "reinforcement learning", "generative AI",
                "large language models", "AI systems", "computational intelligence", "AI algorithms",
                "transformer models", "GPT", "BERT", "supervised learning", "unsupervised learning",
                "semi-supervised learning", "transfer learning", "federated learning", "edge AI",
                "AI assistants", "chatbots", "conversational AI", "AI agents", "autonomous systems",
                "computer reasoning", "knowledge representation", "expert systems", "decision support",
                "predictive analytics", "data mining", "pattern recognition", "image recognition",
                "speech recognition", "natural language understanding", "sentiment analysis",
                "recommendation systems", "personalization algorithms", "AI alignment", "AI safety",
                "responsible AI", "explainable AI", "AI transparency", "AI bias", "AI fairness",
                "AI governance", "AI regulation", "AI policy", "AI research", "AI development",
                "AI deployment", "AI adoption", "AI implementation", "AI integration", "AI solutions",
                "AI applications", "AI use cases", "AI benefits", "AI challenges", "AI limitations",
                "AI potential", "AI future", "AI trends", "AI innovations", "AI breakthroughs",
                "artificial neural networks", "convolutional neural networks", "CNN", "recurrent neural networks", "RNN",
                "LSTM", "GRU", "attention mechanisms", "self-attention", "transformers", "foundation models",
                "prompt engineering", "fine-tuning", "few-shot learning", "zero-shot learning", "one-shot learning",
                "model distillation", "knowledge distillation", "model compression", "quantization", "pruning",
                "neural architecture search", "AutoML", "hyperparameter optimization", "feature engineering",
                "feature extraction", "feature selection", "dimensionality reduction", "principal component analysis",
                "t-SNE", "UMAP", "clustering", "classification", "regression", "anomaly detection", "outlier detection",
                "time series analysis", "time series forecasting", "sequence modeling", "sequence prediction",
                "natural language generation", "text generation", "text summarization", "machine translation",
                "question answering", "information retrieval", "information extraction", "named entity recognition",
                "part-of-speech tagging", "dependency parsing", "semantic parsing", "semantic role labeling",
                "coreference resolution", "discourse analysis", "topic modeling", "document classification",
                "text classification", "image classification", "object detection", "semantic segmentation",
                "instance segmentation", "image generation", "image synthesis", "style transfer",
                "super-resolution", "image inpainting", "image restoration", "image enhancement",
                "image captioning", "visual question answering", "visual reasoning", "3D reconstruction",
                "depth estimation", "pose estimation", "action recognition", "activity recognition",
                "video understanding", "video captioning", "video generation", "video prediction",
                "speech synthesis", "voice conversion", "speaker identification", "speaker verification",
                "speaker diarization", "audio classification", "audio generation", "music generation",
                "sound synthesis", "audio enhancement", "noise reduction", "deep reinforcement learning",
                "Q-learning", "policy gradient", "actor-critic", "model-based reinforcement learning",
                "imitation learning", "inverse reinforcement learning", "multi-agent reinforcement learning",
                "game theory", "multi-agent systems", "swarm intelligence", "evolutionary algorithms",
                "genetic algorithms", "neuroevolution", "bayesian optimization", "bayesian inference",
                "probabilistic programming", "causal inference", "causal reasoning", "knowledge graphs",
                "knowledge bases", "ontologies", "semantic web", "linked data", "graph neural networks",
                "graph convolutional networks", "graph attention networks", "graph representation learning",
                "network analysis", "complex networks", "social network analysis", "recommender systems",
                "collaborative filtering", "content-based filtering", "hybrid recommenders", "personalization",
                "user modeling", "user profiling", "preference learning", "preference elicitation",
                "human-AI interaction", "human-computer interaction", "user experience", "user interface"
            ],
            "cybersecurity": [
                "network security", "data protection", "encryption", "cyber threats",
                "vulnerability assessment", "security breach", "malware", "phishing",
                "ransomware", "zero-day exploit", "security protocols", "cyber defense",
                "information security", "cybersecurity framework", "security operations center",
                "threat intelligence", "incident response", "penetration testing", "ethical hacking",
                "security audit", "risk assessment", "security compliance", "data privacy",
                "identity management", "access control", "authentication", "authorization",
                "multi-factor authentication", "biometric security", "endpoint security",
                "cloud security", "application security", "mobile security", "IoT security",
                "cryptography", "public key infrastructure", "digital certificates", "secure coding",
                "security by design", "defense in depth", "security awareness", "security training",
                "social engineering", "spear phishing", "whaling", "vishing", "smishing",
                "DDoS attacks", "man-in-the-middle attacks", "SQL injection", "cross-site scripting",
                "buffer overflow", "privilege escalation", "backdoor", "rootkit", "keylogger",
                "spyware", "adware", "trojan horse", "worm", "virus", "botnet", "command and control",
                "data breach", "data leakage", "data loss prevention", "security monitoring",
                "security analytics", "security automation", "security orchestration", "SIEM",
                "intrusion detection", "intrusion prevention", "firewall", "web application firewall",
                "VPN", "secure gateway", "secure tunnel", "secure communications", "secure messaging",
                "cyber attack", "cyber defense", "cyber warfare", "cyber espionage", "cyber terrorism",
                "cyber crime", "cyber threat", "cyber risk", "cyber vulnerability", "cyber incident",
                "cyber breach", "cyber intrusion", "cyber compromise", "cyber exploitation", "cyber campaign",
                "cyber operation", "cyber mission", "cyber capability", "cyber weapon", "cyber arsenal",
                "cyber command", "cyber force", "cyber unit", "cyber team", "cyber operator",
                "cyber analyst", "cyber intelligence", "cyber counterintelligence", "cyber attribution",
                "cyber forensics", "cyber investigation", "cyber evidence", "cyber artifact", "cyber indicator",
                "cyber signature", "cyber fingerprint", "cyber footprint", "cyber tradecraft", "cyber methodology",
                "cyber technique", "cyber tactic", "cyber procedure", "cyber protocol", "cyber standard",
                "cyber framework", "cyber policy", "cyber strategy", "cyber doctrine", "cyber law",
                "cyber regulation", "cyber compliance", "cyber governance", "cyber risk management",
                "cyber security management", "cyber security program", "cyber security architecture",
                "cyber security engineering", "cyber security development", "cyber security operations",
                "cyber security monitoring", "cyber security analytics", "cyber security intelligence",
                "cyber threat intelligence", "cyber threat hunting", "cyber threat detection", "cyber threat response",
                "cyber incident response", "cyber crisis management", "cyber disaster recovery", "cyber continuity",
                "cyber resilience", "remote work security", "VPN security", "home network security", "BYOD security",
                "supply chain attacks", "supply chain security", "third-party risk", "vendor risk management",
                "security perimeter", "zero trust security", "zero trust architecture", "zero trust network",
                "security posture", "security hygiene", "security culture", "security awareness training",
                "phishing simulation", "phishing awareness", "phishing prevention", "phishing protection",
                "email security", "spam filtering", "malware detection", "malware prevention", "malware removal",
                "antivirus", "anti-malware", "endpoint protection", "endpoint detection and response", "EDR",
                "extended detection and response", "XDR", "managed detection and response", "MDR",
                "security information and event management", "SIEM", "security orchestration automation and response", "SOAR",
                "threat hunting", "threat detection", "threat prevention", "threat response", "threat remediation",
                "vulnerability management", "vulnerability scanning", "vulnerability assessment", "vulnerability remediation",
                "patch management", "security patching", "security updates", "security fixes", "security hardening"
            ],
            "automotive": [
                "electric vehicles", "autonomous driving", "vehicle safety", "automotive industry",
                "car manufacturing", "fuel efficiency", "connected cars", "vehicle emissions",
                "automotive technology", "car design", "automotive engineering", "vehicle performance",
                "hybrid vehicles", "plug-in hybrids", "battery electric vehicles", "hydrogen fuel cells",
                "EV charging infrastructure", "range anxiety", "battery technology", "battery management",
                "regenerative braking", "self-driving cars", "driver assistance systems", "ADAS",
                "lidar", "radar", "ultrasonic sensors", "camera systems", "vehicle-to-vehicle",
                "vehicle-to-infrastructure", "smart roads", "traffic management", "mobility as a service",
                "ride-sharing", "car-sharing", "micro-mobility", "last-mile transportation",
                "urban mobility", "sustainable transportation", "carbon-neutral vehicles",
                "emission standards", "catalytic converters", "particulate filters", "NOx reduction",
                "lightweight materials", "aerodynamics", "drag coefficient", "rolling resistance",
                "powertrain efficiency", "internal combustion engine", "transmission systems",
                "drivetrain", "suspension", "chassis", "body structure", "crash testing",
                "passive safety", "active safety", "collision avoidance", "emergency braking",
                "lane keeping", "adaptive cruise control", "blind spot detection", "parking assistance"
            ],
            "environment": [
                "climate change", "renewable energy", "sustainability", "carbon emissions",
                "environmental protection", "green technology", "conservation", "biodiversity",
                "pollution control", "ecosystem", "environmental impact", "natural resources",
                "global warming", "greenhouse gases", "carbon footprint", "carbon neutrality",
                "carbon sequestration", "carbon capture", "carbon trading", "carbon tax",
                "climate action", "climate policy", "climate resilience", "climate adaptation",
                "climate mitigation", "Paris Agreement", "IPCC", "net zero emissions",
                "solar energy", "wind energy", "hydroelectric power", "geothermal energy",
                "biomass energy", "tidal energy", "wave energy", "energy storage",
                "smart grid", "energy efficiency", "green building", "LEED certification",
                "circular economy", "waste reduction", "recycling", "upcycling", "composting",
                "zero waste", "plastic pollution", "microplastics", "ocean acidification",
                "coral bleaching", "deforestation", "reforestation", "afforestation",
                "habitat restoration", "wildlife conservation", "endangered species",
                "protected areas", "national parks", "marine reserves", "conservation biology",
                "ecological footprint", "carrying capacity", "sustainable development",
                "environmental justice", "environmental health", "air quality", "water quality",
                "soil health", "land degradation", "desertification", "drought management",
                "flood control", "watershed management", "integrated water resources management"
            ],
            "food": [
                "nutrition", "culinary arts", "food safety", "dietary guidelines", "food production",
                "gastronomy", "food industry", "cooking techniques", "food science", "cuisine",
                "food culture", "food processing", "sustainable food systems", "organic farming",
                "regenerative agriculture", "precision agriculture", "vertical farming",
                "hydroponics", "aquaponics", "aeroponics", "urban farming", "community gardens",
                "farm-to-table", "local food movement", "slow food movement", "food sovereignty",
                "food security", "food access", "food deserts", "food waste", "food recovery",
                "food preservation", "fermentation", "canning", "dehydration", "freezing",
                "food additives", "food coloring", "food flavoring", "food texturizers",
                "food stabilizers", "food emulsifiers", "food fortification", "functional foods",
                "nutraceuticals", "probiotics", "prebiotics", "dietary supplements",
                "macronutrients", "micronutrients", "proteins", "carbohydrates", "fats",
                "vitamins", "minerals", "antioxidants", "phytonutrients", "dietary fiber",
                "food allergies", "food intolerances", "gluten-free", "dairy-free", "vegan",
                "vegetarian", "pescatarian", "flexitarian", "ketogenic diet", "paleo diet",
                "Mediterranean diet", "DASH diet", "intermittent fasting", "caloric restriction",
                "meal planning", "meal prep", "batch cooking", "food pairing", "flavor profiles",
                "culinary techniques", "knife skills", "mise en place", "food presentation"
            ],
            "real estate": [
                "property market", "housing", "commercial real estate", "real estate investment",
                "property development", "real estate agents", "property valuation",
                "real estate market", "property management", "real estate transactions",
                "real estate financing", "property ownership", "mortgage rates", "housing affordability",
                "rental market", "home prices", "real estate bubble", "housing demand", "housing supply",
                "property values", "real estate listings", "housing market volatility", "interest rates",
                "home buying", "property taxes", "real estate trends", "housing crisis", "home equity",
                "real estate development", "housing inventory", "real estate economics", "housing policy",
                "residential real estate", "single-family homes", "multi-family properties",
                "condominiums", "townhouses", "apartments", "co-ops", "vacation properties",
                "investment properties", "rental properties", "fix-and-flip", "buy-and-hold",
                "real estate appreciation", "real estate depreciation", "capital gains",
                "real estate taxes", "property tax assessment", "tax deductions", "1031 exchange",
                "real estate crowdfunding", "REITs", "real estate syndication", "private equity",
                "real estate portfolio", "diversification", "asset allocation", "risk management",
                "cash flow", "cap rate", "ROI", "NOI", "gross rent multiplier", "debt service coverage ratio",
                "loan-to-value ratio", "amortization", "fixed-rate mortgage", "adjustable-rate mortgage",
                "FHA loans", "VA loans", "conventional loans", "jumbo loans", "reverse mortgages",
                "mortgage insurance", "closing costs", "escrow", "title insurance", "home inspection",
                "appraisal", "zoning", "land use", "building codes", "permits", "easements",
                "encroachments", "liens", "foreclosure", "short sale", "real estate owned",
                "housing market trends", "housing market forecast", "housing market analysis",
                "housing market report", "housing market outlook", "housing market prediction",
                "housing market crash", "housing market correction", "housing market recovery",
                "housing market boom", "housing market bust", "housing market cycle",
                "housing market indicators", "housing market metrics", "housing market statistics",
                "housing market data", "housing market research", "housing market study",
                "housing affordability index", "housing affordability crisis", "housing affordability solutions",
                "affordable housing", "affordable housing programs", "affordable housing initiatives",
                "affordable housing policies", "affordable housing development", "affordable housing financing",
                "affordable housing tax credits", "low-income housing", "workforce housing", "public housing",
                "housing subsidies", "housing vouchers", "housing assistance", "housing benefits",
                "first-time homebuyer", "first-time homebuyer programs", "first-time homebuyer incentives",
                "first-time homebuyer tax credits", "first-time homebuyer grants", "first-time homebuyer loans",
                "mortgage lending", "mortgage origination", "mortgage servicing", "mortgage refinancing",
                "mortgage modification", "mortgage forbearance", "mortgage default", "mortgage foreclosure",
                "mortgage backed securities", "mortgage interest rates", "mortgage terms", "mortgage conditions",
                "real estate market analysis", "comparative market analysis", "CMA", "real estate valuation",
                "real estate appraisal methods", "income approach", "sales comparison approach", "cost approach",
                "real estate investment analysis", "real estate due diligence", "real estate acquisition",
                "real estate disposition", "real estate exit strategy", "real estate holding period"
            ],
            "entertainment": [
                "film industry", "music business", "television production", "digital entertainment",
                "streaming services", "gaming industry", "entertainment media", "performing arts",
                "content creation", "entertainment technology", "media production", "creative arts",
                "movie studios", "film production", "cinematography", "film directing",
                "screenwriting", "film editing", "visual effects", "sound design", "film distribution",
                "box office", "film festivals", "award shows", "film criticism", "film genres",
                "blockbusters", "independent films", "documentaries", "animation", "short films",
                "music production", "record labels", "music publishing", "music licensing",
                "music streaming", "concert tours", "live performances", "music festivals",
                "music genres", "music composition", "songwriting", "music recording",
                "music mixing", "music mastering", "music distribution", "music marketing",
                "TV networks", "TV channels", "TV programming", "TV shows", "TV series",
                "TV episodes", "TV seasons", "TV pilots", "TV ratings", "TV advertising",
                "TV syndication", "TV streaming", "TV production companies", "showrunners",
                "video game development", "game design", "game programming", "game art",
                "game audio", "game testing", "game publishing", "game platforms",
                "console gaming", "PC gaming", "mobile gaming", "cloud gaming", "esports",
                "competitive gaming", "game streaming", "game communities", "game monetization",
                "theater", "broadway", "off-broadway", "regional theater", "community theater",
                "acting", "directing", "playwriting", "stage design", "costume design",
                "lighting design", "sound design", "choreography", "dance", "ballet",
                "contemporary dance", "hip-hop dance", "ballroom dance", "folk dance"
            ],
            "virtual reality": [
                "augmented reality", "mixed reality", "immersive technology", "VR headsets",
                "virtual environments", "3D visualization", "spatial computing", "interactive experiences",
                "VR gaming", "virtual worlds", "immersive media", "VR applications",
                "extended reality", "XR", "AR glasses", "smart glasses", "holographic displays",
                "volumetric capture", "motion tracking", "hand tracking", "eye tracking",
                "haptic feedback", "force feedback", "tactile feedback", "spatial audio",
                "3D audio", "binaural audio", "ambisonics", "virtual presence", "telepresence",
                "social VR", "collaborative VR", "multi-user VR", "VR chat", "virtual meetings",
                "virtual conferences", "virtual events", "virtual tourism", "virtual travel",
                "virtual real estate", "virtual property", "metaverse", "digital twins",
                "virtual prototyping", "VR simulation", "VR training", "VR education",
                "VR therapy", "exposure therapy", "pain management", "rehabilitation",
                "VR fitness", "VR exercise", "VR meditation", "VR relaxation",
                "VR entertainment", "VR experiences", "VR storytelling", "VR filmmaking",
                "360-degree video", "stereoscopic 3D", "VR photography", "photogrammetry",
                "3D modeling", "3D scanning", "procedural generation", "real-time rendering",
                "VR development", "VR platforms", "VR frameworks", "VR SDKs", "WebXR",
                "Meta Quest", "Oculus", "HTC Vive", "Valve Index", "PlayStation VR", "PSVR",
                "Windows Mixed Reality", "Apple Vision Pro", "standalone headsets", "tethered headsets",
                "wireless VR", "inside-out tracking", "outside-in tracking", "6DOF", "3DOF",
                "degrees of freedom", "foveated rendering", "passthrough", "room-scale VR",
                "seated VR", "standing VR", "walking VR", "VR locomotion", "smooth locomotion",
                "comfort options", "VR sickness", "VR controllers", "VR gloves", "VR treadmills",
                "VR suits", "full-body tracking", "VR arcades", "VR theme parks", "VR escape rooms",
                "VR fitness", "VR workouts", "VR meditation", "VR therapy", "VR exposure therapy",
                "VR rehabilitation", "VR training", "VR simulation", "VR education", "VR classrooms",
                "VR field trips", "VR museums", "VR art galleries", "VR social", "VR chat",
                "VR productivity", "VR workspaces", "VR desktops", "VR browsers", "WebVR",
                "VR prototyping", "VR testing", "VR user research", "spatial interfaces",
                "VR rendering", "VR performance", "VR optimization", "VR graphics", "VR shaders",
                "VR lighting", "VR materials", "VR textures", "VR physics", "VR collisions",
                "VR gestures", "VR voice commands", "VR storytelling", "VR narratives", "VR cinematography",
                "VR directing", "VR editing", "VR post-production", "VR distribution", "VR stores",
                "VR marketplaces", "VR communities", "VR social networks", "VR creators", "VR developers",
                "VR studios", "VR companies", "VR startups", "VR investments", "VR funding", "VR industry",
                "immersive experiences", "immersive content", "immersive entertainment", "immersive learning",
                "immersive training", "immersive therapy", "immersive design", "immersive development",
                "immersive storytelling", "immersive journalism", "immersive marketing", "immersive advertising",
                "immersive retail", "immersive commerce", "immersive healthcare", "immersive education",
                "immersive collaboration", "immersive meetings", "immersive conferences", "immersive events"
            ],
            "general": [
                "technology", "innovation", "digital transformation", "business", "economy",
                "society", "culture", "education", "health", "science", "research",
                "development", "policy", "governance", "management", "leadership",
                "communication", "collaboration", "productivity", "efficiency",
                "sustainability", "resilience", "adaptation", "growth", "progress",
                "future trends", "emerging technologies", "disruptive innovation",
                "strategic planning", "problem solving", "critical thinking",
                "creative thinking", "decision making", "risk management",
                "performance optimization", "continuous improvement", "best practices",
                "knowledge sharing", "information management", "data analysis",
                "insights generation", "evidence-based approaches", "systems thinking",
                "holistic perspective", "interdisciplinary collaboration", "cross-functional teams"
            ]
        }

        expansions = set()

        if domain is None:
            domain = "general"

        domain_lower = domain.lower()
        matching_domains = []

        if domain_lower in domain_vocabularies:
            matching_domains.append((domain_lower, 1.0))

        for domain_key in domain_vocabularies.keys():
            if domain_key == domain_lower:
                continue

            if domain_key in domain_lower or domain_lower in domain_key:
                overlap_len = len(set(domain_key).intersection(set(domain_lower)))
                match_score = overlap_len / max(len(domain_key), len(domain_lower))
                matching_domains.append((domain_key, match_score))

        if domain_lower != "general" and "general" not in [d[0] for d in matching_domains]:
            matching_domains.append(("general", 0.5))

        matching_domains.sort(key=lambda x: x[1], reverse=True)

        keyphrase_words = keyphrase.lower().split()

        for matching_domain, match_score in matching_domains:
            domain_terms = domain_vocabularies[matching_domain]

            for term in domain_terms:
                term_words = term.lower().split()

                common_words = set(keyphrase_words).intersection(set(term_words))
                if common_words:
                    overlap_score = len(common_words) / min(len(keyphrase_words), len(term_words))

                    if overlap_score > 0.3 or match_score > 0.8:
                        expansions.add(term)

                elif len(keyphrase_words) == 1 and len(keyphrase_words[0]) > 3:
                    keyphrase_word = keyphrase_words[0]
                    for term_word in term_words:
                        if len(term_word) > 3 and (keyphrase_word in term_word or term_word in keyphrase_word):
                            expansions.add(term)
                            break

        return expansions

def test_with_fusion_extractor(text, domain=None, keyphrases=None):
    
    detected_domain = domain

    if keyphrases is None:
        try:
    

            print("Initializing FusionKeyphraseExtractor...")
            fusion_extractor = FusionKeyphraseExtractor(
                use_gpu=True,
                abstractive_weight=0.45,
                extractive_weight=0.55,
                redundancy_threshold=0.68,
                min_score=0.09
            )

            print("\nExtracting keyphrases using fusion extractor...")
            keyphrases = fusion_extractor.extract_keyphrases_with_scores(text)

            if domain is None:
                detected_domain = fusion_extractor.abstractive_extractor.detect_domain(text)
                print(f"Detected domain: {detected_domain}")

        except (ImportError, ModuleNotFoundError):
            print("FusionKeyphraseExtractor not available. Please provide keyphrases manually.")

            if keyphrases is None:
                from sklearn.feature_extraction.text import CountVectorizer

                vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words='english')
                X = vectorizer.fit_transform([text])
                features = vectorizer.get_feature_names_out()

                scores = X.toarray()[0]
                top_indices = scores.argsort()[-10:][::-1]
                keyphrases = [(features[i], float(scores[i])) for i in top_indices]
                print("Using simple n-gram extraction as fallback.")

                if domain is None:
                    detected_domain = _detect_domain_from_text(text)
                    print(f"Detected domain from keywords: {detected_domain}")

    print(f"\nExtracted {len(keyphrases)} keyphrases:")
    for keyphrase, score in keyphrases:
        print(f"- {keyphrase}: {score:.2f}")

    expander = ContextualKeyphraseExpander(
        use_gpu=True,
        similarity_threshold=0.55,
        max_suggestions=5,
        use_phrase_quality_check=True,
        use_collocations=True,
        use_pos_patterns=True,
        use_keybert=True,
        keybert_diversity=0.7,
        model_name="all-mpnet-base-v2"
    )

    print("\nExpanding keyphrases with appropriate quality filtering...")
    expanded_keyphrases = expander.expand_keyphrases(
        keyphrases,
        text,
        domain=detected_domain,
        min_quality_score=0.68,
        num_suggestions=5,
        use_curated=True
    )

    print_expansion_results(
        {'original_keyphrases': keyphrases, 'expanded_keyphrases': expanded_keyphrases},
        detailed=True,
        min_score_threshold=0.68
    )

    try:
        if 'fusion_extractor' in locals():
            fusion_extractor.clean_memory()
    except Exception:
        pass

    return {
        "original_keyphrases": keyphrases,
        "expanded_keyphrases": expanded_keyphrases
    }

def print_expansion_results(results, detailed=True, min_score_threshold=0.68):
    
    print("\nKeyphrase Expansion Results:")

    print("\nOriginal Keyphrases:")
    for i, (keyphrase, score) in enumerate(results['original_keyphrases']):
        if detailed:
            print(f"  {i+1}. {keyphrase} (score: {score:.2f})")
        else:
            print(f"  {i+1}. {keyphrase}")

    print("\nExpanded Keyphrases:")
    for i, (keyphrase, suggestions) in enumerate(results['expanded_keyphrases'].items()):

        quality_suggestions = [(s, score) for s, score in suggestions if score >= min_score_threshold]

        print(f"{keyphrase}:")

        if quality_suggestions:
            for j, (suggestion, score) in enumerate(quality_suggestions):
                if detailed:
                    print(f"  {j+1}. {suggestion} (score: {score:.2f})")
                else:
                    print(f"  {j+1}. {suggestion}")
        else:
            print(f"  -1")

def _detect_domain_from_text(text):
    
    text_lower = text.lower()

    domain_keywords = {
        "artificial intelligence": [
            ("ai", 1.0), ("machine learning", 2.0), ("neural network", 2.0),
            ("deep learning", 2.0), ("nlp", 1.5), ("computer vision", 2.0),
            ("artificial intelligence", 3.0), ("language model", 2.0), ("chatbot", 1.5),
            ("algorithm", 0.5), ("data science", 1.5), ("training data", 1.5),
            ("generative ai", 2.5), ("transformer", 1.5), ("gpt", 2.0), ("bert", 2.0)
        ],
        "cybersecurity": [
            ("security", 0.8), ("cyber", 1.0), ("hack", 1.5), ("breach", 1.5),
            ("malware", 2.0), ("phishing", 2.0), ("ransomware", 2.5), ("encryption", 1.5),
            ("cybersecurity", 3.0), ("vulnerability", 2.0), ("threat", 1.0), ("attack", 1.0),
            ("firewall", 2.0), ("data protection", 1.5), ("password", 1.0), ("authentication", 1.5),
            ("zero-day", 2.5), ("exploit", 1.5), ("security protocol", 2.0), ("cyber attack", 2.5)
        ],
        "automotive": [
            ("car", 1.0), ("vehicle", 1.0), ("automotive", 2.0), ("driving", 0.8),
            ("electric vehicle", 2.0), ("ev", 1.0), ("autonomous", 1.5),
            ("automobile", 2.0), ("self-driving", 2.5), ("fuel", 1.0), ("engine", 1.0),
            ("battery", 0.8), ("charging", 0.8), ("tesla", 1.5), ("toyota", 1.5),
            ("ford", 1.5), ("gm", 1.0), ("volkswagen", 1.5), ("bmw", 1.5)
        ],
        "environment": [
            ("climate", 1.5), ("environment", 1.0), ("sustainability", 2.0),
            ("renewable", 2.0), ("carbon", 1.5), ("green", 0.8), ("pollution", 1.5),
            ("climate change", 2.5), ("global warming", 2.5), ("emissions", 1.5),
            ("solar", 1.5), ("wind power", 2.0), ("fossil fuel", 2.0), ("biodiversity", 2.0),
            ("conservation", 1.5), ("ecosystem", 2.0), ("sustainable", 1.5)
        ],
        "food": [
            ("food", 1.0), ("nutrition", 1.5), ("diet", 1.0), ("cooking", 1.5),
            ("restaurant", 1.5), ("culinary", 2.0), ("ingredient", 1.5),
            ("recipe", 1.5), ("chef", 1.5), ("meal", 1.0), ("cuisine", 2.0),
            ("organic", 1.0), ("vegetarian", 1.5), ("vegan", 1.5), ("flavor", 1.0),
            ("taste", 0.8), ("dining", 1.5), ("food industry", 2.0)
        ],
        "real estate": [
            ("property", 1.0), ("real estate", 2.5), ("housing", 1.5), ("mortgage", 2.0),
            ("rent", 1.0), ("apartment", 1.0), ("home", 0.8), ("house", 0.8),
            ("commercial property", 2.0), ("residential", 1.5), ("housing market", 2.5),
            ("interest rate", 1.0), ("buyer", 1.0), ("seller", 1.0), ("realtor", 2.0),
            ("property value", 2.0), ("housing price", 2.0), ("real estate agent", 2.5),
            ("affordability", 1.5), ("housing supply", 2.0), ("housing demand", 2.0)
        ],
        "entertainment": [
            ("movie", 1.0), ("film", 1.0), ("music", 1.0), ("entertainment", 1.5),
            ("game", 0.8), ("streaming", 1.5), ("television", 1.0), ("tv", 0.8),
            ("hollywood", 2.0), ("actor", 1.0), ("actress", 1.0), ("director", 1.0),
            ("box office", 2.0), ("netflix", 1.5), ("disney", 1.5), ("hbo", 1.5),
            ("concert", 1.5), ("album", 1.0), ("celebrity", 1.5), ("blockbuster", 1.5)
        ],
        "virtual reality": [
            ("vr", 2.0), ("virtual reality", 3.0), ("augmented reality", 2.5), ("ar", 1.0),
            ("mixed reality", 2.5), ("immersive", 1.5), ("headset", 1.5),
            ("oculus", 2.0), ("meta quest", 2.5), ("htc vive", 2.5), ("metaverse", 2.0),
            ("3d", 0.8), ("simulation", 1.0), ("virtual environment", 2.0), ("vr game", 2.0),
            ("virtual world", 2.0), ("immersive experience", 2.0), ("spatial computing", 2.5)
        ]
    }

    domain_scores = {}
    for domain, keywords in domain_keywords.items():
        score = 0
        for keyword, weight in keywords:

            count = text_lower.count(keyword)
            if count > 0:

                score += min(3, count) * weight
        domain_scores[domain] = score

    if domain_scores:
        max_domain = max(domain_scores.items(), key=lambda x: x[1])

        if max_domain[1] >= 2.0:
            return max_domain[0]

    return "general"

def test_with_custom_text(text, domain=None, keyphrases=None):
    

    if domain is None:
        domain = _detect_domain_from_text(text)
        print(f"Detected domain: {domain}")

    results = test_with_fusion_extractor(text, domain=domain, keyphrases=keyphrases)

    print_expansion_results(results, min_score_threshold=0.68)

    return results