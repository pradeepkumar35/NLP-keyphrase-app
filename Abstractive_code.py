import torch
import numpy as np
import re
import gc
import os
import sys
import time
from typing import List, Dict, Tuple, Set, Optional, Union, Any
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import set_seed
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import spacy
from spacy.language import Language
import math
import numpy as np
from nltk import pos_tag
from nltk.tokenize import sent_tokenize
from transformers import pipeline

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

try:
    nltk.data.find('stopwords')
    nltk.data.find('punkt')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

class AbstractiveKeyphraseExtractor:


    PROMPT_TEMPLATES = [
        """Generate keyphrases that capture the main topics and concepts in the text below.
Extract important technical terms, concepts, and topics.
Focus on specific, descriptive phrases rather than generic terms.
Include both single-word and multi-word keyphrases, with emphasis on multi-word phrases.
Provide 15-20 keyphrases, separated by commas.

Text: {text}

Keyphrases:"""

        ,

        """Extract the most important keyphrases from the following text.
Focus on technical terms, named entities, and domain-specific concepts.
Include both single words and multi-word phrases (2-4 words).
Prioritize phrases that capture specific topics rather than general concepts.
Aim for 15-20 keyphrases, separated by commas.

Text: {text}

Keyphrases:"""

        ,

        """Identify the key concepts and terminology in the following text.
Extract specific technical terms, named entities, and important phrases.
Include both single words and multi-word expressions.
Focus on terms that are central to understanding the main topics.
List 15-20 keyphrases, separated by commas.

Text: {text}

Keyphrases:"""

        ,

        """Extract keyphrases that best represent the main topics in this text.
Include technical terms, specific concepts, and important entities.
Prioritize multi-word phrases (2-4 words) over single words when possible.
Focus on specific, descriptive phrases rather than generic terms.
Provide 15-20 keyphrases, separated by commas.

Text: {text}

Keyphrases:"""
        ]

    def __init__(
    self,
    model_name: str = "google/flan-t5-large",
    use_gpu: bool = True,
    use_mdeberta_domain_detection: bool = True,
    max_length: int = 512,
    num_beams: int = 20,
    top_k: int = 100,
    top_p: float = 0.95,
    temperature: float = 0.8,
    repetition_penalty: float = 1.5,
    length_penalty: float = 1.0,
    max_new_tokens: int = 300,
    prompt_template_idx: int = 0,
    batch_size: int = 1,
    seed: int = 42,
    use_8bit: bool = False,
    use_fp16: bool = True,
    max_input_length: int = 1024,
    use_chunking: bool = True,
    chunk_overlap: int = 50,
    post_process: bool = True,
    filter_stopwords: bool = True,
    min_phrase_length: int = 1,
    max_phrase_length: int = 5,
    prioritize_multi_word: bool = True,
    use_lemmatization: bool = True,
    use_ner: bool = True,
    ner_model: str = "en_core_web_sm",
    use_sampling: bool = False,
    num_beam_groups: int = 5,
    diversity_penalty: float = 1.8,
    ):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.max_length = max_length
        self.num_beams = num_beams
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.max_new_tokens = max_new_tokens
        self.prompt_template_idx = prompt_template_idx
        self.batch_size = batch_size
        self.seed = seed
        self.use_8bit = use_8bit
        self.use_fp16 = use_fp16
        self.max_input_length = max_input_length
        self.use_chunking = use_chunking
        self.chunk_overlap = chunk_overlap
        self.post_process = post_process
        self.filter_stopwords = filter_stopwords
        self.min_phrase_length = min_phrase_length
        self.max_phrase_length = max_phrase_length
        self.prioritize_multi_word = prioritize_multi_word
        self.use_lemmatization = use_lemmatization
        self.use_ner = use_ner
        self.ner_model = ner_model
        self.use_sampling = use_sampling
        self.use_mdeberta_domain_detection = use_mdeberta_domain_detection
        self.debug_redundancy = False

        self.pattern_bonuses = {}

        self.has_domain_classifier = False
        self.initialize_domain_components()
        if self.use_ner:
            try:
                print(f"Loading NER model: {ner_model}")
                self.nlp = spacy.load(ner_model)
                self.nlp.disable_pipes(["parser"])
                print("NER model loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load NER model: {str(e)}")
                print("Falling back to basic NER functionality")
                self.use_ner = False

        print(f"Loading sentence transformer model...")
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
        self.use_sampling = use_sampling
        if num_beam_groups > 1:
            self.num_beam_groups = num_beam_groups
            self.diversity_penalty = diversity_penalty

        set_seed(seed)
        self._quality_percentiles = {
        "technology": 5,
        "science": 5,
        "health": 5,
        "business": 5,
        "politics": 5,
        "sports": 5,
        "entertainment": 5,
        "world": 5,
        "default": 5
        }
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        print(f"Using device: {self.device}")

        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Loading model: {model_name}")

        model_kwargs = {
            "device_map": "auto" if self.device == "cuda" else None,
        }

        if use_8bit and self.device == "cuda":
            model_kwargs["load_in_8bit"] = True

        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                **model_kwargs
            )

            if use_fp16 and not use_8bit and self.device == "cuda":
                self.model = self.model.half()

            if self.device == "cuda" and not use_8bit and "device_map" not in model_kwargs:
                self.model = self.model.to(self.device)

            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

        if self.use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()

        if self.filter_stopwords:
            self.stopwords = set(stopwords.words('english'))
            self.stopwords.update([
                'said', 'according', 'reported', 'told', 'says', 'say', 'saying',
                'stated', 'states', 'state', 'added', 'adds', 'add', 'noted',
                'notes', 'note', 'explained', 'explains', 'explain', 'claimed',
                'claims', 'claim', 'announced', 'announces', 'announce'
            ])
        self.domain_generation_params = {
            "technology": {
                "num_beams": 24,
                "num_beam_groups": 6,
                "diversity_penalty": 1.8,
                "max_new_tokens": 300
            },
            "science": {
                "num_beams": 24,
                "num_beam_groups": 6,
                "diversity_penalty": 1.8,
                "max_new_tokens": 300
            },
            "health": {
                "num_beams": 20,
                "num_beam_groups": 5,
                "diversity_penalty": 1.5,
                "max_new_tokens": 250
            },
            "default": {
                "num_beams": 16,
                "num_beam_groups": 4,
                "diversity_penalty": 1.2,
                "max_new_tokens": 200
            }
        }
        self.initialize_generic_terms()
        self.initialize_domain_components()
        print("Abstractive Keyphrase Extractor initialized")

    def optimize_for_raw_candidates(self, articles: List[str], num_articles: int = 3) -> Dict[str, Any]:

        print("\n" + "="*80)
        print("OPTIMIZING FOR MAXIMUM RAW CANDIDATE GENERATION")
        print("="*80)

        results = self.optimize_generation_across_articles(
            articles,
            num_articles=num_articles,
            focus_on_raw_count=True
        )

        best_params = results["best_params"]

        if best_params:
            print("\nBest parameters for maximizing raw candidates:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")

            if articles:
                sample_article = articles[0]
                print("\nTesting best parameters on sample article...")

                original_params = {}
                for param in best_params:
                    if hasattr(self, param):
                        original_params[param] = getattr(self, param)

                self.apply_optimized_parameters(best_params)

                raw_keyphrases = self.generate_keyphrases(sample_article)
                print(f"\nGenerated {len(raw_keyphrases)} raw keyphrases with optimized parameters")

                self.examine_raw_output(sample_article, params=best_params)

                for param, value in original_params.items():
                    setattr(self, param, value)

        return results

    def clean_memory(self):

        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    def preprocess_text(self, text: str) -> str:

        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) > self.max_input_length:
            text = text[:self.max_input_length]

        return text

    def optimize_generation_across_articles(self, articles: List[str], num_articles: int = 5, focus_on_raw_count: bool = True) -> Dict[str, Any]:

        print("\n" + "="*80)
        print("OPTIMIZING GENERATION PARAMETERS ACROSS MULTIPLE ARTICLES")
        print("="*80)

        test_articles = articles[:num_articles] if len(articles) > num_articles else articles
        print(f"Testing on {len(test_articles)} articles")

        param_results = {}

        param_combinations = [
            {
                "name": "Beam Search (8 beams)",
                "params": {
                    "use_sampling": False,
                    "num_beams": 8,
                    "length_penalty": 0.8,
                    "prompt_template_idx": 0,
                    "repetition_penalty": 1.3,
                    "max_new_tokens": 150,
                }
            },
            {
                "name": "Beam Search (12 beams)",
                "params": {
                    "use_sampling": False,
                    "num_beams": 12,
                    "length_penalty": 0.8,
                    "prompt_template_idx": 0,
                    "repetition_penalty": 1.3,
                    "max_new_tokens": 150,
                }
            },
            {
                "name": "Beam Search (16 beams)",
                "params": {
                    "use_sampling": False,
                    "num_beams": 16,
                    "length_penalty": 0.8,
                    "prompt_template_idx": 0,
                    "repetition_penalty": 1.3,
                    "max_new_tokens": 150,
                }
            },
            {
                "name": "Diverse Beam (8 beams, 2 groups, penalty=0.8)",
                "params": {
                    "use_sampling": False,
                    "num_beams": 8,
                    "num_beam_groups": 2,
                    "diversity_penalty": 0.8,
                    "length_penalty": 0.8,
                    "prompt_template_idx": 0,
                    "repetition_penalty": 1.3,
                    "max_new_tokens": 150,
                }
            },
            {
                "name": "Diverse Beam (8 beams, 4 groups, penalty=1.0)",
                "params": {
                    "use_sampling": False,
                    "num_beams": 8,
                    "num_beam_groups": 4,
                    "diversity_penalty": 1.0,
                    "length_penalty": 0.8,
                    "prompt_template_idx": 0,
                    "repetition_penalty": 1.3,
                    "max_new_tokens": 150,
                }
            },
            {
                "name": "Diverse Beam (12 beams, 3 groups, penalty=1.0)",
                "params": {
                    "use_sampling": False,
                    "num_beams": 12,
                    "num_beam_groups": 3,
                    "diversity_penalty": 1.0,
                    "length_penalty": 0.8,
                    "prompt_template_idx": 0,
                    "repetition_penalty": 1.3,
                    "max_new_tokens": 150,
                }
            },
            {
                "name": "Diverse Beam (12 beams, 4 groups, penalty=1.0)",
                "params": {
                    "use_sampling": False,
                    "num_beams": 12,
                    "num_beam_groups": 4,
                    "diversity_penalty": 1.0,
                    "length_penalty": 0.8,
                    "prompt_template_idx": 0,
                    "repetition_penalty": 1.3,
                    "max_new_tokens": 150,
                }
            },
            {
                "name": "Diverse Beam (12 beams, 4 groups, penalty=1.5)",
                "params": {
                    "use_sampling": False,
                    "num_beams": 12,
                    "num_beam_groups": 4,
                    "diversity_penalty": 1.5,
                    "length_penalty": 0.8,
                    "prompt_template_idx": 0,
                    "repetition_penalty": 1.3,
                    "max_new_tokens": 150,
                }
            },
            {
                "name": "Diverse Beam (16 beams, 4 groups, penalty=1.0)",
                "params": {
                    "use_sampling": False,
                    "num_beams": 16,
                    "num_beam_groups": 4,
                    "diversity_penalty": 1.0,
                    "length_penalty": 0.8,
                    "prompt_template_idx": 0,
                    "repetition_penalty": 1.3,
                    "max_new_tokens": 150,
                }
            },
            {
                "name": "Diverse Beam (16 beams, 8 groups, penalty=1.2)",
                "params": {
                    "use_sampling": False,
                    "num_beams": 16,
                    "num_beam_groups": 8,
                    "diversity_penalty": 1.2,
                    "length_penalty": 0.8,
                    "prompt_template_idx": 0,
                    "repetition_penalty": 1.3,
                    "max_new_tokens": 150,
                }
            },
            {
                "name": "Sampling (temp=0.6, top_p=0.95)",
                "params": {
                    "use_sampling": True,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 100,
                    "repetition_penalty": 1.3,
                    "prompt_template_idx": 0,
                    "max_new_tokens": 150,
                }
            },
            {
                "name": "Sampling (temp=0.7, top_p=0.95)",
                "params": {
                    "use_sampling": True,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 100,
                    "repetition_penalty": 1.3,
                    "prompt_template_idx": 0,
                    "max_new_tokens": 150,
                }
            },
            {
                "name": "Sampling (temp=0.8, top_p=0.95)",
                "params": {
                    "use_sampling": True,
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "top_k": 100,
                    "repetition_penalty": 1.3,
                    "prompt_template_idx": 0,
                    "max_new_tokens": 150,
                }
            },
            {
                "name": "Simplified Prompt (Diverse Beam 12/4)",
                "params": {
                    "use_sampling": False,
                    "num_beams": 12,
                    "num_beam_groups": 4,
                    "diversity_penalty": 1.0,
                    "length_penalty": 0.8,
                    "prompt_template_idx": 5 if len(self.PROMPT_TEMPLATES) > 5 else 0,
                    "repetition_penalty": 1.3,
                    "max_new_tokens": 150,
                }
            },
            {
                "name": "Diverse Beam (12/4) with 200 tokens",
                "params": {
                    "use_sampling": False,
                    "num_beams": 12,
                    "num_beam_groups": 4,
                    "diversity_penalty": 1.0,
                    "length_penalty": 0.8,
                    "prompt_template_idx": 0,
                    "repetition_penalty": 1.3,
                    "max_new_tokens": 200,
                }
            },
            {
                "name": "Hybrid (Diverse Beam + Low Temp)",
                "params": {
                    "use_sampling": True,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 100,
                    "num_beams": 8,
                    "num_beam_groups": 4,
                    "diversity_penalty": 1.0,
                    "length_penalty": 0.8,
                    "prompt_template_idx": 0,
                    "repetition_penalty": 1.3,
                    "max_new_tokens": 150,
                }
            },
        ]

        original_params = {
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'length_penalty': self.length_penalty,
            'max_new_tokens': self.max_new_tokens,
            'prompt_template_idx': self.prompt_template_idx,
            'num_beams': self.num_beams,
            'repetition_penalty': self.repetition_penalty,
            'use_sampling': getattr(self, 'use_sampling', True),
        }

        for combo in param_combinations:
            param_results[combo["name"]] = {
                "params": combo["params"],
                "keyphrase_counts": [],
                "raw_counts": [],
                "multi_word_percentages": [],
                "avg_lengths": [],
                "quality_scores": [],
                "processing_times": [],
                "domains": []
            }

        for i, article in enumerate(test_articles):
            print(f"\nTesting article {i+1}/{len(test_articles)}")

            for combo in param_combinations:
                print(f"\n  Testing: {combo['name']}")

                for param, value in combo["params"].items():
                    setattr(self, param, value)

                if "num_beam_groups" in combo["params"] and combo["params"]["num_beam_groups"] > 1:
                    self.num_beam_groups = combo["params"]["num_beam_groups"]
                    self.diversity_penalty = combo["params"]["diversity_penalty"]
                else:
                    if hasattr(self, 'num_beam_groups'):
                        delattr(self, 'num_beam_groups')
                    if hasattr(self, 'diversity_penalty'):
                        delattr(self, 'diversity_penalty')

                start_time = time.time()

                domain = self.detect_domain(article)

                raw_keyphrases = self.generate_keyphrases(article)
                raw_count = len(raw_keyphrases)

                keyphrases = self.extract_keyphrases_with_scores(article)

                end_time = time.time()
                processing_time = end_time - start_time

                num_keyphrases = len(keyphrases)
                multi_word_count = sum(1 for kp, _ in keyphrases if len(kp.split()) > 1)
                multi_word_percentage = multi_word_count / num_keyphrases if num_keyphrases > 0 else 0
                avg_length = sum(len(kp.split()) for kp, _ in keyphrases) / num_keyphrases if num_keyphrases > 0 else 0

                quantity_score = 0.0
                if 15 <= raw_count <= 25:
                    quantity_score = 1.0
                elif 25 < raw_count <= 35:
                    quantity_score = 0.9
                elif 35 < raw_count <= 45:
                    quantity_score = 0.8
                elif 10 <= raw_count < 15:
                    quantity_score = 0.7
                elif 5 <= raw_count < 10:
                    quantity_score = 0.5
                elif raw_count > 45:
                    quantity_score = 0.7

                length_score = 1.0 - abs(avg_length - 2.5) / 2.5 if avg_length > 0 else 0

                quality_score = (
                    (quantity_score * 0.5) +
                    (multi_word_percentage * 0.3) +
                    (length_score * 0.2)
                )

                param_results[combo["name"]]["keyphrase_counts"].append(num_keyphrases)
                param_results[combo["name"]]["raw_counts"].append(raw_count)
                param_results[combo["name"]]["multi_word_percentages"].append(multi_word_percentage)
                param_results[combo["name"]]["avg_lengths"].append(avg_length)
                param_results[combo["name"]]["quality_scores"].append(quality_score)
                param_results[combo["name"]]["processing_times"].append(processing_time)
                param_results[combo["name"]]["domains"].append(domain)

                print(f"    Generated {raw_count} raw keyphrases, {num_keyphrases} final keyphrases")
                print(f"    Domain: {domain}")
                print(f"    Multi-word: {multi_word_percentage:.1%}, Avg length: {avg_length:.1f}, Quality: {quality_score:.2f}")
                print(f"    Processing time: {processing_time:.2f}s")

        best_avg_score = 0
        best_raw_count = 0
        best_combo_name = None
        best_combo_name_raw = None

        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)

        for name, results in param_results.items():
            avg_count = sum(results["keyphrase_counts"]) / len(results["keyphrase_counts"])
            avg_raw_count = sum(results["raw_counts"]) / len(results["raw_counts"])
            avg_multi_word = sum(results["multi_word_percentages"]) / len(results["multi_word_percentages"])
            avg_length = sum(results["avg_lengths"]) / len(results["avg_lengths"])
            avg_quality = sum(results["quality_scores"]) / len(results["quality_scores"])
            avg_time = sum(results["processing_times"]) / len(results["processing_times"])

            results["avg_count"] = avg_count
            results["avg_raw_count"] = avg_raw_count
            results["avg_multi_word"] = avg_multi_word
            results["avg_length"] = avg_length
            results["avg_quality"] = avg_quality
            results["avg_time"] = avg_time

            print(f"\n{name}:")
            print(f"  Avg raw keyphrases: {avg_raw_count:.1f}")
            print(f"  Avg final keyphrases: {avg_count:.1f}")
            print(f"  Avg multi-word: {avg_multi_word:.1%}")
            print(f"  Avg length: {avg_length:.1f} words")
            print(f"  Avg quality score: {avg_quality:.2f}")
            print(f"  Avg processing time: {avg_time:.2f}s")

            if focus_on_raw_count:
                if avg_raw_count > best_raw_count and avg_quality >= 0.5:
                    best_raw_count = avg_raw_count
                    best_combo_name_raw = name

                combined_score = avg_quality * (0.5 + 0.5 * min(1.0, avg_raw_count / 20.0))
                results["combined_score"] = combined_score

                if combined_score > best_avg_score:
                    best_avg_score = combined_score
                    best_combo_name = name
            else:
                combined_score = avg_quality * (0.5 + 0.5 * min(1.0, avg_raw_count / 20.0))
                results["combined_score"] = combined_score

                if combined_score > best_avg_score:
                    best_avg_score = combined_score
                    best_combo_name = name

        if focus_on_raw_count and best_combo_name_raw:
            best_combo_name = best_combo_name_raw

        if best_combo_name:
            print("\n" + "="*80)
            print(f"BEST PARAMETER COMBINATION: {best_combo_name}")
            print("="*80)
            if focus_on_raw_count:
                print(f"Selected based on maximizing raw candidate count")
            else:
                print(f"Selected based on combined quality score")

            print(f"Average raw keyphrases: {param_results[best_combo_name]['avg_raw_count']:.1f}")
            print(f"Average final keyphrases: {param_results[best_combo_name]['avg_count']:.1f}")
            print(f"Average quality score: {param_results[best_combo_name]['avg_quality']:.2f}")
            print(f"Average multi-word percentage: {param_results[best_combo_name]['avg_multi_word']:.1%}")
            print(f"Average phrase length: {param_results[best_combo_name]['avg_length']:.1f} words")
            print(f"Average processing time: {param_results[best_combo_name]['avg_time']:.2f}s")
            print("\nParameters:")
            for param, value in param_results[best_combo_name]["params"].items():
                print(f"  {param}: {value}")

        for param, value in original_params.items():
            setattr(self, param, value)

        if "num_beam_groups" not in original_params and hasattr(self, 'num_beam_groups'):
            delattr(self, 'num_beam_groups')
        if "num_beam_groups" not in original_params and hasattr(self, 'num_beam_groups'):
            delattr(self, 'num_beam_groups')
        if "diversity_penalty" not in original_params and hasattr(self, 'diversity_penalty'):
            delattr(self, 'diversity_penalty')

        print("\nParameter optimization complete. Original parameters restored.")

        top_raw_count_combos = sorted(
            [(name, results["avg_raw_count"]) for name, results in param_results.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]

        print("\nTop 3 parameter sets by raw candidate count:")
        for name, avg_raw_count in top_raw_count_combos:
            print(f"- {name}: {avg_raw_count:.1f} raw candidates")

        print("\nRecommendation: Manually inspect raw output of these top parameter sets using:")
        for name, _ in top_raw_count_combos:
            params_str = ", ".join([f"{k}={v}" for k, v in param_results[name]["params"].items()])
            print(f"extractor.examine_raw_output(text, params={{{params_str}}})")

        return {
            "best_combination": best_combo_name,
            "best_params": param_results[best_combo_name]["params"] if best_combo_name else None,
            "top_raw_count_combos": top_raw_count_combos,
            "all_results": param_results
        }

    def apply_optimized_parameters(self, params: Dict[str, Any]) -> None:

        print("\nApplying optimized parameters:")

        for param, value in params.items():
            if hasattr(self, param):
                print(f"  Setting {param} = {value}")
                setattr(self, param, value)
            else:
                print(f"  Warning: Parameter '{param}' not found in extractor")

        if "num_beam_groups" in params and params["num_beam_groups"] > 1:
            self.num_beam_groups = params["num_beam_groups"]
            if "diversity_penalty" in params:
                self.diversity_penalty = params["diversity_penalty"]
            else:
                self.diversity_penalty = 1.0
            print(f"  Setting num_beam_groups = {self.num_beam_groups}")
            print(f"  Setting diversity_penalty = {self.diversity_penalty}")
        elif hasattr(self, 'num_beam_groups'):
            delattr(self, 'num_beam_groups')
            if hasattr(self, 'diversity_penalty'):
                delattr(self, 'diversity_penalty')

        print("Parameters applied successfully")

    def test_parameter_improvement(self, text: str) -> None:

        print("\n" + "="*80)
        print("TESTING PARAMETER IMPROVEMENT")
        print("="*80)

        parameter_sets = [
            {
                "name": "Default Parameters",
                "params": {
                    "use_sampling": True,
                    "temperature": 0.7,
                    "top_p": 0.92,
                    "top_k": 50,
                    "num_beams": 5,
                    "repetition_penalty": 1.2,
                    "length_penalty": 0.8,
                    "max_new_tokens": 75,
                    "prompt_template_idx": 0
                }
            },
            {
                "name": "Optimized Parameters",
                "params": {
                    "use_sampling": False,
                    "num_beams": 12,
                    "num_beam_groups": 4,
                    "diversity_penalty": 1.0,
                    "repetition_penalty": 1.3,
                    "length_penalty": 0.8,
                    "max_new_tokens": 150,
                    "prompt_template_idx": 0
                }
            }
        ]

        original_params = {
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'length_penalty': self.length_penalty,
            'max_new_tokens': self.max_new_tokens,
            'prompt_template_idx': self.prompt_template_idx,
            'num_beams': self.num_beams,
            'repetition_penalty': self.repetition_penalty,
            'use_sampling': getattr(self, 'use_sampling', True),
        }

        results = {}

        for param_set in parameter_sets:
            print(f"\nTesting: {param_set['name']}")

            for param, value in param_set["params"].items():
                setattr(self, param, value)

            if "num_beam_groups" in param_set["params"] and param_set["params"]["num_beam_groups"] > 1:
                self.num_beam_groups = param_set["params"]["num_beam_groups"]
                self.diversity_penalty = param_set["params"]["diversity_penalty"]
            else:
                if hasattr(self, 'num_beam_groups'):
                    delattr(self, 'num_beam_groups')
                if hasattr(self, 'diversity_penalty'):
                    delattr(self, 'diversity_penalty')

            start_time = time.time()
            keyphrases = self.generate_keyphrases(text)
            end_time = time.time()

            scored_keyphrases = self.score_keyphrases_by_relevance(keyphrases, text)

            num_keyphrases = len(keyphrases)
            multi_word_count = sum(1 for kp in keyphrases if len(kp.split()) > 1)
            multi_word_percentage = multi_word_count / num_keyphrases if num_keyphrases > 0 else 0
            avg_length = sum(len(kp.split()) for kp in keyphrases) / num_keyphrases if num_keyphrases > 0 else 0
            processing_time = end_time - start_time

            results[param_set["name"]] = {
                "keyphrases": scored_keyphrases,
                "count": num_keyphrases,
                "multi_word_percentage": multi_word_percentage,
                "avg_length": avg_length,
                "processing_time": processing_time
            }

            print(f"  Generated {num_keyphrases} keyphrases in {processing_time:.2f} seconds")
            print(f"  Multi-word: {multi_word_percentage:.1%}, Avg length: {avg_length:.1f}")
            print("\n  Top keyphrases:")
            for kp, score in sorted(scored_keyphrases, key=lambda x: x[1], reverse=True)[:10]:
                print(f"    - {kp}: {score:.4f}")

        if "Default Parameters" in results and "Optimized Parameters" in results:
            default_count = results["Default Parameters"]["count"]
            optimized_count = results["Optimized Parameters"]["count"]

            count_improvement = optimized_count - default_count
            count_improvement_percent = (count_improvement / max(1, default_count)) * 100

            print("\n" + "="*80)
            print("IMPROVEMENT SUMMARY")
            print("="*80)
            print(f"Keyphrase count: {default_count} → {optimized_count} ({count_improvement_percent:+.1f}%)")

            default_multi = results["Default Parameters"]["multi_word_percentage"]
            optimized_multi = results["Optimized Parameters"]["multi_word_percentage"]
            multi_improvement = optimized_multi - default_multi
            print(f"Multi-word percentage: {default_multi:.1%} → {optimized_multi:.1%} ({multi_improvement:+.1%})")

            default_time = results["Default Parameters"]["processing_time"]
            optimized_time = results["Optimized Parameters"]["processing_time"]
            time_difference = optimized_time - default_time
            time_difference_percent = (time_difference / default_time) * 100
            print(f"Processing time: {default_time:.2f}s → {optimized_time:.2f}s ({time_difference_percent:+.1f}%)")

        for param, value in original_params.items():
            setattr(self, param, value)

        if "num_beam_groups" not in original_params and hasattr(self, 'num_beam_groups'):
            delattr(self, 'num_beam_groups')
        if "diversity_penalty" not in original_params and hasattr(self, 'diversity_penalty'):
            delattr(self, 'diversity_penalty')

        print("\nTest complete. Original parameters restored.")

    def chunk_text(self, text: str, chunk_size: int = 1024, overlap: int = 50) -> List[str]:

        if not self.use_chunking or len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            if end < len(text):
                last_period = text.rfind('.', start + int(chunk_size * 0.8), end)
                last_newline = text.rfind('\n', start + int(chunk_size * 0.8), end)

                if last_period != -1 or last_newline != -1:
                    end = max(last_period, last_newline) + 1

            chunks.append(text[start:min(end, len(text))])

            start = end - overlap

        return chunks

    def calculate_semantic_diversity(self, keyphrases: List[str], embeddings: Optional[np.ndarray] = None) -> float:

        if not keyphrases or len(keyphrases) < 2:
            return 0.0

        if embeddings is None:
            embeddings = self.sentence_model.encode(keyphrases, show_progress_bar=False)

        similarity_matrix = cosine_similarity(embeddings)

        n = len(keyphrases)
        total_similarity = 0.0
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                total_similarity += similarity_matrix[i, j]
                count += 1

        avg_similarity = total_similarity / max(1, count)

        diversity = 1.0 - avg_similarity

        return diversity

    def enhance_semantic_quality(self, scored_keyphrases: List[Tuple[str, float]], text: str) -> List[Tuple[str, float]]:

        if not scored_keyphrases:
            return scored_keyphrases

        doc_embedding = self.sentence_model.encode([text], show_progress_bar=False)[0]

        keyphrases = [kp for kp, _ in scored_keyphrases]
        keyphrase_embeddings = self.sentence_model.encode(keyphrases, show_progress_bar=False)

        semantic_scores = []
        for i, (kp, score) in enumerate(scored_keyphrases):
            kp_embedding = keyphrase_embeddings[i]
            doc_similarity = cosine_similarity([kp_embedding], [doc_embedding])[0][0]

            combined_score = (score * 0.65) + (doc_similarity * 0.35)

            if doc_similarity > 0.75:
                combined_score += 0.03
            semantic_scores.append((kp, combined_score))

        semantic_scores.sort(key=lambda x: x[1], reverse=True)

        return semantic_scores

    def apply_context_aware_weighting(self, keyphrases: List[str], text: str) -> List[Tuple[str, float]]:

        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            print(f"Warning: Error in sentence tokenization: {str(e)}. Using fallback method.")
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        sentence_weights = {}
        for i, sentence in enumerate(sentences):
            if i < len(sentences) * 0.1:
                sentence_weights[sentence] = 1.25
            elif i < len(sentences) * 0.2:
                sentence_weights[sentence] = 1.15
            elif i > len(sentences) * 0.85:
                sentence_weights[sentence] = 1.1
            else:
                sentence_weights[sentence] = 1.0

        keyphrase_scores = []
        for keyphrase in keyphrases:
            max_weight = 0
            for sentence, weight in sentence_weights.items():
                if keyphrase.lower() in sentence.lower():
                    max_weight = max(max_weight, weight)

            if max_weight == 0:
                max_weight = 0.8

            keyphrase_scores.append((keyphrase, max_weight))

        return keyphrase_scores

    def apply_domain_quality_metrics(self, keyphrases: List[Tuple[str, float]], domain: str) -> List[Tuple[str, float]]:

        domain_specific_adjustments = {
            "artificial intelligence": {
                "technical_terms_boost": 0.30,
                "general_terms_penalty": 0.15,
                "technical_terms": ["neural network", "deep learning", "machine learning", "algorithm",
                                   "model", "training", "dataset", "parameter", "accuracy", "precision",
                                   "transformer", "reinforcement learning", "computer vision", "nlp",
                                   "natural language processing", "ai", "artificial intelligence", "neural",
                                   "classification", "regression", "clustering", "supervised", "unsupervised"]
            },
            "cybersecurity": {
                "technical_terms_boost": 0.30,
                "general_terms_penalty": 0.15,
                "technical_terms": ["vulnerability", "exploit", "malware", "ransomware", "encryption",
                                   "authentication", "firewall", "breach", "attack", "security",
                                   "phishing", "threat", "zero-day", "backdoor", "trojan", "virus",
                                   "worm", "ddos", "mitm", "man-in-the-middle", "penetration testing"]
            },
            "automotive": {
                "technical_terms_boost": 0.30,
                "general_terms_penalty": 0.15,
                "technical_terms": ["electric vehicle", "ev", "autonomous", "self-driving", "hybrid",
                                   "combustion", "engine", "transmission", "chassis", "powertrain",
                                   "battery", "charging", "fuel cell", "hydrogen", "emissions",
                                   "safety system", "driver assistance", "adas"]
            },
            "food": {
                "technical_terms_boost": 0.30,
                "general_terms_penalty": 0.15,
                "technical_terms": ["cuisine", "ingredient", "recipe", "culinary", "nutrition",
                                   "organic", "sustainable", "farm-to-table", "gmo", "processing",
                                   "fermentation", "preservation", "cooking", "baking", "grilling",
                                   "roasting", "flavor", "taste", "texture"]
            },
            "environment": {
                "technical_terms_boost": 0.30,
                "general_terms_penalty": 0.15,
                "technical_terms": ["climate change", "global warming", "carbon", "emissions", "renewable",
                                   "sustainable", "biodiversity", "ecosystem", "conservation", "pollution",
                                   "greenhouse gas", "fossil fuel", "deforestation", "habitat", "species",
                                   "extinction", "recycling", "waste management"]
            },
            "real estate": {
                "technical_terms_boost": 0.30,
                "general_terms_penalty": 0.15,
                "technical_terms": ["property", "mortgage", "interest rate", "housing", "commercial",
                                   "residential", "lease", "rent", "investment", "development",
                                   "appraisal", "valuation", "market", "listing", "agent", "broker",
                                   "closing", "escrow", "title", "deed"]
            },
            "entertainment": {
                "technical_terms_boost": 0.30,
                "general_terms_penalty": 0.15,
                "technical_terms": ["movie", "film", "television", "streaming", "music", "concert",
                                   "performance", "actor", "actress", "director", "producer",
                                   "box office", "rating", "award", "genre", "studio", "platform",
                                   "series", "episode", "season"]
            },
            "default": {
                "technical_terms_boost": 0.08,
                "general_terms_penalty": 0.06,
                "technical_terms": []
            }
        }

        domain_config = domain_specific_adjustments.get(domain.lower(), domain_specific_adjustments["default"])

        adjusted_scores = []
        for keyphrase, score in keyphrases:
            adjustment = 0

            for term in domain_config["technical_terms"]:
                if term in keyphrase.lower():
                    adjustment += domain_config["technical_terms_boost"]
                    break

            general_terms = ["system", "process", "method", "approach", "solution", "technology", "application",
                            "development", "innovation", "advancement", "improvement", "feature", "capability",
                            "function", "service", "product", "device", "tool", "platform"]
            if any(keyphrase.lower() == term for term in general_terms):
                adjustment -= domain_config["general_terms_penalty"]

            adjusted_scores.append((keyphrase, score + adjustment))

        adjusted_scores.sort(key=lambda x: x[1], reverse=True)

        return adjusted_scores

    def enhance_keyphrase_diversity(self, keyphrases: List[Tuple[str, float]], min_diversity_threshold: float = 0.32) -> List[Tuple[str, float]]:

        if len(keyphrases) <= 1:
            return keyphrases

        kps = [kp for kp, _ in keyphrases]
        embeddings = self.sentence_model.encode(kps, show_progress_bar=False)

        similarity_matrix = cosine_similarity(embeddings)

        selected_indices = [0]
        candidate_indices = list(range(1, len(keyphrases)))

        while candidate_indices and len(selected_indices) < len(keyphrases):
            min_avg_similarity = float('inf')
            next_index = -1

            for idx in candidate_indices:
                avg_similarity = sum(similarity_matrix[idx][sel_idx] for sel_idx in selected_indices) / len(selected_indices)

                if avg_similarity < min_avg_similarity:
                    min_avg_similarity = avg_similarity
                    next_index = idx

            if next_index != -1 and min_avg_similarity < (1 - min_diversity_threshold):
                selected_indices.append(next_index)
                candidate_indices.remove(next_index)
            else:
                break

        selected_keyphrases = [keyphrases[idx] for idx in sorted(selected_indices, key=lambda i: keyphrases[i][1], reverse=True)]
        return selected_keyphrases

    def enhance_semantic_diversity(self, scored_keyphrases: List[Tuple[str, float]], text: str, min_cluster_count: int = 3, max_similarity_threshold: float = 0.75) -> List[Tuple[str, float]]:

        if not scored_keyphrases or len(scored_keyphrases) < 3:
            return scored_keyphrases

        keyphrases = [kp for kp, _ in scored_keyphrases]
        scores = [score for _, score in scored_keyphrases]

        embeddings = self.sentence_model.encode(keyphrases, show_progress_bar=False)

        doc_embedding = self.sentence_model.encode([text], show_progress_bar=False)[0]

        similarity_matrix = cosine_similarity(embeddings)

        doc_relevance = cosine_similarity(embeddings, [doc_embedding])

        clusters = {}
        assigned = set()
        cluster_id = 0

        sorted_indices = sorted(range(len(scored_keyphrases)), key=lambda i: scored_keyphrases[i][1], reverse=True)

        for idx in sorted_indices:
            if idx in assigned:
                continue

            clusters[cluster_id] = {
                'center': idx,
                'members': [idx],
                'avg_score': scores[idx],
                'doc_relevance': doc_relevance[idx][0]
            }
            assigned.add(idx)

            for j in range(len(keyphrases)):
                if j in assigned:
                    continue

                if similarity_matrix[idx][j] > max_similarity_threshold:
                    clusters[cluster_id]['members'].append(j)
                    clusters[cluster_id]['avg_score'] = sum(scores[m] for m in clusters[cluster_id]['members']) / len(clusters[cluster_id]['members'])
                    assigned.add(j)

            cluster_id += 1

            if cluster_id >= min_cluster_count and len(assigned) >= len(keyphrases) * 0.8:
                break

        for i in range(len(keyphrases)):
            if i in assigned:
                continue

            best_cluster = -1
            best_similarity = -1

            for c_id, cluster in clusters.items():
                center_idx = cluster['center']
                similarity = similarity_matrix[i][center_idx]

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = c_id

            if best_cluster != -1:
                clusters[best_cluster]['members'].append(i)
                clusters[best_cluster]['avg_score'] = sum(scores[m] for m in clusters[best_cluster]['members']) / len(clusters[best_cluster]['members'])
                assigned.add(i)
            else:
                clusters[cluster_id] = {
                    'center': i,
                    'members': [i],
                    'avg_score': scores[i],
                    'doc_relevance': doc_relevance[i][0]
                }
                assigned.add(i)
                cluster_id += 1

        for c_id, cluster in clusters.items():
            size_factor = min(1.0, len(cluster['members']) / 3)
            cluster['importance'] = (0.4 * cluster['avg_score'] +
                                    0.4 * cluster['doc_relevance'] +
                                    0.2 * size_factor)

        sorted_clusters = sorted(clusters.items(), key=lambda x: x[1]['importance'], reverse=True)

        selected_indices = set()

        for c_id, cluster in sorted_clusters:
            selected_indices.add(cluster['center'])

        for c_id, cluster in sorted_clusters:
            target_reps = min(3, max(1, len(cluster['members']) // 2))

            sorted_members = sorted(cluster['members'], key=lambda i: scores[i], reverse=True)

            count = 1
            for member in sorted_members:
                if member != cluster['center'] and count < target_reps:
                    selected_indices.add(member)
                    count += 1

        if len(selected_indices) < len(scored_keyphrases):
            remaining = [i for i in range(len(keyphrases)) if i not in selected_indices]
            remaining.sort(key=lambda i: scores[i], reverse=True)

            for i in remaining:
                if len(selected_indices) >= len(scored_keyphrases):
                    break
                selected_indices.add(i)

        enhanced_keyphrases = [(keyphrases[i], scores[i]) for i in selected_indices]

        enhanced_keyphrases.sort(key=lambda x: x[1], reverse=True)

        original_diversity = self.calculate_semantic_diversity(keyphrases, embeddings)
        enhanced_diversity = self.calculate_semantic_diversity([kp for kp, _ in enhanced_keyphrases])

        print(f"Semantic diversity: {original_diversity:.3f} → {enhanced_diversity:.3f}")
        print(f"Keyphrase count: {len(scored_keyphrases)} → {len(enhanced_keyphrases)}")

        return enhanced_keyphrases

    def select_diverse_keyphrases(
    self,
    scored_keyphrases: List[Tuple[str, float]],
    target_count: int,
    diversity_weight: float = 0.3
    ) -> List[Tuple[str, float]]:

        if not scored_keyphrases:
            return []

        if len(scored_keyphrases) <= target_count:
            return scored_keyphrases

        all_keyphrases = [kp for kp, _ in scored_keyphrases]
        all_scores = [score for _, score in scored_keyphrases]

        all_embeddings = self.sentence_model.encode(all_keyphrases, show_progress_bar=False)

        selected_indices = [0]
        selected_embeddings = [all_embeddings[0]]

        while len(selected_indices) < target_count:
            best_score = -1
            best_idx = -1

            for i in range(len(all_keyphrases)):
                if i in selected_indices:
                    continue

                candidate_indices = selected_indices + [i]
                candidate_embeddings = selected_embeddings + [all_embeddings[i]]

                candidate_diversity = self.calculate_semantic_diversity(
                    [all_keyphrases[idx] for idx in candidate_indices],
                    np.array(candidate_embeddings)
                )

                relevance = all_scores[i] / max(all_scores)

                combined_score = (1 - diversity_weight) * relevance + diversity_weight * candidate_diversity

                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = i

            if best_idx != -1:
                selected_indices.append(best_idx)
                selected_embeddings.append(all_embeddings[best_idx])
            else:
                break

        return [(all_keyphrases[i], all_scores[i]) for i in selected_indices]

    def generate_keyphrases(self, text: str, debug_output: bool = False, original_domain: str = None) -> List[str]:

        text = self.preprocess_text(text)

        domain = self.detect_domain(text, original_domain=original_domain)
        if debug_output:
            if original_domain:
                print(f"Using original domain for generation: {domain}")
                print(f"IMPORTANT: Using original domain from input data for domain-specific parameters")
            else:
                print(f"Detected domain for generation: {domain}")

        original_params = {
            'num_beams': self.num_beams,
            'max_new_tokens': self.max_new_tokens,
            'repetition_penalty': self.repetition_penalty,
            'length_penalty': self.length_penalty
        }
        if hasattr(self, 'num_beam_groups'):
            original_params['num_beam_groups'] = self.num_beam_groups
            original_params['diversity_penalty'] = self.diversity_penalty

        domain_params_applied = False
        if hasattr(self, 'domain_generation_params') and domain in self.domain_generation_params:
            domain_params = self.domain_generation_params[domain]
            for param, value in domain_params.items():
                setattr(self, param, value)
            domain_params_applied = True
            if debug_output:
                print(f"Applied domain-specific parameters for {domain}:")
                for param, value in domain_params.items():
                    print(f"  - {param}: {value}")
        elif not hasattr(self, 'domain_generation_params'):
            self.domain_generation_params = {
                "technology": {
                    "num_beams": 24,
                    "num_beam_groups": 6,
                    "diversity_penalty": 1.8,
                    "max_new_tokens": 300,
                    "repetition_penalty": 1.5
                },
                "science": {
                    "num_beams": 24,
                    "num_beam_groups": 6,
                    "diversity_penalty": 1.8,
                    "max_new_tokens": 300,
                    "repetition_penalty": 1.5
                },
                "health": {
                    "num_beams": 20,
                    "num_beam_groups": 5,
                    "diversity_penalty": 1.5,
                    "max_new_tokens": 250,
                    "repetition_penalty": 1.4
                },
                "business": {
                    "num_beams": 18,
                    "num_beam_groups": 6,
                    "diversity_penalty": 1.5,
                    "max_new_tokens": 250,
                    "repetition_penalty": 1.4
                },
                "politics": {
                    "num_beams": 18,
                    "num_beam_groups": 6,
                    "diversity_penalty": 1.5,
                    "max_new_tokens": 250,
                    "repetition_penalty": 1.4
                },
                "sports": {
                    "num_beams": 16,
                    "num_beam_groups": 4,
                    "diversity_penalty": 1.2,
                    "max_new_tokens": 200,
                    "repetition_penalty": 1.3
                },
                "entertainment": {
                    "num_beams": 16,
                    "num_beam_groups": 4,
                    "diversity_penalty": 1.2,
                    "max_new_tokens": 200,
                    "repetition_penalty": 1.3
                },
                "default": {
                    "num_beams": 20,
                    "num_beam_groups": 5,
                    "diversity_penalty": 1.5,
                    "max_new_tokens": 250,
                    "repetition_penalty": 1.4
                }
            }

            if domain in self.domain_generation_params:
                domain_params = self.domain_generation_params[domain]
                for param, value in domain_params.items():
                    setattr(self, param, value)
                domain_params_applied = True
                if debug_output:
                    print(f"Applied newly defined domain-specific parameters for {domain}")
            else:
                default_params = self.domain_generation_params["default"]
                for param, value in default_params.items():
                    setattr(self, param, value)
                domain_params_applied = True
                if debug_output:
                    print(f"Applied default parameters (domain {domain} not recognized)")

        named_entities = []
        if self.use_ner:
            named_entities = self.extract_named_entities(text)
            print(f"Extracted {len(named_entities)} named entities")

        chunks = self.chunk_text(text, chunk_size=self.max_input_length, overlap=self.chunk_overlap)

        all_keyphrases = []

        templates_to_try = [self.prompt_template_idx]

        quantity_template_idx = -1
        target_count = 15
        if not hasattr(self, 'QUANTITY_TEMPLATE_ADDED'):
            quantity_template = f"""Extract exactly {target_count} keyphrases from the following text.
Focus on technical terms, named entities, and domain-specific concepts.
Include both single words and multi-word phrases (2-4 words).
Prioritize phrases that capture specific topics rather than general concepts.

Text: {{text}}

Keyphrases:"""

            self.PROMPT_TEMPLATES.append(quantity_template)
            quantity_template_idx = len(self.PROMPT_TEMPLATES) - 1
            self.QUANTITY_TEMPLATE_ADDED = True
            if debug_output:
                print(f"Added quantity-focused template at index {quantity_template_idx}")
        else:
            for i, template in enumerate(self.PROMPT_TEMPLATES):
                if "as many relevant keyphrases as possible" in template and "15-20 keyphrases" in template:
                    quantity_template_idx = i
                    break

        if quantity_template_idx >= 0 and quantity_template_idx != self.prompt_template_idx:
            templates_to_try.append(quantity_template_idx)

        for template_idx in templates_to_try:
            if debug_output and len(templates_to_try) > 1:
                print(f"\nUsing template {template_idx}" + (" (quantity-focused)" if template_idx == quantity_template_idx else ""))

            for chunk in chunks:
                prompt = self.PROMPT_TEMPLATES[template_idx].format(text=chunk)

                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                using_diverse_beams = False  # group beam search is broken in transformers>=4.57 (custom_generate remote code)

                generation_kwargs = {
                    "max_new_tokens": self.max_new_tokens,
                    "num_beams": self.num_beams,
                    "repetition_penalty": self.repetition_penalty,
                    "length_penalty": self.length_penalty,
                    "early_stopping": True,
                    "no_repeat_ngram_size": 2,
                }

                if using_diverse_beams:
                    generation_kwargs.update({
                        "do_sample": False,
                        "num_beam_groups": self.num_beam_groups,
                        "diversity_penalty": self.diversity_penalty,
                    })
                elif self.use_sampling:
                    generation_kwargs.update({
                        "do_sample": True,
                        "top_k": self.top_k,
                        "top_p": self.top_p,
                        "temperature": self.temperature,
                    })
                else:
                    generation_kwargs.update({
                        "do_sample": False,
                    })

                num_return_sequences = min(8, self.num_beams // self.num_beam_groups if using_diverse_beams else self.num_beams)
                generation_kwargs["num_return_sequences"] = num_return_sequences

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **generation_kwargs
                    )

                for i in range(num_return_sequences):
                    generated_text = self.tokenizer.decode(outputs[i], skip_special_tokens=True)

                    if debug_output and i == 0:
                        print("\n" + "="*50)
                        print("RAW MODEL OUTPUT:")
                        print("-"*50)
                        print(generated_text)
                        print("="*50 + "\n")

                    chunk_keyphrases = self.extract_keyphrases_from_generated_text(generated_text)
                    all_keyphrases.extend(chunk_keyphrases)

        all_keyphrases.extend(named_entities)

        self.clean_memory()

        if self.post_process:
            all_keyphrases = self.post_process_keyphrases(all_keyphrases)

        seen = set()
        unique_keyphrases = []
        for kp in all_keyphrases:
            kp_lower = kp.lower()
            if kp_lower not in seen:
                seen.add(kp_lower)
                unique_keyphrases.append(kp)

        if domain_params_applied:
            for param, value in original_params.items():
                if hasattr(self, param):
                    setattr(self, param, value)
            if debug_output:
                print("Restored original generation parameters")

        if debug_output:
            print(f"Generated {len(unique_keyphrases)} unique keyphrases")

        return unique_keyphrases

    def extract_keyphrases_from_generated_text(self, generated_text: str) -> List[str]:

        all_keyphrases = []

        if ',' in generated_text:
            comma_keyphrases = [kp.strip() for kp in generated_text.split(',')]
            comma_keyphrases = [kp for kp in comma_keyphrases if kp and len(kp.split()) <= self.max_phrase_length]
            all_keyphrases.extend(comma_keyphrases)

        if '\n' in generated_text:
            newline_keyphrases = [kp.strip() for kp in generated_text.split('\n')]
            newline_keyphrases = [kp for kp in newline_keyphrases if kp and len(kp.split()) <= self.max_phrase_length]
            all_keyphrases.extend(newline_keyphrases)

        patterns = [
            r'"([^"]+)"',
            r'\'([^\']+)\'',
            r'([A-Z][a-z]+ (?:[A-Z][a-z]+ )?(?:[A-Z][a-z]+)?)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, generated_text)
            matches = [m for m in matches if 1 <= len(m.split()) <= self.max_phrase_length]
            all_keyphrases.extend(matches)

        try:
            words = word_tokenize(generated_text)
            pos_tags = pos_tag(words)

            i = 0
            while i < len(pos_tags):
                if i < len(pos_tags) and pos_tags[i][1].startswith('NN'):
                    start = i
                    while i < len(pos_tags) and pos_tags[i][1].startswith('NN'):
                        i += 1

                    phrase = ' '.join(words[start:i])
                    if 1 <= len(phrase.split()) <= self.max_phrase_length:
                        all_keyphrases.append(phrase)
                    continue

                i += 1
        except Exception as e:
            print(f"Warning: Error in POS tagging: {str(e)}")

        if not all_keyphrases and generated_text.strip():
            words = generated_text.split()
            for i in range(len(words)):
                for j in range(1, min(self.max_phrase_length + 1, len(words) - i + 1)):
                    phrase = ' '.join(words[i:i+j])
                    if j >= self.min_phrase_length and not all(word.lower() in self.stopwords for word in words[i:i+j]):
                        all_keyphrases.append(phrase)

        seen = set()
        unique_keyphrases = []
        for kp in all_keyphrases:
            kp_lower = kp.lower()
            if kp_lower not in seen:
                seen.add(kp_lower)
                unique_keyphrases.append(kp)

        final_keyphrases = []
        for kp in unique_keyphrases:
            kp_cleaned = re.sub(r'^(Keyphrases|Keywords|Concepts|Topics|Entities|Key phrases)[:\s]*', '', kp, flags=re.IGNORECASE).strip()

            kp_cleaned = re.sub(r'^[^\w\s(]+|[^\w\s)]+$', '', kp_cleaned).strip()

            kp_cleaned = re.sub(r'^\d+[\.\)]\s*|\*\s*|•\s*|→\s*|-\s*', '', kp_cleaned).strip()

            if len(kp_cleaned) > 1:
                final_keyphrases.append(kp_cleaned)

        return final_keyphrases

    def enhanced_post_processing(self, keyphrases: List[str]) -> List[str]:

        if not hasattr(self, 'pattern_bonuses'):
            self.pattern_bonuses = {}
        else:
            self.pattern_bonuses.clear()

        print(f"DEBUG: Initialized pattern_bonuses dictionary")
        improved_keyphrases = []

        for kp in keyphrases:
            self.pattern_bonuses[kp] = 0.0

        for keyphrase in keyphrases:
            if not keyphrase.strip():
                continue

            kp = self.clean_keyphrase(keyphrase)

            if not kp:
                continue

            words = kp.split()
            if len(words) < self.min_phrase_length or len(words) > self.max_phrase_length:
                continue

            if self.filter_stopwords and len(words) == 1 and words[0].lower() in self.stopwords:
                continue

            try:
                tokens = word_tokenize(kp)
                pos_tags = pos_tag(tokens)

                pattern_quality = 0
                pattern_bonus = 0

                has_noun = any(tag.startswith('NN') for _, tag in pos_tags)
                if has_noun:
                    pattern_quality += 1
                else:
                    continue

                if len(pos_tags) > 1:
                    pattern_quality += 1
                    pattern_bonus += 0.40

                if len(pos_tags) >= 2 and any(pos_tags[i][1].startswith('JJ') and pos_tags[i+1][1].startswith('NN') for i in range(len(pos_tags)-1)):
                    pattern_quality += 2
                    pattern_bonus += 0.60

                if len(pos_tags) >= 2 and any(pos_tags[i][1].startswith('NN') and pos_tags[i+1][1].startswith('NN') for i in range(len(pos_tags)-1)):
                    pattern_quality += 2
                    pattern_bonus += 0.60

                if len(pos_tags) >= 3 and any(pos_tags[i][1].startswith('NN') and pos_tags[i+1][1] == 'IN' and pos_tags[i+2][1].startswith('NN') for i in range(len(pos_tags)-2)):
                    pattern_quality += 2
                    pattern_bonus += 0.60

                if len(pos_tags) >= 3 and pos_tags[-1][1].startswith('NN'):
                    pattern_quality += 1
                    pattern_bonus += 0.60

                proper_noun_count = sum(1 for _, tag in pos_tags if tag == 'NNP' or tag == 'NNPS')
                if proper_noun_count > 0:
                    pattern_quality += 1

                    if proper_noun_count == 1:
                        pattern_bonus += 0.04
                    elif proper_noun_count == 2:
                        pattern_bonus += 0.06
                    else:
                        pattern_bonus += 0.08

                    if proper_noun_count == len(pos_tags):
                        pattern_bonus += 0.03

                if sum(1 for _, tag in pos_tags if tag.startswith('VB')) > sum(1 for _, tag in pos_tags if tag.startswith('NN')):
                    pattern_quality -= 1

                if pos_tags and pos_tags[0][1] in ['DT', 'IN', 'CC']:
                    pattern_quality -= 1

                if pattern_quality >= 1:
                    improved_keyphrases.append(kp)
                    self.pattern_bonuses[kp] = pattern_bonus
                    print(f"DEBUG: Added pattern bonus {pattern_bonus:.4f} to '{kp}'")
            except Exception as e:
                print(f"Warning: Error in POS tagging: {str(e)}. Using fallback method.")
                improved_keyphrases.append(kp)
                self.pattern_bonuses[kp] = 0.0

        seen = set()
        unique_keyphrases = []
        for kp in improved_keyphrases:
            if kp.lower() not in seen:
                seen.add(kp.lower())
                unique_keyphrases.append(kp)

        if self.prioritize_multi_word:
            indexed_keyphrases = [(i, kp) for i, kp in enumerate(unique_keyphrases)]
            sorted_keyphrases = sorted(
                indexed_keyphrases,
                key=lambda x: (-len(x[1].split()), x[0])
            )
            unique_keyphrases = [kp for _, kp in sorted_keyphrases]

        return unique_keyphrases

    def post_process_keyphrases(self, keyphrases: List[str]) -> List[str]:

        processed_keyphrases = []

        for keyphrase in keyphrases:
            if not keyphrase.strip():
                continue

            kp = self.clean_keyphrase(keyphrase)

            if not kp:
                continue

            words = kp.split()
            if len(words) < self.min_phrase_length or len(words) > self.max_phrase_length:
                continue

            if self.filter_stopwords and len(words) == 1 and words[0].lower() in self.stopwords:
                continue

            processed_keyphrases.append(kp)

        seen = set()
        unique_keyphrases = []
        for kp in processed_keyphrases:
            if kp.lower() not in seen:
                seen.add(kp.lower())
                unique_keyphrases.append(kp)

        if self.prioritize_multi_word:
            indexed_keyphrases = [(i, kp) for i, kp in enumerate(unique_keyphrases)]
            sorted_keyphrases = sorted(
                indexed_keyphrases,
                key=lambda x: (-len(x[1].split()), x[0])
            )
            unique_keyphrases = [kp for _, kp in sorted_keyphrases]

        return unique_keyphrases

    def clean_keyphrase(self, keyphrase: str) -> str:

        kp = keyphrase.strip()

        kp = re.sub(r'^["\']|["\']$', '', kp)

        kp = re.sub(r'^\d+\.\s*|\*\s*', '', kp)

        kp = kp.strip(string.punctuation + ' ')

        if self.use_lemmatization:
            words = word_tokenize(kp)
            lemmatized_words = []
            for word in words:
                if not word[0].isupper():
                    lemmatized_words.append(self.lemmatizer.lemmatize(word.lower()))
                else:
                    lemmatized_words.append(word)
            kp = ' '.join(lemmatized_words)

        return kp

    def extract_named_entities(self, text: str) -> List[str]:

        if not self.use_ner:
            return []

        doc = self.nlp(text)

        entities = []
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART", "EVENT", "LAW", "FAC"]:
                words = ent.text.split()
                if self.min_phrase_length <= len(words) <= self.max_phrase_length:
                    clean_entity = self.clean_keyphrase(ent.text)
                    if clean_entity:
                        entities.append(clean_entity)

        seen = set()
        unique_entities = []
        for entity in entities:
            if entity.lower() not in seen:
                seen.add(entity.lower())
                unique_entities.append(entity)

        return unique_entities

    def get_domain_specific_parameters(self, domain: str, text_length: int) -> Tuple[float, float, int, int]:

        base_threshold = 0.08
        quality_threshold = 0.40
        percentile = 3

        if text_length < 200:
            target_count = 8
        elif text_length < 500:
            target_count = 10
        elif text_length < 1000:
            target_count = 12
        else:
            target_count = 15

        domain = domain.lower() if domain else "general"

        if domain in ["artificial intelligence", "ai", "machine learning", "deep learning", "neural networks"]:
            base_threshold = 0.04
            quality_threshold = 0.32
            percentile = 6
            target_count = 10
        elif domain in ["cybersecurity", "security", "cyber", "hacking", "infosec"]:
            base_threshold = 0.04
            quality_threshold = 0.30
            percentile = 7
            target_count = 9
        elif domain in ["automotive", "cars", "vehicles", "transportation", "auto industry"]:
            base_threshold = 0.045
            quality_threshold = 0.33
            percentile = 6
            target_count = 10
        elif domain in ["food", "cooking", "cuisine", "recipe", "culinary"]:
            base_threshold = 0.05
            quality_threshold = 0.38
            percentile = 5
            target_count = 11
        elif domain in ["environment", "climate", "sustainability", "ecology", "green"]:
            base_threshold = 0.045
            quality_threshold = 0.34
            percentile = 6
            target_count = 10
        elif domain in ["real estate", "property", "housing", "real-estate", "realty"]:
            base_threshold = 0.05
            quality_threshold = 0.36
            percentile = 5
            target_count = 9
        elif domain in ["entertainment", "movies", "film", "music", "television", "tv", "media"]:
            base_threshold = 0.05
            quality_threshold = 0.37
            percentile = 5
            target_count = 11

        elif domain in ["technology", "tech", "digital", "software", "hardware"]:
            base_threshold = 0.09
            quality_threshold = 0.42
            percentile = 5
            target_count = 10

        elif domain in ["business", "finance", "economics", "economy", "market"]:
            base_threshold = 0.08
            quality_threshold = 0.40
            percentile = 4
            target_count = 9

        elif domain in ["health", "medical", "medicine", "healthcare", "wellness"]:
            base_threshold = 0.08
            quality_threshold = 0.40
            percentile = 4
            target_count = 9

        elif domain in ["politics", "government", "policy", "political"]:
            base_threshold = 0.09
            quality_threshold = 0.42
            percentile = 5
            target_count = 9

        elif domain in ["sports", "sport", "athletics", "games"]:
            base_threshold = 0.07
            quality_threshold = 0.38
            percentile = 3
            target_count = 8

        elif domain in ["science", "scientific", "research", "biology", "chemistry", "physics"]:
            base_threshold = 0.09
            quality_threshold = 0.42
            percentile = 4
            target_count = 9

        elif domain in ["space", "astronomy", "cosmos", "astrophysics", "aerospace"]:
            base_threshold = 0.09
            quality_threshold = 0.42
            percentile = 4
            target_count = 9

        elif domain in ["agriculture", "farming", "crops", "livestock"]:
            base_threshold = 0.07
            quality_threshold = 0.38
            percentile = 3
            target_count = 8

        elif domain in ["mental health", "psychology", "psychiatry", "behavioral health"]:
            base_threshold = 0.08
            quality_threshold = 0.40
            percentile = 4
            target_count = 9

        if hasattr(self, 'content_density'):
            content_density = self.content_density
            if content_density > 0.7:
                quality_threshold *= 0.95
                target_count = int(target_count * 1.1)
            elif content_density < 0.5:
                quality_threshold *= 1.05
                target_count = max(8, int(target_count * 0.9))

        if text_length < 200:
            quality_threshold = min(0.45, quality_threshold * 1.1)
        elif text_length > 500:
            quality_threshold = max(0.30, quality_threshold * 0.95)

        if 200 <= text_length < 300:
            target_count = max(7, int(target_count * 0.9))
        elif 300 <= text_length < 400:
            pass
        elif 400 <= text_length < 500:
            target_count = min(14, int(target_count * 1.1))

        if hasattr(self, 'content_density') and self.content_density is not None:
            content_density = self.content_density
            if content_density > 0.7:
                base_threshold = max(0.05, base_threshold - 0.02)
                quality_threshold = max(0.35, quality_threshold - 0.03)
                percentile = max(2, percentile - 1)
            elif content_density < 0.4:
                base_threshold = min(0.12, base_threshold + 0.01)
                quality_threshold = min(0.48, quality_threshold + 0.02)
                percentile = min(10, percentile + 1)

        print(f"\nUsing parameters for domain '{domain}':\n" +
              f"  - base_threshold: {base_threshold:.3f}\n" +
              f"  - quality_threshold: {quality_threshold:.3f}\n" +
              f"  - percentile: {percentile}\n" +
              f"  - target_count: {target_count}")

        return base_threshold, quality_threshold, percentile, target_count

    def extract_keyphrases_with_scores(self, text: str, min_keyphrases: int = 5, max_keyphrases: int = 15, min_score: float = 0.1, optimize_params: bool = False, original_domain: str = None) -> List[Tuple[str, float]]:

        if optimize_params:
            original_use_sampling = getattr(self, 'use_sampling', True)
            original_template_idx = self.prompt_template_idx

            print("Optimizing generation parameters...")
            self._optimize_generation_params(text)

        text_length = len(text.split())

        if original_domain:
            domain = original_domain
            print(f"Using original domain: {domain}")
        else:
            domain = self.detect_domain(text)
            print(f"Detected domain: {domain}")

        keyphrases = self.generate_keyphrases(text, original_domain=original_domain)
        print(f"Generated {len(keyphrases)} candidate keyphrases")

        if len(keyphrases) < min_keyphrases:
            print(f"Too few keyphrases ({len(keyphrases)}), trying different parameters...")

            original_params = {
                'use_sampling': getattr(self, 'use_sampling', True),
                'num_beams': self.num_beams,
                'prompt_template_idx': self.prompt_template_idx
            }

            if not hasattr(self, 'num_beam_groups') or self.num_beam_groups <= 1:
                self.use_sampling = False
                self.num_beams = 12
                self.num_beam_groups = 4
                self.diversity_penalty = 1.0
                print("Trying diverse beam search...")
            else:
                self.use_sampling = False
                self.num_beams = 12
                if hasattr(self, 'num_beam_groups'):
                    delattr(self, 'num_beam_groups')
                if hasattr(self, 'diversity_penalty'):
                    delattr(self, 'diversity_penalty')
                print("Trying pure beam search...")

            self.prompt_template_idx = (self.prompt_template_idx + 1) % len(self.PROMPT_TEMPLATES)

            backup_keyphrases = self.generate_keyphrases(text)
            print(f"Generated {len(backup_keyphrases)} keyphrases with alternative parameters")

            self.use_sampling = original_params['use_sampling']
            self.num_beams = original_params['num_beams']
            self.prompt_template_idx = original_params['prompt_template_idx']

            if 'num_beam_groups' not in original_params and hasattr(self, 'num_beam_groups'):
                delattr(self, 'num_beam_groups')
            if 'diversity_penalty' not in original_params and hasattr(self, 'diversity_penalty'):
                delattr(self, 'diversity_penalty')

            keyphrases.extend(backup_keyphrases)

            seen = set()
            unique_keyphrases = []
            for kp in keyphrases:
                if kp.lower() not in seen:
                    seen.add(kp.lower())
                    unique_keyphrases.append(kp)

            keyphrases = unique_keyphrases
            print(f"Combined unique keyphrases: {len(keyphrases)}")

        scored_keyphrases = self.score_keyphrases_by_relevance(keyphrases, text)

        print(f"DEBUG: Before enhanced post-processing: {len(keyphrases)} keyphrases")
        enhanced_keyphrases = self.enhanced_post_processing(keyphrases)
        print(f"DEBUG: After enhanced post-processing: {len(enhanced_keyphrases)} keyphrases")
        enhanced_scored = [(kp, score) for kp, score in scored_keyphrases if kp in enhanced_keyphrases]
        print(f"DEBUG: After matching with scored keyphrases: {len(enhanced_scored)} keyphrases")

        if len(enhanced_scored) < min_keyphrases and len(scored_keyphrases) >= min_keyphrases:
            print(f"Enhanced post-processing reduced keyphrases too much ({len(enhanced_scored)} < {min_keyphrases}), using original keyphrases")
            enhanced_scored = scored_keyphrases
        else:
            print(f"Enhanced post-processing: {len(scored_keyphrases)} -> {len(enhanced_scored)} keyphrases")

        context_weighted_keyphrases = []
        kp_weights = {kp: weight for kp, weight in self.apply_context_aware_weighting([kp for kp, _ in enhanced_scored], text)}

        for kp, score in enhanced_scored:
            context_weight = kp_weights.get(kp, 1.0)
            adjusted_score = (score * 0.7) + (context_weight * 0.3 * score)

            word_count = len(kp.split())
            if 2 <= word_count <= 4:
                length_bonus = 0.03
                adjusted_score += length_bonus

            context_weighted_keyphrases.append((kp, adjusted_score))

        print(f"Applied context-aware weighting to {len(context_weighted_keyphrases)} keyphrases")

        print(f"DEBUG: Detected domain: '{domain}'")
        domain_adjusted_keyphrases = self.apply_domain_quality_metrics(context_weighted_keyphrases, domain)
        print(f"DEBUG: Applied domain-specific quality metrics to {len(domain_adjusted_keyphrases)} keyphrases")

        print("DEBUG: Top 5 keyphrases after domain adjustment:")
        for i, (kp, score) in enumerate(domain_adjusted_keyphrases[:5]):
            print(f"DEBUG: {i+1}. '{kp}': {score:.4f}")

        quality_enhanced_keyphrases = self.enhance_semantic_quality(domain_adjusted_keyphrases, text)
        print(f"Enhanced semantic quality for {len(quality_enhanced_keyphrases)} keyphrases")

        boosted_keyphrases = self.boost_domain_specific_concepts(quality_enhanced_keyphrases, domain)

        coherent_keyphrases = self.enhance_semantic_coherence(boosted_keyphrases, text)

        diverse_keyphrases = self.enhance_keyphrase_diversity(coherent_keyphrases)
        print(f"Enhanced diversity for {len(diverse_keyphrases)} keyphrases")

        filtered_keyphrases = self.remove_redundant_keyphrases(diverse_keyphrases, domain=domain)

        filtered_keyphrases = [(kp, score) for kp, score in filtered_keyphrases if score >= min_score]

        filtered_keyphrases = self.filter_generic_terms(filtered_keyphrases, domain, text)

        print("DEBUG: Top 5 keyphrases before final filtering:")
        for i, (kp, score) in enumerate(filtered_keyphrases[:5]):
            print(f"DEBUG: {i+1}. '{kp}': {score:.4f}")

        final_keyphrases = self.filter_and_select_by_quality(text, domain, filtered_keyphrases, original_domain=original_domain)

        print("DEBUG: Top 5 keyphrases after final filtering:")
        for i, (kp, score) in enumerate(final_keyphrases[:5]):
            print(f"DEBUG: {i+1}. '{kp}': {score:.4f}")

        if len(final_keyphrases) < min(3, min_keyphrases) and optimize_params:
            backup_template_idx = (self.prompt_template_idx + 1) % len(self.PROMPT_TEMPLATES)
            self.prompt_template_idx = backup_template_idx

            self.use_sampling = not original_use_sampling

            backup_keyphrases = self.generate_keyphrases(text, original_domain=original_domain)

            self.prompt_template_idx = original_template_idx
            self.use_sampling = original_use_sampling

            if backup_keyphrases:
                backup_scored = self.score_keyphrases_by_relevance(backup_keyphrases, text)

                existing_kps = [kp.lower() for kp, _ in final_keyphrases]
                for kp, score in backup_scored:
                    if kp.lower() not in existing_kps and score >= min_score:
                        final_keyphrases.append((kp, score))

                final_keyphrases.sort(key=lambda x: x[1], reverse=True)

        max_reasonable_keyphrases = min(max_keyphrases, max(min_keyphrases, int(text_length / 50)))
        if len(final_keyphrases) > max_reasonable_keyphrases:
            final_keyphrases = final_keyphrases[:max_reasonable_keyphrases]

        return final_keyphrases

    def extract_keyphrases_optimized(self, text: str, original_domain: str = None) -> List[Tuple[str, float]]:

        return self.extract_keyphrases_with_scores(text, optimize_params=True, original_domain=original_domain)

    def extract_keyphrases(self, text: str, original_domain: str = None) -> List[str]:

        scored_keyphrases = self.extract_keyphrases_with_scores(text, original_domain=original_domain)
        keyphrases = [kp for kp, _ in scored_keyphrases]
        final_count = len(keyphrases)
        print(f"Final keyphrase count: {final_count}")
        print(f"Returning {final_count} keyphrases from extract_keyphrases")

        return keyphrases

    def debug_keyphrase_extraction(self, text: str) -> None:

        print("\n" + "="*80)
        print("DEBUGGING KEYPHRASE EXTRACTION")
        print("="*80)

        print("\nSTEP 1: Examining raw T5 output")
        self.examine_raw_output(text)

        print("\nSTEP 2: Testing different generation parameters")
        param_results = self.test_generation_params(text, debug_output=False)

        best_param_key = max(param_results.items(), key=lambda x: len(x[1]), default=(None, []))[0]
        if best_param_key:
            print(f"\nBest parameter combination: {best_param_key}")
            print(f"Generated {len(param_results[best_param_key])} keyphrases")

        print("\nSTEP 3: Extracting keyphrases with optimized parameters")
        keyphrases = self.extract_keyphrases_optimized(text)

        print(f"\nFinal keyphrases ({len(keyphrases)}):")
        for kp, score in keyphrases:
            print(f"- {kp}: {score:.4f}")

        print("\n" + "="*80)

    def initialize_domain_components(self):

        self.add_domain_specific_prompts()

        self.DOMAIN_LABELS = [
            "technology", "business", "health", "politics", "sports",
            "entertainment", "science", "environment", "world", "education",
            "food", "travel", "automotive", "real estate", "cybersecurity",
            "artificial intelligence", "space", "agriculture", "mental health"
        ]

        self.domain_thresholds = {
            'entertainment': 0.3,
            'sports': 0.3,
            'politics': 0.3,
            'education': 0.4,
            'technology': 0.35,
            'real estate': 0.35,
            'food': 0.35,
            'automotive': 0.3,
            'science': 0.3,
            'space': 0.3,
            'health': 0.25,
            'travel': 0.22,
        }

        self.default_threshold = 0.4

        if hasattr(self, 'use_mdeberta_domain_detection') and self.use_mdeberta_domain_detection:
            try:
                print("Initializing mDeBERTa zero-shot classifier for domain detection...")
                device = 0 if (self.use_gpu and torch.cuda.is_available()) else -1
                self.zero_shot_classifier = pipeline(
                    "zero-shot-classification",
                    model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                    device=device
                )
                print("mDeBERTa zero-shot classifier initialized successfully")
            except Exception as e:
                print(f"Error initializing mDeBERTa zero-shot classifier: {str(e)}")
                print("Falling back to alternative domain detection methods")

        print("Domain-specific components initialized")

    def try_all_prompts(self, text: str) -> Dict[int, List[str]]:

        results = {}
        original_template_idx = self.prompt_template_idx

        for i in range(len(self.PROMPT_TEMPLATES)):
            self.prompt_template_idx = i
            results[i] = self.generate_keyphrases(text)

        self.prompt_template_idx = original_template_idx

        return results

    def filter_generic_terms(self, keyphrases: List[Tuple[str, float]], domain: str, text: str) -> List[Tuple[str, float]]:

        if not keyphrases:
            return []

        original_count = len(keyphrases)

        domain_generics = set(self.generic_terms.get(domain, self.generic_terms["general"]))
        general_generics = set(self.generic_terms["general"])
        all_generics = domain_generics.union(general_generics)

        domain_exceptions = {
            "technology": ["digital", "software", "hardware", "platform", "system", "application", "device", "technology", "tech", "solution"],
            "artificial intelligence": ["model", "algorithm", "system", "data", "learning", "intelligence", "ai", "neural", "training"],
            "cybersecurity": ["security", "protection", "threat", "attack", "defense", "vulnerability", "risk", "cyber", "breach"],
            "automotive": ["vehicle", "car", "driver", "driving", "auto", "automotive", "engine", "motor", "fuel"],
            "food": ["food", "cooking", "recipe", "ingredient", "meal", "dish", "flavor", "taste", "cuisine"],
            "environment": ["environment", "climate", "sustainable", "green", "energy", "conservation", "pollution", "emission", "waste"],
            "real estate": ["property", "home", "house", "market", "real estate", "housing", "mortgage", "buyer", "seller"],
            "entertainment": ["movie", "film", "show", "music", "entertainment", "actor", "actress", "director", "performance"],
            "health": ["health", "medical", "patient", "treatment", "doctor", "hospital", "care", "disease", "condition"],
            "science": ["research", "study", "scientist", "experiment", "data", "analysis", "discovery", "finding", "evidence"],
            "sports": ["team", "player", "game", "match", "season", "sport", "championship", "league", "tournament"],
            "politics": ["government", "policy", "political", "election", "official", "administration", "law", "legislation", "regulation"]
        }

        exceptions = set(domain_exceptions.get(domain, []))

        if domain in ["artificial intelligence", "cybersecurity"]:
            exceptions.update(domain_exceptions["technology"])
        elif domain in ["automotive"]:
            exceptions.update(["technology", "engineering", "manufacturing"])

        text_lower = text.lower()
        text_word_count = len(text.split())

        doc_embedding = None
        if hasattr(self, 'sentence_model') and self.sentence_model is not None:
            try:
                doc_embedding = self.sentence_model.encode([text], show_progress_bar=False)[0]
                import numpy as np
            except Exception as e:
                print(f"Warning: Could not create document embedding: {e}")

        words = re.findall(r'\b\w+\b', text_lower)
        word_freq = {}
        for word in words:
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1

        max_freq = max(word_freq.values()) if word_freq else 1

        filtered_keyphrases = []
        filtered_out = []

        keyphrase_importance = {}
        for keyphrase, score in keyphrases:
            kp_lower = keyphrase.lower()
            kp_words = kp_lower.split()

            importance = score

            length_factor = min(1.8, 1.0 + (len(kp_words) - 1) * 0.3)
            importance *= length_factor

            tf_idf_factor = 1.0
            if kp_words:
                term_freqs = [word_freq.get(word, 0) / max_freq for word in kp_words]
                avg_tf = sum(term_freqs) / len(term_freqs) if term_freqs else 0

                if avg_tf > 0:
                    if avg_tf > 0.9:
                        tf_idf_factor = 0.85
                    elif 0.15 <= avg_tf <= 0.8:
                        tf_idf_factor = 1.25

            importance *= tf_idf_factor

            if doc_embedding is not None:
                try:
                    kp_embedding = self.sentence_model.encode([kp_lower], show_progress_bar=False)[0]
                    similarity = np.dot(kp_embedding, doc_embedding) / (
                        np.linalg.norm(kp_embedding) * np.linalg.norm(doc_embedding)
                    )
                    semantic_factor = 0.6 + similarity
                    importance *= semantic_factor
                except Exception:
                    pass

            if any(word in exceptions for word in kp_words):
                domain_boost = 1.3
                importance *= domain_boost

            keyphrase_importance[keyphrase] = importance

        for keyphrase, score in keyphrases:
            kp_lower = keyphrase.lower()
            kp_words = kp_lower.split()

            importance = keyphrase_importance.get(keyphrase, score)

            importance_factor = importance / max(score, 0.1)

            if any(word in exceptions for word in kp_words) and score > 0.3:
                filtered_keyphrases.append((keyphrase, score))
                continue

            if kp_lower in all_generics and importance_factor < 1.1 and kp_lower not in exceptions:
                filtered_out.append((keyphrase, score, "exact generic match"))
                continue

            if len(kp_words) == 1:
                single_word_threshold = 0.32 / importance_factor

                if kp_lower in all_generics and score < single_word_threshold and kp_lower not in exceptions:
                    filtered_out.append((keyphrase, score, "generic single word"))
                    continue

            if len(kp_words) > 1:
                generic_word_count = sum(1 for word in kp_words if word in all_generics and word not in exceptions)
                generic_percentage = generic_word_count / len(kp_words)

                generic_threshold = 0.8 if importance_factor < 1.0 else 0.9
                score_threshold = 0.4 / importance_factor

                if generic_percentage > generic_threshold and score < score_threshold:
                    filtered_out.append((keyphrase, score, f"{int(generic_percentage*100)}% generic words"))
                    continue

            kp_frequency = text_lower.count(kp_lower)
            if kp_frequency > 0:
                frequency_ratio = kp_frequency / (text_word_count / 100)

                frequency_threshold = 2.5 if importance_factor < 1.0 else 3.5
                score_threshold = 0.4 / importance_factor

                if frequency_ratio > frequency_threshold and score < score_threshold:
                    filtered_out.append((keyphrase, score, f"too frequent ({frequency_ratio:.1f}/100 words)"))
                    continue

            vague_patterns = [
                r'^(many|various|different|several|some) [a-z]+s?$',
                r'^(important|significant|major|key) [a-z]+$',
                r'^(new|recent|latest|current) [a-z]+$',
                r'^(high|low|large|small) [a-z]+$',
                r'^[a-z]+ (issues?|concerns?|matters?|aspects?)$',
                r'^(certain|specific|particular) [a-z]+s?$',
                r'^(overall|general|basic) [a-z]+s?$'
            ]

            vague_score_threshold = 0.55 / importance_factor

            if any(re.match(pattern, kp_lower) for pattern in vague_patterns) and score < vague_score_threshold:
                filtered_out.append((keyphrase, score, "vague pattern"))
                continue

            filtered_keyphrases.append((keyphrase, score))

        if filtered_keyphrases and len(filtered_out) > 2.3 * len(filtered_keyphrases):
            filtered_out.sort(key=lambda x: x[1], reverse=True)

            recovery_count = min(len(filtered_out) // 2, 8)

            for i in range(recovery_count):
                if i < len(filtered_out) and filtered_out[i][1] > 0.2:
                    keyphrase, score, reason = filtered_out[i]
                    filtered_keyphrases.append((keyphrase, score))
                    print(f"Recovered keyphrase due to Pareto principle: {keyphrase} ({score:.2f}) - was filtered for: {reason}")

        focus_domains = ["artificial intelligence", "cybersecurity", "automotive",
                        "food", "environment", "real estate", "entertainment"]

        if domain in focus_domains and len(filtered_keyphrases) < 0.7 * original_count:
            additional_recovery = int(0.7 * original_count) - len(filtered_keyphrases)
            if additional_recovery > 0 and filtered_out:
                print(f"Additional recovery for focus domain '{domain}': recovering {additional_recovery} more keyphrases")

                already_recovered = set(kp for kp, _ in filtered_keyphrases)

                candidates = [(kp, score) for kp, score, _ in filtered_out
                             if kp not in already_recovered and score > 0.15]

                candidates.sort(key=lambda x: x[1], reverse=True)

                for i in range(min(additional_recovery, len(candidates))):
                    keyphrase, score = candidates[i]
                    filtered_keyphrases.append((keyphrase, score))
                    print(f"Additional domain-specific recovery: {keyphrase} ({score:.2f})")

        filtered_keyphrases.sort(key=lambda x: x[1], reverse=True)

        if filtered_out:
            print(f"\nGeneric term filtering: {original_count} -> {len(filtered_keyphrases)} keyphrases ({len(filtered_out)} removed)")
            print("\nFiltered out generic terms (examples):")
            for term, score, reason in filtered_out[:5]:
                print(f"- {term} ({score:.2f}): {reason}")
            if len(filtered_out) > 5:
                print(f"...and {len(filtered_out) - 5} more")

        return filtered_keyphrases

    def initialize_generic_terms(self):

        self.generic_terms = {
            "general": [
                "news", "report", "article", "story", "update", "information",
                "according to", "said", "says", "stated", "reported", "announced",

                "today", "yesterday", "recent", "current", "latest",

                "important", "significant", "major", "key", "critical",

                "various", "several", "many", "some", "different",

                "example", "case", "issue", "situation", "development",

                "source", "expert", "official", "spokesperson", "authority"
            ],

            "technology": [
                "technology", "tech", "digital", "solution", "platform", "system",
                "software", "hardware", "device", "application", "app",

                "new", "innovative", "advanced", "smart", "intelligent", "modern",

                "user", "customer", "experience", "market", "industry",

                "performance", "efficient", "fast", "powerful", "high-performance"
            ],

            "artificial intelligence": [
                "ai", "artificial intelligence", "model", "algorithm", "system",
                "machine learning", "deep learning", "neural", "network",

                "intelligent", "automated", "smart", "predictive", "cognitive",

                "data", "dataset", "training", "learning", "inference",

                "performance", "accuracy", "precision", "recall", "efficiency"
            ],

            "cybersecurity": [
                "security", "cybersecurity", "cyber", "protection", "defense",
                "threat", "attack", "vulnerability", "risk", "breach",

                "secure", "protected", "encrypted", "safe", "vulnerable",

                "detect", "prevent", "mitigate", "respond", "protect",

                "hacker", "attacker", "defender", "security team", "analyst"
            ],

            "automotive": [
                "car", "vehicle", "automotive", "auto", "automobile",
                "driver", "driving", "road", "highway", "traffic",

                "engine", "motor", "wheel", "tire", "battery",

                "performance", "speed", "efficiency", "power", "fuel economy",

                "manufacturer", "industry", "market", "production", "sales"
            ],

            "food": [
                "food", "meal", "dish", "recipe", "ingredient",
                "cooking", "cuisine", "chef", "restaurant", "dining",

                "delicious", "tasty", "flavorful", "fresh", "healthy",

                "cook", "bake", "grill", "roast", "fry",

                "dessert", "appetizer", "entree", "beverage", "snack"
            ],

            "environment": [
                "environment", "environmental", "climate", "ecosystem", "sustainable",
                "green", "eco-friendly", "conservation", "preservation", "protection",

                "climate change", "global warming", "carbon", "emission", "greenhouse",

                "energy", "renewable", "resource", "natural resource", "biodiversity",

                "water", "air", "forest", "ocean", "wildlife",

                "pollution", "waste", "impact", "footprint", "sustainability"
            ],

            "real estate": [
                "real estate", "property", "home", "house", "apartment",
                "building", "residential", "commercial", "housing", "market",

                "buyer", "seller", "agent", "broker", "listing",

                "mortgage", "loan", "interest rate", "down payment", "closing cost",

                "spacious", "renovated", "modern", "updated", "luxury"
            ],

            "entertainment": [
                "movie", "film", "show", "series", "program",
                "television", "tv", "streaming", "broadcast", "media",

                "actor", "actress", "director", "producer", "star",
                "cast", "crew", "filmmaker", "celebrity", "performer",

                "entertainment", "Hollywood", "studio", "network", "production",
                "industry", "box office", "audience", "viewer", "fan",

                "performance", "role", "character", "scene", "appearance",

                "hit", "popular", "success", "award", "critically acclaimed",
                "blockbuster", "bestseller", "rating", "review", "critic"
            ],

            "business": [
                "business", "company", "corporation", "firm", "enterprise",

                "market", "industry", "sector", "economy", "economic",

                "financial", "profit", "revenue", "growth", "investment",

                "strategy", "plan", "approach", "decision", "management",

                "success", "result", "performance", "achievement", "development"
            ],

            "health": [
                "health", "healthcare", "medical", "medicine", "treatment",

                "patient", "doctor", "hospital", "clinic", "physician",

                "study", "research", "trial", "finding", "result",

                "condition", "symptom", "effect", "benefit", "risk",

                "care", "therapy", "procedure", "approach", "method"
            ],

            "politics": [
                "government", "administration", "official", "authority", "agency",

                "policy", "legislation", "regulation", "law", "rule",

                "decision", "action", "measure", "initiative", "program",

                "political", "public", "national", "federal", "state",

                "international", "global", "foreign", "diplomatic", "relations"
            ],

            "sports": [
                "sports", "game", "match", "competition", "tournament",

                "team", "player", "athlete", "coach", "manager",

                "performance", "play", "win", "loss", "victory",

                "score", "point", "goal", "record", "statistic",

                "season", "championship", "league", "series", "round"
            ],

            "science": [
                "research", "study", "experiment", "analysis", "investigation",

                "scientist", "researcher", "professor", "expert", "specialist",

                "discovery", "finding", "result", "evidence", "data",

                "scientific", "academic", "theoretical", "experimental", "empirical",

                "university", "laboratory", "institute", "department", "center"
            ],

            "world": [
                "country", "nation", "government", "state", "region",

                "United Nations", "international", "global", "worldwide", "organization",

                "relations", "agreement", "treaty", "cooperation", "alliance",

                "conflict", "crisis", "tension", "dispute", "war",

                "border", "territory", "area", "zone", "continent"
            ]
        }

        cross_domain_generics = [
            "make", "made", "making", "do", "does", "doing", "done",
            "get", "gets", "getting", "got", "take", "takes", "taking",
            "give", "gives", "giving", "gave", "put", "puts", "putting",
            "use", "uses", "using", "used", "provide", "provides", "providing",
            "create", "creates", "creating", "created", "develop", "develops", "developing",

            "good", "great", "best", "better", "worse", "worst",
            "big", "small", "large", "little", "high", "low",
            "new", "old", "recent", "latest", "current", "modern",
            "common", "typical", "standard", "normal", "regular",
            "simple", "complex", "easy", "difficult", "hard",

            "time", "day", "week", "month", "year", "hour", "minute",
            "period", "duration", "interval", "moment", "instant",

            "many", "much", "more", "most", "some", "few", "several",
            "numerous", "multiple", "various", "diverse", "different",

            "in fact", "as well", "in addition", "moreover", "furthermore",
            "for example", "such as", "like", "including", "especially"
        ]

        domain_exceptions = {
            "artificial intelligence": ["model", "algorithm", "data", "learning", "neural", "network"],
            "cybersecurity": ["security", "threat", "attack", "vulnerability", "breach", "protection"],
            "automotive": ["vehicle", "car", "engine", "driver", "fuel", "electric"],
            "food": ["recipe", "ingredient", "dish", "meal", "cooking", "flavor"],
            "environment": ["climate", "sustainable", "energy", "carbon", "emission", "pollution"],
            "real estate": ["property", "home", "house", "market", "mortgage", "buyer"],
            "entertainment": ["movie", "film", "actor", "director", "show", "series"]
        }

        self.domain_exceptions = domain_exceptions

        for domain in self.generic_terms:
            self.generic_terms[domain].extend(cross_domain_generics)
            self.generic_terms[domain] = list(dict.fromkeys(self.generic_terms[domain]))

        print(f"Initialized generic terms for {len(self.generic_terms)} domains using Pareto principle")

    def get_best_prompt(self, text: str) -> int:

        results = self.try_all_prompts(text)

        scores = {}
        for idx, keyphrases in results.items():
            if not keyphrases:
                scores[idx] = 0
                continue

            num_keyphrases = len(keyphrases)

            multi_word_count = sum(1 for kp in keyphrases if len(kp.split()) > 1)

            ngram_penalty = 0
            for i in range(len(keyphrases) - 1):
                kp1 = keyphrases[i].split()
                kp2 = keyphrases[i+1].split()

                if len(kp1) > 0 and len(kp2) > 0:
                    if kp1[-1] == kp2[0]:
                        ngram_penalty += 0.2

            avg_length = sum(len(kp.split()) for kp in keyphrases) / num_keyphrases

            length_score = 1.0 - abs(avg_length - 2.5) / 2.5

            multi_word_percentage = multi_word_count / num_keyphrases if num_keyphrases > 0 else 0

            num_score = min(num_keyphrases / 10, 1.5)

            if num_keyphrases < 3:
                num_score *= 0.5
            elif num_keyphrases > 20:
                num_score *= 0.7

            final_score = (
                (num_score * 0.3) +
                (multi_word_percentage * 0.4) +
                (length_score * 0.3) -
                ngram_penalty
            )

            scores[idx] = final_score

        if not scores:
            return 0

        return max(scores.items(), key=lambda x: x[1])[0]

    def filter_and_select_by_quality(self, text: str, domain: str, scored_keyphrases: List[Tuple[str, float]], original_domain: str = None) -> List[Tuple[str, float]]:

        if not scored_keyphrases:
            return []

        words = text.split()
        word_count = len(words)
        unique_words = len(set(w.lower() for w in words))

        content_density = unique_words / max(word_count, 1)

        self.content_density = content_density

        domain_to_use = original_domain if original_domain else domain

        base_threshold, quality_threshold, percentile_from_domain, target_count = self.get_domain_specific_parameters(domain_to_use, word_count)

        print(f"Using domain '{domain_to_use}' with base threshold: {base_threshold:.3f}, quality threshold: {quality_threshold:.3f}")

        self.quality_threshold = quality_threshold

        scores = [score for _, score in scored_keyphrases]

        domain_percentiles = {
            "artificial intelligence": 5,
            "technology": 5,
            "cybersecurity": 5,
            "automotive": 3,
            "food": 3,
            "environment": 3,
            "real estate": 3,
            "entertainment": 2,
            "science": 5,
            "health": 3,
            "business": 3,
            "politics": 3,
            "sports": 2,
            "world": 2,
            "default": 3
        }

        percentile = domain_percentiles.get(domain_to_use, domain_percentiles["default"])

        percentile = percentile_from_domain if percentile_from_domain else percentile

        if not hasattr(self, '_quality_percentiles'):
            self._quality_percentiles = {}

        self._quality_percentiles[domain_to_use] = percentile

        print(f"Using {percentile}th percentile for domain '{domain_to_use}'")

        if content_density > 0.7:
            percentile = max(5, percentile - 5)
        elif content_density < 0.5:
            percentile = min(20, percentile + 5)

        if len(scores) >= 3:
            import numpy as np
            percentile_threshold = np.percentile(scores, 100 - percentile)

            print(f"- Using {percentile}th percentile from top: {percentile_threshold:.3f}")
        else:
            percentile_threshold = 0
            print(f"- Too few keyphrases, using base threshold only")

        quality_threshold = max(base_threshold, percentile_threshold)

        quality_threshold = min(quality_threshold, 0.35)

        print(f"\nQuality Filtering Stats:")
        print(f"- Domain: {domain}")
        print(f"- Text Length: {word_count} words")
        print(f"- Content Density: {content_density:.2f}")
        print(f"- Initial Candidates: {len(scored_keyphrases)}")
        print(f"- Base Threshold: {base_threshold:.3f}")
        print(f"- Quality Threshold: {quality_threshold:.3f}")

        quality_kps = [(kp, score) for kp, score in scored_keyphrases if score >= quality_threshold]
        print(f"- After Quality Filter: {len(quality_kps)}")

        domain_target_counts = {
            "artificial intelligence": 12,
            "technology": 12,
            "cybersecurity": 12,
            "automotive": 12,
            "food": 12,
            "environment": 12,
            "real estate": 12,
            "entertainment": 12,
            "default": 10
        }

        base_target = domain_target_counts.get(domain_to_use, domain_target_counts["default"])

        base_target = target_count if target_count else base_target

        print(f"Using target count {base_target} for domain '{domain_to_use}'")

        if word_count < 300:
            length_adjustment = -1
        elif word_count > 600:
            length_adjustment = 1
        else:
            length_adjustment = 0

        target_keyphrases = min(12, max(8, base_target + length_adjustment))

        min_keyphrases = 8

        if len(quality_kps) < target_keyphrases and len(scored_keyphrases) >= target_keyphrases:
            print(f"- Quality filter too strict, falling back to top {target_keyphrases} keyphrases")

            lenient_threshold = quality_threshold * 0.5
            lenient_kps = [(kp, score) for kp, score in scored_keyphrases if score >= lenient_threshold]

            if len(lenient_kps) >= target_keyphrases:
                quality_kps = lenient_kps
                print(f"- Using lenient threshold {lenient_threshold:.3f}: {len(quality_kps)} keyphrases")
            else:
                quality_kps = sorted(scored_keyphrases, key=lambda x: x[1], reverse=True)[:target_keyphrases]
                print(f"- Falling back to top {target_keyphrases} keyphrases by score")

        if len(quality_kps) < min_keyphrases and len(scored_keyphrases) >= min_keyphrases:
            print(f"- Too few keyphrases ({len(quality_kps)}), ensuring minimum of {min_keyphrases}")
            quality_kps = sorted(scored_keyphrases, key=lambda x: x[1], reverse=True)[:min_keyphrases]

        diversity_weight = min(0.4, max(0.2, content_density * 0.4))

        max_reasonable_keyphrases = 12

        if len(quality_kps) > max_reasonable_keyphrases:
            print(f"- Too many keyphrases ({len(quality_kps)}), selecting {max_reasonable_keyphrases} diverse ones")
            quality_kps = self.select_diverse_keyphrases(
                quality_kps,
                max_reasonable_keyphrases,
                diversity_weight
            )

        print(f"- Final keyphrase count: {len(quality_kps)}")

        if len(quality_kps) > 1:
            diversity_score = self.calculate_semantic_diversity([kp for kp, _ in quality_kps])
            print(f"- Semantic diversity score: {diversity_score:.2f}")

        return quality_kps

    def run_quality_filter_calibration(self, articles: List[str], num_articles: int = 5) -> Dict[str, Any]:

        print("\n" + "="*80)
        print("QUALITY FILTER CALIBRATION PROCESS")
        print("="*80)

        print("\nSTEP 1: Testing Different Percentile Thresholds")
        calibration_results = self.tune_quality_thresholds(
            articles[:num_articles],
            percentiles=[10, 15, 20, 25, 30]
        )

        print("\nSTEP 2: Applying Recommended Settings")

        best_percentile = None
        for percentile in [10, 15, 20, 25, 30]:
            if percentile in calibration_results:
                avg_count = calibration_results[percentile]["avg_keyphrase_count"]
                avg_min_score = calibration_results[percentile]["avg_min_score"]

                if 7 <= avg_count <= 12 and avg_min_score >= 0.3:
                    best_percentile = percentile
                    break

        if best_percentile is None:
            for percentile in [10, 15, 20, 25, 30]:
                if percentile in calibration_results:
                    avg_count = calibration_results[percentile]["avg_keyphrase_count"]
                    if 7 <= avg_count <= 12:
                        best_percentile = percentile
                        break

        if best_percentile is None:
            best_percentile = 10

        recommended_percentiles = {
            "technology": best_percentile,
            "science": best_percentile,
            "health": best_percentile,
            "business": best_percentile,
            "politics": best_percentile,
            "sports": max(5, best_percentile - 5),
            "entertainment": max(5, best_percentile - 5),
            "world": best_percentile,
            "default": best_percentile
        }

        print(f"\nApplying recommended percentiles:")
        for domain, percentile in recommended_percentiles.items():
            print(f"- {domain}: {percentile}")

        self._quality_percentiles = recommended_percentiles

        print("\nSTEP 3: Testing with New Settings")
        test_articles = articles[:3]

        results = []
        for i, article in enumerate(test_articles):
            print(f"\nArticle {i+1}:")
            domain = self.detect_domain(article)
            print(f"Domain: {domain}")

            keyphrases = self.extract_keyphrases_with_scores(article)
            print(f"Generated {len(keyphrases)} keyphrases:")

            for kp, score in keyphrases:
                print(f"- {kp}: {score:.3f}")

            results.append({
                "domain": domain,
                "count": len(keyphrases),
                "keyphrases": keyphrases
            })

        return {
            "calibration_results": calibration_results,
            "recommended_percentiles": recommended_percentiles,
            "test_results": results
        }
    def tune_quality_thresholds(self, articles: List[str], percentiles: List[int] = [10, 15, 20, 25, 30]) -> Dict[str, Any]:

        results = {}

        original_percentiles = self._quality_percentiles.copy()

        print("\n" + "="*80)
        print("TUNING QUALITY THRESHOLDS")
        print("="*80)

        for percentile in percentiles:
            print(f"\nTesting {percentile}th percentile threshold")

            self._quality_percentiles = {
                "technology": percentile,
                "science": percentile,
                "health": percentile,
                "business": percentile,
                "politics": percentile,
                "sports": percentile,
                "entertainment": percentile,
                "world": percentile,
                "default": percentile
            }

            article_results = []
            domain_counts = {}

            for i, article in enumerate(articles):
                print(f"\nArticle {i+1}/{len(articles)}")

                domain = self.detect_domain(article)
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

                keyphrases = self.extract_keyphrases_with_scores(article)

                min_score = min([score for _, score in keyphrases]) if keyphrases else 0
                max_score = max([score for _, score in keyphrases]) if keyphrases else 0
                avg_score = sum([score for _, score in keyphrases]) / len(keyphrases) if keyphrases else 0

                multi_word_count = sum(1 for kp, _ in keyphrases if len(kp.split()) > 1)
                multi_word_percentage = multi_word_count / len(keyphrases) if keyphrases else 0

                avg_length = sum(len(kp.split()) for kp, _ in keyphrases) / len(keyphrases) if keyphrases else 0

                article_results.append({
                    "domain": domain,
                    "count": len(keyphrases),
                    "min_score": min_score,
                    "max_score": max_score,
                    "avg_score": avg_score,
                    "multi_word_percentage": multi_word_percentage,
                    "avg_length": avg_length,
                    "scores": [score for _, score in keyphrases],
                    "keyphrases": [kp for kp, _ in keyphrases]
                })

                print(f"Domain: {domain}")
                print(f"Keyphrases: {len(keyphrases)}")
                print(f"Min score: {min_score:.3f}, Max score: {max_score:.3f}, Avg score: {avg_score:.3f}")
                print(f"Multi-word: {multi_word_percentage:.1%}, Avg length: {avg_length:.1f} words")

                if keyphrases:
                    sorted_keyphrases = sorted(keyphrases, key=lambda x: x[1])
                    print("\nLowest scoring keyphrases:")
                    for kp, score in sorted_keyphrases[:3]:
                        print(f"- {kp}: {score:.3f}")

            avg_count = sum(r["count"] for r in article_results) / len(article_results)
            avg_min_score = sum(r["min_score"] if r["scores"] else 0 for r in article_results) / len(article_results)
            avg_max_score = sum(r["max_score"] if r["scores"] else 0 for r in article_results) / len(article_results)
            avg_avg_score = sum(r["avg_score"] if r["scores"] else 0 for r in article_results) / len(article_results)
            avg_multi_word = sum(r["multi_word_percentage"] for r in article_results) / len(article_results)
            avg_length = sum(r["avg_length"] for r in article_results) / len(article_results)

            domain_stats = {}
            for domain in domain_counts.keys():
                domain_articles = [r for r in article_results if r["domain"] == domain]
                if domain_articles:
                    domain_avg_count = sum(r["count"] for r in domain_articles) / len(domain_articles)
                    domain_avg_min_score = sum(r["min_score"] if r["scores"] else 0 for r in domain_articles) / len(domain_articles)
                    domain_avg_multi_word = sum(r["multi_word_percentage"] for r in domain_articles) / len(domain_articles)
                    domain_stats[domain] = {
                        "count": len(domain_articles),
                        "avg_keyphrase_count": domain_avg_count,
                        "avg_min_score": domain_avg_min_score,
                        "avg_multi_word": domain_avg_multi_word
                    }

            results[percentile] = {
                "avg_keyphrase_count": avg_count,
                "avg_min_score": avg_min_score,
                "avg_max_score": avg_max_score,
                "avg_avg_score": avg_avg_score,
                "avg_multi_word": avg_multi_word,
                "avg_length": avg_length,
                "article_results": article_results,
                "domain_stats": domain_stats
            }

            print(f"\nPercentile {percentile} Summary:")
            print(f"- Average keyphrase count: {avg_count:.1f}")
            print(f"- Average minimum score: {avg_min_score:.3f}")
            print(f"- Average maximum score: {avg_max_score:.3f}")
            print(f"- Average mean score: {avg_avg_score:.3f}")
            print(f"- Average multi-word percentage: {avg_multi_word:.1%}")
            print(f"- Average phrase length: {avg_length:.1f} words")

            print("\nDomain-specific results:")
            for domain, stats in domain_stats.items():
                print(f"- {domain} ({stats['count']} articles): {stats['avg_keyphrase_count']:.1f} keyphrases, " +
                    f"min score {stats['avg_min_score']:.3f}, multi-word {stats['avg_multi_word']:.1%}")

        self._quality_percentiles = original_percentiles

        print("\n" + "="*80)
        print("PERCENTILE COMPARISON")
        print("="*80)
        print(f"{'Percentile':<10} {'Avg Count':<10} {'Min Score':<10} {'Avg Score':<10} {'Multi-word':<10} {'Avg Length':<10}")
        print("-"*60)

        for percentile in percentiles:
            avg_count = results[percentile]["avg_keyphrase_count"]
            avg_min_score = results[percentile]["avg_min_score"]
            avg_avg_score = results[percentile]["avg_avg_score"]
            avg_multi_word = results[percentile]["avg_multi_word"]
            avg_length = results[percentile]["avg_length"]
            print(f"{percentile:<10} {avg_count:<10.1f} {avg_min_score:<10.3f} {avg_avg_score:<10.3f} {avg_multi_word:<10.1%} {avg_length:<10.1f}")

        target_count = 9
        best_percentile = None
        best_diff = float('inf')

        for percentile in percentiles:
            avg_count = results[percentile]["avg_keyphrase_count"]
            diff = abs(avg_count - target_count)
            if diff < best_diff:
                best_diff = diff
                best_percentile = percentile

        if best_percentile is not None:
            print("\n" + "="*80)
            print(f"RECOMMENDED PERCENTILE: {best_percentile}")
            print("="*80)
            print(f"This percentile produces an average of {results[best_percentile]['avg_keyphrase_count']:.1f} keyphrases")
            print(f"with an average minimum score of {results[best_percentile]['avg_min_score']:.3f}")
            print(f"and average multi-word percentage of {results[best_percentile]['avg_multi_word']:.1%}")

            min_score_acceptable = results[best_percentile]['avg_min_score'] >= 0.3
            print(f"\nQuality check: Average minimum score is {results[best_percentile]['avg_min_score']:.3f}")
            if min_score_acceptable:
                print("✓ This is above the acceptable threshold of 0.3")
            else:
                print("⚠ This is below the acceptable threshold of 0.3")

                for p in sorted(percentiles):
                    if p > best_percentile and results[p]['avg_min_score'] >= 0.3:
                        print(f"Consider using {p} instead for better quality (avg count: {results[p]['avg_keyphrase_count']:.1f})")
                        break

            print("\nRecommended domain-specific percentiles:")
            domain_percentiles = {}

            for domain in domain_counts.keys():
                domain_best_percentile = None
                domain_best_diff = float('inf')

                for percentile in percentiles:
                    if domain in results[percentile]["domain_stats"]:
                        domain_avg_count = results[percentile]["domain_stats"][domain]["avg_keyphrase_count"]
                        domain_avg_min_score = results[percentile]["domain_stats"][domain]["avg_min_score"]

                        if domain_avg_min_score >= 0.3:
                            diff = abs(domain_avg_count - target_count)
                            if diff < domain_best_diff:
                                domain_best_diff = diff
                                domain_best_percentile = percentile

                if domain_best_percentile is None:
                    for percentile in percentiles:
                        if domain in results[percentile]["domain_stats"]:
                            domain_avg_count = results[percentile]["domain_stats"][domain]["avg_keyphrase_count"]
                            diff = abs(domain_avg_count - target_count)
                            if diff < domain_best_diff:
                                domain_best_diff = diff
                                domain_best_percentile = percentile

                if domain_best_percentile is not None:
                    domain_percentiles[domain] = domain_best_percentile
                    domain_avg_count = results[domain_best_percentile]["domain_stats"][domain]["avg_keyphrase_count"]
                    domain_avg_min_score = results[domain_best_percentile]["domain_stats"][domain]["avg_min_score"]
                    print(f"- {domain}: {domain_best_percentile} " +
                        f"({domain_avg_count:.1f} keyphrases, min score: {domain_avg_min_score:.3f})")

            recommended_percentiles = {
                "default": best_percentile
            }
            recommended_percentiles.update(domain_percentiles)

            print("\nRecommended _quality_percentiles setting:")
            print("self._quality_percentiles = {")
            for domain, percentile in recommended_percentiles.items():
                print(f"    \"{domain}\": {percentile},")
            print("}")

            print("\n" + "="*80)
            print("SAMPLE KEYPHRASES WITH RECOMMENDED PERCENTILE")
            print("="*80)

            current_percentiles = self._quality_percentiles.copy()

            self._quality_percentiles = recommended_percentiles

            sample_articles = articles[:3]
            for i, article in enumerate(sample_articles):
                print(f"\nArticle {i+1}:")
                domain = self.detect_domain(article)
                print(f"Domain: {domain}")

                keyphrases = self.extract_keyphrases_with_scores(article)
                print(f"Generated {len(keyphrases)} keyphrases:")

                for kp, score in keyphrases:
                    print(f"- {kp}: {score:.3f}")

            self._quality_percentiles = current_percentiles

        return results

    def calibrate_domain_detection(self, articles: List[str], confidence_thresholds: List[float] = [0.25, 0.30, 0.35, 0.40, 0.45]) -> Dict[str, Any]:

        results = {}

        print("\n" + "="*80)
        print("CALIBRATING DOMAIN DETECTION CONFIDENCE THRESHOLD")
        print("="*80)

        for threshold in confidence_thresholds:
            print(f"\nTesting confidence threshold {threshold:.2f}")

            article_results = []
            zsl_success_count = 0
            fallback_count = 0

            for i, article in enumerate(articles):
                print(f"\nArticle {i+1}/{len(articles)}")

                truncated_text = article[:1500]

                try:
                    if not hasattr(self, 'zero_shot_classifier'):
                        from transformers import pipeline
                        print("Initializing zero-shot domain classifier...")
                        device = 0 if self.device == "cuda" else -1
                        self.zero_shot_classifier = pipeline(
                            "zero-shot-classification",
                            model="facebook/bart-large-mnli",
                            device=device
                        )

                    candidate_domains = [
                        "technology", "business", "health", "politics", "sports",
                        "entertainment", "science", "environment", "world", "education",
                        "food", "travel", "automotive", "real estate", "cybersecurity",
                        "artificial intelligence", "space", "agriculture", "mental health"
                    ]

                    best_score = 0
                    best_domain = None
                    best_result = None

                    hypothesis_templates_to_try = [
                        "This news article is about {}.",
                        "The topic of this document is {}.",
                        "This text discusses {}.",
                        "The main subject of this content is {}."
                    ]

                    for template in hypothesis_templates_to_try:
                        try:
                            result = self.zero_shot_classifier(
                                truncated_text,
                                candidate_domains,
                                multi_label=False,
                                hypothesis_template=template
                            )
                            current_score = result['scores'][0]
                            if current_score > best_score:
                                best_score = current_score
                                best_domain = result['labels'][0]
                                best_result = result
                        except Exception:
                            continue

                    try:
                        result_default = self.zero_shot_classifier(
                            truncated_text,
                            candidate_domains,
                            multi_label=False
                        )
                        default_score = result_default['scores'][0]
                        if default_score > best_score:
                            best_score = default_score
                            best_domain = result_default['labels'][0]
                            best_result = result_default
                    except Exception:
                        pass

                    if best_result is None:
                        fallback_domain = self._keyword_based_domain_detection(article)
                        article_results.append({
                            "zsl_success": False,
                            "zsl_score": 0,
                            "zsl_domain": None,
                            "fallback_domain": fallback_domain,
                            "final_domain": fallback_domain,
                            "used_fallback": True
                        })
                        fallback_count += 1
                    else:
                        zsl_domain = best_domain
                        zsl_score = best_score

                        if zsl_score >= threshold:
                            final_domain = zsl_domain
                            used_fallback = False
                            zsl_success_count += 1
                        else:
                            fallback_domain = self._keyword_based_domain_detection(article)
                            final_domain = fallback_domain
                            used_fallback = True
                            fallback_count += 1

                        article_results.append({
                            "zsl_success": True,
                            "zsl_score": zsl_score,
                            "zsl_domain": zsl_domain,
                            "fallback_domain": fallback_domain if used_fallback else None,
                            "final_domain": final_domain,
                            "used_fallback": used_fallback
                        })

                except Exception as e:
                    print(f"Error in zero-shot domain detection: {str(e)}")
                    fallback_domain = self._keyword_based_domain_detection(article)
                    article_results.append({
                        "zsl_success": False,
                        "zsl_score": 0,
                        "zsl_domain": None,
                        "fallback_domain": fallback_domain,
                        "final_domain": fallback_domain,
                        "used_fallback": True
                    })
                    fallback_count += 1

            zsl_success_rate = zsl_success_count / len(articles)
            fallback_rate = fallback_count / len(articles)

            zsl_scores = [r["zsl_score"] for r in article_results if r["zsl_success"]]
            avg_zsl_score = sum(zsl_scores) / len(zsl_scores) if zsl_scores else 0

            results[threshold] = {
                "zsl_success_rate": zsl_success_rate,
                "fallback_rate": fallback_rate,
                "avg_zsl_score": avg_zsl_score,
                "article_results": article_results
            }

            print(f"\nConfidence Threshold {threshold:.2f} Summary:")
            print(f"- ZSL Success Rate: {zsl_success_rate:.1%}")
            print(f"- Fallback Rate: {fallback_rate:.1%}")
            print(f"- Average ZSL Score: {avg_zsl_score:.3f}")

        print("\n" + "="*80)
        print("CONFIDENCE THRESHOLD COMPARISON")
        print("="*80)
        print(f"{'Threshold':<10} {'ZSL Rate':<10} {'Fallback':<10} {'Avg Score':<10}")
        print("-"*40)

        for threshold in confidence_thresholds:
            zsl_rate = results[threshold]["zsl_success_rate"]
            fallback_rate = results[threshold]["fallback_rate"]
            avg_score = results[threshold]["avg_zsl_score"]
            print(f"{threshold:<10.2f} {zsl_rate:<10.1%} {fallback_rate:<10.1%} {avg_score:<10.3f}")

        target_fallback_rate = 0.2
        best_threshold = None
        best_diff = float('inf')

        for threshold in confidence_thresholds:
            fallback_rate = results[threshold]["fallback_rate"]
            diff = abs(fallback_rate - target_fallback_rate)
            if diff < best_diff:
                best_diff = diff
                best_threshold = threshold

        if best_threshold is not None:
            print("\n" + "="*80)
            print(f"RECOMMENDED CONFIDENCE THRESHOLD: {best_threshold:.2f}")
            print("="*80)
            print(f"This threshold results in a ZSL success rate of {results[best_threshold]['zsl_success_rate']:.1%}")
            print(f"and a fallback rate of {results[best_threshold]['fallback_rate']:.1%}")
            print(f"with an average ZSL score of {results[best_threshold]['avg_zsl_score']:.3f}")

            print("\nTo apply this threshold, update the confidence_threshold variable in the detect_domain method:")
            print(f"confidence_threshold = {best_threshold:.2f}")

        return results

    def create_labeled_dataset(self, articles: List[str], manual_review: bool = False) -> List[Tuple[str, str]]:

        print("\n" + "="*80)
        print("CREATING LABELED DATASET FOR DOMAIN CLASSIFIER")
        print("="*80)

        labeled_data = []
        high_confidence_count = 0

        for i, article in enumerate(articles):
            print(f"\nArticle {i+1}/{len(articles)}")

            try:
                if not hasattr(self, 'zero_shot_classifier'):
                    from transformers import pipeline
                    print("Initializing zero-shot domain classifier...")
                    device = 0 if self.device == "cuda" else -1
                    self.zero_shot_classifier = pipeline(
                        "zero-shot-classification",
                        model="facebook/bart-large-mnli",
                        device=device
                    )

                truncated_text = article[:1500]

                candidate_domains = [
                    "technology", "business", "health", "politics", "sports",
                    "entertainment", "science", "environment", "world", "education",
                    "food", "travel", "automotive", "real estate", "cybersecurity",
                    "artificial intelligence", "space", "agriculture", "mental health"
                ]

                best_score = 0
                best_domain = None

                hypothesis_templates_to_try = [
                    "This news article is about {}.",
                    "The topic of this document is {}.",
                    "This text discusses {}.",
                    "The main subject of this content is {}."
                ]

                for template in hypothesis_templates_to_try:
                    try:
                        result = self.zero_shot_classifier(
                            truncated_text,
                            candidate_domains,
                            multi_label=False,
                            hypothesis_template=template
                        )
                        current_score = result['scores'][0]
                        if current_score > best_score:
                            best_score = current_score
                            best_domain = result['labels'][0]
                    except Exception:
                        continue

                try:
                    result_default = self.zero_shot_classifier(
                        truncated_text,
                        candidate_domains,
                        multi_label=False
                    )
                    default_score = result_default['scores'][0]
                    if default_score > best_score:
                        best_score = default_score
                        best_domain = result_default['labels'][0]
                except Exception:
                    pass

                domain_mapping = {
                    "artificial intelligence": "technology",
                    "cybersecurity": "technology",
                    "space": "science",
                    "agriculture": "science",
                    "mental health": "health",
                    "automotive": "technology",
                    "real estate": "business",
                    "food": "lifestyle"
                }

                if best_domain in domain_mapping:
                    best_domain = domain_mapping[best_domain]

                high_confidence_threshold = 0.6

                if best_score >= high_confidence_threshold:
                    print(f"High confidence label: {best_domain} (Score: {best_score:.4f})")
                    domain = best_domain
                    high_confidence_count += 1
                else:
                    print(f"Low confidence label: {best_domain} (Score: {best_score:.4f})")
                    keyword_domain = self._keyword_based_domain_detection(article)

                    if manual_review:
                        excerpt = article[:300] + "..." if len(article) > 300 else article
                        print("\nArticle excerpt:")
                        print("-" * 40)
                        print(excerpt)
                        print("-" * 40)
                        print(f"ZSL domain: {best_domain} (Score: {best_score:.4f})")
                        print(f"Keyword domain: {keyword_domain}")

                        domain_options = ", ".join(sorted(set(candidate_domains) - {"artificial intelligence", "cybersecurity", "space", "agriculture", "mental health", "automotive", "real estate"}))
                        manual_domain = input(f"Enter domain ({domain_options}), or press Enter to accept ZSL domain: ")

                        if manual_domain.strip():
                            domain = manual_domain.strip()
                        else:
                            domain = best_domain
                    else:
                        if best_score >= 0.4:
                            domain = best_domain
                        else:
                            domain = keyword_domain

                labeled_data.append((article, domain))

            except Exception as e:
                print(f"Error processing article: {str(e)}")
                continue

        print("\n" + "="*80)
        print(f"LABELED DATASET CREATED: {len(labeled_data)} articles")
        print(f"High confidence labels: {high_confidence_count} ({high_confidence_count/len(labeled_data):.1%})")

        domain_counts = {}
        for _, domain in labeled_data:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        print("\nDomain distribution:")
        for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"- {domain}: {count} articles ({count/len(labeled_data):.1%})")

        return labeled_data

    def improve_domain_detection(self, articles: List[str], num_articles: int = 10) -> Dict[str, Any]:
        print("\n" + "="*80)
        print("DOMAIN DETECTION IMPROVEMENT PROCESS")
        print("="*80)

        print("\nSTEP 1: Calibrating Confidence Threshold")
        calibration_results = self.calibrate_domain_detection(
            articles[:num_articles],
            confidence_thresholds=[0.25, 0.30, 0.35, 0.40, 0.45]
        )

        print("\nSTEP 2: Creating Labeled Dataset")
        labeled_data = self.create_labeled_dataset(articles)

        print("\nSTEP 3: Training Fallback Classifier")
        self.train_domain_classifier(labeled_data)

        print("\nTesting Improved Domain Detection")
        test_articles = articles[:5]

        results = []
        for i, article in enumerate(test_articles):
            print(f"\nArticle {i+1}:")
            domain = self.detect_domain(article)
            print(f"Detected domain: {domain}")
            results.append(domain)

        return {
            "calibration_results": calibration_results,
            "labeled_data_size": len(labeled_data),
            "has_classifier": self.has_domain_classifier,
            "test_results": results
        }

    def apply_optimized_parameters(self, params: Dict[str, Any]) -> None:

        print("\nApplying optimized parameters:")

        for param, value in params.items():
            if hasattr(self, param):
                print(f"  Setting {param} = {value}")
                setattr(self, param, value)
            else:
                print(f"  Warning: Parameter '{param}' not found in extractor")

        if "num_beam_groups" in params and params["num_beam_groups"] > 1:
            self.num_beam_groups = params["num_beam_groups"]
            if "diversity_penalty" in params:
                self.diversity_penalty = params["diversity_penalty"]
            else:
                self.diversity_penalty = 1.0
            print(f"  Setting num_beam_groups = {self.num_beam_groups}")
            print(f"  Setting diversity_penalty = {self.diversity_penalty}")
        elif hasattr(self, 'num_beam_groups'):
            delattr(self, 'num_beam_groups')
            if hasattr(self, 'diversity_penalty'):
                delattr(self, 'diversity_penalty')

        print("Parameters applied successfully")

    def run_phase2_optimization(self, articles: List[str], num_articles: int = 5) -> Dict[str, Any]:

        print("\n" + "="*80)
        print("PHASE 2 OPTIMIZATION")
        print("="*80)

        print("\nSTEP 1: Benchmarking Generation Parameters")
        generation_results = self.optimize_generation_across_articles(articles, num_articles=num_articles)

        if generation_results["best_params"]:
            self.apply_optimized_parameters(generation_results["best_params"])

        print("\nSTEP 2: Calibrating Quality Filter Thresholds")
        threshold_results = self.tune_quality_thresholds(articles[:num_articles])

        print("\nSTEP 3: Calibrating Domain Detection Confidence")
        confidence_results = self.calibrate_domain_detection(articles[:num_articles])

        print("\nOptimization complete. Use the recommended settings to update your code.")

        return {
            "generation_results": generation_results,
            "threshold_results": threshold_results,
            "confidence_results": confidence_results
        }

    def find_best_params(self, text: str) -> Tuple[Dict[str, Any], List[str]]:

        param_results = self.test_generation_params(text)

        best_score = -1
        best_params = None
        best_keyphrases = []

        for param_key, keyphrases in param_results.items():
            if not keyphrases:
                continue

            num_keyphrases = len(keyphrases)

            multi_word_count = sum(1 for kp in keyphrases if len(kp.split()) > 1)

            ngram_penalty = 0
            for i in range(len(keyphrases) - 1):
                kp1 = keyphrases[i].split()
                kp2 = keyphrases[i+1].split()

                if len(kp1) > 0 and len(kp2) > 0:
                    if kp1[-1] == kp2[0]:
                        ngram_penalty += 0.2

            avg_length = sum(len(kp.split()) for kp in keyphrases) / num_keyphrases

            length_score = 1.0 - abs(avg_length - 2.5) / 2.5

            multi_word_percentage = multi_word_count / num_keyphrases if num_keyphrases > 0 else 0

            num_score = min(num_keyphrases / 10, 1.5)

            if num_keyphrases < 3:
                num_score *= 0.5
            elif num_keyphrases > 20:
                num_score *= 0.7

            final_score = (
                (num_score * 0.3) +
                (multi_word_percentage * 0.4) +
                (length_score * 0.3) -
                ngram_penalty
            )

            if final_score > best_score:
                best_score = final_score
                best_params = param_key
                best_keyphrases = keyphrases

        if best_params:
            param_dict = {}
            for param in best_params.split(', '):
                key, value = param.split('=')
                if key == 'temp':
                    param_dict['temperature'] = float(value)
                elif key == 'top_p':
                    param_dict['top_p'] = float(value)
                elif key == 'len_pen':
                    param_dict['length_penalty'] = float(value)
                elif key == 'prompt':
                    param_dict['prompt_template_idx'] = int(value)
        else:
            param_dict = {
                'temperature': 0.7,
                'top_p': 0.92,
                'length_penalty': 0.8,
                'prompt_template_idx': 0
            }

        return param_dict, best_keyphrases

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

    def score_keyphrases_by_relevance(self, keyphrases: List[str], text: str) -> List[Tuple[str, float]]:

        if not keyphrases:
            return []

        if not hasattr(self, 'pattern_bonuses'):
            self.pattern_bonuses = {}

        for kp in keyphrases:
            if kp in self.pattern_bonuses:
                continue

            self.pattern_bonuses[kp] = 0.0

            if len(kp.split()) <= 1:
                continue

            try:
                tokens = word_tokenize(kp)
                pos_tags = pos_tag(tokens)

                pattern_bonus = 0.0

                pattern_bonus += 0.40

                if len(pos_tags) >= 2:
                    if any(pos_tags[i][1].startswith('JJ') and pos_tags[i+1][1].startswith('NN') for i in range(len(pos_tags)-1)):
                        pattern_bonus += 0.60

                    if any(pos_tags[i][1].startswith('NN') and pos_tags[i+1][1].startswith('NN') for i in range(len(pos_tags)-1)):
                        pattern_bonus += 0.60

                if len(pos_tags) >= 3:
                    pattern_bonus += 0.60

                self.pattern_bonuses[kp] = pattern_bonus
                print(f"DEBUG: Added pattern bonus {pattern_bonus:.4f} to '{kp}' in score_keyphrases_by_relevance")
            except Exception as e:
                print(f"Warning: Error in POS tagging for '{kp}': {str(e)}. Using no pattern bonus.")

        doc_embedding = self.sentence_model.encode([text], show_progress_bar=False)[0]

        self.title, self.lead_paragraph, self.first_paragraphs = self.extract_title_and_lead(text)

        keyphrase_embeddings = self.sentence_model.encode(keyphrases, show_progress_bar=False)

        named_entities = set()
        if self.use_ner:
            named_entities = set(self.extract_named_entities(text))

        sentences = self._split_into_sentences(text)

        sentence_embeddings = self.sentence_model.encode(sentences, show_progress_bar=False)
        scored_keyphrases = []
        for i, kp in enumerate(keyphrases):
            kp_embedding = keyphrase_embeddings[i]
            kp_lower = kp.lower()

            doc_similarity = cosine_similarity(
                doc_embedding.reshape(1, -1),
                kp_embedding.reshape(1, -1)
            )[0][0]

            containing_sentences = []
            containing_sentence_indices = []

            for j, sentence in enumerate(sentences):
                if kp_lower in sentence.lower():
                    containing_sentences.append(sentence)
                    containing_sentence_indices.append(j)

            sentence_similarity = 0.0
            if containing_sentences:
                sentence_similarities = []
                for idx in containing_sentence_indices:
                    sent_embedding = sentence_embeddings[idx]
                    sim = cosine_similarity(
                        sent_embedding.reshape(1, -1),
                        kp_embedding.reshape(1, -1)
                    )[0][0]
                    sentence_similarities.append(sim)

                sentence_similarity = max(sentence_similarities)

            if containing_sentences:
                combined_similarity = 0.4 * doc_similarity + 0.6 * sentence_similarity
            else:
                combined_similarity = 0.9 * doc_similarity

            if hasattr(self, 'pattern_bonuses'):
                print(f"DEBUG: pattern_bonuses attribute exists with {len(self.pattern_bonuses)} entries")
                if kp in self.pattern_bonuses:
                    pattern_bonus = self.pattern_bonuses[kp]
                    old_score = combined_similarity
                    combined_similarity += pattern_bonus
                    print(f"DEBUG: Applied pattern bonus {pattern_bonus:.4f} to '{kp}', score: {old_score:.4f} -> {combined_similarity:.4f}")
                else:
                    print(f"DEBUG: No pattern bonus found for '{kp}'")
            else:
                print("DEBUG: No pattern_bonuses attribute found")

            final_score = combined_similarity

            if kp in named_entities:
                final_score = min(final_score * 1.1, 1.0)

            contextual_importance = self.calculate_contextual_importance(kp, text)

            final_score = min(final_score * contextual_importance, 1.0)

            word_count = len(kp.split())
            if word_count > 1:
                if word_count == 2:
                    length_boost = 0.10
                elif word_count == 3:
                    length_boost = 0.15
                elif word_count >= 4:
                    length_boost = 0.20

                final_score = min(final_score * (1 + length_boost), 1.0)

            scored_keyphrases.append((kp, final_score))

        scored_keyphrases.sort(key=lambda x: x[1], reverse=True)

        return scored_keyphrases

    def _split_into_sentences(self, text: str) -> List[str]:

        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
        except Exception as e:
            print(f"Warning: NLTK sent_tokenize failed: {e}. Falling back to basic split.")
            sentences = re.split(r'(?<=\.|\?|\!)\s+', text)

        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences and text.strip():
            sentences = [text.strip()]

        return sentences

    def enhance_semantic_coherence(self, keyphrases: List[Tuple[str, float]], text: str) -> List[Tuple[str, float]]:

        if not keyphrases or len(keyphrases) < 2:
            return keyphrases

        kps = [kp for kp, _ in keyphrases]
        scores = [score for _, score in keyphrases]

        kp_embeddings = self.sentence_model.encode(kps, show_progress_bar=False)
        doc_embedding = self.sentence_model.encode([text], show_progress_bar=False)[0]

        similarity_matrix = cosine_similarity(kp_embeddings)

        coherence_scores = []
        for i in range(len(kps)):
            similarities = [similarity_matrix[i][j] for j in range(len(kps)) if i != j]
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0

            doc_relevance = cosine_similarity([kp_embeddings[i]], [doc_embedding])[0][0]

            coherence_score = 0.7 * doc_relevance + 0.3 * avg_similarity
            coherence_scores.append(coherence_score)

        adjusted_scores = [0.7 * scores[i] + 0.3 * coherence_scores[i] for i in range(len(scores))]

        enhanced_keyphrases = [(kps[i], adjusted_scores[i]) for i in range(len(kps))]

        enhanced_keyphrases.sort(key=lambda x: x[1], reverse=True)

        return enhanced_keyphrases

    def calculate_contextual_importance(self, keyphrase: str, text: str) -> float:

        title, lead_paragraph, first_paragraphs = self.extract_title_and_lead(text)

        importance = 1.0

        if keyphrase.lower() in title.lower():
            importance *= 2.0
        elif keyphrase.lower() in lead_paragraph.lower():
            importance *= 1.5
        elif keyphrase.lower() in first_paragraphs.lower():
            importance *= 1.3

        doc_embedding = self.sentence_model.encode([text], show_progress_bar=False)[0]
        kp_embedding = self.sentence_model.encode([keyphrase], show_progress_bar=False)[0]
        semantic_centrality = cosine_similarity([kp_embedding], [doc_embedding])[0][0]

        importance *= (0.5 + 0.5 * semantic_centrality)

        text_lower = text.lower()
        kp_lower = keyphrase.lower()

        count = 0
        for pattern in [f" {kp_lower} ", f"{kp_lower} ", f" {kp_lower}", f"{kp_lower}"]:
            count += text_lower.count(pattern)

        text_length = len(text.split())
        normalized_frequency = count / (text_length / 1000)

        if normalized_frequency > 0:
            frequency_factor = min(1.5, 1.0 + 0.1 * math.log(1 + normalized_frequency))
            importance *= frequency_factor

        importance = min(2.0, importance)

        return importance

    def boost_domain_specific_concepts(self, keyphrases: List[Tuple[str, float]], domain: str) -> List[Tuple[str, float]]:

        if not keyphrases:
            return keyphrases

        normalized_domain = domain.lower()

        domain_mapping = {
            "artificial intelligence": "technology",
            "cybersecurity": "technology",
            "automotive": "technology",
            "real estate": "business",
            "food": "food",
            "environment": "environment",
            "entertainment": "entertainment"
        }

        lookup_domain = domain_mapping.get(normalized_domain, normalized_domain)

        domain_keywords_list = []
        if hasattr(self, 'domain_keywords') and lookup_domain in self.domain_keywords:
            domain_keywords_list = self.domain_keywords.get(lookup_domain, [])
        elif not domain_keywords_list:
            try:
                if lookup_domain == "technology" and "TECHNOLOGY_KEYWORDS" in globals():
                    domain_keywords_list = globals()["TECHNOLOGY_KEYWORDS"]
                elif lookup_domain == "business" and "BUSINESS_KEYWORDS" in globals():
                    domain_keywords_list = globals()["BUSINESS_KEYWORDS"]
                elif lookup_domain == "entertainment" and "ENTERTAINMENT_KEYWORDS" in globals():
                    domain_keywords_list = globals()["ENTERTAINMENT_KEYWORDS"]
                elif lookup_domain == "environment" and "ENVIRONMENT_KEYWORDS" in globals():
                    domain_keywords_list = globals()["ENVIRONMENT_KEYWORDS"]
                elif lookup_domain == "food" and "FOOD_KEYWORDS" in globals():
                    domain_keywords_list = globals()["FOOD_KEYWORDS"]
            except Exception as e:
                print(f"Warning: Error accessing global domain keywords: {str(e)}")

        if not domain_keywords_list:
            default_domain_keywords = {
                "artificial intelligence": [
                    "ai", "artificial intelligence", "machine learning", "deep learning", "neural network",
                    "algorithm", "data", "model", "training", "prediction", "classification", "computer vision",
                    "natural language processing", "nlp", "transformer", "generative ai", "large language model"
                ],
                "cybersecurity": [
                    "cybersecurity", "security", "hack", "breach", "vulnerability", "threat", "malware",
                    "ransomware", "phishing", "encryption", "firewall", "authentication", "zero-day",
                    "exploit", "attack", "defense", "protection", "data breach", "cyber attack"
                ],
                "automotive": [
                    "automotive", "car", "vehicle", "electric vehicle", "ev", "autonomous", "self-driving",
                    "battery", "charging", "engine", "motor", "transmission", "fuel", "emissions",
                    "safety", "driver", "passenger", "manufacturer", "model", "brand"
                ],
                "food": [
                    "food", "cuisine", "dish", "meal", "recipe", "ingredient", "cooking", "chef",
                    "restaurant", "taste", "flavor", "culinary", "nutrition", "diet", "organic",
                    "sustainable", "farm", "agriculture", "produce", "meat", "vegetable", "fruit"
                ],
                "environment": [
                    "environment", "climate", "climate change", "global warming", "carbon", "emission",
                    "pollution", "renewable", "sustainable", "conservation", "ecosystem", "biodiversity",
                    "species", "wildlife", "forest", "ocean", "water", "energy", "green", "recycling"
                ],
                "real estate": [
                    "real estate", "property", "housing", "home", "house", "apartment", "condo",
                    "commercial", "residential", "mortgage", "loan", "interest rate", "market",
                    "buyer", "seller", "agent", "broker", "development", "construction", "investment"
                ],
                "entertainment": [
                    "entertainment", "movie", "film", "television", "tv", "show", "series", "streaming",
                    "actor", "actress", "director", "producer", "celebrity", "music", "song", "artist",
                    "band", "concert", "performance", "box office", "audience", "viewer", "subscriber"
                ]
            }
            domain_keywords_list = default_domain_keywords.get(normalized_domain, [])

        if not domain_keywords_list:
            return keyphrases

        domain_keywords_set = set(kw.lower() for kw in domain_keywords_list)

        if hasattr(self, 'sentence_model'):
            try:
                kp_texts = [kp for kp, _ in keyphrases]
                kp_embeddings = self.sentence_model.encode(kp_texts, show_progress_bar=False)

                max_domain_keywords = 100
                domain_sample = domain_keywords_list[:max_domain_keywords] if len(domain_keywords_list) > max_domain_keywords else domain_keywords_list
                domain_embeddings = self.sentence_model.encode(domain_sample, show_progress_bar=False)

                avg_domain_embedding = np.mean(domain_embeddings, axis=0)

                domain_similarities = cosine_similarity(kp_embeddings, [avg_domain_embedding])

                boosted_keyphrases = []
                for i, (kp, score) in enumerate(keyphrases):
                    kp_lower = kp.lower()

                    boost_factor = 1.0

                    if kp_lower in domain_keywords_set:
                        boost_factor = 1.3
                    elif any(keyword.lower() in kp_lower for keyword in domain_keywords_list):
                        boost_factor = 1.2

                    semantic_boost = 0.1 * domain_similarities[i][0]
                    boost_factor += semantic_boost

                    boosted_score = min(score * boost_factor, 1.0)
                    boosted_keyphrases.append((kp, boosted_score))

                boosted_keyphrases.sort(key=lambda x: x[1], reverse=True)

                return boosted_keyphrases

            except Exception as e:
                print(f"Warning: Error in semantic domain boosting: {str(e)}")

        boosted_keyphrases = []
        for kp, score in keyphrases:
            kp_lower = kp.lower()

            if kp_lower in domain_keywords_set:
                boosted_score = min(score * 1.3, 1.0)
                boosted_keyphrases.append((kp, boosted_score))
            elif any(keyword.lower() in kp_lower for keyword in domain_keywords_list):
                boosted_score = min(score * 1.2, 1.0)
                boosted_keyphrases.append((kp, boosted_score))
            else:
                boosted_keyphrases.append((kp, score))

        boosted_keyphrases.sort(key=lambda x: x[1], reverse=True)

        return boosted_keyphrases

    def remove_redundant_keyphrases(self, scored_keyphrases: List[Tuple[str, float]], base_threshold: float = 0.70, domain: str = "general") -> List[Tuple[str, float]]:

        if not scored_keyphrases:
            return []

        sorted_keyphrases = sorted(scored_keyphrases, key=lambda x: x[1], reverse=True)

        keyphrases = [kp for kp, _ in sorted_keyphrases]
        scores = [score for _, score in sorted_keyphrases]

        embeddings = self.sentence_model.encode(keyphrases, show_progress_bar=False)

        similarity_matrix = cosine_similarity(embeddings)

        to_keep = [True] * len(keyphrases)

        substring_removals = []
        semantic_removals = []
        similarity_debug = []

        for i in range(len(keyphrases)):
            if not to_keep[i]:
                continue

            kp_i = keyphrases[i].lower()
            kp_i_words = set(kp_i.split())

            kp_i_normalized = self._normalize_keyphrase(kp_i)

            i_acronym = self._extract_acronym(kp_i)

            for j in range(i + 1, len(keyphrases)):
                if not to_keep[j]:
                    continue

                kp_j = keyphrases[j].lower()
                kp_j_words = set(kp_j.split())

                kp_j_normalized = self._normalize_keyphrase(kp_j)

                j_acronym = self._extract_acronym(kp_j)

                is_i_in_j = kp_i in kp_j and kp_i != kp_j
                is_j_in_i = kp_j in kp_i and kp_i != kp_j

                if not (is_i_in_j or is_j_in_i):
                    is_i_in_j = kp_i_normalized in kp_j_normalized and kp_i_normalized != kp_j_normalized
                    is_j_in_i = kp_j_normalized in kp_i_normalized and kp_i_normalized != kp_j_normalized

                is_plural_variant = False
                if not (is_i_in_j or is_j_in_i):
                    if kp_i + "s" == kp_j or kp_i + "es" == kp_j:
                        is_plural_variant = True
                        is_i_in_j = True
                    elif kp_j + "s" == kp_i or kp_j + "es" == kp_i:
                        is_plural_variant = True
                        is_j_in_i = True
                    elif (len(kp_i) > 3 and len(kp_j) > 3 and
                        kp_i[:-1] == kp_j[:-1] and
                        similarity_matrix[i, j] > 0.9):
                        is_plural_variant = True
                        if len(kp_i) > len(kp_j):
                            is_j_in_i = True
                        else:
                            is_i_in_j = True

                is_possessive_variant = False
                if not (is_i_in_j or is_j_in_i or is_plural_variant):
                    if kp_i + "'s" == kp_j or kp_i + "s'" == kp_j:
                        is_possessive_variant = True
                        is_i_in_j = True
                    elif kp_j + "'s" == kp_i or kp_j + "s'" == kp_i:
                        is_possessive_variant = True
                        is_j_in_i = True

                is_hyphenated_variant = False
                if not (is_i_in_j or is_j_in_i or is_plural_variant or is_possessive_variant):
                    kp_i_no_hyphen = kp_i.replace('-', ' ')
                    kp_j_no_hyphen = kp_j.replace('-', ' ')

                    if kp_i_no_hyphen == kp_j or kp_i == kp_j_no_hyphen:
                        is_hyphenated_variant = True
                        if '-' in kp_i:
                            is_j_in_i = True
                        else:
                            is_i_in_j = True

                is_acronym_pair = False
                if not (is_i_in_j or is_j_in_i or is_plural_variant or is_possessive_variant or is_hyphenated_variant):
                    if i_acronym and i_acronym == kp_j:
                        is_acronym_pair = True
                        is_j_in_i = True
                    elif j_acronym and j_acronym == kp_i:
                        is_acronym_pair = True
                        is_i_in_j = True

                has_article_difference = False
                if not (is_i_in_j or is_j_in_i or is_plural_variant or is_possessive_variant or is_hyphenated_variant or is_acronym_pair):
                    articles = ["the ", "a ", "an "]
                    for article in articles:
                        if kp_i.startswith(article) and kp_i[len(article):] == kp_j:
                            is_j_in_i = True
                            has_article_difference = True
                            break
                        elif kp_j.startswith(article) and kp_j[len(article):] == kp_i:
                            is_i_in_j = True
                            has_article_difference = True
                            break

                if is_i_in_j or is_j_in_i or is_plural_variant or is_possessive_variant or is_hyphenated_variant or is_acronym_pair:
                    if len(kp_i_words) > 0 and len(kp_j_words) > 0:
                        kp_i_word_set = set(kp_i_words)
                        kp_j_word_set = set(kp_j_words)
                        _ = len(kp_i_word_set.intersection(kp_j_word_set)) / len(kp_i_words)
                        _ = len(kp_i_word_set.intersection(kp_j_word_set)) / len(kp_j_words)

                    score_diff_threshold = 1.05
                    if domain in ["artificial intelligence", "technology", "science", "cybersecurity"]:
                        score_diff_threshold = 1.03
                    elif domain in ["entertainment", "sports"]:
                        score_diff_threshold = 1.08

                    if is_i_in_j:
                        if (len(kp_j) > len(kp_i) * 1.5 and scores[j] > scores[i] * 0.85) or has_article_difference:
                            to_keep[i] = False
                            substring_removals.append((i, j, "substring"))
                        else:
                            if scores[i] > scores[j] * score_diff_threshold:
                                to_keep[j] = False
                                substring_removals.append((j, i, "substring"))
                            else:
                                if domain in ["artificial intelligence", "technology", "science", "cybersecurity"] and scores[i] > 0.5 and scores[j] > 0.5:
                                    pass
                                else:
                                    to_keep[i] = False
                                    substring_removals.append((i, j, "substring"))
                    else:
                        if (len(kp_i) > len(kp_j) * 1.5 and scores[i] > scores[j] * 0.85) or has_article_difference:
                            to_keep[j] = False
                            substring_removals.append((j, i, "substring"))
                        else:
                            if scores[j] > scores[i] * score_diff_threshold:
                                to_keep[i] = False
                                substring_removals.append((i, j, "substring"))
                            else:
                                if domain in ["artificial intelligence", "technology", "science", "cybersecurity"] and scores[i] > 0.5 and scores[j] > 0.5:
                                    pass
                                else:
                                    to_keep[j] = False
                                    substring_removals.append((j, i, "substring"))

        for i in range(len(keyphrases)):
            if not to_keep[i]:
                continue

            kp_i = keyphrases[i].lower()
            kp_i_words = kp_i.split()

            position_factor = max(0.0, min(0.03, 0.03 * (1.0 - i / max(10, len(keyphrases)))))

            length_i = len(kp_i_words)
            length_factor = 0.0
            if length_i == 1:
                length_factor = 0.05
            elif length_i == 2:
                length_factor = 0.02

            domain_factor = 0.0
            if domain in ["technology", "science", "health", "artificial intelligence", "cybersecurity", "automotive"]:
                domain_factor = -0.03
            elif domain in ["entertainment", "sports"]:
                domain_factor = 0.01

            score_factor = 0.0
            if scores[i] > 0.7:
                score_factor = 0.03
            elif scores[i] > 0.5:
                score_factor = 0.01

            adjusted_threshold = base_threshold - position_factor - length_factor + domain_factor - score_factor

            lower_bound = max(0.62, base_threshold - 0.08)
            upper_bound = min(0.82, base_threshold + 0.12)
            adjusted_threshold = max(lower_bound, min(upper_bound, adjusted_threshold))

            for j in range(i + 1, len(keyphrases)):
                if not to_keep[j]:
                    continue

                kp_j = keyphrases[j].lower()
                kp_j_words = kp_j.split()

                similarity = similarity_matrix[i, j]

                if similarity > adjusted_threshold - 0.1:
                    similarity_debug.append((i, j, kp_i, kp_j, similarity, adjusted_threshold))

                if similarity > adjusted_threshold:
                    if domain in ["technology", "science", "artificial intelligence", "cybersecurity", "automotive"]:
                        if (scores[i] > 0.55 and scores[j] > 0.55 and
                            similarity < 0.85 and
                            (length_i > 1 or len(kp_j_words) > 1)):
                            continue

                    if (length_i == 1 and len(kp_j_words) == 1 and
                        kp_i != kp_j and
                        scores[i] > 0.5 and scores[j] > 0.5 and
                        similarity < 0.85):
                        continue

                    if abs(scores[i] - scores[j]) < 0.05 and similarity < 0.78:
                        continue

                    if scores[i] >= scores[j]:
                        to_keep[j] = False
                        semantic_removals.append((j, i, similarity, adjusted_threshold))
                    else:
                        to_keep[i] = False
                        semantic_removals.append((i, j, similarity, adjusted_threshold))
                        break

        filtered_count = sum(1 for k in to_keep if k)
        original_count = len(keyphrases)

        if filtered_count < original_count * 0.6:
            candidates_for_recovery = [(i, scores[i]) for i in range(len(keyphrases)) if not to_keep[i]]
            candidates_for_recovery.sort(key=lambda x: x[1], reverse=True)

            target_recovery = int(original_count * 0.7) - filtered_count

            for i in range(min(target_recovery, len(candidates_for_recovery))):
                idx, _ = candidates_for_recovery[i]
                to_keep[idx] = True
                print(f"Recovered keyphrase: {keyphrases[idx]} (score: {scores[idx]:.4f})")

        filtered_keyphrases = []
        for i in range(len(keyphrases)):
            if to_keep[i]:
                filtered_keyphrases.append((keyphrases[i], scores[i]))

        if hasattr(self, 'debug_redundancy') and self.debug_redundancy:
            print(f"\nRedundancy filtering: {len(sorted_keyphrases)} -> {len(filtered_keyphrases)} keyphrases")
            print(f"Substring removals: {len(substring_removals)}")
            print(f"Semantic similarity removals: {len(semantic_removals)}")

            if substring_removals:
                print("\nSubstring removal examples:")
                for i, j, _ in substring_removals[:3]:
                    print(f"  Removed '{keyphrases[i]}' (score: {scores[i]:.4f}) in favor of '{keyphrases[j]}' (score: {scores[j]:.4f})")

            if semantic_removals:
                print("\nSemantic removal examples:")
                for i, j, sim, adj_threshold in semantic_removals[:3]:
                    print(f"  Removed '{keyphrases[i]}' (score: {scores[i]:.4f}) in favor of '{keyphrases[j]}' (score: {scores[j]:.4f})")
                    print(f"  Similarity: {sim:.4f}, Adjusted threshold: {adj_threshold:.4f}")

        return filtered_keyphrases

    def _normalize_keyphrase(self, keyphrase: str) -> str:

        normalized = keyphrase.lower()

        normalized = re.sub(r"'s$", "", normalized)
        normalized = re.sub(r"s'$", "s", normalized)

        normalized = normalized.replace("-", " ")

        normalized = re.sub(r"^(the|a|an) ", "", normalized)

        normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized

    def _extract_acronym(self, phrase: str) -> str:

        words = phrase.split()

        if len(words) < 2:
            return ""

        if len(phrase) <= 5 and phrase.isupper():
            return ""

        acronym = "".join(word[0] for word in words if word and not word.lower() in ['a', 'an', 'the', 'and', 'or', 'of', 'for', 'in', 'on', 'by', 'to'])

        if len(acronym) >= 2:
            return acronym.upper()

        return ""

    def test_generation_params(self, text: str, debug_output: bool = False) -> Dict[str, List[str]]:

        results = {}

        original_params = {
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'length_penalty': self.length_penalty,
            'max_new_tokens': self.max_new_tokens,
            'prompt_template_idx': self.prompt_template_idx,
            'num_beams': self.num_beams,
            'use_sampling': getattr(self, 'use_sampling', True),
            'num_beam_groups': getattr(self, 'num_beam_groups', 1),
            'diversity_penalty': getattr(self, 'diversity_penalty', 0.0),
            'repetition_penalty': self.repetition_penalty
        }

        test_text = text
        if len(text.split()) > 500:
            test_text = ' '.join(text.split()[:500])

        print("\n" + "="*80)
        print("TESTING GENERATION PARAMETERS")
        print("="*80)

        print("\nTesting Pure Beam Search:")
        print("-"*40)

        beam_search_configs = [
            {"num_beams": 5, "length_penalty": 0.8},
            {"num_beams": 8, "length_penalty": 0.8},
            {"num_beams": 8, "length_penalty": 1.0},
            {"num_beams": 10, "length_penalty": 0.8},
            {"num_beams": 12, "length_penalty": 1.0},
        ]

        for config in beam_search_configs:
            self.num_beams = config["num_beams"]
            self.length_penalty = config["length_penalty"]
            self.use_sampling = False

            if hasattr(self, 'num_beam_groups'):
                delattr(self, 'num_beam_groups')
            if hasattr(self, 'diversity_penalty'):
                delattr(self, 'diversity_penalty')

            for prompt_idx in range(min(3, len(self.PROMPT_TEMPLATES))):
                self.prompt_template_idx = prompt_idx

                keyphrases = self.generate_keyphrases(test_text, debug_output=debug_output)

                param_key = f"beam_search, beams={config['num_beams']}, len_pen={config['length_penalty']}, prompt={prompt_idx}"
                results[param_key] = keyphrases

                print(f"Params: {param_key} → {len(keyphrases)} keyphrases")
                if len(keyphrases) > 0:
                    print(f"  Sample: {', '.join(keyphrases[:3])}" + ("..." if len(keyphrases) > 3 else ""))

        print("\nTesting Diverse Beam Search:")
        print("-"*40)

        diverse_beam_configs = [
            {"num_beams": 8, "num_beam_groups": 4, "diversity_penalty": 1.0},
            {"num_beams": 8, "num_beam_groups": 2, "diversity_penalty": 1.5},
            {"num_beams": 12, "num_beam_groups": 4, "diversity_penalty": 1.0},
            {"num_beams": 12, "num_beam_groups": 3, "diversity_penalty": 1.5},
            {"num_beams": 16, "num_beam_groups": 4, "diversity_penalty": 1.2},
        ]

        for config in diverse_beam_configs:
            if config["num_beams"] % config["num_beam_groups"] != 0:
                continue

            self.num_beams = config["num_beams"]
            self.num_beam_groups = config["num_beam_groups"]
            self.diversity_penalty = config["diversity_penalty"]
            self.use_sampling = False

            for prompt_idx in [0, 1]:
                self.prompt_template_idx = prompt_idx

                keyphrases = self.generate_keyphrases(test_text, debug_output=debug_output)

                param_key = f"diverse_beam, beams={config['num_beams']}, groups={config['num_beam_groups']}, div_pen={config['diversity_penalty']}, prompt={prompt_idx}"
                results[param_key] = keyphrases

                print(f"Params: {param_key} → {len(keyphrases)} keyphrases")
                if len(keyphrases) > 0:
                    print(f"  Sample: {', '.join(keyphrases[:3])}" + ("..." if len(keyphrases) > 3 else ""))

        print("\nTesting Sampling:")
        print("-"*40)

        sampling_configs = [
            {"temperature": 0.7, "top_p": 0.92, "top_k": 50, "repetition_penalty": 1.2},
            {"temperature": 0.8, "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.2},
            {"temperature": 0.9, "top_p": 0.95, "top_k": 100, "repetition_penalty": 1.3},
            {"temperature": 1.0, "top_p": 0.98, "top_k": 100, "repetition_penalty": 1.3},
            {"temperature": 0.6, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.1},
        ]

        for config in sampling_configs:
            self.temperature = config["temperature"]
            self.top_p = config["top_p"]
            self.top_k = config["top_k"]
            self.repetition_penalty = config["repetition_penalty"]
            self.use_sampling = True

            if hasattr(self, 'num_beam_groups'):
                delattr(self, 'num_beam_groups')
            if hasattr(self, 'diversity_penalty'):
                delattr(self, 'diversity_penalty')

            for prompt_idx in range(min(3, len(self.PROMPT_TEMPLATES))):
                self.prompt_template_idx = prompt_idx

                keyphrases = self.generate_keyphrases(test_text, debug_output=debug_output)

                param_key = f"sampling, temp={config['temperature']}, top_p={config['top_p']}, top_k={config['top_k']}, rep_pen={config['repetition_penalty']}, prompt={prompt_idx}"
                results[param_key] = keyphrases

                print(f"Params: {param_key} → {len(keyphrases)} keyphrases")
                if len(keyphrases) > 0:
                    print(f"  Sample: {', '.join(keyphrases[:3])}" + ("..." if len(keyphrases) > 3 else ""))

        best_param_key = None
        best_count = 0
        best_quality_score = 0

        for param_key, keyphrases in results.items():
            if len(keyphrases) < 5:
                continue

            num_keyphrases = len(keyphrases)
            multi_word_count = sum(1 for kp in keyphrases if len(kp.split()) > 1)
            multi_word_percentage = multi_word_count / num_keyphrases if num_keyphrases > 0 else 0
            avg_length = sum(len(kp.split()) for kp in keyphrases) / num_keyphrases

            quantity_score = 0.0
            if 15 <= num_keyphrases <= 25:
                quantity_score = 1.0
            elif 10 <= num_keyphrases < 15:
                quantity_score = 0.8
            elif 25 < num_keyphrases <= 35:
                quantity_score = 0.8
            elif 5 <= num_keyphrases < 10:
                quantity_score = 0.6
            elif num_keyphrases > 35:
                quantity_score = 0.5

            length_score = 1.0 - abs(avg_length - 2.5) / 2.5

            quality_score = (
                (quantity_score * 0.5) +
                (multi_word_percentage * 0.3) +
                (length_score * 0.2)
            )

            if quality_score > best_quality_score:
                best_quality_score = quality_score
                best_param_key = param_key
                best_count = num_keyphrases

        if best_param_key:
            print("\nBest Parameter Combination:")
            print(f"- {best_param_key}")
            print(f"- Generated {best_count} keyphrases with quality score {best_quality_score:.2f}")
            print(f"- Sample: {', '.join(results[best_param_key][:5])}" + ("..." if len(results[best_param_key]) > 5 else ""))
        else:
            print("\nNo optimal parameter combination found.")

        self.temperature = original_params['temperature']
        self.top_p = original_params['top_p']
        self.top_k = original_params['top_k']
        self.length_penalty = original_params['length_penalty']
        self.max_new_tokens = original_params['max_new_tokens']
        self.prompt_template_idx = original_params['prompt_template_idx']
        self.num_beams = original_params['num_beams']
        self.repetition_penalty = original_params['repetition_penalty']

        if 'use_sampling' in original_params:
            self.use_sampling = original_params['use_sampling']

        if original_params['num_beam_groups'] > 1:
            self.num_beam_groups = original_params['num_beam_groups']
            self.diversity_penalty = original_params['diversity_penalty']
        else:
            if hasattr(self, 'num_beam_groups'):
                delattr(self, 'num_beam_groups')
            if hasattr(self, 'diversity_penalty'):
                delattr(self, 'diversity_penalty')

        print("\nParameter testing complete. Original parameters restored.")

        return results

    def examine_raw_output(self, text: str, params: Dict[str, Any] = None) -> None:

        text = self.preprocess_text(text)

        truncated_text = text
        if len(text.split()) > 500:
            truncated_text = ' '.join(text.split()[:500])
            print(f"Note: Text truncated to 500 words for analysis")

        print("\n" + "="*80)
        print("EXAMINING RAW T5 OUTPUT")
        print("="*80)

        if params:
            parameter_sets = [{
                "name": "Custom Parameters",
                "params": params
            }]
        else:
            parameter_sets = [
                {
                    "name": "Default Sampling",
                    "params": {
                        "do_sample": True,
                        "temperature": 0.7,
                        "top_p": 0.92,
                        "top_k": 50,
                        "repetition_penalty": 1.2,
                        "max_new_tokens": self.max_new_tokens
                    }
                },
                {
                    "name": "Pure Beam Search",
                    "params": {
                        "do_sample": False,
                        "num_beams": 8,
                        "repetition_penalty": 1.2,
                        "length_penalty": 0.8,
                        "max_new_tokens": self.max_new_tokens
                    }
                },
                {
                    "name": "Diverse Beam Search",
                    "params": {
                        "do_sample": False,
                        "num_beams": 12,
                        "num_beam_groups": 4,
                        "diversity_penalty": 1.0,
                        "repetition_penalty": 1.2,
                        "length_penalty": 0.8,
                        "max_new_tokens": self.max_new_tokens
                    }
                }
            ]

        for prompt_idx in range(min(2, len(self.PROMPT_TEMPLATES))):
            print(f"\nPROMPT TEMPLATE {prompt_idx}:")
            print("-"*80)

            prompt = self.PROMPT_TEMPLATES[prompt_idx].format(text=truncated_text)

            print("PROMPT:")
            if len(prompt) > 300:
                print(prompt[:300] + "...")
            else:
                print(prompt)

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            for param_set in parameter_sets:
                print(f"\n{param_set['name']}:")
                print("-"*40)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **param_set['params'],
                        early_stopping=True,
                    )

                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                print("RAW OUTPUT:")
                print(generated_text)
                print("-"*40)

                keyphrases = self.extract_keyphrases_from_generated_text(generated_text)
                print(f"EXTRACTED KEYPHRASES ({len(keyphrases)}):")
                for kp in keyphrases:
                    print(f"- {kp}")

                print("\nANALYSIS:")
                if len(keyphrases) < 5:
                    print("- LOW KEYPHRASE COUNT: Model is not generating enough candidates")

                    if "," not in generated_text:
                        print("- NO COMMAS: Model is not using comma separation as expected")

                    if any(example in generated_text.lower() for example in ["example", "article:"]):
                        print("- EXAMPLE REPETITION: Model is repeating example text from prompt")

                    if len(generated_text) < 50:
                        print("- SHORT OUTPUT: Model is generating very little text")
                elif len(keyphrases) > 30:
                    print("- HIGH KEYPHRASE COUNT: Model is generating many candidates")

                    similar_count = 0
                    for i in range(len(keyphrases)):
                        for j in range(i+1, len(keyphrases)):
                            if keyphrases[i].lower() in keyphrases[j].lower() or keyphrases[j].lower() in keyphrases[i].lower():
                                similar_count += 1

                    if similar_count > len(keyphrases) * 0.3:
                        print(f"- HIGH REDUNDANCY: {similar_count} similar keyphrases detected")
                    else:
                        print("- GOOD DIVERSITY: Low redundancy in generated keyphrases")

                if len(keyphrases) > 0:
                    multi_word_count = sum(1 for kp in keyphrases if len(kp.split()) > 1)
                    multi_word_percentage = multi_word_count / len(keyphrases)
                    print(f"- Multi-word phrases: {multi_word_percentage:.1%}")

                    avg_length = sum(len(kp.split()) for kp in keyphrases) / len(keyphrases)
                    print(f"- Average phrase length: {avg_length:.1f} words")

        print("\n" + "="*80)

    def optimize_generation_params(self, text: str) -> Dict[str, Any]:

        print("Optimizing generation parameters...")

        param_results = self.test_generation_params(text, debug_output=False)

        best_score = -1
        best_params_key = None
        best_keyphrases = []

        for param_key, keyphrases in param_results.items():
            if not keyphrases:
                continue

            num_keyphrases = len(keyphrases)

            if num_keyphrases < 5:
                continue

            multi_word_count = sum(1 for kp in keyphrases if len(kp.split()) > 1)
            multi_word_percentage = multi_word_count / num_keyphrases if num_keyphrases > 0 else 0

            avg_length = sum(len(kp.split()) for kp in keyphrases) / num_keyphrases

            length_score = 1.0 - abs(avg_length - 2.5) / 2.5

            quantity_score = 0.0
            if 10 <= num_keyphrases <= 20:
                quantity_score = 1.0
            elif 5 <= num_keyphrases < 10:
                quantity_score = 0.7
            elif 20 < num_keyphrases <= 30:
                quantity_score = 0.8
            elif num_keyphrases > 30:
                quantity_score = 0.5

            final_score = (
                (quantity_score * 0.4) +
                (multi_word_percentage * 0.4) +
                (length_score * 0.2)
            )

            if final_score > best_score:
                best_score = final_score
                best_params_key = param_key
                best_keyphrases = keyphrases

        if best_params_key:
            print(f"Best parameters: {best_params_key}")
            print(f"Generated {len(best_keyphrases)} keyphrases with score {best_score:.2f}")

            if "beam_search" in best_params_key:
                params = {}
                parts = best_params_key.split(", ")
                for part in parts:
                    if "beams=" in part:
                        params["num_beams"] = int(part.split("=")[1])
                    elif "len_pen=" in part:
                        params["length_penalty"] = float(part.split("=")[1])
                    elif "prompt=" in part:
                        params["prompt_template_idx"] = int(part.split("=")[1])

                params["use_sampling"] = False
                return params

            elif "diverse_beam" in best_params_key:
                params = {}
                parts = best_params_key.split(", ")
                for part in parts:
                    if "beams=" in part:
                        params["num_beams"] = int(part.split("=")[1])
                    elif "groups=" in part:
                        params["num_beam_groups"] = int(part.split("=")[1])
                    elif "div_pen=" in part:
                        params["diversity_penalty"] = float(part.split("=")[1])
                    elif "prompt=" in part:
                        params["prompt_template_idx"] = int(part.split("=")[1])

                params["use_sampling"] = False
                return params

            elif "sampling" in best_params_key:
                params = {}
                parts = best_params_key.split(", ")
                for part in parts:
                    if "temp=" in part:
                        params["temperature"] = float(part.split("=")[1])
                    elif "top_p=" in part:
                        params["top_p"] = float(part.split("=")[1])
                    elif "top_k=" in part:
                        params["top_k"] = int(part.split("=")[1])
                    elif "prompt=" in part:
                        params["prompt_template_idx"] = int(part.split("=")[1])

                params["use_sampling"] = True
                return params

        print("No optimal parameters found, using defaults")
        return {
            "temperature": 0.7,
            "top_p": 0.92,
            "top_k": 50,
            "length_penalty": 0.8,
            "prompt_template_idx": 0,
            "use_sampling": True,
            "num_beams": 5
        }

    def add_domain_specific_prompts(self):

        self.DOMAIN_PROMPTS = {
            "technology": """Extract keyphrases from this technology article.
Focus on technical terms, innovations, and specific technologies mentioned.
Include both single words and multi-word phrases (2-4 words).
Prioritize specific technical concepts rather than general terms.
Provide 15-20 keyphrases, separated by commas.

Text: {text}

Keyphrases:""",

            "business": """Extract keyphrases from this business article.
Focus on business terms, company names, financial concepts, and market trends.
Include both single words and multi-word phrases (2-4 words).
Prioritize specific business concepts rather than general terms.
Provide 15-20 keyphrases, separated by commas.

Text: {text}

Keyphrases:""",

            "health": """Extract keyphrases from this health article.
Focus on medical terms, health conditions, treatments, and healthcare concepts.
Include both single words and multi-word phrases (2-4 words).
Prioritize specific health concepts rather than general terms.
Provide 15-20 keyphrases, separated by commas.

Text: {text}

Keyphrases:""",

            "politics": """Extract keyphrases from this political article.
Focus on political terms, policy concepts, government entities, and political figures.
Include both single words and multi-word phrases (2-4 words).
Prioritize specific political concepts rather than general terms.
Provide 15-20 keyphrases, separated by commas.

Text: {text}

Keyphrases:""",

            "sports": """Extract keyphrases from this sports article.
Focus on sports terms, team names, athlete names, and sporting events.
Include both single words and multi-word phrases (2-4 words).
Prioritize specific sports concepts rather than general terms.
Provide 15-20 keyphrases, separated by commas.

Text: {text}

Keyphrases:""",

            "entertainment": """Extract keyphrases from this entertainment article.
Focus on entertainment terms, movie/show titles, celebrity names, and media concepts.
Include both single words and multi-word phrases (2-4 words).
Prioritize specific entertainment concepts rather than general terms.
Provide 15-20 keyphrases, separated by commas.

Text: {text}

Keyphrases:""",

            "science": """Extract keyphrases from this scientific article.
Focus on scientific terms, research concepts, methodologies, and findings.
Include both single words and multi-word phrases (2-4 words).
Prioritize specific scientific concepts rather than general terms.
Provide 15-20 keyphrases, separated by commas.

Text: {text}

Keyphrases:""",

            "environment": """Extract keyphrases from this environmental article.
Focus on environmental terms, ecological concepts, sustainability issues, and climate topics.
Include both single words and multi-word phrases (2-4 words).
Prioritize specific environmental concepts rather than general terms.
Provide 15-20 keyphrases, separated by commas.

Text: {text}

Keyphrases:""",

            "world": """Extract keyphrases from this world news article.
Focus on international terms, country names, global issues, and geopolitical concepts.
Include both single words and multi-word phrases (2-4 words).
Prioritize specific international concepts rather than general terms.
Provide 15-20 keyphrases, separated by commas.

Text: {text}

Keyphrases:"""
        }

    def evaluate_domain_detection(self, articles_with_domains: List[Tuple[str, str]]) -> Dict[str, float]:

        results = {
            "total": len(articles_with_domains),
            "correct": 0,
            "zsl_used": 0,
            "keyword_used": 0,
            "domains": {}
        }

        for text, true_domain in articles_with_domains:
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

            zsl_used = True

            try:
                detected_domain = self.detect_domain(text)

                if not hasattr(self, 'zero_shot_classifier') or detected_domain != self._keyword_based_domain_detection(text):
                    zsl_used = False
            except Exception:
                detected_domain = self._keyword_based_domain_detection(text)
                zsl_used = False

            sys.stdout.close()
            sys.stdout = original_stdout

            if detected_domain == true_domain:
                results["correct"] += 1

            if zsl_used:
                results["zsl_used"] += 1
            else:
                results["keyword_used"] += 1

            if true_domain not in results["domains"]:
                results["domains"][true_domain] = {"total": 0, "correct": 0}

            results["domains"][true_domain]["total"] += 1
            if detected_domain == true_domain:
                results["domains"][true_domain]["correct"] += 1

        results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
        results["zsl_percentage"] = results["zsl_used"] / results["total"] if results["total"] > 0 else 0

        for domain in results["domains"]:
            domain_total = results["domains"][domain]["total"]
            domain_correct = results["domains"][domain]["correct"]
            results["domains"][domain]["accuracy"] = domain_correct / domain_total if domain_total > 0 else 0

        return results

    def train_domain_classifier(self, articles_with_domains: List[Tuple[str, str]]) -> None:

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.svm import LinearSVC
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import train_test_split, GridSearchCV
            from sklearn.metrics import classification_report, confusion_matrix
            import numpy as np
            import pandas as pd

            print("\n" + "="*80)
            print("TRAINING FALLBACK DOMAIN CLASSIFIER")
            print("="*80)

            texts = [text for text, _ in articles_with_domains]
            domains = [domain for _, domain in articles_with_domains]

            domain_counts = {}
            for domain in domains:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

            print("Domain distribution:")
            for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"- {domain}: {count} articles ({count/len(domains):.1%})")

            X_train, X_test, y_train, y_test = train_test_split(
                texts, domains, test_size=0.2, random_state=42, stratify=domains
            )

            print(f"\nTraining on {len(X_train)} articles, testing on {len(X_test)} articles")

            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('clf', LinearSVC(class_weight='balanced'))
            ])

            param_grid = {
                'tfidf__max_features': [3000, 5000, 7000],
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'clf__C': [0.1, 1.0, 10.0]
            }

            print("Performing grid search for hyperparameter tuning...")
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_
            print(f"\nBest parameters: {best_params}")

            self.domain_classifier = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=best_params['tfidf__max_features'],
                    ngram_range=best_params['tfidf__ngram_range']
                )),
                ('clf', LinearSVC(C=best_params['clf__C'], class_weight='balanced'))
            ])

            self.domain_classifier.fit(X_train, y_train)

            y_pred = self.domain_classifier.predict(X_test)
            accuracy = np.mean(y_pred == y_test)

            print(f"\nDomain classifier trained with accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            print("\nConfusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            domains_unique = sorted(set(domains))
            cm_df = pd.DataFrame(cm, index=domains_unique, columns=domains_unique)
            print(cm_df)

            self.has_domain_classifier = True

            print("\nDomain classifier training complete. It will now be used as a fallback when zero-shot classification has low confidence.")

        except Exception as e:
            print(f"Error training domain classifier: {str(e)}")
            print("Domain classifier will not be available")
            self.has_domain_classifier = False

    def detect_domain_with_bart(self, text: str) -> Tuple[str, float]:

        if not hasattr(self, 'bart_classifier'):
            try:
                from transformers import pipeline
                self.bart_classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=0 if torch.cuda.is_available() else -1
                )
                print("Initialized BART-large for improved topic classification")
            except Exception as e:
                print(f"Error initializing BART-large: {str(e)}")
                return None, 0.0

        candidate_domains = [
            "artificial intelligence", "cybersecurity", "automotive",
            "food", "environment", "real estate", "entertainment",

            "technology", "business", "health", "politics", "sports",
            "science", "world", "education", "travel", "space",
            "agriculture", "mental health"
        ]

        try:
            truncated_text = text[:2000]

            result = self.bart_classifier(
                truncated_text,
                candidate_domains,
                hypothesis_template="This text is about {}."
            )

            priority_domains = [
                "artificial intelligence", "automotive", "cybersecurity",
                "food", "environment", "real estate", "entertainment"
            ]

            priority_domain_found = False
            priority_domain = None
            priority_score = 0.0

            for i in range(min(5, len(result['labels']))):
                domain = result['labels'][i]
                score = result['scores'][i]

                if domain in priority_domains and score > priority_score:
                    priority_domain = domain
                    priority_score = score
                    priority_domain_found = True

            if priority_domain_found:
                print(f"Priority domain found in BART results: {priority_domain} (Score: {priority_score:.4f})")
                return priority_domain, priority_score

            return result['labels'][0], result['scores'][0]
        except Exception as e:
            print(f"Error in BART domain detection: {str(e)}")
            return None, 0.0

    def detect_domain(self, text: str, original_domain: str = None) -> str:

        if original_domain:
            print(f"Using provided original domain: {original_domain}")
            print("IMPORTANT: Using original domain from input data instead of detecting domain")
            return original_domain

        try:
            domain, confidence = self.detect_domain_with_bart(text)
            if domain and confidence >= 0.5:
                print(f"Domain detected using BART: {domain} (confidence: {confidence:.4f})")

                domain_mapping = {
                    "machine learning": "artificial intelligence",
                    "deep learning": "artificial intelligence",
                    "natural language processing": "artificial intelligence",
                    "computer vision": "artificial intelligence",

                    "network security": "cybersecurity",
                    "information security": "cybersecurity",
                    "data security": "cybersecurity",

                    "electric vehicles": "automotive",
                    "self-driving cars": "automotive",
                    "autonomous vehicles": "automotive",

                    "nutrition": "food",
                    "cooking": "food",
                    "cuisine": "food",

                    "climate": "environment",
                    "sustainability": "environment",
                    "renewable energy": "environment",

                    "housing": "real estate",
                    "property": "real estate",
                    "mortgage": "real estate",

                    "film": "entertainment",
                    "movie": "entertainment",
                    "television": "entertainment",
                    "streaming": "entertainment",
                    "gaming": "entertainment"
                }

                if domain in domain_mapping:
                    mapped_domain = domain_mapping[domain]
                    print(f"Mapped specific domain '{domain}' to broader category '{mapped_domain}'")
                    return mapped_domain

                return domain
        except Exception as e:
            print(f"Error using BART for domain detection: {str(e)}")

        if not hasattr(self, 'zero_shot_classifier'):
            try:
                from transformers import pipeline
                print("Initializing mDeBERTa zero-shot domain classifier (backup method)...")
                device = 0 if self.device == "cuda" else -1
                self.zero_shot_classifier = pipeline(
                    "zero-shot-classification",
                    model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                    device=device
                )
                print("mDeBERTa zero-shot classifier initialized successfully")
            except Exception as e:
                print(f"Error initializing mDeBERTa zero-shot classifier: {str(e)}")
                print("Falling back to alternative domain detection methods")

                if hasattr(self, 'has_domain_classifier') and self.has_domain_classifier:
                    try:
                        domain = self.domain_classifier.predict([text])[0]
                        print(f"Domain detected using trained classifier: {domain}")
                        return domain
                    except Exception:
                        pass

                if 'DomainFallbackClassifier' in globals():
                    try:
                        if not hasattr(self, 'domain_fallback_classifier'):
                            self.domain_fallback_classifier = globals()['DomainFallbackClassifier']()
                            print("Initialized domain fallback classifier from global namespace")

                        domain, confidence = self.domain_fallback_classifier.detect_domain(text)
                        if domain is not None and confidence >= 0.3:
                            print(f"Domain detected using fallback classifier: {domain} (confidence: {confidence:.4f})")
                            return domain
                        else:
                            print(f"Fallback classifier result: {domain} (confidence: {confidence:.4f} - too low)")
                    except Exception as e:
                        print(f"Error using fallback classifier: {str(e)}")

                return self._keyword_based_domain_detection(text)

        candidate_domains = [
            "artificial intelligence", "cybersecurity", "automotive",
            "food", "environment", "real estate", "entertainment",

            "technology", "business", "health", "politics", "sports",
            "science", "world", "education", "travel", "space",
            "agriculture", "mental health"
        ]

        domain_mapping = {
            "machine learning": "artificial intelligence",
            "deep learning": "artificial intelligence",
            "natural language processing": "artificial intelligence",
            "computer vision": "artificial intelligence",

            "network security": "cybersecurity",
            "information security": "cybersecurity",
            "data security": "cybersecurity",

            "electric vehicles": "automotive",
            "self-driving cars": "automotive",
            "autonomous vehicles": "automotive",

            "nutrition": "food",
            "cooking": "food",
            "cuisine": "food",

            "climate": "environment",
            "sustainability": "environment",
            "renewable energy": "environment",

            "housing": "real estate",
            "property": "real estate",
            "mortgage": "real estate",

            "film": "entertainment",
            "movie": "entertainment",
            "television": "entertainment",
            "streaming": "entertainment",
            "gaming": "entertainment"
        }

        try:
            truncated_text = text[:1500]

            domain_thresholds = {
                'artificial intelligence': 0.35,
                'cybersecurity': 0.32,
                'automotive': 0.30,
                'food': 0.28,
                'environment': 0.28,
                'real estate': 0.32,
                'entertainment': 0.35,

                'politics': 0.5,
                'technology': 0.45,
                'business': 0.45,
                'sports': 0.45,
                'science': 0.4,
                'health': 0.35,
                'world': 0.4,
                'education': 0.4,
                'space': 0.3,
                'travel': 0.25,
                'agriculture': 0.25,
                'mental health': 0.25
            }

            default_threshold = 0.4

            result = self.zero_shot_classifier(
                truncated_text,
                candidate_domains,
                hypothesis_template="This text is about {}."
            )

            priority_domains = [
                "artificial intelligence", "automotive", "cybersecurity",
                "food", "environment", "real estate", "entertainment"
            ]

            priority_domain_found = False
            priority_domain = None
            priority_score = 0.0

            for i in range(min(5, len(result['labels']))):
                domain = result['labels'][i]
                score = result['scores'][i]

                if domain in priority_domains and score > priority_score:
                    priority_domain = domain
                    priority_score = score
                    priority_domain_found = True

            if priority_domain_found:
                top_domain = priority_domain
                top_score = priority_score
                print(f"Priority domain found in top 5: {top_domain} (Score: {top_score:.4f})")
            else:
                top_domain = result['labels'][0]
                top_score = result['scores'][0]

            print("\nZero-Shot Domain Detection Results:")
            print("-" * 40)
            print(f"Top domain: {top_domain} (Score: {top_score:.4f})")


            threshold = domain_thresholds.get(top_domain, default_threshold)
            print(f"Threshold for '{top_domain}': {threshold:.2f}")

            if top_score >= threshold:
                if top_domain in domain_mapping:
                    mapped_domain = domain_mapping[top_domain]
                    print(f"Mapped specific domain '{top_domain}' to broader category '{mapped_domain}'")
                    return mapped_domain

                return top_domain

            for i in range(1, min(3, len(result['labels']))):
                domain = result['labels'][i]
                score = result['scores'][i]
                domain_threshold = domain_thresholds.get(domain, default_threshold)

                if score >= domain_threshold:
                    print(f"Using secondary domain '{domain}' (Score: {score:.4f}, Threshold: {domain_threshold:.2f})")

                    if domain in domain_mapping:
                        mapped_domain = domain_mapping[domain]
                        print(f"Mapped specific domain '{domain}' to broader category '{mapped_domain}'")
                        return mapped_domain

                    return domain

            print(f"Low confidence ({top_score:.4f}) for domain '{top_domain}'.")

            if hasattr(self, 'has_domain_classifier') and self.has_domain_classifier:
                try:
                    domain = self.domain_classifier.predict([text])[0]
                    print(f"Domain detected using trained classifier: {domain}")
                    return domain
                except Exception as e:
                    print(f"Error using trained classifier: {str(e)}")

            if top_score >= 0.25:
                print(f"Score is above 0.25, using detected domain despite being below threshold")

                if top_domain in domain_mapping:
                    mapped_domain = domain_mapping[top_domain]
                    print(f"Mapped specific domain '{top_domain}' to broader category '{mapped_domain}'")
                    return mapped_domain

                return top_domain

            if 'DomainFallbackClassifier' in globals():
                try:
                    if not hasattr(self, 'domain_fallback_classifier'):
                        self.domain_fallback_classifier = globals()['DomainFallbackClassifier']()
                        print("Initialized domain fallback classifier from global namespace")

                    domain, confidence = self.domain_fallback_classifier.detect_domain(text)
                    if domain is not None and confidence >= 0.3:
                        print(f"Domain detected using fallback classifier: {domain} (confidence: {confidence:.4f})")
                        return domain
                    else:
                        print(f"Fallback classifier result: {domain} (confidence: {confidence:.4f} - too low)")
                except Exception as e:
                    print(f"Error using fallback classifier: {str(e)}")

            print("Applying domain confidence booster before keyword fallback")
            boosted_domain = self._apply_domain_confidence_booster(text, top_domain, top_score)
            if boosted_domain:
                print(f"Domain boosted to '{boosted_domain}' based on high-precision keywords")

                if boosted_domain in domain_mapping:
                    mapped_domain = domain_mapping[boosted_domain]
                    print(f"Mapped boosted domain '{boosted_domain}' to broader category '{mapped_domain}'")
                    return mapped_domain

                return boosted_domain

            print("Using keyword-based domain detection as fallback")
            keyword_domain = self._keyword_based_domain_detection(text)

            if keyword_domain != top_domain:
                if top_score >= 0.2:
                    print(f"Using ZSL domain '{top_domain}' (score: {top_score:.4f}) instead of keyword domain '{keyword_domain}'")

                    if top_domain in domain_mapping:
                        mapped_domain = domain_mapping[top_domain]
                        print(f"Mapped specific domain '{top_domain}' to broader category '{mapped_domain}'")
                        return mapped_domain

                    return top_domain

            return keyword_domain

        except Exception as e:
            print(f"Error in mDeBERTa domain detection: {str(e)}")

            if hasattr(self, 'has_domain_classifier') and self.has_domain_classifier:
                try:
                    domain = self.domain_classifier.predict([text])[0]
                    print(f"Domain detected using trained classifier: {domain}")
                    return domain
                except Exception as e:
                    print(f"Error using trained classifier: {str(e)}")

            if 'DomainFallbackClassifier' in globals():
                try:
                    if not hasattr(self, 'domain_fallback_classifier'):
                        self.domain_fallback_classifier = globals()['DomainFallbackClassifier']()
                        print("Initialized domain fallback classifier from global namespace")

                    domain, confidence = self.domain_fallback_classifier.detect_domain(text)
                    if domain is not None and confidence >= 0.3:
                        print(f"Domain detected using fallback classifier: {domain} (confidence: {confidence:.4f})")
                        return domain
                    else:
                        print(f"Fallback classifier result: {domain} (confidence: {confidence:.4f} - too low)")
                except Exception as e:
                    print(f"Error using fallback classifier: {str(e)}")

            print("Falling back to keyword-based domain detection")
            return self._keyword_based_domain_detection(text)

    def train_domain_classifier(self, articles_with_domains: List[Tuple[str, str]]) -> None:

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.svm import LinearSVC
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import train_test_split

            print("\n" + "="*80)
            print("TRAINING FALLBACK DOMAIN CLASSIFIER")
            print("="*80)

            texts = [text for text, _ in articles_with_domains]
            domains = [domain for _, domain in articles_with_domains]

            domain_counts = {}
            for domain in domains:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

            print("Domain distribution:")
            for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"- {domain}: {count} articles ({count/len(domains):.1%})")

            X_train, X_test, y_train, y_test = train_test_split(
                texts, domains, test_size=0.2, random_state=42, stratify=domains
            )

            print(f"\nTraining on {len(X_train)} articles, testing on {len(X_test)} articles")

            self.domain_classifier = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('clf', LinearSVC(C=1.0, class_weight='balanced'))
            ])

            self.domain_classifier.fit(X_train, y_train)

            accuracy = self.domain_classifier.score(X_test, y_test)
            print(f"Domain classifier trained with accuracy: {accuracy:.4f}")

            self.has_domain_classifier = True

            print("\nDomain classifier training complete. It will now be used as a fallback when zero-shot classification has low confidence.")

        except Exception as e:
            print(f"Error training domain classifier: {str(e)}")
            print("Domain classifier will not be available")
            self.has_domain_classifier = False

    def _apply_domain_confidence_booster(self, text: str, candidate_domain: str, candidate_score: float) -> str:

        text_lower = text.lower()

        high_precision_keywords = {
            "artificial intelligence": [
                "neural network", "deep learning", "machine learning", "natural language processing",
                "computer vision", "reinforcement learning", "transformer model", "large language model",
                "generative ai", "gpt", "bert", "llm", "diffusion model", "stable diffusion"
            ],
            "cybersecurity": [
                "ransomware", "zero-day", "vulnerability", "data breach", "phishing", "malware",
                "ddos attack", "firewall", "encryption", "cyber attack", "penetration testing",
                "security breach", "threat actor", "infosec", "cybersecurity"
            ],
            "automotive": [
                "electric vehicle", "autonomous driving", "self-driving", "ev charging",
                "battery electric", "hybrid vehicle", "automotive industry", "automaker",
                "vehicle emissions", "car manufacturer", "automotive supplier", "powertrain"
            ],
            "food": [
                "culinary", "cuisine", "ingredient", "recipe", "restaurant", "chef", "food safety",
                "nutrition", "dietary", "food processing", "food production", "food industry",
                "food supply", "food security", "food system", "food waste", "sustainable food"
            ],
            "environment": [
                "climate change", "global warming", "carbon emissions", "greenhouse gas",
                "renewable energy", "sustainability", "biodiversity", "conservation",
                "environmental impact", "pollution", "ecosystem", "carbon footprint",
                "environmental protection", "climate crisis", "climate action"
            ],
            "real estate": [
                "property market", "housing market", "real estate market", "commercial property",
                "residential property", "mortgage rate", "home buyer", "property value",
                "real estate investment", "property development", "real estate agent",
                "housing affordability", "rental market", "property price"
            ],
            "entertainment": [
                "box office", "streaming service", "film industry", "movie studio", "television series",
                "entertainment industry", "media company", "production studio", "streaming platform",
                "theatrical release", "content creator", "media streaming", "subscription service"
            ]
        }

        if candidate_score >= 0.2:
            if candidate_domain in high_precision_keywords:
                for keyword in high_precision_keywords[candidate_domain]:
                    if keyword in text_lower:
                        print(f"Found high-precision keyword '{keyword}' confirming domain '{candidate_domain}'")
                        return candidate_domain

        for domain, keywords in high_precision_keywords.items():
            matches = [keyword for keyword in keywords if keyword in text_lower]
            if len(matches) >= 3:
                print(f"Found {len(matches)} high-precision keywords for domain '{domain}': {', '.join(matches[:3])}...")
                return domain
            elif len(matches) >= 2 and candidate_score < 0.3:
                print(f"Found {len(matches)} high-precision keywords for domain '{domain}': {', '.join(matches)}")
                return domain

        return None

    def _keyword_based_domain_detection(self, text: str) -> str:

        try:
            from domain_keywords_expansion import (
                TECHNOLOGY_KEYWORDS, BUSINESS_KEYWORDS, HEALTH_KEYWORDS, POLITICS_KEYWORDS,
                SPORTS_KEYWORDS, ENTERTAINMENT_KEYWORDS, SCIENCE_KEYWORDS, ENVIRONMENT_KEYWORDS,
                WORLD_KEYWORDS, EDUCATION_KEYWORDS, FOOD_KEYWORDS, TRAVEL_KEYWORDS,
                AUTOMOTIVE_KEYWORDS, REAL_ESTATE_KEYWORDS, CYBERSECURITY_KEYWORDS,
                AI_KEYWORDS, SPACE_KEYWORDS, AGRICULTURE_KEYWORDS, MENTAL_HEALTH_KEYWORDS
            )
            print("Loaded comprehensive domain keyword lists")
        except ImportError:
            print("Expanded domain keyword lists not found, using default lists")

        domain_keywords = {
            "technology": [
                "technology", "software", "hardware", "AI", "artificial intelligence", "machine learning",
                "digital", "data", "algorithm", "computing", "chip", "processor", "semiconductor",
                "app", "application", "device", "platform", "cloud", "network", "internet", "cyber",
                "tech", "innovation", "startup", "smartphone", "computer", "robot", "automation",
                "programming", "code", "developer", "encryption", "blockchain", "cryptocurrency",
                "virtual reality", "augmented reality", "IoT", "5G", "broadband", "server", "database"
            ],
            "business": [
                "business", "company", "market", "industry", "corporate", "firm", "enterprise",
                "CEO", "executive", "management", "strategy", "investment", "investor", "stock",
                "finance", "financial", "economy", "economic", "revenue", "profit", "earnings",
                "growth", "startup", "entrepreneur", "venture capital", "merger", "acquisition",
                "IPO", "quarterly", "fiscal", "shareholder", "stakeholder", "consumer", "customer",
                "client", "retail", "wholesale", "supply chain", "logistics", "manufacturing", "product"
            ],
            "health": [
                "health", "medical", "medicine", "doctor", "physician", "patient", "hospital",
                "clinic", "treatment", "therapy", "disease", "condition", "symptom", "diagnosis",
                "drug", "medication", "pharmaceutical", "surgery", "vaccine", "vaccination",
                "immune", "immunity", "virus", "bacterial", "infection", "cancer", "diabetes",
                "heart disease", "obesity", "mental health", "psychology", "psychiatry", "wellness",
                "nutrition", "diet", "exercise", "fitness", "healthcare", "clinical", "trial"
            ],
            "politics": [
                "politics", "government", "policy", "election", "vote", "voter", "campaign",
                "candidate", "president", "senator", "representative", "congress", "parliament",
                "democrat", "republican", "liberal", "conservative", "progressive", "legislation",
                "law", "bill", "regulation", "federal", "state", "local", "national", "international",
                "administration", "White House", "Supreme Court", "constitutional", "democracy",
                "democratic", "authoritarian", "populist", "political", "politician", "party"
            ],
            "sports": [
                "sports", "game", "team", "player", "coach", "championship", "tournament", "match",
                "season", "league", "score", "win", "lose", "victory", "defeat", "record", "fan",
                "stadium", "arena", "field", "court", "NFL", "NBA", "MLB", "NHL", "soccer", "football",
                "basketball", "baseball", "hockey", "tennis", "golf", "Olympic", "Olympics", "medal",
                "athlete", "athletic", "performance", "contract", "draft", "trade", "championship"
            ],
            "entertainment": [
                "entertainment", "movie", "film", "television", "TV", "show", "series", "episode",
                "actor", "actress", "director", "producer", "Hollywood", "celebrity", "star", "fame",
                "award", "Oscar", "Emmy", "Grammy", "Golden Globe", "box office", "streaming",
                "Netflix", "Disney", "HBO", "Amazon", "Hulu", "music", "album", "song", "artist",
                "band", "concert", "tour", "performance", "release", "review", "critic", "rating"
            ],
            "science": [
                "science", "scientific", "research", "researcher", "study", "discovery", "experiment",
                "laboratory", "lab", "theory", "hypothesis", "data", "analysis", "evidence", "journal",
                "publication", "peer-review", "academic", "university", "college", "professor",
                "student", "education", "degree", "PhD", "STEM", "physics", "chemistry", "biology",
                "astronomy", "geology", "mathematics", "statistics", "engineering", "technology",
                "innovation", "breakthrough", "grant", "funding", "collaboration"
            ],
            "environment": [
                "environment", "environmental", "climate", "climate change", "global warming",
                "carbon", "emission", "greenhouse gas", "pollution", "renewable", "sustainable",
                "sustainability", "conservation", "ecosystem", "biodiversity", "species", "wildlife",
                "habitat", "forest", "deforestation", "ocean", "marine", "water", "air quality",
                "energy", "solar", "wind", "hydroelectric", "fossil fuel", "coal", "natural gas",
                "oil", "recycling", "waste", "plastic", "EPA", "regulation", "policy", "agreement",
                "Paris Agreement", "drought", "flood", "extreme weather", "temperature"
            ],
            "world": [
                "international", "global", "world", "foreign", "country", "nation", "region",
                "United Nations", "UN", "NATO", "EU", "European Union", "treaty", "agreement",
                "diplomacy", "diplomatic", "embassy", "ambassador", "foreign minister",
                "foreign policy", "trade", "tariff", "sanction", "conflict", "war", "peace",
                "military", "army", "defense", "security", "terrorism", "refugee", "immigration",
                "border", "summit", "bilateral", "multilateral", "alliance", "coalition",
                "Middle East", "Asia", "Africa", "Europe", "North America", "South America"
            ],
            "food": [
                "food", "cuisine", "dish", "meal", "recipe", "ingredient", "cooking", "baking",
                "chef", "restaurant", "dining", "menu", "taste", "flavor", "culinary", "kitchen",
                "breakfast", "lunch", "dinner", "appetizer", "dessert", "snack", "beverage", "drink"
            ],
            "agriculture": [
                "agriculture", "farming", "farm", "crop", "harvest", "cultivation", "livestock",
                "agricultural", "farmer", "ranch", "plantation", "orchard", "vineyard", "greenhouse",
                "soil", "irrigation", "fertilizer", "pesticide", "organic farming", "sustainable agriculture"
            ],
            "education": [
                "education", "school", "university", "college", "student", "teacher", "professor",
                "classroom", "curriculum", "course", "degree", "learning", "teaching", "academic",
                "study", "research", "lecture", "assignment", "exam", "test", "grade", "graduation"
            ],
            "mental health": [
                "mental health", "psychology", "psychiatry", "therapy", "counseling", "depression",
                "anxiety", "stress", "trauma", "disorder", "wellbeing", "emotional health", "mindfulness",
                "self-care", "mental illness", "psychiatric", "psychological", "therapist", "counselor"
            ]
        }

        try:
            if 'TECHNOLOGY_KEYWORDS' in globals():
                domain_keywords["technology"] = globals()['TECHNOLOGY_KEYWORDS']
                domain_keywords["business"] = globals()['BUSINESS_KEYWORDS']
                domain_keywords["health"] = globals()['HEALTH_KEYWORDS']
                domain_keywords["politics"] = globals()['POLITICS_KEYWORDS']
                domain_keywords["sports"] = globals()['SPORTS_KEYWORDS']
                domain_keywords["entertainment"] = globals()['ENTERTAINMENT_KEYWORDS']
                domain_keywords["science"] = globals()['SCIENCE_KEYWORDS']
                domain_keywords["environment"] = globals()['ENVIRONMENT_KEYWORDS']
                domain_keywords["world"] = globals()['WORLD_KEYWORDS']
                domain_keywords["education"] = globals()['EDUCATION_KEYWORDS']
                domain_keywords["food"] = globals()['FOOD_KEYWORDS']
                domain_keywords["travel"] = globals()['TRAVEL_KEYWORDS']
                domain_keywords["automotive"] = globals()['AUTOMOTIVE_KEYWORDS']
                domain_keywords["real estate"] = globals()['REAL_ESTATE_KEYWORDS']
                domain_keywords["cybersecurity"] = globals()['CYBERSECURITY_KEYWORDS']
                domain_keywords["artificial intelligence"] = globals()['AI_KEYWORDS']
                domain_keywords["space"] = globals()['SPACE_KEYWORDS']
                domain_keywords["agriculture"] = globals()['AGRICULTURE_KEYWORDS']
                domain_keywords["mental health"] = globals()['MENTAL_HEALTH_KEYWORDS']

                print("Successfully updated domain keywords from global namespace")
            elif 'TECHNOLOGY_KEYWORDS' in locals():
                domain_keywords["technology"] = TECHNOLOGY_KEYWORDS
                domain_keywords["business"] = BUSINESS_KEYWORDS
                domain_keywords["health"] = HEALTH_KEYWORDS
                domain_keywords["politics"] = POLITICS_KEYWORDS
                domain_keywords["sports"] = SPORTS_KEYWORDS
                domain_keywords["entertainment"] = ENTERTAINMENT_KEYWORDS
                domain_keywords["science"] = SCIENCE_KEYWORDS
                domain_keywords["environment"] = ENVIRONMENT_KEYWORDS
                domain_keywords["world"] = WORLD_KEYWORDS
                domain_keywords["education"] = EDUCATION_KEYWORDS
                domain_keywords["food"] = FOOD_KEYWORDS
                domain_keywords["travel"] = TRAVEL_KEYWORDS
                domain_keywords["automotive"] = AUTOMOTIVE_KEYWORDS
                domain_keywords["real estate"] = REAL_ESTATE_KEYWORDS
                domain_keywords["cybersecurity"] = CYBERSECURITY_KEYWORDS
                domain_keywords["artificial intelligence"] = AI_KEYWORDS
                domain_keywords["space"] = SPACE_KEYWORDS
                domain_keywords["agriculture"] = AGRICULTURE_KEYWORDS
                domain_keywords["mental health"] = MENTAL_HEALTH_KEYWORDS

                print("Successfully updated domain keywords from local namespace")
            else:
                print("No expanded domain keywords found in global or local namespace, using default keywords")

            if 'TECHNOLOGY_KEYWORDS' in globals() or 'TECHNOLOGY_KEYWORDS' in locals():
                print("\nDomain keyword statistics:")
                print(f"Technology: {len(domain_keywords['technology'])} terms")
                print(f"Business: {len(domain_keywords['business'])} terms")
                print(f"Health: {len(domain_keywords['health'])} terms")
                print(f"Politics: {len(domain_keywords['politics'])} terms")
                print(f"Sports: {len(domain_keywords['sports'])} terms")
                print(f"Entertainment: {len(domain_keywords['entertainment'])} terms")
                print(f"Science: {len(domain_keywords['science'])} terms")
                print(f"Environment: {len(domain_keywords['environment'])} terms")
                print(f"World: {len(domain_keywords['world'])} terms")
                print(f"Education: {len(domain_keywords['education'])} terms")
                print(f"Food: {len(domain_keywords['food'])} terms")
                print(f"Travel: {len(domain_keywords['travel'])} terms")
                print(f"Automotive: {len(domain_keywords['automotive'])} terms")
                print(f"Real Estate: {len(domain_keywords['real estate'])} terms")
                print(f"Cybersecurity: {len(domain_keywords['cybersecurity'])} terms")
                print(f"AI: {len(domain_keywords['artificial intelligence'])} terms")
                print(f"Space: {len(domain_keywords['space'])} terms")
                print(f"Agriculture: {len(domain_keywords['agriculture'])} terms")
                print(f"Mental Health: {len(domain_keywords['mental health'])} terms")
        except Exception as e:
            print(f"Error applying expanded keywords: {str(e)}")
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        domain_scores = {}
        domain_keyword_matches = {}

        for domain, keywords in domain_keywords.items():
            score = 0
            keyword_matches = []

            for keyword in keywords:
                keyword_lower = keyword.lower()

                exact_count = text_lower.count(keyword_lower)
                if exact_count > 0:
                    match_score = exact_count * 2
                    score += match_score
                    keyword_matches.append((keyword, match_score))
                    continue

                if ' ' in keyword_lower:
                    keyword_parts = keyword_lower.split()
                    if all(part in words for part in keyword_parts):
                        score += 1
                        keyword_matches.append((keyword, 1))
                        continue

                    if len(keyword_parts) > 2 and sum(1 for part in keyword_parts if part in words) >= len(keyword_parts) * 0.7:
                        score += 0.5
                        keyword_matches.append((keyword, 0.5))

                elif keyword_lower in words:
                    score += 1
                    keyword_matches.append((keyword, 1))

            domain_scores[domain] = score
            domain_keyword_matches[domain] = keyword_matches

        print("\nKeyword-Based Domain Detection Results")
        print("-" * 40)
        for domain, score in sorted(domain_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"{domain}: {score:.4f}")

            if score > 0 and len(domain_keyword_matches[domain]) > 0:
                sorted_matches = sorted(domain_keyword_matches[domain], key=lambda x: x[1], reverse=True)
                top_matches = sorted_matches[:5]
                print(f"  Top keywords: {', '.join([f'{kw} ({count})' for kw, count in top_matches])}")
                if len(sorted_matches) > 5:
                    print(f"  ...and {len(sorted_matches) - 5} more matches")
        print("-" * 40)

        if max(domain_scores.values(), default=0) > 0:
            detected_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            print(f"Detected domain: {detected_domain}\n")
            return detected_domain
        else:
            print("No domain detected, using 'general'\n")
            return "general"