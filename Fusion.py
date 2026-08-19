import torch
import numpy as np
import re
import gc
from typing import List, Dict, Tuple, Set, Optional, Union, Any
from tqdm.auto import tqdm
import time
import sys
import os

class FusionKeyphraseExtractor:

    def __init__(
        self,
        abstractive_extractor: AbstractiveKeyphraseExtractor = None,
        extractive_extractor: HybridExtractiveKeyphraseExtractor = None,
        use_gpu: bool = True,
        abstractive_weight: float = 0.5,
        extractive_weight: float = 0.5,
        redundancy_threshold: float = 0.72,
        min_score: float = 0.1,
    ):

        if abstractive_extractor is None:
            print("Initializing abstractive extractor with optimized parameters...")
            self.abstractive_extractor = AbstractiveKeyphraseExtractor(
                 model_name="google/flan-t5-large",
                use_gpu=True,
                max_length=512,
                num_beams=24,
                top_k=100,
                top_p=0.95,
                temperature=0.8,
                repetition_penalty=1.5,
                length_penalty=1.0,
                max_new_tokens=300,
                prompt_template_idx=0,
                use_fp16=True,
                max_input_length=1024,
                use_chunking=True,
                post_process=True,
                filter_stopwords=True,
                min_phrase_length=1,
                max_phrase_length=5,
                prioritize_multi_word=True,
                use_lemmatization=True,
                use_ner=True,
                ner_model="en_core_web_sm",
                use_sampling=False,
                num_beam_groups=6,
                diversity_penalty=1.7,
                use_mdeberta_domain_detection=False
            )
        else:
            self.abstractive_extractor = abstractive_extractor

        if extractive_extractor is None:
            print("Initializing extractive extractor with optimized parameters...")
            self.extractive_extractor = HybridExtractiveKeyphraseExtractor(
             model_name="all-mpnet-base-v2",
                use_gpu=True,
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
                    'keybert': 0.38,
                    'multipartiterank': 0.27,
                    'yake': 0.18,
                    'textrank': 0.17
                }
                )
        else:
            self.extractive_extractor = extractive_extractor

        self.use_gpu = use_gpu
        self.abstractive_weight = abstractive_weight
        self.extractive_weight = extractive_weight
        self.redundancy_threshold = redundancy_threshold
        self.min_score = min_score

        print("Fusion Keyphrase Extractor initialized")

    def normalize_keyphrase_count(self, keyphrases: List[Tuple[str, float]], text: str, domain: str,
                               target_min: int = 12, target_max: int = 18, debug: bool = False) -> List[Tuple[str, float]]:
        if not keyphrases:
            return []

        word_count = len(text.split())

        sorted_keyphrases = sorted(keyphrases, key=lambda x: x[1], reverse=True)
        current_count = len(sorted_keyphrases)

        domain_quality_thresholds = {
            "artificial intelligence": 0.60,
            "technology": 0.55,
            "cybersecurity": 0.55,
            "automotive": 0.50,
            "food": 0.40,
            "environment": 0.45,
            "real estate": 0.40,
            "entertainment": 0.35,
            "default": 0.40
        }

        normalized_domain = domain.lower().strip()

        domain_mapping = {
            "ai": "artificial intelligence",
            "tech": "technology",
            "cyber": "cybersecurity",
            "auto": "automotive",
            "cars": "automotive",
            "nutrition": "food",
            "climate": "environment",
            "housing": "real estate",
            "property": "real estate",
            "media": "entertainment",
            "movie": "entertainment",
            "film": "entertainment"
        }

        for key, mapped_domain in domain_mapping.items():
            if key in normalized_domain:
                normalized_domain = mapped_domain
                break

        quality_threshold = domain_quality_thresholds.get(normalized_domain, domain_quality_thresholds["default"])
        print(f"Using domain '{normalized_domain}' with quality threshold: {quality_threshold:.2f}")

        if word_count < 300:
            quality_threshold *= 0.9
        elif word_count > 600:
            quality_threshold *= 1.1

        if debug:
            print(f"\nNormalizing keyphrase count for {domain} domain:")
            print(f"- Current count: {current_count}")
            print(f"- Target range: {target_min}-{target_max}")
            print(f"- Quality threshold: {quality_threshold:.2f}")
            print(f"- Text length: {word_count} words")

        if current_count < target_min:
            if debug:
                print(f"Too few keyphrases ({current_count} < {target_min}), lowering quality threshold")

            remaining_needed = target_min - current_count

            for reduction_factor in [0.9, 0.8, 0.7, 0.6, 0.5]:
                adjusted_threshold = quality_threshold * reduction_factor

                additional_keyphrases = []

                abstractive_candidates = self.abstractive_extractor.extract_keyphrases_with_scores(text)
                for kp, score in abstractive_candidates:
                    kp_lower = kp.lower()
                    if not any(kp_lower == existing_kp.lower() for existing_kp, _ in sorted_keyphrases):
                        if score >= adjusted_threshold:
                            additional_keyphrases.append((kp, score))

                if len(additional_keyphrases) >= remaining_needed:
                    additional_keyphrases = sorted(additional_keyphrases, key=lambda x: x[1], reverse=True)
                    sorted_keyphrases.extend(additional_keyphrases[:remaining_needed])

                    if debug:
                        print(f"Added {len(additional_keyphrases[:remaining_needed])} keyphrases with threshold {adjusted_threshold:.2f}")

                    break

            sorted_keyphrases = sorted(sorted_keyphrases, key=lambda x: x[1], reverse=True)

        elif current_count > target_max:
            if debug:
                print(f"Too many keyphrases ({current_count} > {target_max}), applying stricter filtering")

            diversity_weight = 0.3

            sorted_keyphrases = self.abstractive_extractor.select_diverse_keyphrases(
                sorted_keyphrases,
                target_max,
                diversity_weight
            )

            if debug:
                print(f"Selected {len(sorted_keyphrases)} diverse keyphrases")

        final_keyphrases = [(kp, score) for kp, score in sorted_keyphrases if score >= quality_threshold * 0.8]

        if len(final_keyphrases) < target_min and len(sorted_keyphrases) > len(final_keyphrases):
            remaining_needed = target_min - len(final_keyphrases)

            filtered_out = [(kp, score) for kp, score in sorted_keyphrases if (kp, score) not in final_keyphrases]
            filtered_out = sorted(filtered_out, key=lambda x: x[1], reverse=True)

            final_keyphrases.extend(filtered_out[:remaining_needed])

            final_keyphrases = sorted(final_keyphrases, key=lambda x: x[1], reverse=True)

            if debug:
                print(f"Added back {min(remaining_needed, len(filtered_out))} keyphrases to reach minimum target")

        if debug:
            print(f"Final normalized count: {len(final_keyphrases)}")

        return final_keyphrases

    def extract_keyphrases_with_scores(self, text: str, debug: bool = False, original_domain: str = None) -> List[Tuple[str, float]]:
        start_time = time.time()

        if debug:
            print("\n" + "="*80)
            print("FUSION KEYPHRASE EXTRACTION")
            print("="*80)

        if debug:
            print("\nStep 1: Generating independent candidate lists...")

        abstractive_start = time.time()
        abstractive_candidates = self.abstractive_extractor.extract_keyphrases_with_scores(text, original_domain=original_domain)
        abstractive_time = time.time() - abstractive_start

        if debug:
            print(f"Generated {len(abstractive_candidates)} abstractive candidates in {abstractive_time:.2f}s")
            if abstractive_candidates:
                print("Sample abstractive candidates:")
                for kp, score in abstractive_candidates[:3]:
                    print(f"- {kp}: {score:.4f}")
                if len(abstractive_candidates) > 3:
                    print(f"... and {len(abstractive_candidates) - 3} more")

        extractive_start = time.time()
        extractive_candidates = self.extractive_extractor.extract_keyphrases_with_scores(text)
        extractive_time = time.time() - extractive_start

        if debug:
            print(f"Generated {len(extractive_candidates)} extractive candidates in {extractive_time:.2f}s")
            if extractive_candidates:
                print("Sample extractive candidates:")
                for kp, score in extractive_candidates[:3]:
                    print(f"- {kp}: {score:.4f}")
                if len(extractive_candidates) > 3:
                    print(f"... and {len(extractive_candidates) - 3} more")

        if debug:
            print("\nStep 2: Pooling all candidates...")

        pooled_candidates = []
        pooled_candidates.extend([(kp, score, 'abstractive') for kp, score in abstractive_candidates])
        pooled_candidates.extend([(kp, score, 'extractive') for kp, score in extractive_candidates])

        unique_keyphrases = {}
        for kp, score, source in pooled_candidates:
            kp_lower = kp.lower()
            if kp_lower not in unique_keyphrases:
                unique_keyphrases[kp_lower] = (kp, score, source)
            else:
                if score > unique_keyphrases[kp_lower][1]:
                    unique_keyphrases[kp_lower] = (kp, score, source)

        pooled_candidates = list(unique_keyphrases.values())

        if debug:
            print(f"Pooled {len(pooled_candidates)} unique candidates")

        if debug:
            print("\nStep 3: Uniform re-scoring of all candidates...")

        candidate_phrases = [kp for kp, _, _ in pooled_candidates]

        rescoring_start = time.time()
        rescored_candidates = self.abstractive_extractor.score_keyphrases_by_relevance(candidate_phrases, text)

        boosted_multiword = []
        for kp, score in rescored_candidates:
            word_count = len(kp.split())
            if word_count > 1:
                if word_count == 2:
                    length_boost = 0.25
                    words = kp.lower().split()
                    if len(set(words)) == len(words):
                        length_boost += 0.05
                elif word_count == 3:
                    length_boost = 0.35
                    words = kp.lower().split()
                    if len(set(words)) == len(words):
                        length_boost += 0.07
                elif word_count >= 4:
                    length_boost = 0.45
                    words = kp.lower().split()
                    if len(set(words)) == len(words):
                        length_boost += 0.10

                boosted_score = min(score * (1 + length_boost), 1.0)
                boosted_multiword.append((kp, boosted_score))
            else:
                boosted_multiword.append((kp, score))

        rescored_candidates = boosted_multiword

        if original_domain:
            domain = original_domain
            if debug:
                print(f"Using original domain: {domain}")
        else:
            domain = self.abstractive_extractor.detect_domain(text)
            if debug:
                print(f"Detected domain: {domain}")

        boosted_candidates = self.abstractive_extractor.boost_domain_specific_concepts(rescored_candidates, domain)

        coherent_candidates = self.abstractive_extractor.enhance_semantic_coherence(boosted_candidates, text)

        diverse_candidates = self.abstractive_extractor.enhance_semantic_diversity(coherent_candidates, text)
        rescoring_time = time.time() - rescoring_start

        if debug:
            print(f"Re-scored, boosted, enhanced, and diversified {len(diverse_candidates)} candidates in {rescoring_time:.2f}s")
            if diverse_candidates:
                print("Sample diverse candidates:")
                for kp, score in sorted(diverse_candidates, key=lambda x: x[1], reverse=True)[:3]:
                    print(f"- {kp}: {score:.4f}")

        if debug:
            print("\nStep 4: Applying enhanced redundancy filtering...")

        redundancy_start = time.time()

        domain_redundancy_thresholds = {
            "artificial intelligence": 0.45,
            "cybersecurity": 0.45,
            "automotive": 0.45,
            "food": 0.45,
            "environment": 0.45,
            "real estate": 0.45,
            "entertainment": 0.45,
            "default": 0.50
        }

        domain_threshold = domain_redundancy_thresholds.get(
            domain.lower(), domain_redundancy_thresholds["default"]
        )

        if debug:
            print(f"Using domain-specific redundancy threshold for '{domain}': {domain_threshold:.2f}")

        domain_descriptions = {
            "artificial intelligence": "Artificial intelligence, machine learning, neural networks, deep learning, computer vision, natural language processing, algorithms, data science, predictive models",
            "cybersecurity": "Cybersecurity, information security, network security, data protection, encryption, threats, vulnerabilities, malware, hacking, privacy, authentication",
            "automotive": "Automotive industry, vehicles, cars, electric vehicles, autonomous driving, transportation, engines, mobility, fuel efficiency, safety systems",
            "food": "Food, cuisine, cooking, recipes, ingredients, nutrition, culinary arts, gastronomy, restaurants, dietary, flavors, meals",
            "environment": "Environment, climate change, sustainability, renewable energy, conservation, pollution, ecosystems, biodiversity, green technology, carbon emissions",
            "real estate": "Real estate, property, housing market, commercial property, residential homes, mortgages, investment properties, land development, construction",
            "entertainment": "Entertainment, movies, music, television, streaming, gaming, celebrities, media, performances, shows, films, production, audience"
        }

        normalized_domain = domain.lower().strip()
        domain_mapping = {
            "ai": "artificial intelligence",
            "tech": "technology",
            "cyber": "cybersecurity",
            "auto": "automotive",
            "cars": "automotive",
            "nutrition": "food",
            "climate": "environment",
            "housing": "real estate",
            "property": "real estate",
            "media": "entertainment",
            "movie": "entertainment",
            "film": "entertainment"
        }

        for key, mapped_domain in domain_mapping.items():
            if key in normalized_domain:
                normalized_domain = mapped_domain
                break

        domain_boosted = []

        domain_desc = domain_descriptions.get(normalized_domain, "")
        if not domain_desc and normalized_domain in domain_mapping:
            mapped_domain = domain_mapping[normalized_domain]
            domain_desc = domain_descriptions.get(mapped_domain, "")

        contrastive_domains = []
        for d_name, d_desc in domain_descriptions.items():
            if d_name != normalized_domain and d_name != mapped_domain:
                contrastive_domains.append(d_desc)

        if len(contrastive_domains) > 3:
            contrastive_domains = contrastive_domains[:3]

        if domain_desc:
            try:
                all_texts_to_encode = [domain_desc] + contrastive_domains
                all_embeddings = self.abstractive_extractor.sentence_model.encode(all_texts_to_encode)
                domain_embedding = all_embeddings[0]
                contrastive_embeddings = all_embeddings[1:] if len(all_embeddings) > 1 else []

                keyphrase_texts = [kp for kp, _ in diverse_candidates]
                keyphrase_embeddings = self.abstractive_extractor.sentence_model.encode(keyphrase_texts)

                for i, ((kp, score), kp_embedding) in enumerate(zip(diverse_candidates, keyphrase_embeddings)):
                    target_similarity = np.dot(domain_embedding, kp_embedding) / (np.linalg.norm(domain_embedding) * np.linalg.norm(kp_embedding))

                    contrastive_score = 0
                    if contrastive_embeddings:
                        other_similarities = []
                        for contrast_emb in contrastive_embeddings:
                            contrast_sim = np.dot(contrast_emb, kp_embedding) / (np.linalg.norm(contrast_emb) * np.linalg.norm(kp_embedding))
                            other_similarities.append(contrast_sim)
                        avg_other_sim = sum(other_similarities) / len(other_similarities)

                        contrastive_score = max(0, target_similarity - avg_other_sim)

                    combined_similarity = 0.7 * target_similarity + 0.3 * contrastive_score

                    if combined_similarity > 0.45:
                        domain_boost = (combined_similarity - 0.45) * 0.85 + 0.15
                        old_score = score
                        boosted_score = min(score * (1 + domain_boost), 1.0)
                        domain_boosted.append((kp, boosted_score))
                        if debug:
                            print(f"DEBUG: Boosted domain-relevant term '{kp}' (similarity: {combined_similarity:.4f}, contrastive: {contrastive_score:.4f}): {old_score:.4f} -> {boosted_score:.4f}")
                        continue

                    domain_boosted.append((kp, score))

                if debug:
                    print(f"Applied domain embedding similarity boosting for '{normalized_domain}' domain")
            except Exception as e:
                print(f"Warning: Domain embedding similarity failed: {str(e)}. Using no domain boost.")
                domain_boosted = diverse_candidates
        else:
            domain_boosted = diverse_candidates
            if debug:
                print(f"No domain description available for '{normalized_domain}'. Using no domain boost.")

        diverse_candidates = domain_boosted

        try:
            keyphrase_texts = [kp for kp, _ in diverse_candidates]
            if len(keyphrase_texts) > 1:
                keyphrase_embeddings = self.abstractive_extractor.sentence_model.encode(keyphrase_texts)

                similarity_matrix = np.zeros((len(keyphrase_texts), len(keyphrase_texts)))
                for i in range(len(keyphrase_texts)):
                    for j in range(len(keyphrase_texts)):
                        if i != j:
                            similarity = np.dot(keyphrase_embeddings[i], keyphrase_embeddings[j]) / \
                                        (np.linalg.norm(keyphrase_embeddings[i]) * np.linalg.norm(keyphrase_embeddings[j]))
                            similarity_matrix[i, j] = similarity

                centrality_scores = np.sum(similarity_matrix, axis=1) / (len(keyphrase_texts) - 1)

                related_candidates = []

                for i, ((kp1, score1), emb1) in enumerate(zip(diverse_candidates, keyphrase_embeddings)):
                    if len(kp1.split()) > 1:
                        centrality_boost = centrality_scores[i] * 0.25
                        word_count_factor = min(0.05 * len(kp1.split()), 0.15)
                        total_boost = centrality_boost + word_count_factor

                        old_score = score1
                        boosted_score = min(score1 * (1 + total_boost), 1.0)
                        related_candidates.append((kp1, boosted_score))
                        if debug:
                            print(f"DEBUG: Boosted central multi-word term '{kp1}' (centrality: {centrality_scores[i]:.4f}): {old_score:.4f} -> {boosted_score:.4f}")

                clusters = []
                assigned = set()

                for i in range(len(keyphrase_texts)):
                    if i in assigned:
                        continue

                    cluster = [i]
                    assigned.add(i)

                    for j in range(len(keyphrase_texts)):
                        if j not in assigned and similarity_matrix[i, j] > 0.65:
                            cluster.append(j)
                            assigned.add(j)

                    if len(cluster) > 1:
                        clusters.append(cluster)

                for cluster in clusters:
                    cluster_centrality = [centrality_scores[i] for i in cluster]
                    most_central_idx = cluster[np.argmax(cluster_centrality)]

                    kp, score = diverse_candidates[most_central_idx]
                    cluster_size_boost = min(0.05 * len(cluster), 0.15)
                    old_score = score
                    boosted_score = min(score * (1 + 0.15 + cluster_size_boost), 1.0)
                    related_candidates.append((kp, boosted_score))
                    if debug:
                        print(f"DEBUG: Boosted cluster representative '{kp}' (cluster size: {len(cluster)}): {old_score:.4f} -> {boosted_score:.4f}")

                for related, score in related_candidates:
                    for i, (kp, old_score) in enumerate(diverse_candidates):
                        if kp == related:
                            diverse_candidates[i] = (kp, max(old_score, score))
                            if debug and max(old_score, score) > old_score:
                                print(f"DEBUG: Updated semantically related term '{kp}' score: {old_score:.4f} -> {max(old_score, score):.4f}")
                            break

                if debug and related_candidates:
                    print(f"Applied advanced semantic relationship boosting to {len(related_candidates)} keyphrases")
        except Exception as e:
            print(f"Warning: Advanced semantic relationship detection failed: {str(e)}. Skipping this step.")

        if debug:
            print(f"Applied domain-specific term boosting for '{normalized_domain}' domain")

        deduplicated_candidates = self.abstractive_extractor.remove_redundant_keyphrases(
            diverse_candidates,
            base_threshold=domain_threshold,
            domain=domain
        )
        redundancy_time = time.time() - redundancy_start

        if debug:
            print(f"Removed redundancy, {len(deduplicated_candidates)} candidates remaining ({redundancy_time:.2f}s)")

        if debug:
            print("\nStep 5: Applying final quality and generic filtering...")

        filtering_start = time.time()
        filtered_candidates = self.abstractive_extractor.filter_generic_terms(deduplicated_candidates, domain, text)

        quality_filtered_keyphrases = self.abstractive_extractor.filter_and_select_by_quality(text, domain, filtered_candidates)
        filtering_time = time.time() - filtering_start

        if debug:
            print(f"Applied quality filtering, {len(quality_filtered_keyphrases)} keyphrases remaining ({filtering_time:.2f}s)")

        if debug:
            print("\nStep 6: Applying keyphrase count normalization...")

        normalization_start = time.time()
        final_keyphrases = self.normalize_keyphrase_count(
            quality_filtered_keyphrases,
            text,
            domain,
            target_min=12,
            target_max=18,
            debug=debug
        )
        normalization_time = time.time() - normalization_start

        if debug:
            print(f"Applied count normalization in {normalization_time:.2f}s")

        total_time = time.time() - start_time

        if debug:
            print("\n" + "="*80)
            print("FUSION EXTRACTION COMPLETE")
            print("="*80)
            print(f"Total processing time: {total_time:.2f}s")
            print(f"Final keyphrases ({len(final_keyphrases)}):")
            for kp, score in final_keyphrases:
                print(f"- {kp}: {score:.4f}")

        return final_keyphrases

    def extract_keyphrases(self, text: str, original_domain: str = None) -> List[str]:
        
        scored_keyphrases = self.extract_keyphrases_with_scores(text, original_domain=original_domain)
        keyphrases = [kp for kp, _ in scored_keyphrases]
        print(f"Returning {len(keyphrases)} keyphrases from extract_keyphrases")
        return keyphrases

    def benchmark(self, articles: List[str], num_articles: int = 5) -> Dict[str, Any]:
        
        print("\n" + "="*80)
        print("BENCHMARKING FUSION EXTRACTOR")
        print("="*80)

        test_articles = articles[:num_articles] if len(articles) > num_articles else articles
        print(f"Testing on {len(test_articles)} articles")

        results = {
            "abstractive": {
                "counts": [],
                "times": [],
                "multi_word_percentages": [],
                "avg_lengths": [],
                "keyphrases": []
            },
            "extractive": {
                "counts": [],
                "times": [],
                "multi_word_percentages": [],
                "avg_lengths": [],
                "keyphrases": []
            },
            "fusion": {
                "counts": [],
                "times": [],
                "multi_word_percentages": [],
                "avg_lengths": [],
                "keyphrases": []
            }
        }

        for i, article in enumerate(test_articles):
            print(f"\nTesting article {i+1}/{len(test_articles)}")

            print("Testing abstractive extractor...")
            abstractive_start = time.time()
            abstractive_keyphrases = self.abstractive_extractor.extract_keyphrases_with_scores(article)
            abstractive_time = time.time() - abstractive_start

            abstractive_count = len(abstractive_keyphrases)
            abstractive_multi_word = sum(1 for kp, _ in abstractive_keyphrases if len(kp.split()) > 1)
            abstractive_multi_word_percentage = abstractive_multi_word / abstractive_count if abstractive_count > 0 else 0
            abstractive_avg_length = sum(len(kp.split()) for kp, _ in abstractive_keyphrases) / abstractive_count if abstractive_count > 0 else 0

            results["abstractive"]["counts"].append(abstractive_count)
            results["abstractive"]["times"].append(abstractive_time)
            results["abstractive"]["multi_word_percentages"].append(abstractive_multi_word_percentage)
            results["abstractive"]["avg_lengths"].append(abstractive_avg_length)
            results["abstractive"]["keyphrases"].append(abstractive_keyphrases)

            print(f"Abstractive: {abstractive_count} keyphrases in {abstractive_time:.2f}s")
            print(f"Multi-word: {abstractive_multi_word_percentage:.1%}, Avg length: {abstractive_avg_length:.1f}")

            print("Testing extractive extractor...")
            extractive_start = time.time()
            extractive_keyphrases = self.extractive_extractor.extract_keyphrases_with_scores(article)
            extractive_time = time.time() - extractive_start

            extractive_count = len(extractive_keyphrases)
            extractive_multi_word = sum(1 for kp, _ in extractive_keyphrases if len(kp.split()) > 1)
            extractive_multi_word_percentage = extractive_multi_word / extractive_count if extractive_count > 0 else 0
            extractive_avg_length = sum(len(kp.split()) for kp, _ in extractive_keyphrases) / extractive_count if extractive_count > 0 else 0

            results["extractive"]["counts"].append(extractive_count)
            results["extractive"]["times"].append(extractive_time)
            results["extractive"]["multi_word_percentages"].append(extractive_multi_word_percentage)
            results["extractive"]["avg_lengths"].append(extractive_avg_length)
            results["extractive"]["keyphrases"].append(extractive_keyphrases)

            print(f"Extractive: {extractive_count} keyphrases in {extractive_time:.2f}s")
            print(f"Multi-word: {extractive_multi_word_percentage:.1%}, Avg length: {extractive_avg_length:.1f}")

            print("Testing fusion extractor...")
            fusion_start = time.time()
            fusion_keyphrases = self.extract_keyphrases_with_scores(article)
            fusion_time = time.time() - fusion_start

            fusion_count = len(fusion_keyphrases)
            fusion_multi_word = sum(1 for kp, _ in fusion_keyphrases if len(kp.split()) > 1)
            fusion_multi_word_percentage = fusion_multi_word / fusion_count if fusion_count > 0 else 0
            fusion_avg_length = sum(len(kp.split()) for kp, _ in fusion_keyphrases) / fusion_count if fusion_count > 0 else 0

            results["fusion"]["counts"].append(fusion_count)
            results["fusion"]["times"].append(fusion_time)
            results["fusion"]["multi_word_percentages"].append(fusion_multi_word_percentage)
            results["fusion"]["avg_lengths"].append(fusion_avg_length)
            results["fusion"]["keyphrases"].append(fusion_keyphrases)

            print(f"Fusion: {fusion_count} keyphrases in {fusion_time:.2f}s")
            print(f"Multi-word: {fusion_multi_word_percentage:.1%}, Avg length: {fusion_avg_length:.1f}")

        for extractor in ["abstractive", "extractive", "fusion"]:
            results[extractor]["avg_count"] = sum(results[extractor]["counts"]) / len(results[extractor]["counts"])
            results[extractor]["avg_time"] = sum(results[extractor]["times"]) / len(results[extractor]["times"])
            results[extractor]["avg_multi_word"] = sum(results[extractor]["multi_word_percentages"]) / len(results[extractor]["multi_word_percentages"])
            results[extractor]["avg_length"] = sum(results[extractor]["avg_lengths"]) / len(results[extractor]["avg_lengths"])

        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        print(f"{'Extractor':<15} {'Avg Count':<10} {'Avg Time':<10} {'Multi-word':<10} {'Avg Length':<10}")
        print("-"*55)

        for extractor in ["abstractive", "extractive", "fusion"]:
            print(f"{extractor:<15} {results[extractor]['avg_count']:<10.1f} {results[extractor]['avg_time']:<10.2f}s {results[extractor]['avg_multi_word']:<10.1%} {results[extractor]['avg_length']:<10.1f}")

        print("\nKeyphrase Overlap Analysis:")

        abstractive_extractive_overlap = []
        abstractive_fusion_overlap = []
        extractive_fusion_overlap = []

        for i in range(len(test_articles)):
            abstractive_kps = set(kp.lower() for kp, _ in results["abstractive"]["keyphrases"][i])
            extractive_kps = set(kp.lower() for kp, _ in results["extractive"]["keyphrases"][i])
            fusion_kps = set(kp.lower() for kp, _ in results["fusion"]["keyphrases"][i])

            if abstractive_kps and extractive_kps:
                overlap = len(abstractive_kps.intersection(extractive_kps)) / len(abstractive_kps.union(extractive_kps))
                abstractive_extractive_overlap.append(overlap)

            if abstractive_kps and fusion_kps:
                overlap = len(abstractive_kps.intersection(fusion_kps)) / len(abstractive_kps.union(fusion_kps))
                abstractive_fusion_overlap.append(overlap)

            if extractive_kps and fusion_kps:
                overlap = len(extractive_kps.intersection(fusion_kps)) / len(extractive_kps.union(fusion_kps))
                extractive_fusion_overlap.append(overlap)

        if abstractive_extractive_overlap:
            avg_overlap = sum(abstractive_extractive_overlap) / len(abstractive_extractive_overlap)
            print(f"Abstractive-Extractive overlap: {avg_overlap:.1%}")

        if abstractive_fusion_overlap:
            avg_overlap = sum(abstractive_fusion_overlap) / len(abstractive_fusion_overlap)
            print(f"Abstractive-Fusion overlap: {avg_overlap:.1%}")

        if extractive_fusion_overlap:
            avg_overlap = sum(extractive_fusion_overlap) / len(extractive_fusion_overlap)
            print(f"Extractive-Fusion overlap: {avg_overlap:.1%}")

        print("\nFusion Contribution Analysis:")
        fusion_unique = []
        abstractive_unique = []
        extractive_unique = []

        for i in range(len(test_articles)):
            abstractive_kps = set(kp.lower() for kp, _ in results["abstractive"]["keyphrases"][i])
            extractive_kps = set(kp.lower() for kp, _ in results["extractive"]["keyphrases"][i])
            fusion_kps = set(kp.lower() for kp, _ in results["fusion"]["keyphrases"][i])

            fusion_unique_kps = fusion_kps - (abstractive_kps.union(extractive_kps))
            abstractive_unique_kps = abstractive_kps - fusion_kps
            extractive_unique_kps = extractive_kps - fusion_kps

            if fusion_kps:
                fusion_unique.append(len(fusion_unique_kps) / len(fusion_kps))

            if abstractive_kps:
                abstractive_unique.append(len(abstractive_unique_kps) / len(abstractive_kps))

            if extractive_kps:
                extractive_unique.append(len(extractive_unique_kps) / len(extractive_kps))

        if fusion_unique:
            avg_unique = sum(fusion_unique) / len(fusion_unique)
            print(f"Fusion unique keyphrases: {avg_unique:.1%}")

        if abstractive_unique:
            avg_unique = sum(abstractive_unique) / len(abstractive_unique)
            print(f"Abstractive keyphrases not in fusion: {avg_unique:.1%}")

        if extractive_unique:
            avg_unique = sum(extractive_unique) / len(extractive_unique)
            print(f"Extractive keyphrases not in fusion: {avg_unique:.1%}")

        return results

    def analyze_article(self, text: str) -> Dict[str, Any]:
        print("\n" + "="*80)
        print("DETAILED ARTICLE ANALYSIS")
        print("="*80)

        print("\nExtracting keyphrases from all methods...")

        abstractive_start = time.time()
        abstractive_keyphrases = self.abstractive_extractor.extract_keyphrases_with_scores(text)
        abstractive_time = time.time() - abstractive_start

        extractive_start = time.time()
        extractive_keyphrases = self.extractive_extractor.extract_keyphrases_with_scores(text)
        extractive_time = time.time() - extractive_start

        fusion_start = time.time()
        fusion_keyphrases = self.extract_keyphrases_with_scores(text)
        fusion_time = time.time() - fusion_start

        print("\nSummary:")
        print(f"- Abstractive: {len(abstractive_keyphrases)} keyphrases in {abstractive_time:.2f}s")
        print(f"- Extractive: {len(extractive_keyphrases)} keyphrases in {extractive_time:.2f}s")
        print(f"- Fusion: {len(fusion_keyphrases)} keyphrases in {fusion_time:.2f}s")

        domain = self.abstractive_extractor.detect_domain(text)
        print(f"\nDetected domain: {domain}")

        print("\nAbstractive Keyphrases:")
        for kp, score in abstractive_keyphrases:
            print(f"- {kp}: {score:.4f}")

        print("\nExtractive Keyphrases:")
        for kp, score in extractive_keyphrases:
            print(f"- {kp}: {score:.4f}")

        print("\nFusion Keyphrases:")
        for kp, score in fusion_keyphrases:
            print(f"- {kp}: {score:.4f}")

        abstractive_kps = set(kp.lower() for kp, _ in abstractive_keyphrases)
        extractive_kps = set(kp.lower() for kp, _ in extractive_keyphrases)
        fusion_kps = set(kp.lower() for kp, _ in fusion_keyphrases)

        abstractive_extractive_overlap = len(abstractive_kps.intersection(extractive_kps))
        abstractive_fusion_overlap = len(abstractive_kps.intersection(fusion_kps))
        extractive_fusion_overlap = len(extractive_kps.intersection(fusion_kps))

        fusion_unique_kps = fusion_kps - (abstractive_kps.union(extractive_kps))
        abstractive_unique_kps = abstractive_kps - extractive_kps
        extractive_unique_kps = extractive_kps - abstractive_kps

        print("\nOverlap Analysis:")
        print(f"- Abstractive-Extractive overlap: {abstractive_extractive_overlap} keyphrases")
        print(f"- Abstractive-Fusion overlap: {abstractive_fusion_overlap} keyphrases")
        print(f"- Extractive-Fusion overlap: {extractive_fusion_overlap} keyphrases")

        print("\nUnique Keyphrases:")
        print(f"- Fusion unique: {len(fusion_unique_kps)} keyphrases")
        if fusion_unique_kps:
            print("  " + ", ".join(fusion_unique_kps))

        print(f"- Abstractive unique (not in Extractive): {len(abstractive_unique_kps)} keyphrases")
        if abstractive_unique_kps:
            print("  " + ", ".join(abstractive_unique_kps))

        print(f"- Extractive unique (not in Abstractive): {len(extractive_unique_kps)} keyphrases")
        if extractive_unique_kps:
            print("  " + ", ".join(extractive_unique_kps))

        return {
            "abstractive": {
                "keyphrases": abstractive_keyphrases,
                "time": abstractive_time
            },
            "extractive": {
                "keyphrases": extractive_keyphrases,
                "time": extractive_time
            },
            "fusion": {
                "keyphrases": fusion_keyphrases,
                "time": fusion_time
            },
            "domain": domain,
            "overlap": {
                "abstractive_extractive": abstractive_extractive_overlap,
                "abstractive_fusion": abstractive_fusion_overlap,
                "extractive_fusion": extractive_fusion_overlap
            },
            "unique": {
                "fusion": list(fusion_unique_kps),
                "abstractive": list(abstractive_unique_kps),
                "extractive": list(extractive_unique_kps)
            }
        }

    def clean_memory(self):
        
        gc.collect()
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()

