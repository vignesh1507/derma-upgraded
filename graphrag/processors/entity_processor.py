import re
from typing import List, Tuple
from graphrag.utils.logger import get_logger

logger = get_logger(__name__)


class EntityProcessor:

    @staticmethod
    def process_matches(
        matches: list,
        priority_entity_types: List[str] | None = None,
        boost_drug_pairs: bool = False,
        query: str = "",
    ) -> Tuple[str, List[str], List[str]]:
        """
        Extract entities and build vector context string from reranked chunks.

        Parameters
        ----------
        matches              : reranked Pinecone match dicts
        priority_entity_types: entity types to surface first (from QueryConfig)
        boost_drug_pairs     : if True, re-rank chunks that contain BOTH drug
                               names detected in the query (drug_interaction mode)
        query                : original query text (used for drug-pair detection)
        """
        if not matches:
            logger.info("❌ No chunks to process.")
            return "No medical chunks found.", [], []

        priority_entity_types = priority_entity_types or []

        # ── Optional: drug-pair chunk boosting ──────────────────────────────
        if boost_drug_pairs and query:
            matches = EntityProcessor._boost_drug_pair_chunks(matches, query)

        extracted_entities: set[str] = set()
        priority_entities: set[str] = set()
        chunk_summaries: List[str] = []

        for match in matches:
            md = match.get("metadata", {})
            chunk_summaries.append(md.get("summary", ""))

            for ent_str in md.get("entities", []):
                # Two corpus conventions are in use:
                #   typed   — "disease: psoriasis"  (e.g. the `enervera` index)
                #   untyped — "psoriasis"           (e.g. `enervara-specialists`)
                # Skipping untyped entities discarded EVERY entity in an
                # untyped corpus, leaving extracted_entities empty. That is
                # invisible while GRAPH_RETRIEVAL_ENABLED is false, but the
                # moment graph traversal is switched on it would be skipped
                # forever with only a "no entities" line in the log.
                if ":" in ent_str:
                    ent_type, ent_name = ent_str.split(":", 1)
                    ent_type_clean = ent_type.strip().lower()
                else:
                    ent_name = ent_str
                    ent_type_clean = ""

                ent_name_clean = ent_name.strip().lower()
                if not ent_name_clean:
                    continue

                extracted_entities.add(ent_name_clean)

                if (
                    ent_type_clean
                    and priority_entity_types
                    and ent_type_clean in priority_entity_types
                ):
                    priority_entities.add(ent_name_clean)

        # ── Build final entity list: priority first, then rest (cap 30) ────
        rest = extracted_entities - priority_entities
        final_entities = list(priority_entities)[:30] + list(rest)[:max(0, 30 - len(priority_entities))]

        vector_context_str = "\n".join([f"- {s}" for s in chunk_summaries if s])

        logger.info(
            f"✅ {len(matches)} chunks processed  |  "
            f"{len(final_entities)} entities  |  "
            f"{len(priority_entities)} priority ({', '.join(priority_entity_types) or 'none'})"
        )

        return vector_context_str, final_entities, chunk_summaries

    # -------------------------------------------------------------------------

    @staticmethod
    def _boost_drug_pair_chunks(matches: list, query: str) -> list:
        """
        For drug_interaction queries: re-rank chunks so that those containing
        BOTH detected drug names in their entities come first.
        """
        detected_drugs = EntityProcessor._extract_drug_names(query)
        if len(detected_drugs) < 2:
            return matches  # can't boost without a pair

        logger.info(f"💊 Drug-pair boost active — detected drugs: {detected_drugs}")

        def drug_hit_count(match):
            entities_text = " ".join(match.get("metadata", {}).get("entities", [])).lower()
            return sum(1 for d in detected_drugs if d in entities_text)

        return sorted(matches, key=drug_hit_count, reverse=True)

    @staticmethod
    def _extract_drug_names(query: str) -> List[str]:
        """
        Lightweight heuristic: extract multi-word tokens that look like drug names
        (capitalized or all-alpha strings after 'take', 'with', 'and', 'between').
        """
        pattern = r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b"
        candidates = re.findall(pattern, query)

        # Common stop-words to filter out
        stopwords = {
            "What", "When", "Where", "How", "Why", "Can", "Does", "The",
            "This", "That", "Drug", "Medication", "Medicine", "Patient",
        }
        drugs = [c for c in candidates if c not in stopwords]
        return [d.lower() for d in drugs]
