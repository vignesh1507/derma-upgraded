from pinecone import Pinecone
from graphrag.config.settings import Config
from graphrag.utils.logger import get_logger
from graphrag.utils.rate_limit import call_with_retries

logger = get_logger(__name__)


class PineconeRetriever:
    def __init__(self):
        if not Config.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is not set.")
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index = self.pc.Index(Config.PINECONE_INDEX_NAME)
        # Empty string means "default namespace", matching Pinecone's own
        # semantics and preserving the previous behaviour for single-namespace
        # indexes.
        self.namespace = getattr(Config, "PINECONE_NAMESPACE", "") or ""

    def retrieve(self, query_text: str, vector_top_k: int = 15, reranker_top_k: int = 5):
        """
        1. Embed query
        2. Fetch `vector_top_k` candidates from Pinecone
        3. Rerank with bge-reranker-v2-m3
        4. Return top `reranker_top_k` as plain dicts
        """
        logger.info(
            f"\n🔍 [1/3] Vector search  →  candidates: {vector_top_k}  |  "
            f"keep after rerank: {reranker_top_k}"
        )

        # ── 1. Embed ────────────────────────────────────────────────────────
        response = call_with_retries(
            self.pc.inference.embed,
            model="llama-text-embed-v2",
            inputs=[query_text],
            parameters={"input_type": "query", "truncate": "END"},
            operation="Pinecone query embedding",
        )
        query_vector = response[0]["values"]

        # ── 2. Retrieve candidates ──────────────────────────────────────────
        query_kwargs = {
            "vector": query_vector,
            "top_k": vector_top_k,
            "include_metadata": True,
        }
        if self.namespace:
            query_kwargs["namespace"] = self.namespace

        search_result = self.index.query(**query_kwargs)
        matches = search_result.get("matches", [])

        if not matches:
            # Distinguish "corpus has nothing relevant" from "we queried the
            # wrong partition". A namespaced index returns zero for every query
            # when the namespace is wrong or unset, which otherwise looks
            # identical to a genuine miss and silently degrades the service to
            # ungrounded answers.
            logger.info(
                "❌ No matches found in vector store "
                f"(index={Config.PINECONE_INDEX_NAME}, "
                f"namespace={self.namespace or '<default>'})."
            )
            return []

        logger.info(f"🔄 {len(matches)} candidates retrieved — reranking...")

        # ── 3. Prepare reranker documents ───────────────────────────────────
        documents = []
        for match in matches:
            md = match.get("metadata", {})
            combined = (
                f"Summary: {md.get('summary', '')}\n"
                f"Key Content (Entities): {', '.join(md.get('entities', []))}"
            )
            documents.append({"id": match["id"], "text": combined})

        # ── 4. Rerank ────────────────────────────────────────────────────────
        try:
            rerank_resp = call_with_retries(
                self.pc.inference.rerank,
                model="bge-reranker-v2-m3",
                query=query_text,
                documents=documents,
                rank_fields=["text"],
                top_n=reranker_top_k,
                return_documents=False,
                operation="Pinecone rerank",
            )

            reranked = []
            for result in rerank_resp.data:
                match = matches[result["index"]]
                match_dict = match.to_dict() if hasattr(match, "to_dict") else dict(match)
                match_dict["score"] = result["score"]
                match_dict["reranked"] = True
                reranked.append(match_dict)

            logger.info(f"✅ Reranking complete → {len(reranked)} chunks selected.")
            return reranked

        except Exception as e:
            logger.error(f"❌ Reranker failed: {e} — falling back to vector scores.")
            fallback = []
            for match in matches[:reranker_top_k]:
                d = match.to_dict() if hasattr(match, "to_dict") else dict(match)
                d["reranked"] = False
                fallback.append(d)
            return fallback
