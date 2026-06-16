"""Policy clause vector store with Milvus hybrid search.

Pre-indexes all article_content from policy data files into Milvus Lite (embedded),
supporting hybrid (keyword + vector similarity) search for clause retrieval
during DPO sample repair.

Architecture:
  1. index_all() — walks policy JSON files, embeds each article via SentenceTransformer,
     inserts into local Milvus Lite collection.
  2. search()    — vectors the prompt, queries Milvus filtered by policy_id,
     then reranks by keyword overlap between prompt and article_content.
  3. _extract_keywords() — uses CLAIM_KEYWORDS from validator plus simple tokenization
     to build a keyword set for scoring.


┌─────────────────────────────────────────────────────┐
│                  Pipeline.init_from_config()         │
│                        │                            │
│     policy_store 配置存在? ──No──▶ _policy_store=None │
│              │Yes                                    │
│        PolicyStore(ps_cfg)                           │
│              │                                       │
│     ensure_ready() ──▶ 失败 ──▶ _policy_store=None   │
│              │Yes                                    │
│    _policy_store.ready=True                          │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│          _repair_missing_reference(sample, store)    │
│                                                     │
│  ① Milvus 混合检索（优先）                            │
│     ├─ policy_id 存在 + store.ready                  │
│     ├─ prompt 向量化 → ANN 检索 filter by policy_id   │
│     ├─ CLAIM_KEYWORDS 关键词重叠 rerank               │
│     └─ 追加条款原文: "依据POL-CRIT-001：第2.1条（等待期）│
│        ：自本合同生效日起90日内..."                     │
│                                                     │
│  ② ID 兜底（store 不可用但有 policy_id）               │
│     └─ "具体参见POL-CRIT-001相关条款及保单约定。"       │
│                                                     │
│  ③ 通用兜底（无 policy_id）                           │
│     └─ "具体参见相关保险条款及保单约定。"               │
└─────────────────────────────────────────────────────┘



Dependencies (optional, graceful degradation if missing):
  - pymilvus >= 2.4 (includes Milvus Lite)
  - sentence-transformers (for embedding model)
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from m_data.validator import CLAIM_KEYWORDS

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Embedding model defaults (Chinese retrieval-optimised)
# ------------------------------------------------------------------
_DEFAULT_EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"
_DEFAULT_EMBEDDING_DIM: int = 512
_DEFAULT_COLLECTION: str = "policy_clauses"
_DEFAULT_TOP_K: int = 3
_MAX_ARTICLE_LEN: int = 2000  # max chars stored per article (truncated for vector DB)
_MAX_SEARCH_PRE_FETCH: int = 15  # fetch more candidates before keyword rerank


class PolicyStore:
    """
    Milvus-backed policy clause store with hybrid (dense + sparse) search.

    Usage inside Pipeline::

        store = PolicyStore(config["policy_store"])
        clauses = store.search(policy_id="POL-CRIT-001", prompt="等待期理赔？")
        # clauses → [{"article_id": "2.1", "title": "等待期", "content": "...", ...}, ...]
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Args:
            config: Sub-dict from insurance_dpo_gen.yaml → policy_store section.
                    Expected keys:
                        - enabled (bool)
                        - data_dir (str)          — policy JSON files directory
                        - embedding_model (str)   — HF model name or local path
                        - embedding_dim (int)     — must match the model
                        - milvus_db_path (str)    — local Milvus Lite file path
                        - collection_name (str)
                        - top_k (int)             — number of clauses to return
        """
        cfg = config or {}
        self._enabled: bool = cfg.get("enabled", True)
        self._data_dir: Path = Path(cfg.get("data_dir", "data/insurance/raw/policies"))
        self._model_name: str = cfg.get("embedding_model", _DEFAULT_EMBEDDING_MODEL)
        self._embedding_dim: int = cfg.get("embedding_dim", _DEFAULT_EMBEDDING_DIM)
        self._milvus_path: str = cfg.get("milvus_db_path", "./milvus_data/policy_store.db")
        self._collection_name: str = cfg.get("collection_name", _DEFAULT_COLLECTION)
        self._top_k: int = cfg.get("top_k", _DEFAULT_TOP_K)

        self._model: Any = None          # SentenceTransformer instance
        self._client: Any = None         # MilvusClient instance
        self._ready: bool = False        # True after successful init + index
        self._init_error: str | None = None
        self._server_error: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """Whether the store is fully initialised and searchable."""
        return self._ready

    @property
    def init_error(self) -> str | None:
        """Human-readable reason why the store is not ready, if any."""
        return self._init_error

    @property
    def server_error(self) -> str | None:
        """Human-readable reason for a server error, if any."""
        return self._server_error

    def ensure_ready(self) -> bool:
        """Lazily initialise Milvus client + embedding model + index.

        Safe to call multiple times — only initialises once.

        Returns:
            True if the store is ready for search.
        """
        if self._ready:
            return True
        if not self._enabled:
            self._init_error = "policy_store.enabled=false"
            return False

        try:
            self._init_milvus()
            self._init_embedding()
            self._ensure_indexed()
            self._ready = True
            logger.info(
                "PolicyStore ready: collection=%s, dim=%d, model=%s",
                self._collection_name,
                self._embedding_dim,
                self._model_name,
            )
        except Exception as exc:
            self._init_error = str(exc)
            logger.warning("PolicyStore init failed (will fall back to template repair): %s", exc)
            self._ready = False

        return self._ready

    def index_all(self, force: bool = False) -> int:
        """(Re-)index all policy articles from the data directory.

        Args:
            force: If True, drop existing collection and re-create.

        Returns:
            Number of articles inserted.
        """
        if not self._client:
            self._init_milvus()

        if force and self._client.has_collection(self._collection_name):
            self._client.drop_collection(self._collection_name)
            logger.info("Dropped existing collection '%s'", self._collection_name)

        if not self._client.has_collection(self._collection_name):
            self._create_collection()

        articles = self._load_policy_articles()
        if not articles:
            logger.warning("No policy articles found in %s", self._data_dir)
            return 0

        # Batch embed & insert （每 32 条批一次，控制内存）
        batch_size = 32
        total = 0
        for i in range(0, len(articles), batch_size):
            batch = articles[i : i + batch_size]
            contents = [a["article_content"][:_MAX_ARTICLE_LEN] for a in batch]
            embeddings = self._encode_batch(contents)
            rows = []
            for j, art in enumerate(batch):
                rows.append({
                    "policy_id": art["policy_id"],
                    "article_id": art["article_id"],
                    "article_title": art["article_title"],
                    "article_content": art["article_content"][:_MAX_ARTICLE_LEN],
                    "embedding": embeddings[j].tolist(),
                    "source_file": art["source_file"],
                })
            self._client.insert(collection_name=self._collection_name, data=rows)
            total += len(rows)
            logger.debug("Indexed %d/%d articles", total, len(articles))

        # Flush to disk
        self._client.flush(self._collection_name)
        logger.info("PolicyStore indexed %d articles into '%s'", total, self._collection_name)
        return total

    def search(
        self,
        policy_id: str,
        prompt: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search for policy clauses most relevant to *prompt*.

        Steps:
          1. Embed *prompt* → vector.
          2. ANN search in Milvus filtered by *policy_id* (fetch top_k × 5 candidates).
          3. Rerank candidates by keyword overlap between *prompt* and *article_content*.
          4. Return final top_k.

        Args:
            policy_id: Policy ID to filter on (e.g. "POL-CRIT-001").
            prompt:   User question / DPO prompt text.
            top_k:    Number of clauses to return (defaults to self._top_k).

        Returns:
            List of clause dicts with keys: article_id, article_title, article_content,
            policy_id, score (combined score 0-1, higher is better).
        """
        if not self.ensure_ready():
            return []

        k = top_k or self._top_k
        fetch_k = max(k * 5, _MAX_SEARCH_PRE_FETCH)

        # Step 1: embed prompt
        prompt_embedding = self._encode_single(prompt)

        # Step 2: ANN search
        try:
            ann_results = self._client.search(
                collection_name=self._collection_name,
                data=[prompt_embedding.tolist()],
                filter=f'policy_id == "{policy_id}"',
                limit=fetch_k,
                output_fields=["policy_id", "article_id", "article_title", "article_content"],
            )
        except Exception as exc:
            logger.warning("Milvus search failed for policy_id=%s: %s", policy_id, exc)
            return []

        if not ann_results or not ann_results[0]:
            logger.debug("No ANN results for policy_id=%s, prompt='%s'", policy_id, prompt)
            return []

        # Step 3: keyword rerank
        prompt_keywords = self._extract_keywords(prompt)
        candidates: list[dict[str, Any]] = []
        for hit in ann_results[0]:
            entity = hit.get("entity", {})
            vec_score = 1.0 - hit.get("distance", 0.0)  # COSINE: distance = 1 - similarity
            kw_score = self._keyword_score(
                entity.get("article_content", ""),
                prompt_keywords,
            )
            combined = 0.65 * vec_score + 0.35 * kw_score
            candidates.append({
                "policy_id": entity.get("policy_id", policy_id),
                "article_id": entity.get("article_id", ""),
                "article_title": entity.get("article_title", ""),
                "article_content": entity.get("article_content", ""),
                "score": round(combined, 4),
            })

        # Sort by combined score descending, take top_k
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:k]

    # ------------------------------------------------------------------
    # Internal: init helpers
    # ------------------------------------------------------------------

    def _init_milvus(self) -> None:
        """Connect to Milvus Lite (embedded, file-based)."""
        try:
            from pymilvus import MilvusClient  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "pymilvus is required for PolicyStore. "
                "Install with: pip install pymilvus>=2.4"
            )

        os.makedirs(os.path.dirname(self._milvus_path) or ".", exist_ok=True)
        self._client = MilvusClient(self._milvus_path)
        logger.info("Milvus Lite connected: %s", self._milvus_path)

    def _init_embedding(self) -> None:
        """Load SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for PolicyStore. "
                "Install with: pip install sentence-transformers"
            )

        self._model = SentenceTransformer(self._model_name)
        # Verify embedding dimension matches config
        actual_dim = self._model.get_sentence_embedding_dimension()
        if actual_dim != self._embedding_dim:
            logger.warning(
                "Embedding dim mismatch: config=%d, model=%d. Using model dim.",
                self._embedding_dim,
                actual_dim,
            )
            self._embedding_dim = actual_dim

    def _create_collection(self) -> None:
        """Create Milvus collection with schema + IVF_FLAT index."""
        from pymilvus import DataType  # type: ignore[import-untyped]

        # Build schema
        schema = self._client.create_schema(
            auto_id=True,
            enable_dynamic_field=False,
        )
        schema.add_field(
            field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True,
        )
        schema.add_field(
            field_name="policy_id", datatype=DataType.VARCHAR, max_length=64,
        )
        schema.add_field(
            field_name="article_id", datatype=DataType.VARCHAR, max_length=32,
        )
        schema.add_field(
            field_name="article_title", datatype=DataType.VARCHAR, max_length=256,
        )
        schema.add_field(
            field_name="article_content", datatype=DataType.VARCHAR, max_length=_MAX_ARTICLE_LEN,
        )
        schema.add_field(
            field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self._embedding_dim,
        )
        schema.add_field(
            field_name="source_file", datatype=DataType.VARCHAR, max_length=512,
        )

        # IVF_FLAT (Inverted File + Flat Storage) 索引配置说明：
        #   1. IVF (倒排文件索引)：使用 K-Means 算法将高维向量空间划分为 nlist 个聚类簇（Voronoi 单元）。
        #      检索时，系统先计算查询向量与所有簇中心的距离，快速定位到最近的若干个候选簇，
        #      随后仅在这些局部簇内部进行比对。这避免了全量暴力扫描，显著提升检索速度。
        #   2. FLAT (平坦/原始存储)：落入同一簇的向量不进行任何压缩或量化（如 PQ），直接保留原始浮点格式。
        #      搜索时进行精确的向量距离计算，因此召回精度极高，但内存占用会随数据量线性增长。
        # 适用场景：中小型数据集（通常 < 100 万条）。在精度要求高、硬件内存充足时是首选方案。
        # 参数 nlist=64：预设的聚类中心（桶）数量。工程经验通常建议设置为 4 * sqrt(总数据量) 附近。
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 64},
        )

        self._client.create_collection(
            collection_name=self._collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info("Created Milvus collection '%s'", self._collection_name)

    def _ensure_indexed(self) -> None:
        """Check if collection has data; if empty, trigger full index."""
        try:
            stats = self._client.get_collection_stats(self._collection_name)
            row_count = stats.get("row_count", 0)
        except Exception:
            row_count = 0

        if row_count == 0:
            logger.info("Collection '%s' is empty, running full index...", self._collection_name)
            self.index_all()

    # ------------------------------------------------------------------
    # Internal: data loading
    # ------------------------------------------------------------------

    def _load_policy_articles(self) -> list[dict[str, Any]]:
        """
        Walk data_dir, parse JSON policy files, extract all articles.

        Returns:
            List of article dicts with keys:
            policy_id, article_id, article_title, article_content, source_file.
        """
        articles: list[dict[str, Any]] = []
        if not self._data_dir.exists():
            logger.warning("Policy data dir not found: %s", self._data_dir)
            return articles

        for json_path in sorted(self._data_dir.rglob("*.json")):
            # Skip non-policy files (faq, tickets etc.)
            if any(kw in str(json_path).lower() for kw in ("faq", "ticket", "smoke")):
                continue
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to parse %s: %s", json_path, exc)
                continue

            policy_id = data.get("policy_id", json_path.stem)
            policy_name = data.get("policy_name", "")
            for art in data.get("articles", []):
                content = art.get("content", "") or art.get("article_content", "")
                if not content.strip():
                    continue
                articles.append({
                    "policy_id": policy_id,
                    "policy_name": policy_name,
                    "article_id": art.get("article_id", ""),
                    "article_title": art.get("title", "") or art.get("article_title", ""),
                    "article_content": content.strip(),
                    "source_file": str(json_path),
                })

        logger.info("Loaded %d articles from %s", len(articles), self._data_dir)
        return articles

    # ------------------------------------------------------------------
    # Internal: embedding helpers
    # ------------------------------------------------------------------

    def _encode_single(self, text: str) -> Any:
        """Encode a single text string → normalised vector."""
        return self._model.encode(
            text.strip(),
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _encode_batch(self, texts: list[str]) -> Any:
        """Encode a batch of texts → normalised vectors (numpy array)."""
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=len(texts),
        )

    # ------------------------------------------------------------------
    # Internal: keyword scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        """Extract insurance-relevant keywords from text.

        Uses:
          1. CLAIM_KEYWORDS from validator (pre-defined insurance terms).
          2. Simple tokenisation for additional meaningful words.
        """
        keywords: set[str] = set()

        # 1. Pre-defined insurance keywords
        text_lower = text.lower()
        for kw in CLAIM_KEYWORDS:
            if kw in text:
                keywords.add(kw)

        # 2. Simple tokenisation: split on common delimiters, keep >=2-char tokens
        tokens = re.split(r"[，。！？；、\s,.!?;:：]+", text)
        for token in tokens:
            token = token.strip()
            if len(token) >= 2:
                keywords.add(token)

        return keywords

    @staticmethod
    def _keyword_score(content: str, prompt_keywords: set[str]) -> float:
        """Compute keyword overlap score between content and prompt keywords.

        Returns:
            Score in [0, 1]. 1 means all prompt keywords appear in content.
        """
        if not prompt_keywords:
            return 0.0
        content_lower = content.lower()
        matches = sum(1 for kw in prompt_keywords if kw.lower() in content_lower)
        return matches / len(prompt_keywords)
