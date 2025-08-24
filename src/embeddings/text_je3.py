from typing import List

from sentence_transformers import SentenceTransformer


_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # 1024-D multilingual embeddings
        _model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True, cache_folder="./models_cache")
    return _model


def encode_texts(texts: List[str]) -> List[List[float]]:
    model = _get_model()
    emb = model.encode(
        texts,
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return emb.tolist()


