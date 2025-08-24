from typing import List

from PIL import Image
from sentence_transformers import SentenceTransformer


_clip = None


def _get_clip() -> SentenceTransformer:
    global _clip
    if _clip is None:
        # Multilingual CLIP v2, 1024-D unified space
        _clip = SentenceTransformer("jinaai/jina-clip-v2", trust_remote_code=True)
    return _clip


def embed_text(texts: List[str]) -> List[List[float]]:
    model = _get_clip()
    embs = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    return embs.tolist()


def embed_images(images: List[Image.Image]) -> List[List[float]]:
    model = _get_clip()
    embs = model.encode(images, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return embs.tolist()


