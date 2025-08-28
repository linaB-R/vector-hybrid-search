from typing import List

from sentence_transformers import SentenceTransformer
import torch

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(
            "intfloat/multilingual-e5-small",
            device=device,
            trust_remote_code=True,
        )
    return _model


def encode_texts(texts: List[str], batch_size: int = 256) -> List[List[float]]:
    model = _get_model()
    embs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_tensor=True,
    )
    return embs.float().cpu().numpy().tolist()


