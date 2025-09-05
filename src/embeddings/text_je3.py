from typing import List
import os
# ---- optimizaciones runtime para GPU T4 ----
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false") # para evitar overhead/aviso de warning
os.environ.setdefault("HF_HOME", "/content/drive/MyDrive/hf_cache")  # evita usar TRANSFORMERS_CACHE
# ---- Reduce memory fragmentation VRAM cuando se sube/baja el batch size ----
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True") # PyTorch CUDA allocator que permite reutilizar segmentos de memoria

from sentence_transformers import SentenceTransformer
import torch

# Fuerza a PyTorch a NO usar rutas 'flash' ni 'mem_efficient' (no válidas en T4)
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)   # usa implementación 'math'
except Exception:
    pass

torch.backends.cudnn.benchmark = True # autotune convolution algorithms, aplica a todas las operaciones de cuda
torch.set_float32_matmul_precision("high") # permite rutas matmul en fp32 para multiplicaciones de matrices 

_model = None




def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # 1024-D multilingual embeddings
        # para correr localmente
        # _model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True, cache_folder="./models_cache")

        # 1024-D multilingual embeddings
        # Use with CUDA and detect automatically if there is a GPU available
        # aseguremonos de cargar el modelo directamente a cuda
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(
            "jinaai/jina-embeddings-v3",
            device=device,
            trust_remote_code=True,
            cache_folder=os.environ.get("HF_HOME", "./models_cache"),
            model_kwargs={
                "dtype": torch.float16 if device == "cuda" else torch.float32,
                "attn_implementation": "eager"}
        )

        # Usar FP16 si hay GPU.
        # Al ser un modelo Transformers estándar, puedes castear el modelo de SBERT a bfloat16() o half() tras moverlo a CUDA
        # (métodos nativos de SentenceTransformers).
        # Solo si hay GPU (en CPU puede ralentizar).
        if device == "cuda":
            _model.half()

        # limitar max_seq_length si mis textos son cortos
        _model.max_seq_length = 256

        # (opcional) warn-up para estabilizar kernels/planificacion
        try:
            with torch.inference_mode():
                _ = _model.encode(
                    ["warnup"] * 32,
                    batch_size=32,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                )
        except Exception: # pass
            pass

    return _model


# def encode_texts(texts: List[str]) -> List[List[float]]:
def encode_texts(texts: List[str], batch_size: int = 256) -> List[List[float]]:
    model = _get_model()
    emb = model.encode(
        texts,
        # batch_size=128, # default
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False, # tqdm lo maneja en el backfill
        convert_to_tensor=True
    )
    # baja a CPU SOLO AL FINAL (minimiza el overhead de transferencia)
    return emb.float().cpu().numpy().tolist()


