from typing import List
import os

# ---- optimizaciones runtime para GPU T4 ----
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false") # para evitar overhead/aviso de warning
os.environ.setdefault("HF_HOME", "/content/drive/MyDrive/hf_cache")  # evita usar TRANSFORMERS_CACHE
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True") # PyTorch CUDA allocator que permite reutilizar segmentos de memoria

from PIL import Image
from sentence_transformers import SentenceTransformer
import torch

try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)   # usa implementación 'math'
except Exception:
    pass

torch.backends.cudnn.benchmark = True # autotune convolution algorithms, aplica a todas las operaciones de cuda
torch.set_float32_matmul_precision("high") # permite rutas matmul en fp32 para multiplicaciones de matrices

_clip = None


def _get_clip() -> SentenceTransformer:
    global _clip
    if _clip is None:
        # # Multilingual CLIP v2, 1024-D unified space
        # # para correr localmente
        # _clip = SentenceTransformer("jinaai/jina-clip-v2", trust_remote_code=True)

        # Multilingual CLIP v2, 1024-D unified space
        # Usar con CUDA y detectar automáticamente si hay una GPU disponible
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip = SentenceTransformer(
            "jinaai/jina-clip-v2", 
            device=device,
            trust_remote_code=True, 
            cache_folder=os.environ.get("HF_HOME", "./models_cache"),
            model_kwargs={
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                "attn_implementation": "sdpa"} # <- clave para evitar flash-attn
            )

        # Usar FP16 si hay GPU.
        # Al ser un modelo Transformers estándar, puedes castear el modelo de SBERT a bfloat16() o half() tras moverlo a CUDA
        # (métodos nativos de SentenceTransformers).
        # Solo si hay GPU (en CPU puede ralentizar).
        if device == "cuda":
            _clip.half()

        # limitar max_seq_length si mis textos son cortos
        _clip.max_seq_length = 256

        # (opcional) warm-up texto e imagen para precalentar kernels/planificacion   
        try:
            from PIL import Image
            import numpy as np
            with torch.inference_mode():
                # se usa un batch de 16 para precalentar el modelo y evitar el overhead de transferencia
                _ = _clip.encode(["warmup"] * 16, batch_size=16, normalize_embeddings=False, show_progress_bar=False)
                # el dummy es una imagen de 512x512 con un color gris medio (127) 
                dummy = Image.fromarray((np.ones((512,512,3))*127).astype("uint8"))
                # el batch de 8 sirve para precalentar
                _ = _clip.encode([dummy]*8, batch_size=8, normalize_embeddings=False, show_progress_bar=False)
        except Exception: # pass
            pass

    return _clip

# # Para usar sin cuda, usar batch_size=128
# def embed_text(texts: List[str]) -> List[List[float]]:
def embed_text(texts: List[str], batch_size: int = 256) -> List[List[float]]:
    model = _get_clip()
    # # Asi lo usabamos antes
    # embs = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    embs = model.encode(
        texts, 
        batch_size=batch_size, 
        normalize_embeddings=True, 
        show_progress_bar=False, # tqdm lo maneja en el backfill
        convert_to_tensor=True
        ) 
    # baja a CPU SOLO AL FINAL (minimiza el overhead de transferencia)
    return embs.float().cpu().numpy().tolist()

# # Para usar sin cuda, usar batch_size=128
# def embed_images(images: List[Image.Image]) -> List[List[float]]:
def embed_images(images: List[Image.Image], batch_size: int = 128) -> List[List[float]]:
    model = _get_clip()
    # embs = model.encode(images, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    embs = model.encode(
        images, 
        batch_size=batch_size, 
        normalize_embeddings=True, 
        show_progress_bar=False,
        convert_to_tensor=True
        ) 
    # baja a CPU SOLO AL FINAL (minimiza el overhead de transferencia)
    return embs.float().cpu().numpy().tolist()


