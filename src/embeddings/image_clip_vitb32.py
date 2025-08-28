from typing import List

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

_model = None
_processor = None


def _get_model_processor():
    global _model, _processor
    if _model is None or _processor is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _model = _model.to(device)
        _model.eval()
    return _model, _processor


def encode_images(images: List[Image.Image], batch_size: int = 64) -> List[List[float]]:
    model, processor = _get_model_processor()
    device = next(model.parameters()).device
    all_vecs = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        all_vecs.extend(feats.float().cpu().numpy().tolist())
    return all_vecs


