import os
import threading 
import torch
import open_clip


from django.conf import settings


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

index_path = settings.IMAGE_INDEX_PATH

_model = None
_tokenizer = None
_preprocess = None
_image_embeddings = None
_image_paths = None

# Simple lock for first-time lazy init
_init_lock = threading.Lock()
_initialized = False

def _lazy_init():
    global _initialized, _model, _tokenizer, _preprocess, _image_embeddings, _image_paths

    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

    
    model_name = "ViT-B-32"
    pretrained = "openai"
    _model, _, _preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    _tokenizer = open_clip.get_tokenizer(model_name)

    
    idx = torch.load(index_path, map_location=_DEVICE)
    _image_embeddings = idx["embeddings"].to(_DEVICE)
    _image_paths = idx["paths"]

    _image_embeddings = _image_embeddings / _image_embeddings.norm(dim=-1, keepdim=True)

    _initialized = True

def search_images(query: str, top_k: int = 3):
    """Returns list of (image_path, score) tuples for a text query."""
    _lazy_init()

    with torch.no_grad():
        # Tokenize text
        text_tokens = _tokenizer([query]).to(_DEVICE)

        # Encode & normalize
        text_emb = _model.encode_text(text_tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        # Cosine similarity (dot product because both sides are normalized)
        # You can use raw similarity or softmax'd probabilities. Here: raw.
        similarity = (text_emb @ _image_embeddings.T)  # shape [1, N]
        values, indices = torch.topk(similarity[0], k=min(top_k, similarity.shape[-1]))

        # Convert to float & list
        results = [( _image_paths[i], float(values[j].item()) ) for j, i in enumerate(indices)]
        return results