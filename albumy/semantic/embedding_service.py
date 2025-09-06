import os
import numpy as np
from PIL import Image

class EmbeddingService:
    """Encodes images and text into a shared embedding space using CLIP via sentence-transformers if available.
    Falls back to a simple hashed bag-of-words for text and zeros for images when the model is unavailable.
    """
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv('SEMANTIC_MODEL_NAME', 'clip-ViT-B-32')
        self._st_model = None
        try:
            from sentence_transformers import SentenceTransformer
            # Sentence-Transformers can encode PIL Images for CLIP models
            self._st_model = SentenceTransformer(self.model_name, device='cpu')
        except Exception:
            self._st_model = None

    def encode_text(self, text: str) -> np.ndarray:
        if self._st_model is not None:
            vec = self._st_model.encode([text], normalize_embeddings=True)
            return vec.astype('float32')[0]
        # Fallback: hashed bag-of-words
        tokens = [t for t in re_tokenize(text)]
        dim = 384
        v = np.zeros(dim, dtype='float32')
        for tok in tokens:
            h = abs(hash(tok)) % dim
            v[h] += 1.0
        n = np.linalg.norm(v) or 1.0
        return (v / n).astype('float32')

    def encode_image(self, image_path: str) -> np.ndarray:
        if self._st_model is not None:
            img = Image.open(image_path).convert('RGB')
            vec = self._st_model.encode([img], normalize_embeddings=True)
            return vec.astype('float32')[0]
        # Fallback: zero vector
        return np.zeros(384, dtype='float32')

def re_tokenize(text: str):
    # very small tokenizer: lowercase and split on non-alnum
    import re
    return [t for t in re.split(r"[^0-9a-zA-Z]+", text.lower()) if t]
