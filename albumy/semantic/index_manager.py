import os
import numpy as np
from typing import List, Tuple


class IndexManager:
    """Vector index backed by FAISS (if installed) else NumPy cosine similarity."""
    def __init__(self, dim: int = 512, index_path: str = None, mapping_path: str = None):
        self.dim = int(dim)
        self.index_path = index_path or os.getenv('SEMANTIC_INDEX_PATH', 'data/semantic.index')
        self.mapping_path = mapping_path or os.getenv('SEMANTIC_MAPPING_PATH', 'data/semantic.ids.npy')
        self.use_faiss = False
        self._faiss = None
        self._index = None      # FAISS index
        self._vectors = None    # NumPy matrix (N, dim)
        self._ids = np.array([], dtype='int64')

        # Try to import FAISS
        try:
            import faiss
            self._faiss = faiss
            self.use_faiss = True
        except Exception:
            self._faiss = None
            self.use_faiss = False

        # Load existing index if present
        self._load()

    def _load(self):
        ids_path = self.mapping_path
        if self.use_faiss and os.path.exists(self.index_path) and os.path.exists(ids_path):
            self._index = self._faiss.read_index(self.index_path)
            self._ids = np.load(ids_path).astype('int64')
        else:
            # NumPy fallback
            if os.path.exists(self.index_path) and os.path.exists(ids_path):
                self._vectors = np.load(self.index_path).astype('float32')
                self._ids = np.load(ids_path).astype('int64')
            else:
                self._vectors = np.zeros((0, self.dim), dtype='float32')
                self._ids = np.array([], dtype='int64')
            if self.use_faiss:
                self._index = self._faiss.IndexFlatIP(self.dim)

    def save(self):
        os.makedirs(os.path.dirname(self.index_path) or '.', exist_ok=True)
        if self.use_faiss:
            if self._index is None:
                self._index = self._faiss.IndexFlatIP(self.dim)
            self._faiss.write_index(self._index, self.index_path)
            np.save(self.mapping_path, self._ids)
        else:
            np.save(self.index_path, self._vectors.astype('float32'))
            np.save(self.mapping_path, self._ids.astype('int64'))

    @property
    def size(self) -> int:
        return int(self._ids.shape[0])

    def rebuild(self, vectors: np.ndarray, ids: np.ndarray):
        assert vectors.ndim == 2 and vectors.shape[1] == self.dim, "vectors dim mismatch"
        assert vectors.shape[0] == ids.shape[0], "N mismatch"
        # normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        vectors = (vectors / norms).astype('float32')
        ids = ids.astype('int64')

        if self.use_faiss:
            self._index = self._faiss.IndexFlatIP(self.dim)
            if vectors.shape[0] > 0:
                self._index.add(vectors)
            self._ids = ids
        else:
            self._vectors = vectors
            self._ids = ids

    def add(self, vectors: np.ndarray, ids: np.ndarray):
        if vectors.size == 0:
            return
        assert vectors.shape[1] == self.dim, "vectors dim mismatch"
        assert vectors.shape[0] == ids.shape[0], "N mismatch"
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        vectors = (vectors / norms).astype('float32')
        ids = ids.astype('int64')

        if self.use_faiss:
            if self._index is None:
                self._index = self._faiss.IndexFlatIP(self.dim)
            self._index.add(vectors)
            self._ids = np.concatenate([self._ids, ids])
        else:
            if self._vectors is None or self._vectors.shape[0] == 0:
                self._vectors = vectors
            else:
                self._vectors = np.vstack([self._vectors, vectors])
            self._ids = np.concatenate([self._ids, ids])

    def query(self, query_vec: np.ndarray, topk: int = 12) -> List[Tuple[int, float]]:
        q = query_vec.astype('float32').reshape(1, -1)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-10)
        if self.use_faiss:
            if self._index is None or self.size == 0:
                return []
            D, I = self._index.search(q, min(topk, self.size))
            return [(int(self._ids[i]), float(d)) for i, d in zip(I[0], D[0]) if i != -1]
        else:
            if self._vectors is None or self._vectors.shape[0] == 0:
                return []
            sims = (self._vectors @ q.T).reshape(-1)
            idxs = np.argsort(-sims)[:topk]
            return [(int(self._ids[i]), float(sims[i])) for i in idxs]
