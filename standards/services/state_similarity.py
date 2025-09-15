"""Analyze similarity between states based on mean atom embeddings."""
from typing import Dict, List, Tuple
import numpy as np

from .base import BaseService
from ..models import State, StandardAtom


class StateSimilarityAnalyzer(BaseService):
    def create_state_vectors(self) -> Dict[str, np.ndarray]:
        vectors: Dict[str, np.ndarray] = {}
        for state in State.objects.all():
            vals: List[List[float]] = list(
                StandardAtom.objects.filter(standard__state=state, embedding__isnull=False)
                .values_list('embedding', flat=True)
            )
            if not vals:
                continue
            arr = np.array(vals)
            vectors[state.code] = arr.mean(axis=0)
        return vectors

    def similarity_matrix(self, vectors: Dict[str, np.ndarray]) -> Tuple[List[str], np.ndarray]:
        states = list(vectors.keys())
        if not states:
            return [], np.zeros((0, 0))
        X = np.stack([vectors[s] for s in states])
        # cosine similarity
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        sim = Xn @ Xn.T
        return states, sim

    def clusters(self, sim_matrix: np.ndarray, states: List[str], threshold: float = 0.25) -> Dict:
        # simple agglomerative thresholding; no external deps
        # Convert to distance
        dist = 1.0 - sim_matrix
        n = len(states)
        visited = [False] * n
        groups: Dict[int, List[str]] = {}
        gid = 0
        for i in range(n):
            if visited[i]:
                continue
            group = [states[i]]
            visited[i] = True
            for j in range(n):
                if not visited[j] and dist[i, j] <= threshold:
                    group.append(states[j])
                    visited[j] = True
            groups[gid] = group
            gid += 1
        # stats
        stats = {}
        for k, members in groups.items():
            idxs = [states.index(s) for s in members]
            vals = []
            for a in idxs:
                for b in idxs:
                    if a < b:
                        vals.append(sim_matrix[a, b])
            avg = float(np.mean(vals)) if vals else 1.0
            stats[k] = {'states': members, 'size': len(members), 'avg_similarity': avg}
        return {'groups': groups, 'stats': stats, 'threshold': threshold}



