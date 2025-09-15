"""Clustering parameter optimizer (grid search)."""
from typing import List, Dict
from .base import BaseService
from .clustering import ClusteringService


class ClusteringOptimizer(BaseService):
    def __init__(self):
        super().__init__()
        self.results: List[Dict] = []

    def grid_search(self, min_cluster_sizes: List[int] = [6, 8, 10], epsilons: List[float] = [0.10, 0.15, 0.20]) -> Dict:
        for mcs in min_cluster_sizes:
            for eps in epsilons:
                try:
                    svc = ClusteringService()
                    res = svc.run_full(min_cluster_size=mcs, epsilon=eps)
                    eff = self._efficiency(res)
                    self.results.append({
                        'min_cluster': mcs,
                        'epsilon': eps,
                        'n_clusters': res['n_clusters'],
                        'coverage': res['coverage'],
                        'noise_ratio': res['n_noise'] / max(1, res['n_total']),
                        'efficiency': eff,
                    })
                except Exception as e:
                    continue
        optimal = self._optimal()
        return {'all_results': self.results, 'optimal': optimal}

    def _efficiency(self, res: Dict) -> float:
        if not res['n_clusters']:
            return 0.0
        noise_ratio = res['n_noise'] / max(1, res['n_total'])
        eff = res['coverage'] / (res['n_clusters'] / 100.0)
        return eff * (1 - 0.5 * noise_ratio)

    def _optimal(self) -> Dict:
        if not self.results:
            return {}
        return sorted(self.results, key=lambda x: x['efficiency'], reverse=True)[0]



