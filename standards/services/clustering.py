"""UMAP + HDBSCAN clustering over StandardAtom embeddings to create ProxyStandards."""
from typing import Tuple, List, Dict, Optional

import numpy as np
from django.db import transaction
from sklearn.metrics.pairwise import cosine_similarity

try:  # optional
    import umap
    import hdbscan
except Exception:  # pragma: no cover
    umap = None
    hdbscan = None

from .base import BaseService
from ..models import StandardAtom, ProxyStandard, ProxyStateCoverage, State, GradeLevel, Standard


class ClusteringService(BaseService):
    def __init__(self):
        super().__init__()
        self.embeddings = None
        self.atoms: List[StandardAtom] = []

    def load_embeddings(self, grade_levels: List[int] = None) -> Tuple[np.ndarray, List[StandardAtom]]:
        """Load embeddings with optional grade level filtering.
        
        Args:
            grade_levels: List of grade level numeric values to filter by (e.g., [3, 4])
        """
        atoms_query = StandardAtom.objects.filter(embedding__isnull=False).select_related(
            'standard__state', 'standard__subject_area'
        ).prefetch_related('standard__grade_levels')
        
        # Filter by grade levels if specified
        if grade_levels:
            atoms_query = atoms_query.filter(
                standard__grade_levels__grade_numeric__in=grade_levels
            ).distinct()
        
        atoms = list(atoms_query)
        if not atoms:
            raise ValueError("No StandardAtom embeddings found for the specified criteria")
        
        X = np.array([a.embedding for a in atoms])
        self.embeddings = X
        self.atoms = atoms
        return X, atoms

    def run_umap(self, X: np.ndarray, n_components: int = 25) -> np.ndarray:
        if umap is None:
            raise RuntimeError("umap-learn not installed")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42,
        )
        return reducer.fit_transform(X)

    def run_hdbscan(self, X_reduced: np.ndarray, min_cluster_size: int = 8, epsilon: float = 0.15) -> np.ndarray:
        if hdbscan is None:
            raise RuntimeError("hdbscan not installed")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=epsilon,
            metric='euclidean',
            cluster_selection_method='eom',
        )
        return clusterer.fit_predict(X_reduced)

    def find_medoids(self, labels: np.ndarray) -> Dict[int, Dict]:
        medoids: Dict[int, Dict] = {}
        for label in set(labels):
            if label == -1:
                continue
            idxs = np.where(labels == label)[0]
            cluster_vecs = self.embeddings[idxs]
            centroid = cluster_vecs.mean(axis=0)
            sims = cosine_similarity(cluster_vecs, [centroid]).flatten()
            medoid_idx_in_cluster = sims.argmax()
            medoid_idx_global = idxs[medoid_idx_in_cluster]
            states = {self.atoms[i].standard.state for i in idxs}
            medoids[label] = {
                'medoid_idx': int(medoid_idx_global),
                'member_indices': idxs.tolist(),
                'member_count': len(idxs),
                'centroid': centroid,
                'avg_similarity': float(sims.mean()),
                'states_covered': list(states),
                'state_count': len(states),
            }
        return medoids

    def run_full(self, min_cluster_size: int = 8, epsilon: float = 0.15, grade_levels: List[int] = None) -> Dict:
        """Run full clustering pipeline with optional grade level filtering.
        
        Args:
            min_cluster_size: Minimum cluster size for HDBSCAN
            epsilon: Epsilon value for HDBSCAN
            grade_levels: List of grade level numeric values to filter by
        """
        X, _ = self.load_embeddings(grade_levels=grade_levels)
        Xr = self.run_umap(X)
        labels = self.run_hdbscan(Xr, min_cluster_size=min_cluster_size, epsilon=epsilon)
        medoids = self.find_medoids(labels)
        all_states = State.objects.count()
        covered_states = set()
        for c in medoids.values():
            for s in c['states_covered']:
                covered_states.add(s.id)
        coverage = (len(covered_states) / all_states) if all_states else 0
        return {
            'labels': labels.tolist(),
            'medoids': medoids,
            'n_clusters': len(medoids),
            'n_total': len(labels),
            'n_noise': int((np.array(labels) == -1).sum()),
            'coverage': coverage,
            'grade_levels': grade_levels,
        }

    @transaction.atomic
    def persist_proxies(self, results: Dict) -> int:
        created_count = 0
        grade_levels = results.get('grade_levels', [])
        
        for cluster_id, data in results['medoids'].items():
            medoid_atom = self.atoms[data['medoid_idx']]
            members = [self.atoms[i] for i in data['member_indices']]
            
            # Calculate grade range from members
            min_grade, max_grade = self._calculate_grade_range(members)
            
            # Generate proxy ID with grade information
            proxy_id = self._generate_proxy_id(cluster_id, min_grade, max_grade)
            
            defaults = {
                'cluster_id': int(cluster_id),
                'medoid_atom': medoid_atom,
                'centroid_embedding': data['centroid'].tolist(),
                'coverage_count': int(data['state_count']),
                'avg_similarity': float(data['avg_similarity']),
                'min_grade': min_grade,
                'max_grade': max_grade,
            }
            proxy, created = ProxyStandard.objects.get_or_create(
                proxy_id=proxy_id,
                defaults=defaults,
            )
            if not created:
                # Update fields on rerun to be idempotent
                for key, value in defaults.items():
                    setattr(proxy, key, value)
                proxy.save(update_fields=list(defaults.keys()))
            else:
                created_count += 1

            # Update members set each run
            proxy.member_atoms.set(members)
            
            # Update grade levels
            proxy.update_grade_range_from_atoms()
            proxy.save()

            # Recompute per-state coverage counts for this proxy (clear then insert)
            ProxyStateCoverage.objects.filter(proxy=proxy).delete()
            per_state: Dict[str, int] = {}
            for a in members:
                code = a.standard.state.code
                per_state[code] = per_state.get(code, 0) + 1
            for code, count in per_state.items():
                state_obj = next((a.standard.state for a in members if a.standard.state.code == code), None)
                if state_obj:
                    ProxyStateCoverage.objects.create(proxy=proxy, state=state_obj, atom_count=count)

        return created_count
    
    def _calculate_grade_range(self, atoms: List[StandardAtom]) -> Tuple[int, int]:
        """Calculate min and max grade levels from atoms."""
        grade_levels = GradeLevel.objects.filter(
            standards__atoms__in=atoms
        ).distinct().values_list('grade_numeric', flat=True)
        
        if grade_levels:
            return min(grade_levels), max(grade_levels)
        return None, None
    
    def _generate_proxy_id(self, cluster_id: int, min_grade: int = None, max_grade: int = None) -> str:
        """Generate proxy ID with grade level information."""
        if min_grade is not None and max_grade is not None:
            if min_grade == max_grade:
                grade_part = f"Grade{min_grade}"
            else:
                grade_part = f"Grade{min_grade}-{max_grade}"
        else:
            grade_part = "AllGrades"
        
        # Find the next sequence number for this grade range
        existing_count = ProxyStandard.objects.filter(
            proxy_id__startswith=f"PS-{grade_part}-"
        ).count()
        
        sequence = existing_count + 1
        return f"PS-{grade_part}-{sequence:03d}"


class StandardClusteringService(BaseService):
    """Service for clustering Standard objects (not StandardAtoms) to create ProxyStandards"""
    
    def __init__(self):
        super().__init__()
        self.embeddings = None
        self.standards: List['Standard'] = []

    def load_standard_embeddings(self, grade_levels: List[int] = None) -> Tuple[np.ndarray, List['Standard']]:
        """Load embeddings from Standard objects with optional grade level filtering.
        
        Args:
            grade_levels: List of grade level numeric values to filter by (e.g., [3, 4])
        """
        standards_query = Standard.objects.filter(embedding__isnull=False).select_related(
            'state', 'subject_area'
        ).prefetch_related('grade_levels')
        
        # Filter by grade levels if specified
        if grade_levels:
            standards_query = standards_query.filter(
                grade_levels__grade_numeric__in=grade_levels
            ).distinct()
        
        standards = list(standards_query)
        if not standards:
            raise ValueError("No Standard embeddings found for the specified criteria")
        
        X = np.array([s.embedding for s in standards])
        self.embeddings = X
        self.standards = standards
        return X, standards

    def run_umap(self, X: np.ndarray, n_components: int = 25) -> np.ndarray:
        """Run UMAP dimensionality reduction on Standard embeddings"""
        if umap is None:
            raise RuntimeError("umap-learn not installed")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42,
        )
        return reducer.fit_transform(X)

    def run_hdbscan(self, X_reduced: np.ndarray, min_cluster_size: int = 8, epsilon: float = 0.15) -> np.ndarray:
        """Run HDBSCAN clustering on reduced Standard embeddings"""
        if hdbscan is None:
            raise RuntimeError("hdbscan not installed")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=epsilon,
            metric='euclidean',
            cluster_selection_method='eom',
        )
        return clusterer.fit_predict(X_reduced)

    def find_medoids(self, labels: np.ndarray) -> Dict[int, Dict]:
        """Find medoid standards for each cluster"""
        medoids: Dict[int, Dict] = {}
        for label in set(labels):
            if label == -1:
                continue
            idxs = np.where(labels == label)[0]
            cluster_vecs = self.embeddings[idxs]
            centroid = cluster_vecs.mean(axis=0)
            sims = cosine_similarity(cluster_vecs, [centroid]).flatten()
            medoid_idx_in_cluster = sims.argmax()
            medoid_idx_global = idxs[medoid_idx_in_cluster]
            states = {self.standards[i].state for i in idxs}
            medoids[label] = {
                'medoid_idx': int(medoid_idx_global),
                'member_indices': idxs.tolist(),
                'member_count': len(idxs),
                'centroid': centroid,
                'avg_similarity': float(sims.mean()),
                'states_covered': list(states),
                'state_count': len(states),
            }
        return medoids

    def run_full(self, min_cluster_size: int = 8, epsilon: float = 0.15, grade_levels: List[int] = None) -> Dict:
        """Run full standard-level clustering pipeline with optional grade level filtering.
        
        Args:
            min_cluster_size: Minimum cluster size for HDBSCAN
            epsilon: Epsilon value for HDBSCAN
            grade_levels: List of grade level numeric values to filter by
        """
        X, _ = self.load_standard_embeddings(grade_levels=grade_levels)
        Xr = self.run_umap(X)
        labels = self.run_hdbscan(Xr, min_cluster_size=min_cluster_size, epsilon=epsilon)
        medoids = self.find_medoids(labels)
        
        all_states = State.objects.count()
        covered_states = set()
        for c in medoids.values():
            for s in c['states_covered']:
                covered_states.add(s.id)
        coverage = (len(covered_states) / all_states) if all_states else 0
        
        return {
            'labels': labels.tolist(),
            'medoids': medoids,
            'n_clusters': len(medoids),
            'n_total': len(labels),
            'n_noise': int((np.array(labels) == -1).sum()),
            'coverage': coverage,
            'grade_levels': grade_levels,
        }

    @transaction.atomic
    def persist_standard_proxies(self, results: Dict) -> int:
        """Create ProxyStandards from clustered Standard objects"""
        created_count = 0
        grade_levels = results.get('grade_levels', [])
        
        for cluster_id, data in results['medoids'].items():
            medoid_standard = self.standards[data['medoid_idx']]
            members = [self.standards[i] for i in data['member_indices']]
            
            # Calculate grade range from members
            min_grade, max_grade = self._calculate_grade_range_from_standards(members)
            
            # Generate proxy ID with grade information and STD prefix
            proxy_id = self._generate_standard_proxy_id(cluster_id, min_grade, max_grade)
            
            defaults = {
                'source_type': 'standards',
                'cluster_id': int(cluster_id),
                'medoid_standard': medoid_standard,
                'centroid_embedding': data['centroid'].tolist(),
                'coverage_count': int(data['state_count']),
                'avg_similarity': float(data['avg_similarity']),
                'min_grade': min_grade,
                'max_grade': max_grade,
            }
            
            proxy, created = ProxyStandard.objects.get_or_create(
                proxy_id=proxy_id,
                defaults=defaults,
            )
            if not created:
                # Update fields on rerun to be idempotent
                for key, value in defaults.items():
                    setattr(proxy, key, value)
                proxy.save(update_fields=list(defaults.keys()))
            else:
                created_count += 1

            # Update members set each run
            proxy.member_standards.set(members)
            
            # Update grade levels
            proxy.update_grade_range_from_members()
            proxy.save()

            # Recompute per-state coverage counts for this proxy (clear then insert)
            ProxyStateCoverage.objects.filter(proxy=proxy).delete()
            per_state: Dict[str, int] = {}
            for standard in members:
                code = standard.state.code
                per_state[code] = per_state.get(code, 0) + 1
            for code, count in per_state.items():
                state_obj = next((s.state for s in members if s.state.code == code), None)
                if state_obj:
                    ProxyStateCoverage.objects.create(proxy=proxy, state=state_obj, atom_count=count)

        return created_count
    
    def _calculate_grade_range_from_standards(self, standards: List['Standard']) -> Tuple[int, int]:
        """Calculate min and max grade levels from standards."""
        grade_levels = GradeLevel.objects.filter(
            standards__in=standards
        ).distinct().values_list('grade_numeric', flat=True)
        
        if grade_levels:
            return min(grade_levels), max(grade_levels)
        return None, None
    
    def _generate_standard_proxy_id(self, cluster_id: int, min_grade: int = None, max_grade: int = None) -> str:
        """Generate proxy ID with grade level information for standard-level clustering."""
        if min_grade is not None and max_grade is not None:
            if min_grade == max_grade:
                grade_part = f"Grade{min_grade}"
            else:
                grade_part = f"Grade{min_grade}-{max_grade}"
        else:
            grade_part = "AllGrades"
        
        # Find the next sequence number for this grade range (STD prefix)
        existing_count = ProxyStandard.objects.filter(
            proxy_id__startswith=f"PS-STD-{grade_part}-"
        ).count()
        
        sequence = existing_count + 1
        return f"PS-STD-{grade_part}-{sequence:03d}"



