"""
Storyline discovery service for finding common educational threads across states
"""
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from django.db.models import Count, Q, Avg
from pgvector.django import CosineDistance
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA

from .base import BaseService
from .embedding import EmbeddingService
from ..models import Standard, State, SubjectArea, GradeLevel, TopicCluster


class StorylineDiscoveryService(BaseService):
    """Service for discovering common educational storylines and progressions"""
    
    def __init__(self):
        super().__init__()
        self.embedding_service = EmbeddingService()
    
    def discover_storylines(
        self,
        subject_area: SubjectArea,
        start_grade: int = 0,  # K
        end_grade: int = 12
    ) -> Dict[str, Any]:
        """
        Discover educational storylines that progress through grade levels.
        
        This finds common threads that run through the curriculum from
        elementary to high school across multiple states.
        
        Args:
            subject_area: Subject area to analyze
            start_grade: Starting grade level (0 for K)
            end_grade: Ending grade level
            
        Returns:
            Dictionary containing discovered storylines and progressions
        """
        storylines = []
        
        # Get standards for the subject across all grades
        standards_by_grade = {}
        for grade_num in range(start_grade, end_grade + 1):
            grade_level = GradeLevel.objects.filter(grade_numeric=grade_num).first()
            if grade_level:
                standards = Standard.objects.filter(
                    subject_area=subject_area,
                    grade_levels=grade_level
                ).select_related('state')
                standards_by_grade[grade_num] = list(standards)
        
        # Cluster standards at each grade level
        clusters_by_grade = {}
        for grade_num, standards in standards_by_grade.items():
            if standards:
                clusters = self._cluster_standards(standards)
                clusters_by_grade[grade_num] = clusters
        
        # Trace connections between clusters across grades
        storylines = self._trace_cluster_connections(clusters_by_grade)
        
        # Identify major themes
        major_themes = self._identify_major_themes(storylines, standards_by_grade)
        
        # Find prerequisite relationships
        prerequisites = self._find_prerequisite_relationships(standards_by_grade)
        
        return {
            'storylines': storylines,
            'major_themes': major_themes,
            'prerequisites': prerequisites,
            'grade_progression': self._create_grade_progression_map(storylines),
            'coverage_by_state': self._analyze_state_coverage(storylines, standards_by_grade)
        }
    
    def find_common_threads(
        self,
        min_state_coverage: int = 30,
        subject_area: Optional[SubjectArea] = None
    ) -> List[Dict[str, Any]]:
        """
        Find educational concepts that appear across many states.
        
        Args:
            min_state_coverage: Minimum number of states that must cover a concept
            subject_area: Optional subject area filter
            
        Returns:
            List of common threads with state coverage information
        """
        # Get all standards with embeddings
        query = Standard.objects.filter(embedding__isnull=False)
        if subject_area:
            query = query.filter(subject_area=subject_area)
        
        standards = query.select_related('state', 'subject_area').prefetch_related('grade_levels')
        
        # Group standards by similarity
        if not standards.exists():
            return []
        
        # Extract embeddings and perform clustering
        embeddings = []
        standard_list = []
        for standard in standards:
            if standard.embedding:
                embeddings.append(standard.embedding)
                standard_list.append(standard)
        
        if not embeddings:
            return []
        
        # Perform clustering
        clusters = self._perform_clustering(np.array(embeddings))
        
        # Analyze each cluster for common threads
        common_threads = []
        for cluster_id in set(clusters):
            if cluster_id == -1:  # Skip noise points
                continue
            
            cluster_standards = [s for i, s in enumerate(standard_list) if clusters[i] == cluster_id]
            
            # Count unique states in this cluster
            unique_states = set(s.state.code for s in cluster_standards)
            
            if len(unique_states) >= min_state_coverage:
                # Extract common themes from cluster
                theme = self._extract_cluster_theme(cluster_standards)
                
                common_threads.append({
                    'theme': theme,
                    'state_coverage': len(unique_states),
                    'states': list(unique_states),
                    'total_standards': len(cluster_standards),
                    'grade_distribution': self._get_grade_distribution(cluster_standards),
                    'example_standards': self._get_example_standards(cluster_standards, limit=5)
                })
        
        # Sort by state coverage
        common_threads.sort(key=lambda x: x['state_coverage'], reverse=True)
        
        return common_threads
    
    def analyze_regional_patterns(self) -> Dict[str, Any]:
        """
        Analyze regional patterns in educational standards.
        
        Returns:
            Dictionary containing regional analysis
        """
        # Define regions (simplified US regions)
        regions = {
            'Northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
            'Southeast': ['DE', 'MD', 'DC', 'VA', 'WV', 'KY', 'TN', 'NC', 'SC', 'GA', 'FL', 'AL', 'MS', 'LA', 'AR'],
            'Midwest': ['OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
            'Southwest': ['TX', 'OK', 'NM', 'AZ'],
            'West': ['CO', 'WY', 'MT', 'ID', 'UT', 'NV', 'CA', 'OR', 'WA', 'AK', 'HI']
        }
        
        regional_analysis = {}
        
        for region_name, state_codes in regions.items():
            # Get standards for this region
            regional_standards = Standard.objects.filter(
                state__code__in=state_codes,
                embedding__isnull=False
            )
            
            if not regional_standards.exists():
                continue
            
            # Find unique patterns in this region
            unique_patterns = self._find_unique_regional_patterns(
                regional_standards,
                state_codes
            )
            
            # Calculate similarity within region
            intra_regional_similarity = self._calculate_regional_similarity(
                regional_standards
            )
            
            regional_analysis[region_name] = {
                'states': state_codes,
                'total_standards': regional_standards.count(),
                'unique_patterns': unique_patterns,
                'intra_regional_similarity': intra_regional_similarity,
                'common_topics': self._get_common_regional_topics(regional_standards)
            }
        
        # Find cross-regional patterns
        cross_regional = self._analyze_cross_regional_patterns(regions)
        
        return {
            'regional_analysis': regional_analysis,
            'cross_regional_patterns': cross_regional,
            'most_consistent_region': max(
                regional_analysis.items(),
                key=lambda x: x[1].get('intra_regional_similarity', 0)
            )[0] if regional_analysis else None
        }
    
    def create_learning_pathways(
        self,
        target_concept: str,
        subject_area: SubjectArea,
        max_grade_span: int = 3
    ) -> Dict[str, Any]:
        """
        Create learning pathways that lead to a target concept.
        
        Args:
            target_concept: The target learning concept
            subject_area: Subject area
            max_grade_span: Maximum number of grades in the pathway
            
        Returns:
            Dictionary containing learning pathways
        """
        # Generate embedding for target concept
        target_embedding = self.embedding_service.generate_embedding(target_concept)
        if not target_embedding:
            return {'error': 'Could not generate embedding for target concept'}
        
        # Find standards matching the target concept
        target_standards = self._find_similar_standards(
            target_embedding,
            subject_area=subject_area,
            threshold=0.8
        )
        
        if not target_standards:
            return {'error': 'No standards found matching the target concept'}
        
        # Group target standards by grade
        target_grades = {}
        for standard in target_standards:
            for grade in standard.grade_levels.all():
                if grade.grade_numeric not in target_grades:
                    target_grades[grade.grade_numeric] = []
                target_grades[grade.grade_numeric].append(standard)
        
        # Build pathways for each target grade
        pathways = []
        for target_grade, standards in target_grades.items():
            # Find prerequisite concepts for lower grades
            pathway = self._build_learning_pathway(
                target_grade=target_grade,
                target_standards=standards,
                subject_area=subject_area,
                max_grade_span=max_grade_span
            )
            pathways.append(pathway)
        
        return {
            'target_concept': target_concept,
            'pathways': pathways,
            'coverage_analysis': self._analyze_pathway_coverage(pathways),
            'recommended_pathway': self._select_optimal_pathway(pathways)
        }
    
    def _cluster_standards(self, standards: List[Standard]) -> List[Dict[str, Any]]:
        """Cluster standards based on their embeddings"""
        if not standards:
            return []
        
        embeddings = []
        valid_standards = []
        
        for standard in standards:
            if standard.embedding:
                embeddings.append(standard.embedding)
                valid_standards.append(standard)
        
        if len(embeddings) < 2:
            return [{'standards': valid_standards, 'cluster_id': 0}]
        
        # Use KMeans for clustering
        n_clusters = min(5, len(embeddings) // 10 + 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings)
        
        # Group standards by cluster
        cluster_groups = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(valid_standards[i])
        
        return [
            {
                'cluster_id': cid,
                'standards': stds,
                'centroid': kmeans.cluster_centers_[cid].tolist(),
                'size': len(stds)
            }
            for cid, stds in cluster_groups.items()
        ]
    
    def _trace_cluster_connections(
        self,
        clusters_by_grade: Dict[int, List[Dict]]
    ) -> List[Dict[str, Any]]:
        """Trace connections between clusters across grade levels"""
        storylines = []
        
        grade_levels = sorted(clusters_by_grade.keys())
        if len(grade_levels) < 2:
            return storylines
        
        # Start from the lowest grade
        for start_cluster in clusters_by_grade.get(grade_levels[0], []):
            storyline = {
                'start_grade': grade_levels[0],
                'clusters': [start_cluster],
                'progression': []
            }
            
            # Trace through subsequent grades
            current_centroid = np.array(start_cluster['centroid'])
            
            for i in range(1, len(grade_levels)):
                grade = grade_levels[i]
                next_clusters = clusters_by_grade.get(grade, [])
                
                if not next_clusters:
                    continue
                
                # Find most similar cluster in next grade
                best_match = None
                best_similarity = -1
                
                for next_cluster in next_clusters:
                    next_centroid = np.array(next_cluster['centroid'])
                    similarity = np.dot(current_centroid, next_centroid) / (
                        np.linalg.norm(current_centroid) * np.linalg.norm(next_centroid)
                    )
                    
                    if similarity > best_similarity and similarity > 0.6:
                        best_similarity = similarity
                        best_match = next_cluster
                
                if best_match:
                    storyline['clusters'].append(best_match)
                    storyline['progression'].append({
                        'from_grade': grade_levels[i-1],
                        'to_grade': grade,
                        'similarity': float(best_similarity)
                    })
                    current_centroid = np.array(best_match['centroid'])
            
            if len(storyline['clusters']) > 1:
                storylines.append(storyline)
        
        return storylines
    
    def _identify_major_themes(
        self,
        storylines: List[Dict],
        standards_by_grade: Dict[int, List[Standard]]
    ) -> List[Dict[str, Any]]:
        """Identify major themes from storylines"""
        themes = []
        
        for storyline in storylines:
            # Collect all standards in this storyline
            all_standards = []
            for cluster in storyline['clusters']:
                all_standards.extend(cluster['standards'])
            
            if not all_standards:
                continue
            
            # Extract common keywords and concepts
            keywords = {}
            domains = {}
            
            for standard in all_standards:
                # Count keywords
                if hasattr(standard, 'keywords') and standard.keywords:
                    for keyword in standard.keywords:
                        keywords[keyword] = keywords.get(keyword, 0) + 1
                
                # Count domains
                if standard.domain:
                    domains[standard.domain] = domains.get(standard.domain, 0) + 1
            
            # Identify theme based on most common elements
            theme_name = max(domains, key=domains.get) if domains else "General"
            top_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:5]
            
            themes.append({
                'name': theme_name,
                'keywords': [k for k, _ in top_keywords],
                'grade_span': f"{storyline['start_grade']}-{storyline['clusters'][-1]['cluster_id']}",
                'standard_count': len(all_standards),
                'state_coverage': len(set(s.state.code for s in all_standards))
            })
        
        return themes
    
    def _find_prerequisite_relationships(
        self,
        standards_by_grade: Dict[int, List[Standard]]
    ) -> List[Dict[str, Any]]:
        """Find prerequisite relationships between concepts across grades"""
        prerequisites = []
        
        grade_levels = sorted(standards_by_grade.keys())
        
        for i in range(len(grade_levels) - 1):
            current_grade = grade_levels[i]
            next_grade = grade_levels[i + 1]
            
            current_standards = standards_by_grade[current_grade]
            next_standards = standards_by_grade[next_grade]
            
            # Find strongly connected standard pairs
            for curr_std in current_standards[:10]:  # Limit for performance
                if not curr_std.embedding:
                    continue
                
                for next_std in next_standards[:10]:
                    if not next_std.embedding:
                        continue
                    
                    # Calculate similarity
                    similarity = np.dot(curr_std.embedding, next_std.embedding) / (
                        np.linalg.norm(curr_std.embedding) * np.linalg.norm(next_std.embedding)
                    )
                    
                    if similarity > 0.75:  # Strong connection
                        prerequisites.append({
                            'prerequisite': {
                                'grade': current_grade,
                                'standard': curr_std.code,
                                'title': curr_std.title
                            },
                            'leads_to': {
                                'grade': next_grade,
                                'standard': next_std.code,
                                'title': next_std.title
                            },
                            'strength': float(similarity)
                        })
        
        # Sort by strength
        prerequisites.sort(key=lambda x: x['strength'], reverse=True)
        
        return prerequisites[:20]  # Return top 20 relationships
    
    def _perform_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Perform DBSCAN clustering on embeddings"""
        # Reduce dimensionality for better clustering
        if embeddings.shape[0] > 50:
            pca = PCA(n_components=min(50, embeddings.shape[0] // 2))
            embeddings_reduced = pca.fit_transform(embeddings)
        else:
            embeddings_reduced = embeddings
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=0.3, min_samples=5)
        clusters = clustering.fit_predict(embeddings_reduced)
        
        return clusters
    
    def _extract_cluster_theme(self, standards: List[Standard]) -> str:
        """Extract the main theme from a cluster of standards"""
        # Count domains and keywords
        domains = {}
        keywords = {}
        
        for standard in standards:
            if standard.domain:
                domains[standard.domain] = domains.get(standard.domain, 0) + 1
            
            if hasattr(standard, 'keywords') and standard.keywords:
                for keyword in standard.keywords:
                    keywords[keyword] = keywords.get(keyword, 0) + 1
        
        # Get most common domain
        if domains:
            theme = max(domains, key=domains.get)
        elif keywords:
            theme = max(keywords, key=keywords.get)
        else:
            theme = "General Concept"
        
        return theme
    
    def _get_grade_distribution(self, standards: List[Standard]) -> Dict[str, int]:
        """Get distribution of standards across grade levels"""
        distribution = {}
        
        for standard in standards:
            for grade in standard.grade_levels.all():
                grade_name = f"Grade {grade.grade}"
                distribution[grade_name] = distribution.get(grade_name, 0) + 1
        
        return distribution
    
    def _get_example_standards(self, standards: List[Standard], limit: int = 5) -> List[Dict]:
        """Get example standards from a list"""
        examples = []
        
        for standard in standards[:limit]:
            examples.append({
                'state': standard.state.code,
                'code': standard.code,
                'title': standard.title,
                'grade': [g.grade for g in standard.grade_levels.all()]
            })
        
        return examples
    
    def _find_similar_standards(
        self,
        embedding: List[float],
        subject_area: Optional[SubjectArea] = None,
        threshold: float = 0.7
    ) -> List[Standard]:
        """Find standards similar to given embedding"""
        try:
            query = Standard.objects.filter(embedding__isnull=False)
            
            if subject_area:
                query = query.filter(subject_area=subject_area)
            
            results = query.annotate(
                distance=CosineDistance('embedding', embedding)
            ).filter(
                distance__lt=1 - threshold
            ).select_related('state').prefetch_related('grade_levels')
            
            return list(results)
        except Exception as e:
            print(f"Error finding similar standards: {e}")
            return []
    
    def _create_grade_progression_map(self, storylines: List[Dict]) -> Dict[str, Any]:
        """Create a map of concept progression through grades"""
        progression_map = {}
        
        for storyline in storylines:
            for i, progression in enumerate(storyline.get('progression', [])):
                key = f"Grade {progression['from_grade']} to {progression['to_grade']}"
                if key not in progression_map:
                    progression_map[key] = []
                
                progression_map[key].append({
                    'storyline_id': i,
                    'similarity': progression['similarity']
                })
        
        return progression_map
    
    def _analyze_state_coverage(
        self,
        storylines: List[Dict],
        standards_by_grade: Dict
    ) -> Dict[str, Any]:
        """Analyze how well states are covered by storylines"""
        state_coverage = {}
        
        for storyline in storylines:
            for cluster in storyline['clusters']:
                for standard in cluster['standards']:
                    state_code = standard.state.code
                    if state_code not in state_coverage:
                        state_coverage[state_code] = 0
                    state_coverage[state_code] += 1
        
        return state_coverage
    
    def _find_unique_regional_patterns(
        self,
        regional_standards,
        state_codes: List[str]
    ) -> List[Dict]:
        """Find patterns unique to a region"""
        # This would compare regional standards against national patterns
        # For now, return a simplified version
        return []
    
    def _calculate_regional_similarity(self, regional_standards) -> float:
        """Calculate average similarity within a region"""
        # This would calculate pairwise similarities
        # For now, return a placeholder
        return 0.75
    
    def _get_common_regional_topics(self, regional_standards) -> List[str]:
        """Get common topics in a region"""
        domains = {}
        for standard in regional_standards[:100]:  # Sample for performance
            if standard.domain:
                domains[standard.domain] = domains.get(standard.domain, 0) + 1
        
        return sorted(domains.keys(), key=lambda x: domains[x], reverse=True)[:5]
    
    def _analyze_cross_regional_patterns(self, regions: Dict[str, List[str]]) -> Dict:
        """Analyze patterns across regions"""
        return {
            'common_patterns': [],
            'regional_differences': []
        }
    
    def _build_learning_pathway(
        self,
        target_grade: int,
        target_standards: List[Standard],
        subject_area: SubjectArea,
        max_grade_span: int
    ) -> Dict[str, Any]:
        """Build a learning pathway to target standards"""
        pathway = {
            'target_grade': target_grade,
            'target_standards': [
                {'code': s.code, 'title': s.title}
                for s in target_standards[:3]
            ],
            'steps': []
        }
        
        # Build pathway backwards from target grade
        for grade_offset in range(1, min(max_grade_span + 1, target_grade + 1)):
            prereq_grade = target_grade - grade_offset
            
            # Find related standards at this grade level
            grade_level = GradeLevel.objects.filter(grade_numeric=prereq_grade).first()
            if grade_level:
                prereq_standards = Standard.objects.filter(
                    subject_area=subject_area,
                    grade_levels=grade_level,
                    embedding__isnull=False
                )[:10]  # Limit for performance
                
                if prereq_standards:
                    pathway['steps'].insert(0, {
                        'grade': prereq_grade,
                        'concepts': [s.title for s in prereq_standards[:3]]
                    })
        
        return pathway
    
    def _analyze_pathway_coverage(self, pathways: List[Dict]) -> Dict:
        """Analyze coverage of learning pathways"""
        return {
            'total_pathways': len(pathways),
            'average_steps': sum(len(p['steps']) for p in pathways) / max(len(pathways), 1)
        }
    
    def _select_optimal_pathway(self, pathways: List[Dict]) -> Optional[Dict]:
        """Select the optimal learning pathway"""
        if not pathways:
            return None
        
        # For now, return the pathway with most steps (most comprehensive)
        return max(pathways, key=lambda p: len(p['steps']))