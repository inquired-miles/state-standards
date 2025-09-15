"""
Coverage analysis service for standards alignment system
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
from django.db.models import Count, Q, Avg, Max, Min
from django.utils import timezone
from .base import BaseService
from ..models import (
    State, SubjectArea, GradeLevel, Standard, Concept, 
    CoverageAnalysis, TopicCluster
)


class CoverageAnalysisService(BaseService):
    """Service for analyzing standards coverage across states"""
    
    def __init__(self):
        super().__init__()
        self.similarity_threshold = self.get_edtech_setting('DEFAULT_SIMILARITY_THRESHOLD', 0.8)
    
    def analyze_state_coverage(
        self, 
        state: Optional[State] = None,
        subject_area: Optional[SubjectArea] = None,
        grade_level: Optional[GradeLevel] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze coverage for a specific state, subject, or grade level
        
        Args:
            state: Optional state to analyze
            subject_area: Optional subject area to analyze
            grade_level: Optional grade level to analyze
            force_refresh: Whether to force refresh cached results
            
        Returns:
            Dictionary containing coverage analysis results
        """
        # Generate cache key
        cache_key = self.generate_cache_key(
            "coverage_analysis",
            state_id=state.id if state else "all",
            subject_id=subject_area.id if subject_area else "all",
            grade_id=grade_level.id if grade_level else "all"
        )
        
        if not force_refresh:
            cached_result = self.get_cached_result(cache_key)
            if cached_result:
                return cached_result
        
        # Build queryset
        standards_qs = Standard.objects.all()
        
        if state:
            standards_qs = standards_qs.filter(state=state)
        if subject_area:
            standards_qs = standards_qs.filter(subject_area=subject_area)
        if grade_level:
            standards_qs = standards_qs.filter(grade_levels=grade_level)
        
        # Perform analysis
        result, execution_time = self.measure_execution_time(
            self._perform_coverage_analysis,
            standards_qs, state, subject_area, grade_level
        )
        
        # Cache result
        self.set_cached_result(cache_key, result)
        
        # Log performance
        self.log_performance(
            "analyze_state_coverage",
            execution_time,
            state=state.code if state else "all",
            subject=subject_area.name if subject_area else "all",
            standards_count=standards_qs.count()
        )
        
        return result
    
    def _perform_coverage_analysis(
        self,
        standards_qs,
        state: Optional[State],
        subject_area: Optional[SubjectArea],
        grade_level: Optional[GradeLevel]
    ) -> Dict[str, Any]:
        """Perform the actual coverage analysis"""
        
        total_standards = standards_qs.count()
        
        if total_standards == 0:
            return {
                'total_standards': 0,
                'covered_concepts': 0,
                'coverage_percentage': 0.0,
                'bell_curve_data': [],
                'gap_analysis': {},
                'benchmark_comparison': {},
                'analysis_metadata': {
                    'state': state.name if state else "All States",
                    'subject': subject_area.name if subject_area else "All Subjects",
                    'grade': grade_level.grade if grade_level else "All Grades",
                    'analysis_date': timezone.now().isoformat()
                }
            }
        
        # Analyze concept coverage
        concept_coverage = self._analyze_concept_coverage(standards_qs)
        
        # Generate bell curve data
        bell_curve_data = self._generate_bell_curve_data(concept_coverage)
        
        # Perform gap analysis
        gap_analysis = self._perform_gap_analysis(standards_qs, state, subject_area, grade_level)
        
        # Generate benchmark comparison
        benchmark_comparison = self._generate_benchmark_comparison(
            standards_qs, state, subject_area, grade_level
        )
        
        # Calculate coverage metrics
        covered_concepts = len([c for c in concept_coverage.values() if c['coverage'] > 0])
        coverage_percentage = (covered_concepts / len(concept_coverage)) * 100 if concept_coverage else 0
        
        return {
            'total_standards': total_standards,
            'covered_concepts': covered_concepts,
            'coverage_percentage': coverage_percentage,
            'bell_curve_data': bell_curve_data,
            'gap_analysis': gap_analysis,
            'benchmark_comparison': benchmark_comparison,
            'concept_coverage': concept_coverage,
            'analysis_metadata': {
                'state': state.name if state else "All States",
                'subject': subject_area.name if subject_area else "All Subjects",
                'grade': grade_level.grade if grade_level else "All Grades",
                'analysis_date': timezone.now().isoformat()
            }
        }
    
    def _analyze_concept_coverage(self, standards_qs) -> Dict[str, Dict[str, Any]]:
        """Analyze coverage of concepts within the standards"""
        concept_coverage = defaultdict(lambda: {
            'coverage': 0,
            'standards': [],
            'states': set(),
            'domains': set(),
            'clusters': set()
        })
        
        # Extract concepts from keywords and skills
        for standard in standards_qs.select_related('state', 'subject_area'):
            concepts = []
            
            # Add keywords as concepts
            if standard.keywords:
                concepts.extend(standard.keywords)
            
            # Add skills as concepts
            if standard.skills:
                concepts.extend(standard.skills)
            
            # Add domain and cluster as high-level concepts
            if standard.domain:
                concepts.append(f"Domain: {standard.domain}")
            if standard.cluster:
                concepts.append(f"Cluster: {standard.cluster}")
            
            # Update concept coverage
            for concept in concepts:
                concept_key = concept.lower().strip()
                concept_coverage[concept_key]['coverage'] += 1
                concept_coverage[concept_key]['standards'].append({
                    'id': str(standard.id),
                    'code': standard.code,
                    'title': standard.title,
                    'state': standard.state.code
                })
                concept_coverage[concept_key]['states'].add(standard.state.code)
                if standard.domain:
                    concept_coverage[concept_key]['domains'].add(standard.domain)
                if standard.cluster:
                    concept_coverage[concept_key]['clusters'].add(standard.cluster)
        
        # Convert sets to lists for JSON serialization
        for concept_data in concept_coverage.values():
            concept_data['states'] = list(concept_data['states'])
            concept_data['domains'] = list(concept_data['domains'])
            concept_data['clusters'] = list(concept_data['clusters'])
        
        return dict(concept_coverage)
    
    def _generate_bell_curve_data(self, concept_coverage: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate bell curve distribution data for visualization"""
        coverage_counts = [data['coverage'] for data in concept_coverage.values()]
        
        if not coverage_counts:
            return []
        
        # Create histogram data
        hist, bin_edges = np.histogram(coverage_counts, bins=20)
        
        bell_curve_data = []
        for i in range(len(hist)):
            bell_curve_data.append({
                'bin_start': float(bin_edges[i]),
                'bin_end': float(bin_edges[i + 1]),
                'frequency': int(hist[i]),
                'percentage': float(hist[i] / len(coverage_counts) * 100)
            })
        
        return bell_curve_data
    
    def _perform_gap_analysis(
        self,
        standards_qs,
        state: Optional[State],
        subject_area: Optional[SubjectArea],
        grade_level: Optional[GradeLevel]
    ) -> Dict[str, Any]:
        """Identify gaps in standards coverage"""
        
        # Compare with other states if analyzing a specific state
        if state:
            # Find standards from other states in the same subject/grade
            other_states_qs = Standard.objects.exclude(state=state)
            if subject_area:
                other_states_qs = other_states_qs.filter(subject_area=subject_area)
            if grade_level:
                other_states_qs = other_states_qs.filter(grade_levels=grade_level)
            
            # Find common concepts in other states that are missing
            other_concepts = self._extract_concepts_from_standards(other_states_qs)
            current_concepts = self._extract_concepts_from_standards(standards_qs)
            
            missing_concepts = set(other_concepts.keys()) - set(current_concepts.keys())
            
            gap_analysis = {
                'missing_concepts': list(missing_concepts)[:20],  # Top 20 missing
                'underrepresented_concepts': self._find_underrepresented_concepts(
                    current_concepts, other_concepts
                ),
                'recommendations': self._generate_gap_recommendations(
                    missing_concepts, current_concepts, other_concepts
                )
            }
        else:
            # General gap analysis across all data
            gap_analysis = {
                'low_coverage_concepts': self._find_low_coverage_concepts(standards_qs),
                'uneven_distribution': self._analyze_uneven_distribution(standards_qs),
                'recommendations': []
            }
        
        return gap_analysis
    
    def _extract_concepts_from_standards(self, standards_qs) -> Dict[str, int]:
        """Extract concepts from standards queryset"""
        concepts = Counter()
        
        for standard in standards_qs:
            if standard.keywords:
                concepts.update(keyword.lower().strip() for keyword in standard.keywords)
            if standard.skills:
                concepts.update(skill.lower().strip() for skill in standard.skills)
        
        return dict(concepts)
    
    def _find_underrepresented_concepts(
        self, 
        current_concepts: Dict[str, int], 
        other_concepts: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """Find concepts that are underrepresented compared to other states"""
        underrepresented = []
        
        for concept, other_count in other_concepts.items():
            current_count = current_concepts.get(concept, 0)
            if current_count < other_count * 0.5:  # Less than 50% of other states
                underrepresented.append({
                    'concept': concept,
                    'current_count': current_count,
                    'benchmark_count': other_count,
                    'gap_percentage': ((other_count - current_count) / other_count * 100)
                })
        
        return sorted(underrepresented, key=lambda x: x['gap_percentage'], reverse=True)[:10]
    
    def _find_low_coverage_concepts(self, standards_qs) -> List[Dict[str, Any]]:
        """Find concepts with low coverage across all standards"""
        concept_coverage = self._analyze_concept_coverage(standards_qs)
        
        low_coverage = []
        avg_coverage = np.mean([data['coverage'] for data in concept_coverage.values()])
        
        for concept, data in concept_coverage.items():
            if data['coverage'] < avg_coverage * 0.5:  # Less than 50% of average
                low_coverage.append({
                    'concept': concept,
                    'coverage': data['coverage'],
                    'states_count': len(data['states'])
                })
        
        return sorted(low_coverage, key=lambda x: x['coverage'])[:10]
    
    def _analyze_uneven_distribution(self, standards_qs) -> Dict[str, Any]:
        """Analyze uneven distribution of standards across grades/domains"""
        # Grade distribution
        grade_distribution = defaultdict(int)
        domain_distribution = defaultdict(int)
        
        for standard in standards_qs:
            for grade in standard.grade_levels.all():
                grade_distribution[grade.grade] += 1
            if standard.domain:
                domain_distribution[standard.domain] += 1
        
        return {
            'grade_distribution': dict(grade_distribution),
            'domain_distribution': dict(domain_distribution),
            'grade_imbalance_score': self._calculate_distribution_imbalance(grade_distribution),
            'domain_imbalance_score': self._calculate_distribution_imbalance(domain_distribution)
        }
    
    def _calculate_distribution_imbalance(self, distribution: Dict[str, int]) -> float:
        """Calculate how imbalanced a distribution is (0 = perfectly balanced, 1 = completely imbalanced)"""
        if not distribution:
            return 0.0
        
        values = list(distribution.values())
        if len(values) <= 1:
            return 0.0
        
        # Calculate coefficient of variation
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if mean_val == 0:
            return 0.0
        
        cv = std_val / mean_val
        # Normalize to 0-1 scale (values above 1 are capped at 1)
        return min(cv, 1.0)
    
    def _generate_gap_recommendations(
        self,
        missing_concepts: set,
        current_concepts: Dict[str, int],
        other_concepts: Dict[str, int]
    ) -> List[str]:
        """Generate recommendations for addressing gaps"""
        recommendations = []
        
        if missing_concepts:
            top_missing = sorted(
                missing_concepts, 
                key=lambda x: other_concepts.get(x, 0), 
                reverse=True
            )[:5]
            
            for concept in top_missing:
                recommendations.append(
                    f"Consider adding standards related to '{concept}' "
                    f"(appears in {other_concepts.get(concept, 0)} other state standards)"
                )
        
        # Add more specific recommendations based on analysis
        if len(current_concepts) < len(other_concepts) * 0.7:
            recommendations.append(
                "Standards coverage appears to be below average. "
                "Consider reviewing and expanding the standards framework."
            )
        
        return recommendations
    
    def _generate_benchmark_comparison(
        self,
        standards_qs,
        state: Optional[State],
        subject_area: Optional[SubjectArea],
        grade_level: Optional[GradeLevel]
    ) -> Dict[str, Any]:
        """Generate benchmark comparison with similar contexts"""
        
        if not state:
            return {'message': 'Benchmark comparison requires a specific state'}
        
        # Find similar states (same subject and grade level)
        similar_states_qs = Standard.objects.exclude(state=state)
        if subject_area:
            similar_states_qs = similar_states_qs.filter(subject_area=subject_area)
        if grade_level:
            similar_states_qs = similar_states_qs.filter(grade_levels=grade_level)
        
        # Calculate metrics for comparison
        current_count = standards_qs.count()
        
        # Get stats from similar states
        similar_stats = similar_states_qs.values('state').annotate(
            standards_count=Count('id')
        ).order_by('-standards_count')
        
        if not similar_stats:
            return {'message': 'No comparable data available'}
        
        benchmark_data = {
            'current_state_count': current_count,
            'peer_states': list(similar_stats[:10]),  # Top 10 peer states
            'percentile_rank': self._calculate_percentile_rank(
                current_count, 
                [s['standards_count'] for s in similar_stats]
            ),
            'recommendations': self._generate_benchmark_recommendations(
                current_count, similar_stats
            )
        }
        
        return benchmark_data
    
    def _calculate_percentile_rank(self, value: int, peer_values: List[int]) -> float:
        """Calculate percentile rank of a value among peers"""
        if not peer_values:
            return 50.0
        
        peer_values_sorted = sorted(peer_values)
        below_count = sum(1 for v in peer_values_sorted if v < value)
        
        percentile = (below_count / len(peer_values_sorted)) * 100
        return round(percentile, 1)
    
    def _generate_benchmark_recommendations(
        self, 
        current_count: int, 
        similar_stats: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on peer comparison"""
        recommendations = []
        
        if not similar_stats:
            return recommendations
        
        peer_counts = [s['standards_count'] for s in similar_stats]
        avg_peer_count = np.mean(peer_counts)
        top_25_percentile = np.percentile(peer_counts, 75)
        
        if current_count < avg_peer_count * 0.8:
            recommendations.append(
                f"Current standards count ({current_count}) is below peer average "
                f"({avg_peer_count:.0f}). Consider expanding standards coverage."
            )
        
        if current_count < top_25_percentile:
            recommendations.append(
                f"To reach top quartile performance, consider increasing to "
                f"{top_25_percentile:.0f} standards."
            )
        
        return recommendations
    
    def generate_coverage_report(
        self,
        state: Optional[State] = None,
        subject_area: Optional[SubjectArea] = None,
        grade_level: Optional[GradeLevel] = None
    ) -> CoverageAnalysis:
        """
        Generate and save a comprehensive coverage analysis report
        
        Returns:
            CoverageAnalysis instance
        """
        analysis_data = self.analyze_state_coverage(state, subject_area, grade_level)
        
        # Determine analysis type
        if state and subject_area and grade_level:
            analysis_type = 'comprehensive'
        elif state:
            analysis_type = 'state'
        elif subject_area:
            analysis_type = 'subject'
        elif grade_level:
            analysis_type = 'grade'
        else:
            analysis_type = 'comprehensive'
        
        # Create CoverageAnalysis record
        coverage_analysis = CoverageAnalysis.objects.create(
            state=state,
            subject_area=subject_area,
            grade_level=grade_level,
            analysis_type=analysis_type,
            total_standards=analysis_data['total_standards'],
            covered_concepts=analysis_data['covered_concepts'],
            coverage_percentage=analysis_data['coverage_percentage'],
            bell_curve_data=analysis_data['bell_curve_data'],
            gap_analysis=analysis_data['gap_analysis'],
            benchmark_comparison=analysis_data['benchmark_comparison']
        )
        
        return coverage_analysis