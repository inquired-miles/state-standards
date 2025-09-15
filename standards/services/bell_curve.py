"""
Bell curve analysis service for educational standards coverage
"""
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from django.db.models import Count, Q
from pgvector.django import CosineDistance

from .base import BaseService
from .embedding import EmbeddingService
from ..models import Standard, State, Concept, SubjectArea, GradeLevel


class BellCurveAnalysisService(BaseService):
    """Service for analyzing coverage distribution and finding optimal concept sets"""
    
    def __init__(self):
        super().__init__()
        self.embedding_service = EmbeddingService()
    
    def calculate_bell_curve(
        self,
        concepts: List[str],
        subject_area: Optional[SubjectArea] = None,
        grade_levels: Optional[List[GradeLevel]] = None,
        target_states: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate bell curve distribution for concept coverage across states.
        
        This answers: "If we cover X concepts, how many states will we cover?"
        
        Args:
            concepts: List of concept names or descriptions
            subject_area: Optional subject area filter
            grade_levels: Optional grade level filters
            target_states: Optional list of target state codes (default: all 50)
            
        Returns:
            Dictionary containing bell curve data and coverage analysis
        """
        # Get all states or filter by target
        if target_states:
            states = State.objects.filter(code__in=target_states)
        else:
            states = State.objects.all()
        
        # Initialize coverage tracking
        coverage_by_concept_count = {}
        state_coverage_details = {}
        
        # For each concept combination size (1 to len(concepts))
        for num_concepts in range(1, len(concepts) + 1):
            covered_states = set()
            concept_subset = concepts[:num_concepts]
            
            # Find standards matching these concepts
            for concept in concept_subset:
                # Generate embedding for the concept
                concept_embedding = self.embedding_service.generate_embedding(concept)
                
                if concept_embedding:
                    # Find matching standards across states
                    matching_standards = self._find_matching_standards(
                        concept_embedding,
                        subject_area=subject_area,
                        grade_levels=grade_levels,
                        threshold=0.7
                    )
                    
                    # Track which states are covered
                    for standard in matching_standards:
                        covered_states.add(standard.state.code)
            
            # Record coverage for this concept count
            coverage_percentage = (len(covered_states) / len(states)) * 100
            coverage_by_concept_count[num_concepts] = {
                'concepts_used': concept_subset,
                'states_covered': len(covered_states),
                'coverage_percentage': coverage_percentage,
                'covered_states': list(covered_states)
            }
            
            # Track detailed state coverage
            for state in states:
                if state.code not in state_coverage_details:
                    state_coverage_details[state.code] = {
                        'state_name': state.name,
                        'first_covered_at': None,
                        'concepts_needed': []
                    }
                
                if state.code in covered_states and state_coverage_details[state.code]['first_covered_at'] is None:
                    state_coverage_details[state.code]['first_covered_at'] = num_concepts
                    state_coverage_details[state.code]['concepts_needed'] = concept_subset
        
        # Calculate bell curve statistics
        bell_curve_stats = self._calculate_distribution_stats(coverage_by_concept_count)
        
        # Find optimal coverage points
        optimal_points = self._find_optimal_coverage_points(coverage_by_concept_count)
        
        return {
            'coverage_by_concept_count': coverage_by_concept_count,
            'state_coverage_details': state_coverage_details,
            'bell_curve_stats': bell_curve_stats,
            'optimal_points': optimal_points,
            'total_states': len(states),
            'total_concepts': len(concepts)
        }
    
    def find_minimum_viable_coverage(
        self,
        target_coverage_percentage: float = 80.0,
        subject_area: Optional[SubjectArea] = None,
        grade_levels: Optional[List[GradeLevel]] = None,
        max_concepts: int = 20
    ) -> Dict[str, Any]:
        """
        Find the minimum set of concepts needed to achieve target coverage.
        
        Args:
            target_coverage_percentage: Target percentage of states to cover (0-100)
            subject_area: Optional subject area filter
            grade_levels: Optional grade level filters
            max_concepts: Maximum number of concepts to consider
            
        Returns:
            Dictionary containing optimal concept set and coverage analysis
        """
        # Get high-impact concepts from the database
        concepts = self._identify_high_impact_concepts(
            subject_area=subject_area,
            grade_levels=grade_levels,
            limit=max_concepts
        )
        
        # Use greedy algorithm to find minimum concept set
        selected_concepts = []
        covered_states = set()
        all_states = State.objects.all()
        target_state_count = int(all_states.count() * target_coverage_percentage / 100)
        
        for concept in concepts:
            if len(covered_states) >= target_state_count:
                break
            
            # Check how many new states this concept would cover
            concept_embedding = self.embedding_service.generate_embedding(concept['name'])
            if concept_embedding:
                matching_standards = self._find_matching_standards(
                    concept_embedding,
                    subject_area=subject_area,
                    grade_levels=grade_levels,
                    threshold=0.7
                )
                
                new_states = set()
                for standard in matching_standards:
                    if standard.state.code not in covered_states:
                        new_states.add(standard.state.code)
                
                if new_states:
                    selected_concepts.append({
                        'concept': concept['name'],
                        'new_states_covered': len(new_states),
                        'cumulative_coverage': len(covered_states) + len(new_states),
                        'coverage_percentage': ((len(covered_states) + len(new_states)) / all_states.count()) * 100
                    })
                    covered_states.update(new_states)
        
        return {
            'target_coverage': target_coverage_percentage,
            'achieved_coverage': (len(covered_states) / all_states.count()) * 100,
            'concepts_needed': len(selected_concepts),
            'selected_concepts': selected_concepts,
            'covered_states': list(covered_states),
            'uncovered_states': list(set(s.code for s in all_states) - covered_states),
            'efficiency_score': (len(covered_states) / max(len(selected_concepts), 1))  # States per concept
        }
    
    def analyze_coverage_distribution(
        self,
        subject_area: Optional[SubjectArea] = None,
        grade_levels: Optional[List[GradeLevel]] = None
    ) -> Dict[str, Any]:
        """
        Analyze the distribution of concept coverage across all states.
        
        Returns:
            Dictionary containing distribution analysis and visualiation data
        """
        # Get coverage data for all concepts
        concepts = Concept.objects.all()
        
        if subject_area:
            concepts = concepts.filter(subject_areas=subject_area)
        if grade_levels:
            concepts = concepts.filter(grade_levels__in=grade_levels)
        
        # Create distribution bins
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        distribution = {f"{bins[i]}-{bins[i+1]}%": 0 for i in range(len(bins)-1)}
        
        # Categorize concepts by coverage
        for concept in concepts:
            coverage_pct = concept.coverage_percentage
            for i in range(len(bins)-1):
                if bins[i] <= coverage_pct < bins[i+1]:
                    distribution[f"{bins[i]}-{bins[i+1]}%"] += 1
                    break
        
        # Calculate statistics
        coverage_values = [c.coverage_percentage for c in concepts]
        if coverage_values:
            mean_coverage = np.mean(coverage_values)
            std_coverage = np.std(coverage_values)
            median_coverage = np.median(coverage_values)
        else:
            mean_coverage = std_coverage = median_coverage = 0
        
        # Find concepts at different percentiles
        percentile_concepts = {}
        for percentile in [25, 50, 75, 90, 95]:
            if coverage_values:
                threshold = np.percentile(coverage_values, percentile)
                matching_concepts = concepts.filter(
                    coverage_percentage__gte=threshold
                ).order_by('-coverage_percentage')[:5]
                percentile_concepts[f"p{percentile}"] = [
                    {
                        'name': c.name,
                        'coverage': c.coverage_percentage,
                        'states_covered': c.states_covered
                    }
                    for c in matching_concepts
                ]
        
        return {
            'distribution': distribution,
            'statistics': {
                'mean': mean_coverage,
                'std_deviation': std_coverage,
                'median': median_coverage,
                'total_concepts': concepts.count()
            },
            'percentile_concepts': percentile_concepts,
            'visualization_data': self._prepare_visualization_data(coverage_values)
        }
    
    def _find_matching_standards(
        self,
        embedding: List[float],
        subject_area: Optional[SubjectArea] = None,
        grade_levels: Optional[List[GradeLevel]] = None,
        threshold: float = 0.7
    ) -> List[Standard]:
        """Find standards matching the given embedding"""
        try:
            query = Standard.objects.filter(embedding__isnull=False)
            
            if subject_area:
                query = query.filter(subject_area=subject_area)
            if grade_levels:
                query = query.filter(grade_levels__in=grade_levels)
            
            results = query.annotate(
                distance=CosineDistance('embedding', embedding)
            ).filter(
                distance__lt=1 - threshold
            ).select_related('state')
            
            return list(results)
        except Exception as e:
            print(f"Error finding matching standards: {e}")
            return []
    
    def _identify_high_impact_concepts(
        self,
        subject_area: Optional[SubjectArea] = None,
        grade_levels: Optional[List[GradeLevel]] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Identify concepts with highest coverage impact"""
        concepts = Concept.objects.all()
        
        if subject_area:
            concepts = concepts.filter(subject_areas=subject_area)
        if grade_levels:
            concepts = concepts.filter(grade_levels__in=grade_levels)
        
        # Order by coverage percentage and states covered
        concepts = concepts.order_by('-coverage_percentage', '-states_covered')[:limit]
        
        return [
            {
                'name': c.name,
                'coverage_percentage': c.coverage_percentage,
                'states_covered': c.states_covered,
                'complexity_score': c.complexity_score
            }
            for c in concepts
        ]
    
    def _calculate_distribution_stats(
        self,
        coverage_data: Dict[int, Dict]
    ) -> Dict[str, Any]:
        """Calculate statistical measures for the coverage distribution"""
        if not coverage_data:
            return {}
        
        percentages = [data['coverage_percentage'] for data in coverage_data.values()]
        
        return {
            'mean': np.mean(percentages),
            'std_deviation': np.std(percentages),
            'median': np.median(percentages),
            'min': np.min(percentages),
            'max': np.max(percentages),
            'range': np.max(percentages) - np.min(percentages)
        }
    
    def _find_optimal_coverage_points(
        self,
        coverage_data: Dict[int, Dict]
    ) -> List[Dict[str, Any]]:
        """Find optimal points in the coverage curve (best ROI)"""
        optimal_points = []
        
        # Calculate marginal coverage gain for each additional concept
        for i in sorted(coverage_data.keys()):
            if i == 1:
                marginal_gain = coverage_data[i]['coverage_percentage']
            else:
                marginal_gain = (
                    coverage_data[i]['coverage_percentage'] - 
                    coverage_data[i-1]['coverage_percentage']
                )
            
            optimal_points.append({
                'concept_count': i,
                'total_coverage': coverage_data[i]['coverage_percentage'],
                'marginal_gain': marginal_gain,
                'efficiency': coverage_data[i]['coverage_percentage'] / i
            })
        
        # Sort by efficiency to find best ROI points
        optimal_points.sort(key=lambda x: x['efficiency'], reverse=True)
        
        return optimal_points[:5]  # Return top 5 optimal points
    
    def _prepare_visualization_data(
        self,
        coverage_values: List[float]
    ) -> Dict[str, Any]:
        """Prepare data for bell curve visualization"""
        if not coverage_values:
            return {}
        
        # Create histogram data
        hist, bin_edges = np.histogram(coverage_values, bins=20)
        
        return {
            'histogram': {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist(),
                'bin_centers': [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
            },
            'normal_curve': {
                'mean': float(np.mean(coverage_values)),
                'std': float(np.std(coverage_values)),
                'x_values': np.linspace(0, 100, 100).tolist()
            }
        }