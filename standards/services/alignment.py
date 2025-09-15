"""
Content alignment service for analyzing educational content
"""
import hashlib
from typing import List, Dict, Any, Optional
from .base import BaseService
from .embedding import EmbeddingService
from .search import SearchService
from ..models import ContentAlignment, Standard, State


class ContentAlignmentService(BaseService):
    """Service for analyzing content alignment with standards"""
    
    def __init__(self):
        super().__init__()
        self.embedding_service = EmbeddingService()
        self.search_service = SearchService()
    
    def analyze_content_alignment(
        self,
        content_text: str,
        content_title: str,
        target_states: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze how well content aligns with standards
        
        Args:
            content_text: The content to analyze
            content_title: Title of the content
            target_states: Optional list of target state codes
            
        Returns:
            Dictionary containing alignment analysis
        """
        # Generate content hash
        content_hash = hashlib.sha256(content_text.encode()).hexdigest()
        
        # Generate embedding for content
        content_embedding = self.embedding_service.generate_embedding(
            f"{content_title}. {content_text}"
        )
        
        if not content_embedding:
            return {
                'error': 'Could not generate embedding for content',
                'alignment_score': 0.0
            }
        
        # Find matching standards using semantic search
        matching_standards = self.search_service.semantic_search(
            content_text,
            threshold=0.6,  # Lower threshold for content alignment
            limit=50
        )
        
        # Calculate alignment metrics
        alignment_analysis = self._calculate_alignment_metrics(
            matching_standards, target_states
        )
        
        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(
            alignment_analysis, matching_standards
        )
        
        return {
            'content_hash': content_hash,
            'total_standards_analyzed': len(matching_standards),
            'alignment_analysis': alignment_analysis,
            'improvement_suggestions': suggestions,
            'matching_standards': matching_standards[:20]  # Top 20 matches
        }
    
    def _calculate_alignment_metrics(
        self,
        matching_standards: List[Dict[str, Any]],
        target_states: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Calculate alignment metrics from matching standards"""
        
        if not matching_standards:
            return {
                'overall_score': 0.0,
                'exact_matches': 0,
                'semantic_matches': 0,
                'state_coverage': {},
                'subject_coverage': {}
            }
        
        # Categorize matches by similarity score
        exact_matches = sum(1 for m in matching_standards if m['similarity_score'] >= 0.9)
        semantic_matches = sum(1 for m in matching_standards if 0.7 <= m['similarity_score'] < 0.9)
        
        # Calculate state coverage
        state_coverage = {}
        for match in matching_standards:
            state_code = match['standard'].state.code
            if state_code not in state_coverage:
                state_coverage[state_code] = {
                    'standards_count': 0,
                    'max_similarity': 0.0,
                    'avg_similarity': 0.0
                }
            
            state_coverage[state_code]['standards_count'] += 1
            state_coverage[state_code]['max_similarity'] = max(
                state_coverage[state_code]['max_similarity'],
                match['similarity_score']
            )
        
        # Calculate average similarities
        for state_data in state_coverage.values():
            state_matches = [
                m for m in matching_standards 
                if m['standard'].state.code in state_coverage
            ]
            if state_matches:
                state_data['avg_similarity'] = sum(
                    m['similarity_score'] for m in state_matches
                ) / len(state_matches)
        
        # Calculate overall alignment score
        if matching_standards:
            overall_score = sum(m['similarity_score'] for m in matching_standards) / len(matching_standards)
        else:
            overall_score = 0.0
        
        return {
            'overall_score': overall_score * 100,  # Convert to percentage
            'exact_matches': exact_matches,
            'semantic_matches': semantic_matches,
            'state_coverage': state_coverage,
            'total_states_covered': len(state_coverage)
        }
    
    def _generate_improvement_suggestions(
        self,
        alignment_analysis: Dict[str, Any],
        matching_standards: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate suggestions for improving content alignment"""
        
        suggestions = []
        
        overall_score = alignment_analysis['overall_score']
        
        if overall_score < 60:
            suggestions.append(
                "Consider revising content to better align with educational standards. "
                "Focus on incorporating key learning objectives and skill development."
            )
        
        if alignment_analysis['exact_matches'] == 0:
            suggestions.append(
                "No exact matches found. Review content against specific state standards "
                "to identify gaps in curriculum alignment."
            )
        
        if alignment_analysis['total_states_covered'] < 10:
            suggestions.append(
                "Content aligns with fewer than 10 states. Consider broadening content "
                "to address more common educational themes across states."
            )
        
        # Analyze missing subjects or topics
        if matching_standards:
            common_domains = {}
            for match in matching_standards:
                domain = match['standard'].domain
                if domain:
                    common_domains[domain] = common_domains.get(domain, 0) + 1
            
            if common_domains:
                top_domain = max(common_domains, key=common_domains.get)
                suggestions.append(
                    f"Content shows strong alignment with '{top_domain}' standards. "
                    f"Consider expanding coverage in this area."
                )
        
        return suggestions