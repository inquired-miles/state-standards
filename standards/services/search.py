"""
Advanced search service for standards and content
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from django.db.models import Q
from pgvector.django import CosineDistance
from .base import BaseService
from .embedding import EmbeddingService
from ..models import Standard, Concept, ContentAlignment

logger = logging.getLogger(__name__)


class SearchService(BaseService):
    """Advanced search service with semantic capabilities"""
    
    def __init__(self):
        super().__init__()
        self.embedding_service = EmbeddingService()
        self.default_threshold = self.get_edtech_setting('DEFAULT_SIMILARITY_THRESHOLD', 0.8)
        self.max_results = self.get_edtech_setting('MAX_SIMILAR_STANDARDS', 50)
    
    def semantic_search(
        self,
        query: str,
        threshold: float = None,
        limit: int = 20,
        state_code: Optional[str] = None,
        subject_id: Optional[int] = None,
        grade_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search on standards using vector similarity
        
        Args:
            query: Search query text
            threshold: Minimum similarity threshold (now only used for logging, not filtering)
            limit: Maximum number of results
            state_code: Optional state filter
            subject_id: Optional subject filter
            grade_id: Optional grade filter
            
        Returns:
            List of search results with similarity scores and alignment indicators
        """
        threshold = threshold or self.default_threshold
        
        # Validate inputs
        if not query or not query.strip():
            logger.warning("Empty query provided to semantic search")
            return []
        
        # Check if embedding service is available
        if not self.embedding_service.is_available():
            logger.error("OpenAI API key not configured - semantic search unavailable")
            return []
        
        # Generate embedding for query
        cache_key = self.generate_cache_key("query_embedding", query=query)
        query_embedding = self.embedding_service.generate_embedding(query, cache_key)
        
        if not query_embedding:
            logger.error(f"Failed to generate embedding for query: {query}")
            return []
        
        # Build base queryset
        standards_qs = Standard.objects.filter(embedding__isnull=False)
        
        # Check if any standards have embeddings
        if not standards_qs.exists():
            logger.warning("No standards with embeddings found in database")
            return []
        
        # Apply filters
        if state_code:
            standards_qs = standards_qs.filter(state__code=state_code)
        if subject_id:
            standards_qs = standards_qs.filter(subject_area_id=subject_id)
        if grade_id:
            standards_qs = standards_qs.filter(grade_levels__grade_numeric=grade_id)
        
        # Perform vector similarity search - return top N results without threshold filtering
        similar_standards = standards_qs.annotate(
            distance=CosineDistance('embedding', query_embedding)
        ).order_by('distance')[:limit]
        
        # Format results with alignment indicators
        results = []
        for standard in similar_standards:
            similarity_score = 1 - standard.distance  # Convert back to similarity
            
            # Add alignment category based on percentage ranges
            similarity_percentage = similarity_score * 100
            
            if similarity_percentage >= 35:
                alignment_category = 'aligned'
                alignment_label = f'Aligned: {similarity_percentage:.1f}%'
            elif similarity_percentage >= 30:
                alignment_category = 'could_align'
                alignment_label = f'Could Align: {similarity_percentage:.1f}%'
            elif similarity_percentage >= 25:
                alignment_category = 'stretch'
                alignment_label = f'Stretch: {similarity_percentage:.1f}%'
            else:
                alignment_category = 'not_aligned'
                alignment_label = f'Not Aligned: {similarity_percentage:.1f}%'
            
            results.append({
                'standard': standard,
                'similarity_score': similarity_score,
                'alignment_category': alignment_category,
                'alignment_label': alignment_label,
                'match_explanation': self._generate_match_explanation(
                    query, standard, similarity_score
                )
            })
        
        return results
    
    def find_similar_standards(
        self,
        reference_standard: Standard,
        threshold: float = None,
        limit: int = 10,
        exclude_same_state: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find standards similar to a reference standard
        
        Args:
            reference_standard: The reference standard
            threshold: Minimum similarity threshold
            limit: Maximum number of results
            exclude_same_state: Whether to exclude standards from the same state
            
        Returns:
            List of similar standards with similarity scores
        """
        threshold = threshold or self.default_threshold
        
        if not reference_standard.embedding:
            return []
        
        # Build queryset
        standards_qs = Standard.objects.filter(
            embedding__isnull=False
        ).exclude(id=reference_standard.id)
        
        if exclude_same_state:
            standards_qs = standards_qs.exclude(state=reference_standard.state)
        
        # Perform similarity search
        similar_standards = standards_qs.annotate(
            distance=CosineDistance('embedding', reference_standard.embedding)
        ).filter(
            distance__lt=1 - threshold
        ).order_by('distance')[:limit]
        
        # Format results
        results = []
        for standard in similar_standards:
            similarity_score = 1 - standard.distance
            
            results.append({
                'standard': standard,
                'similarity_score': similarity_score,
                'match_explanation': self._generate_standard_match_explanation(
                    reference_standard, standard, similarity_score
                )
            })
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        **filters
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword matching
        
        Args:
            query: Search query
            semantic_weight: Weight for semantic similarity
            keyword_weight: Weight for keyword matching
            **filters: Additional filters
            
        Returns:
            List of search results with combined scores
        """
        # Get semantic results
        semantic_results = self.semantic_search(query, **filters)
        
        # Get keyword results
        keyword_results = self._keyword_search(query, **filters)
        
        # Combine and rank results
        combined_results = self._combine_search_results(
            semantic_results, keyword_results, semantic_weight, keyword_weight
        )
        
        return combined_results
    
    def _keyword_search(self, query: str, **filters) -> List[Dict[str, Any]]:
        """Perform keyword-based search"""
        query_terms = query.lower().split()
        
        # Build Q objects for searching
        q_objects = Q()
        for term in query_terms:
            q_objects |= (
                Q(title__icontains=term) |
                Q(description__icontains=term) |
                Q(keywords__icontains=term) |
                Q(skills__icontains=term) |
                Q(domain__icontains=term) |
                Q(cluster__icontains=term)
            )
        
        # Build queryset with filters
        standards_qs = Standard.objects.filter(q_objects)
        
        # Apply additional filters
        state_code = filters.get('state_code')
        if state_code:
            standards_qs = standards_qs.filter(state__code=state_code)
        
        subject_id = filters.get('subject_id')
        if subject_id:
            standards_qs = standards_qs.filter(subject_area_id=subject_id)
        
        grade_id = filters.get('grade_id')
        if grade_id:
            standards_qs = standards_qs.filter(grade_levels__grade_numeric=grade_id)
        
        # Calculate keyword match scores
        results = []
        for standard in standards_qs[:50]:  # Limit for performance
            score = self._calculate_keyword_score(query_terms, standard)
            if score > 0:
                results.append({
                    'standard': standard,
                    'similarity_score': score,
                    'match_explanation': f"Keyword match (score: {score:.2f})"
                })
        
        return sorted(results, key=lambda x: x['similarity_score'], reverse=True)
    
    def _calculate_keyword_score(self, query_terms: List[str], standard: Standard) -> float:
        """Calculate keyword match score for a standard"""
        score = 0.0
        total_terms = len(query_terms)
        
        if total_terms == 0:
            return 0.0
        
        # Check each field for matches
        fields_to_check = [
            (standard.title or '', 3.0),        # Title matches weighted higher
            (standard.description or '', 1.0),  # Description matches
            (standard.domain or '', 2.0),       # Domain matches weighted higher
            (standard.cluster or '', 1.5),      # Cluster matches
        ]
        
        # Add keywords and skills
        if standard.keywords:
            fields_to_check.append((' '.join(standard.keywords), 2.5))
        if standard.skills:
            fields_to_check.append((' '.join(standard.skills), 2.0))
        
        # Calculate matches
        matches = 0
        for term in query_terms:
            term_lower = term.lower()
            for field_text, weight in fields_to_check:
                if term_lower in field_text.lower():
                    score += weight
                    matches += 1
                    break  # Don't double-count same term
        
        # Normalize score
        max_possible_score = total_terms * 3.0  # Max weight
        normalized_score = min(score / max_possible_score, 1.0)
        
        return normalized_score
    
    def _combine_search_results(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        semantic_weight: float,
        keyword_weight: float
    ) -> List[Dict[str, Any]]:
        """Combine semantic and keyword search results"""
        
        # Create lookup for combining results
        combined_scores = {}
        
        # Add semantic results
        for result in semantic_results:
            standard_id = result['standard'].id
            combined_scores[standard_id] = {
                'standard': result['standard'],
                'semantic_score': result['similarity_score'],
                'keyword_score': 0.0,
                'semantic_explanation': result['match_explanation'],
                'keyword_explanation': ''
            }
        
        # Add keyword results
        for result in keyword_results:
            standard_id = result['standard'].id
            if standard_id in combined_scores:
                combined_scores[standard_id]['keyword_score'] = result['similarity_score']
                combined_scores[standard_id]['keyword_explanation'] = result['match_explanation']
            else:
                combined_scores[standard_id] = {
                    'standard': result['standard'],
                    'semantic_score': 0.0,
                    'keyword_score': result['similarity_score'],
                    'semantic_explanation': '',
                    'keyword_explanation': result['match_explanation']
                }
        
        # Calculate combined scores
        final_results = []
        for data in combined_scores.values():
            combined_score = (
                data['semantic_score'] * semantic_weight +
                data['keyword_score'] * keyword_weight
            )
            
            # Create combined explanation
            explanations = []
            if data['semantic_explanation']:
                explanations.append(f"Semantic: {data['semantic_explanation']}")
            if data['keyword_explanation']:
                explanations.append(f"Keyword: {data['keyword_explanation']}")
            
            final_results.append({
                'standard': data['standard'],
                'similarity_score': combined_score,
                'match_explanation': '; '.join(explanations)
            })
        
        # Sort by combined score
        return sorted(final_results, key=lambda x: x['similarity_score'], reverse=True)
    
    def _generate_match_explanation(
        self, 
        query: str, 
        standard: Standard, 
        similarity_score: float
    ) -> str:
        """Generate explanation for why a standard matched the query"""
        
        explanations = []
        
        # Check for direct text matches
        query_lower = query.lower()
        
        if query_lower in (standard.title or '').lower():
            explanations.append("title match")
        if query_lower in (standard.description or '').lower():
            explanations.append("description match")
        if standard.keywords and any(query_lower in kw.lower() for kw in standard.keywords):
            explanations.append("keyword match")
        if standard.skills and any(query_lower in skill.lower() for skill in standard.skills):
            explanations.append("skill match")
        
        # Add semantic similarity info
        if similarity_score > 0.9:
            explanations.append("high semantic similarity")
        elif similarity_score > 0.8:
            explanations.append("good semantic similarity")
        else:
            explanations.append("moderate semantic similarity")
        
        if explanations:
            return f"Match based on: {', '.join(explanations)} (score: {similarity_score:.2f})"
        else:
            return f"Semantic similarity (score: {similarity_score:.2f})"
    
    def _generate_standard_match_explanation(
        self,
        reference_standard: Standard,
        matched_standard: Standard,
        similarity_score: float
    ) -> str:
        """Generate explanation for why two standards are similar"""
        
        explanations = []
        
        # Check for common elements
        if reference_standard.subject_area == matched_standard.subject_area:
            explanations.append("same subject")
        
        if reference_standard.domain and matched_standard.domain:
            if reference_standard.domain.lower() == matched_standard.domain.lower():
                explanations.append("same domain")
        
        # Check for common keywords/skills
        if reference_standard.keywords and matched_standard.keywords:
            common_keywords = set(kw.lower() for kw in reference_standard.keywords) & \
                            set(kw.lower() for kw in matched_standard.keywords)
            if common_keywords:
                explanations.append(f"common keywords: {', '.join(list(common_keywords)[:3])}")
        
        # Add similarity level
        if similarity_score > 0.9:
            explanations.append("very high similarity")
        elif similarity_score > 0.8:
            explanations.append("high similarity")
        else:
            explanations.append("moderate similarity")
        
        if explanations:
            return f"Similar due to: {', '.join(explanations)} (score: {similarity_score:.2f})"
        else:
            return f"Semantic similarity (score: {similarity_score:.2f})"