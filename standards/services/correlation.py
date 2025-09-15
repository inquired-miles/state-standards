"""
Correlation service for creating standard correlations
"""
import numpy as np
from typing import List, Tuple
from django.db.models import Q
from standards.models import Standard, StandardCorrelation
from .embedding import EmbeddingService


class StandardCorrelationService:
    """Service for managing standard correlations"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
    
    def find_similar_standards(self, standard: Standard, limit: int = 10, 
                              threshold: float = 0.7) -> List[Tuple[Standard, float]]:
        """Find standards similar to the given standard"""
        if standard.embedding is None or len(standard.embedding) == 0:
            return []
        
        # Get all standards with embeddings from different states
        other_standards = Standard.objects.filter(
            embedding__isnull=False
        ).exclude(
            state=standard.state
        ).exclude(
            id=standard.id
        )
        
        similarities = []
        try:
            query_embedding = np.array(standard.embedding)
            
            for other_standard in other_standards:
                try:
                    if other_standard.embedding is not None and len(other_standard.embedding) > 0:
                        # Calculate cosine similarity
                        other_embedding = np.array(other_standard.embedding)
                        
                        # Check dimensions match
                        if query_embedding.shape != other_embedding.shape:
                            continue
                            
                        # Normalize vectors
                        query_norm = np.linalg.norm(query_embedding)
                        other_norm = np.linalg.norm(other_embedding)
                        
                        if query_norm > 0 and other_norm > 0:
                            cosine_similarity = np.dot(query_embedding, other_embedding) / (query_norm * other_norm)
                            
                            # Convert to Python float and check threshold
                            similarity_score = float(cosine_similarity)
                            if similarity_score >= threshold:
                                similarities.append((other_standard, similarity_score))
                except Exception as e:
                    # Skip this standard if there's an error
                    continue
        except Exception as e:
            print(f"Error processing embeddings for {standard.code}: {e}")
            return []
        
        # Sort by similarity (highest first) and limit results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
    
    def create_correlations_for_standard(self, standard: Standard, 
                                       similarity_threshold: float = 0.8):
        """Create correlations for a standard with other similar standards"""
        similar_standards = self.find_similar_standards(
            standard, limit=20, threshold=similarity_threshold
        )
        
        created_count = 0
        for similar_standard, similarity_score in similar_standards:
            # Check if correlation already exists (either direction)
            existing_correlation = StandardCorrelation.objects.filter(
                Q(standard_1=standard, standard_2=similar_standard) |
                Q(standard_1=similar_standard, standard_2=standard)
            ).first()
            
            if not existing_correlation:
                # Determine correlation type based on similarity score
                if similarity_score >= 0.95:
                    correlation_type = 'exact'
                elif similarity_score >= 0.85:
                    correlation_type = 'similar'
                elif similarity_score >= 0.75:
                    correlation_type = 'related'
                else:
                    correlation_type = 'partial'
                
                StandardCorrelation.objects.create(
                    standard_1=standard,
                    standard_2=similar_standard,
                    similarity_score=similarity_score,
                    correlation_type=correlation_type
                )
                created_count += 1
                print(f"Created correlation: {standard.code} <-> {similar_standard.code} ({similarity_score:.3f})")
        
        if created_count == 0:
            print(f"No new correlations created for {standard.code}")
    
    def batch_create_correlations(self, similarity_threshold: float = 0.8):
        """Create correlations for all standards in batch"""
        standards_with_embeddings = Standard.objects.filter(embedding__isnull=False)
        total_standards = standards_with_embeddings.count()
        
        print(f"Processing {total_standards} standards with embeddings...")
        
        for i, standard in enumerate(standards_with_embeddings, 1):
            print(f"[{i}/{total_standards}] Creating correlations for {standard.code}")
            self.create_correlations_for_standard(standard, similarity_threshold)
        
        total_correlations = StandardCorrelation.objects.count()
        print(f"Correlation creation complete. Total correlations: {total_correlations}")