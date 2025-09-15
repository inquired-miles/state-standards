"""
Services for handling vector operations and database integration
"""
import os
from typing import List, Tuple, Optional

import numpy as np
import openai
from django.conf import settings
from django.db import connection
from pgvector.django import CosineDistance

from .models import Standard, StandardCorrelation


class EmbeddingService:
    """Service for generating text embeddings using OpenAI's API"""
    
    def __init__(self):
        # Initialize OpenAI client (you'll need to set OPENAI_API_KEY in your environment)
        self.openai_client = None
        if os.getenv('OPENAI_API_KEY'):
            openai.api_key = os.getenv('OPENAI_API_KEY')
            self.openai_client = openai
    
    def generate_embedding(self, text: str, model: str = "text-embedding-3-small") -> Optional[List[float]]:
        """
        Generate embedding for given text using OpenAI's embedding API
        
        Args:
            text: The text to embed
            model: The embedding model to use
            
        Returns:
            List of floats representing the embedding, or None if API unavailable
        """
        if not self.openai_client:
            print("Warning: OpenAI API key not configured. Cannot generate embeddings.")
            return None
            
        try:
            # Combine title and description for more comprehensive embedding
            response = self.openai_client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str], model: str = "text-embedding-3-small") -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in a single API call (batch processing)
        OpenAI API supports up to 2048 embeddings per request
        
        Args:
            texts: List of texts to embed
            model: The embedding model to use
            
        Returns:
            List of embeddings (or None for failed items)
        """
        if not self.openai_client:
            print("Warning: OpenAI API key not configured. Cannot generate embeddings.")
            return [None] * len(texts)
        
        if not texts:
            return []
        
        # OpenAI recommends batches of up to 100 for optimal performance
        max_batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), max_batch_size):
            batch = texts[i:i + max_batch_size]
            
            try:
                # Make batch API call
                response = self.openai_client.embeddings.create(
                    input=batch,
                    model=model
                )
                
                # Extract embeddings in order
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error generating batch embeddings: {e}")
                # Return None for failed batch items
                all_embeddings.extend([None] * len(batch))
        
        return all_embeddings
    
    def generate_standards_embeddings_batch(self, standards: List[Standard]) -> dict:
        """
        Generate embeddings for multiple standards efficiently using batch processing
        
        Args:
            standards: List of Standard instances to process
            
        Returns:
            Dictionary with results: {'successful': int, 'failed': int, 'embeddings': dict}
        """
        if not standards:
            return {'successful': 0, 'failed': 0, 'embeddings': {}}
        
        # Prepare texts for all standards
        texts = []
        standard_ids = []
        
        for standard in standards:
            # Build embedding text for each standard
            parts = []
            
            if standard.code:
                parts.append(f"Standard: {standard.code}")
            if standard.description:
                parts.append(standard.description)
            if standard.domain:
                parts.append(f"Domain: {standard.domain}")
            if standard.cluster:
                parts.append(f"Cluster: {standard.cluster}")
            if hasattr(standard, 'keywords') and standard.keywords:
                parts.append(f"Keywords: {', '.join(standard.keywords)}")
            if standard.title and standard.title.lower() not in (standard.description or '').lower():
                parts.append(f"Also known as: {standard.title}")
            
            text = " | ".join(parts)
            texts.append(text)
            standard_ids.append(standard.id)
        
        # Generate embeddings in batch
        embeddings = self.generate_embeddings_batch(texts)
        
        # Map embeddings to standards
        results = {'successful': 0, 'failed': 0, 'embeddings': {}}
        
        for standard_id, embedding in zip(standard_ids, embeddings):
            if embedding:
                results['embeddings'][standard_id] = embedding
                results['successful'] += 1
            else:
                results['failed'] += 1
        
        return results
    
    def generate_standard_embedding(self, standard: Standard) -> Optional[List[float]]:
        """
        Generate embedding for a standard using code, description, and other standardized fields
        
        Args:
            standard: The Standard instance to embed
            
        Returns:
            List of floats representing the embedding
        """
        # Build embedding text prioritizing standardized content
        parts = []
        
        # Always include code as primary identifier
        if standard.code:
            parts.append(f"Standard: {standard.code}")
        
        # Always include description (most standardized content)
        if standard.description:
            parts.append(standard.description)
        
        # Include domain and cluster for context
        if standard.domain:
            parts.append(f"Domain: {standard.domain}")
        if standard.cluster:
            parts.append(f"Cluster: {standard.cluster}")
        if standard.keywords:
            parts.append(f"Keywords: {', '.join(standard.keywords)}")
        
        # Only include title if it adds semantic value not already in description
        if standard.title and standard.title.lower() not in standard.description.lower():
            parts.append(f"Also known as: {standard.title}")
        
        text_to_embed = " | ".join(parts)
        return self.generate_embedding(text_to_embed)


class DatabaseService:
    """Service for interacting with PostgreSQL database with pgvector"""
    
    def __init__(self):
        # Check if we're using PostgreSQL with pgvector
        self.has_pgvector = False
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                self.has_pgvector = cursor.fetchone() is not None
        except Exception as e:
            print(f"Warning: Could not check for pgvector extension: {e}")
    
    def is_connected(self) -> bool:
        """Check if database is properly configured with pgvector"""
        return self.has_pgvector
    
    def store_standard_vector(self, standard: Standard, embedding: List[float]) -> bool:
        """
        Store a standard's vector embedding in PostgreSQL
        
        Args:
            standard: The Standard instance
            embedding: The vector embedding
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update the standard's embedding field in the database
            standard.embedding = embedding
            standard.save(update_fields=['embedding'])
            return True
        except Exception as e:
            print(f"Error storing vector: {e}")
            return False
    
    def similarity_search(self, query_embedding: List[float], limit: int = 10, 
                         threshold: float = 0.7, exclude_states: List[str] = None) -> List[Tuple[Standard, float]]:
        """
        Perform similarity search using vector embeddings
        
        Args:
            query_embedding: The query vector
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold
            exclude_states: Optional list of state codes to exclude
            
        Returns:
            List of tuples containing (Standard, similarity_score)
        """
        if not query_embedding or not self.has_pgvector:
            return []
            
        try:
            # Build query
            query = Standard.objects.filter(embedding__isnull=False)
            
            # Exclude states if specified
            if exclude_states:
                query = query.exclude(state__code__in=exclude_states)
            
            # Perform similarity search
            results = query.annotate(
                distance=CosineDistance('embedding', query_embedding)
            ).filter(
                distance__lt=1 - threshold  # Convert similarity to distance
            ).order_by('distance')[:limit]
            
            # Convert distance back to similarity score
            return [(standard, 1 - standard.distance) for standard in results]
            
        except ImportError:
            print("Error: pgvector not installed. Please install with: pip install pgvector")
            return []
        except Exception as e:
            print(f"Error performing similarity search: {e}")
            return []


class StandardCorrelationService:
    """Service for managing standard correlations"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.database_service = DatabaseService()
    
    def generate_embeddings_for_all_standards(self):
        """Generate embeddings for all standards that don't have them"""
        standards_without_embeddings = Standard.objects.filter(embedding__isnull=True)
        
        for standard in standards_without_embeddings:
            print(f"Generating embedding for {standard.code}")
            embedding = self.embedding_service.generate_standard_embedding(standard)
            
            if embedding:
                self.database_service.store_standard_vector(standard, embedding)
                print(f"✓ Embedding generated and stored for {standard.code}")
            else:
                print(f"✗ Failed to generate embedding for {standard.code}")
    
    def find_similar_standards(self, standard: Standard, limit: int = 10, 
                              threshold: float = 0.7) -> List[Tuple[Standard, float]]:
        """
        Find standards similar to the given standard
        
        Args:
            standard: The reference standard
            limit: Maximum number of similar standards to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of tuples containing (Standard, similarity_score)
        """
        if not standard.embedding:
            print(f"Standard {standard.code} has no embedding. Generating one...")
            embedding = self.embedding_service.generate_standard_embedding(standard)
            if embedding:
                self.database_service.store_standard_vector(standard, embedding)
            else:
                return []
        
        # Exclude the standard itself and standards from the same state
        similar_standards = self.database_service.similarity_search(
            standard.embedding, limit + 10, threshold, 
            exclude_states=[standard.state.code]
        )
        
        # Filter out the same standard and same state standards
        filtered_results = []
        for similar_standard, score in similar_standards:
            if (similar_standard.id != standard.id and 
                similar_standard.state != standard.state):
                filtered_results.append((similar_standard, score))
                
            if len(filtered_results) >= limit:
                break
        
        return filtered_results
    
    def create_correlations_for_standard(self, standard: Standard, 
                                       similarity_threshold: float = 0.8):
        """
        Create correlations for a standard with other similar standards
        
        Args:
            standard: The standard to find correlations for
            similarity_threshold: Minimum similarity score for creating correlations
        """
        similar_standards = self.find_similar_standards(
            standard, limit=20, threshold=similarity_threshold
        )
        
        for similar_standard, similarity_score in similar_standards:
            # Check if correlation already exists
            existing_correlation = StandardCorrelation.objects.filter(
                standard_1=standard,
                standard_2=similar_standard
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
                
                print(f"Created correlation: {standard.code} <-> {similar_standard.code} ({similarity_score:.3f})")
    
    def batch_create_correlations(self, similarity_threshold: float = 0.8):
        """
        Create correlations for all standards in batch
        
        Args:
            similarity_threshold: Minimum similarity score for creating correlations
        """
        standards_with_embeddings = Standard.objects.filter(embedding__isnull=False)
        
        for standard in standards_with_embeddings:
            print(f"Creating correlations for {standard.code}")
            self.create_correlations_for_standard(standard, similarity_threshold)