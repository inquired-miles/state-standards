"""
Enhanced embedding service for text-to-vector conversion
"""
import os
try:
    # Prefer modern client if available
    from openai import OpenAI  # type: ignore
    _HAS_OPENAI_CLIENT = True
except Exception:
    OpenAI = None  # type: ignore
    _HAS_OPENAI_CLIENT = False
import openai
import numpy as np
from typing import List, Optional, Dict, Any
from django.conf import settings
from .base import BaseService
from ..models import Standard, Concept, ContentAlignment, StandardAtom


class EmbeddingService(BaseService):
    """Enhanced service for generating and managing text embeddings"""
    
    def __init__(self):
        super().__init__()
        self.openai_client = None
        self.model = "text-embedding-3-small"
        self.dimensions = 1536
        self.batch_size = self.get_edtech_setting('EMBEDDING_BATCH_SIZE', 100)
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            try:
                if _HAS_OPENAI_CLIENT and OpenAI is not None:
                    self.openai_client = OpenAI(api_key=api_key)
                else:
                    openai.api_key = api_key
                    self.openai_client = openai
            except Exception:
                # Fallback to legacy module usage
                openai.api_key = api_key
                self.openai_client = openai
    
    def is_available(self) -> bool:
        """Check if embedding service is available"""
        return self.openai_client is not None
    
    def generate_embedding(self, text: str, cache_key: Optional[str] = None) -> Optional[List[float]]:
        """
        Generate embedding for given text using OpenAI's embedding API
        
        Args:
            text: The text to embed
            cache_key: Optional cache key for storing result
            
        Returns:
            List of floats representing the embedding, or None if API unavailable
        """
        if not self.is_available():
            self.handle_service_error(
                "generate_embedding", 
                ValueError("OpenAI API key not configured"),
                text_length=len(text)
            )
            return None
        
        # Check cache first
        if cache_key:
            cached_embedding = self.get_cached_result(cache_key)
            if cached_embedding:
                return cached_embedding
        
        try:
            # Clean and truncate text; ensure non-empty
            cleaned_text = self._clean_text(text or "")
            if not cleaned_text:
                return None

            response = self.openai_client.embeddings.create(
                model=self.model,
                input=cleaned_text
            )

            # Extract embedding from response
            embedding = None
            if hasattr(response, 'data') and response.data:
                first = response.data[0]
                if hasattr(first, 'embedding'):
                    embedding = first.embedding
                elif isinstance(first, dict) and 'embedding' in first:
                    embedding = first['embedding']
            elif isinstance(response, dict):
                # Fallback for dict responses
                data = response.get('data', [])
                if data and isinstance(data[0], dict):
                    embedding = data[0].get('embedding')
            
            if embedding is None:
                return None
            
            # Cache the result
            if cache_key:
                self.set_cached_result(cache_key, embedding)
            
            return embedding
            
        except Exception as e:
            self.handle_service_error(
                "generate_embedding",
                e,
                text_length=len(text),
                model=self.model
            )
            return None
    
    def generate_batch_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in batch
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings (or None for failed embeddings)
        """
        if not self.is_available():
            return [None] * len(texts)
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._process_batch(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def _process_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Process a batch of texts"""
        try:
            cleaned_texts = [self._clean_text(text or "") for text in texts]
            # Some OpenAI deployments can be strict about input types; ensure strings
            cleaned_texts = [t if isinstance(t, str) else ("" if t is None else str(t)) for t in cleaned_texts]
            # Filter out any empty strings to avoid API 400
            non_empty_indices = [i for i, t in enumerate(cleaned_texts) if t]
            if not non_empty_indices:
                return [None] * len(texts)
            batch_inputs = [cleaned_texts[i] for i in non_empty_indices]
            response = self.openai_client.embeddings.create(
                model=self.model,
                input=batch_inputs
            )
            # Map results back to original positions
            results = [None] * len(texts)
            data_list = getattr(response, 'data', None) or (response.get('data') if isinstance(response, dict) else None) or []
            for out_idx, in_idx in enumerate(non_empty_indices):
                item = data_list[out_idx] if out_idx < len(data_list) else None
                vec = getattr(item, 'embedding', None) if item is not None else None
                if vec is None and isinstance(item, dict):
                    vec = item.get('embedding')
                results[in_idx] = vec
            return results
            
        except Exception as e:
            self.handle_service_error(
                "_process_batch",
                e,
                batch_size=len(texts)
            )
            return [None] * len(texts)
    
    def generate_standard_embedding(self, standard: Standard) -> Optional[List[float]]:
        """
        Generate embedding for a standard using its title and description
        
        Args:
            standard: The Standard instance to embed
            
        Returns:
            List of floats representing the embedding
        """
        # Generate cache key
        cache_key = self.generate_cache_key(
            "standard_embedding",
            standard_id=str(standard.id),
            title_hash=hash(standard.title),
            description_hash=hash(standard.description)
        )
        
        # Combine relevant text fields for embedding
        text_components = [standard.title, standard.description]
        
        if standard.domain:
            text_components.append(f"Domain: {standard.domain}")
        if standard.cluster:
            text_components.append(f"Cluster: {standard.cluster}")
        if standard.keywords:
            text_components.append(f"Keywords: {', '.join(standard.keywords)}")
        if standard.skills:
            text_components.append(f"Skills: {', '.join(standard.skills)}")
        
        text_to_embed = ". ".join(text_components)
        
        return self.generate_embedding(text_to_embed, cache_key)
    
    def generate_concept_embedding(self, concept: Concept) -> Optional[List[float]]:
        """
        Generate embedding for a concept
        
        Args:
            concept: The Concept instance to embed
            
        Returns:
            List of floats representing the embedding
        """
        cache_key = self.generate_cache_key(
            "concept_embedding",
            concept_id=str(concept.id),
            name_hash=hash(concept.name),
            description_hash=hash(concept.description or "")
        )
        
        text_components = [concept.name]
        
        if concept.description:
            text_components.append(concept.description)
        if concept.keywords:
            text_components.append(f"Keywords: {', '.join(concept.keywords)}")
        
        text_to_embed = ". ".join(text_components)
        
        return self.generate_embedding(text_to_embed, cache_key)
    
    def generate_content_embedding(self, content_alignment: ContentAlignment) -> Optional[List[float]]:
        """
        Generate embedding for content alignment
        
        Args:
            content_alignment: The ContentAlignment instance to embed
            
        Returns:
            List of floats representing the embedding
        """
        cache_key = self.generate_cache_key(
            "content_embedding",
            content_hash=content_alignment.content_hash
        )
        
        # Use title and truncated content text
        text_components = [content_alignment.content_title]
        
        # Truncate content text to manageable size
        max_content_length = self.get_edtech_setting('MAX_CONTENT_LENGTH', 50000)
        content_text = content_alignment.content_text[:max_content_length]
        text_components.append(content_text)
        
        text_to_embed = ". ".join(text_components)
        
        return self.generate_embedding(text_to_embed, cache_key)

    # --- Atom-specific helpers ---
    def generate_atom_embedding(self, atom: StandardAtom) -> Optional[List[float]]:
        cache_key = self.generate_cache_key(
            "atom_embedding",
            atom_code=atom.atom_code,
        )
        return self.generate_embedding(atom.text, cache_key)

    def update_embeddings_for_atoms(self, atoms_queryset=None) -> Dict[str, int]:
        if atoms_queryset is None:
            atoms_queryset = StandardAtom.objects.filter(embedding__isnull=True)
        atoms = list(atoms_queryset)
        texts = [a.text for a in atoms]
        batch = self.generate_batch_embeddings(texts)
        updated, failed = 0, 0
        for atom, emb in zip(atoms, batch):
            if emb:
                atom.embedding = emb
                updated += 1
            else:
                failed += 1
        StandardAtom.objects.bulk_update(atoms, ['embedding'])
        return {"processed": len(atoms), "successful": updated, "failed": failed}
    
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
        embeddings = self.generate_batch_embeddings(texts)
        
        # Map embeddings to standards
        results = {'successful': 0, 'failed': 0, 'embeddings': {}}
        
        for standard_id, embedding in zip(standard_ids, embeddings):
            if embedding:
                results['embeddings'][standard_id] = embedding
                results['successful'] += 1
            else:
                results['failed'] += 1
        
        return results
    
    def update_embeddings_for_standards(self, standards_queryset=None) -> Dict[str, int]:
        """
        Update embeddings for standards that don't have them
        
        Args:
            standards_queryset: Optional queryset of standards to process
            
        Returns:
            Dictionary with update statistics
        """
        if standards_queryset is None:
            from ..models import Standard
            standards_queryset = Standard.objects.filter(embedding__isnull=True)
        
        standards = list(standards_queryset)
        successful_updates = 0
        failed_updates = 0
        
        for standard in standards:
            try:
                embedding = self.generate_standard_embedding(standard)
                if embedding:
                    standard.embedding = embedding
                    standard.save(update_fields=['embedding'])
                    successful_updates += 1
                else:
                    failed_updates += 1
            except Exception as e:
                self.handle_service_error(
                    "update_embeddings_for_standards",
                    e,
                    standard_id=str(standard.id)
                )
                failed_updates += 1
        
        return {
            'processed': len(standards),
            'successful': successful_updates,
            'failed': failed_updates
        }
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            
            if norm_product == 0:
                return 0.0
            
            similarity = dot_product / norm_product
            
            # Ensure similarity is between 0 and 1
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            self.handle_service_error(
                "calculate_similarity",
                e,
                vec1_length=len(embedding1),
                vec2_length=len(embedding2)
            )
            return 0.0
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text for embedding generation
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        cleaned = " ".join(text.split())
        
        # Truncate if too long (OpenAI has token limits)
        max_length = 8000  # Conservative limit
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "..."
        
        return cleaned