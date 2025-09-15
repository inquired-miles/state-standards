"""
Token counting utilities for estimating API token usage.
"""
import json
import logging
from typing import List, Dict, Any, Optional
import tiktoken

logger = logging.getLogger(__name__)


class TokenCounter:
    """Utility class for estimating token counts for standards data."""
    
    # Model encoding mappings
    MODEL_ENCODINGS = {
        'gpt-4': 'cl100k_base',
        'gpt-4-turbo': 'cl100k_base',
        'gpt-4o': 'o200k_base',
        'gpt-4o-mini': 'o200k_base',
        'gpt-4.1': 'o200k_base',
        'gpt-4.1-mini': 'o200k_base',
        'gpt-5': 'o200k_base',
        'gpt-5-mini': 'o200k_base',
        'gpt-3.5-turbo': 'cl100k_base',
    }
    
    # Token limit configurations for different models
    MODEL_LIMITS = {
        'gpt-4o-mini': {
            'context_window': 128000,
            'max_output': 16384
        },
        'gpt-4o': {
            'context_window': 128000,
            'max_output': 16384
        },
        'gpt-4': {
            'context_window': 128000,
            'max_output': 4096
        },
        'gpt-4-turbo': {
            'context_window': 128000,
            'max_output': 4096
        },
        'gpt-4.1': {
            'context_window': 1047576,
            'max_output': 32768
        },
        'gpt-4.1-mini': {
            'context_window': 1047576,
            'max_output': 32768
        },
        'gpt-5': {
            'context_window': 400000,
            'max_output': 128000
        },
        'gpt-5-mini': {
            'context_window': 400000,
            'max_output': 128000
        },
        'gpt-3.5-turbo': {
            'context_window': 16385,
            'max_output': 4096
        }
    }
    
    def __init__(self, model_name: str = 'gpt-4o-mini'):
        """
        Initialize TokenCounter with a specific model.
        
        Args:
            model_name: Name of the model to use for token counting
        """
        self.model_name = model_name
        self.encoding_name = self.MODEL_ENCODINGS.get(model_name, 'o200k_base')
        
        try:
            self.encoder = tiktoken.get_encoding(self.encoding_name)
        except Exception as e:
            logger.warning(f"Failed to load encoding {self.encoding_name}: {e}. Using default.")
            self.encoder = tiktoken.get_encoding('cl100k_base')
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        try:
            return len(self.encoder.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Fallback estimation: ~4 characters per token
            return len(text) // 4
    
    def estimate_standard_tokens(self, standard: Dict[str, Any]) -> int:
        """
        Estimate token count for a single standard.
        
        Args:
            standard: Standard dictionary with title, description, code, etc.
            
        Returns:
            Estimated token count
        """
        # Build the text representation of the standard
        text_parts = []
        
        if 'code' in standard:
            text_parts.append(f"Code: {standard['code']}")
        if 'title' in standard:
            text_parts.append(f"Title: {standard['title']}")
        if 'description' in standard:
            text_parts.append(f"Description: {standard['description']}")
        if 'domain' in standard:
            text_parts.append(f"Domain: {standard['domain']}")
        if 'cluster' in standard:
            text_parts.append(f"Cluster: {standard['cluster']}")
        
        text = '\n'.join(text_parts)
        return self.count_tokens(text)
    
    def estimate_standards_batch_tokens(self, standards: List[Dict[str, Any]]) -> int:
        """
        Estimate token count for a batch of standards.
        
        Args:
            standards: List of standard dictionaries
            
        Returns:
            Total estimated token count
        """
        total_tokens = 0
        for standard in standards:
            total_tokens += self.estimate_standard_tokens(standard)
        
        # Add overhead for JSON formatting (approximately 10-20 tokens per standard)
        total_tokens += len(standards) * 15
        
        return total_tokens
    
    def estimate_prompt_overhead(self, topic_hierarchy: Optional[str] = None) -> int:
        """
        Legacy method - use _calculate_accurate_prompt_overhead for new code.
        """
        return self._calculate_accurate_prompt_overhead(topic_hierarchy, "basic")
    
    def _calculate_accurate_prompt_overhead(self, topic_hierarchy: Optional[str] = None, mode: str = "enhanced") -> int:
        """
        Calculate accurate prompt overhead for different processing modes.
        
        Args:
            topic_hierarchy: Optional topic hierarchy text to include
            mode: Processing mode ("enhanced" or "basic")
            
        Returns:
            Accurate prompt overhead tokens
        """
        # Base system prompt overhead varies significantly by mode
        if mode == "enhanced":
            # Enhanced mode has comprehensive system prompts, JSON schema, pre-analysis requirements
            base_overhead = 2500  # Much more realistic for enhanced categorization
        else:
            # Basic mode has simpler prompts
            base_overhead = 800
        
        # Add topic hierarchy tokens if provided
        if topic_hierarchy:
            hierarchy_tokens = self.count_tokens(topic_hierarchy)
            base_overhead += hierarchy_tokens
            logger.debug(f"Added {hierarchy_tokens} tokens for topic hierarchy")
        
        # Add realistic formatting overhead based on mode
        if mode == "enhanced":
            # Enhanced mode needs substantial JSON schema and response formatting
            formatting_overhead = 1500
        else:
            # Basic mode has simpler formatting
            formatting_overhead = 300
        
        base_overhead += formatting_overhead
        
        logger.debug(f"Calculated prompt overhead: {base_overhead} tokens (mode: {mode})")
        return base_overhead
    
    def estimate_response_tokens(self, num_standards: int, mode: str = "enhanced") -> int:
        """
        Estimate output tokens needed for categorization response.
        
        Args:
            num_standards: Number of standards being categorized
            mode: Processing mode ("enhanced" or "basic")
            
        Returns:
            Estimated output tokens needed
        """
        if mode == "enhanced":
            # Enhanced mode includes:
            # - Pre-analysis section: ~500 tokens
            # - Per standard: ID (10) + category (20) + confidence (5) + reasoning (50) + key_concepts (30) = ~115 tokens
            # - Outliers with detailed reasoning: ~80 tokens each (assume 10% outlier rate)
            # - Summary section: ~200 tokens
            tokens_per_standard = 115
            outlier_rate = 0.1
            outlier_tokens = int(num_standards * outlier_rate * 80)
            
            structure_overhead = 700  # Pre-analysis + summary + JSON structure
            total_tokens = (num_standards * tokens_per_standard) + outlier_tokens + structure_overhead
        else:
            # Basic mode - simpler categorizations
            tokens_per_standard = 40
            structure_overhead = 100
            total_tokens = (num_standards * tokens_per_standard) + structure_overhead
        
        logger.debug(f"Estimated response tokens: {total_tokens} for {num_standards} standards ({mode} mode)")
        return total_tokens
    
    def estimate_hierarchy_response_tokens(self) -> int:
        """
        Estimate output tokens needed for hierarchy generation response.
        
        Returns:
            Estimated output tokens needed for topic hierarchy
        """
        # Hierarchy structure estimation:
        # - 5-8 topics, each with ~20 tokens for name + description
        # - 3-6 sub-topics per topic, each with ~25 tokens  
        # - 4-8 sub-sub-topics per sub-topic, each with ~10 tokens
        # - JSON structure overhead
        
        # Conservative estimate: 8 topics × 6 sub-topics × 8 sub-sub-topics
        topics = 8
        sub_topics_per_topic = 6
        sub_sub_topics_per_sub = 8
        
        topic_tokens = topics * 20
        sub_topic_tokens = topics * sub_topics_per_topic * 25
        sub_sub_topic_tokens = topics * sub_topics_per_topic * sub_sub_topics_per_sub * 10
        
        # JSON structure and formatting overhead
        json_overhead = 500
        
        total_tokens = topic_tokens + sub_topic_tokens + sub_sub_topic_tokens + json_overhead
        
        logger.info(f"Estimated hierarchy response tokens: {total_tokens}")
        return total_tokens
    
    def _get_representative_sample(self, standards: List[Dict[str, Any]], sample_size: int = 30) -> List[Dict[str, Any]]:
        """
        Get a representative sample from beginning, middle, and end of standards list.
        
        Args:
            standards: Full list of standards
            sample_size: Number of standards to sample
            
        Returns:
            Representative sample of standards
        """
        if len(standards) <= sample_size:
            return standards
        
        # Sample from beginning, middle, and end
        third = sample_size // 3
        beginning = standards[:third]
        middle_start = len(standards) // 2 - third // 2
        middle = standards[middle_start:middle_start + third]
        end = standards[-third:]
        
        sample = beginning + middle + end
        return sample[:sample_size]  # Ensure exact sample size
    
    def _estimate_actual_tokens_for_standards(self, standards: List[Dict[str, Any]], mode: str = "enhanced") -> int:
        """
        Estimate actual tokens needed for a list of standards in the prompt.
        
        Args:
            standards: List of standards to estimate
            mode: Processing mode ("enhanced" or "basic")
            
        Returns:
            Total estimated tokens for these standards
        """
        total_tokens = 0
        
        for standard in standards:
            # Format as it appears in the actual prompt
            if mode == "enhanced":
                # Enhanced format includes grade context and subject context
                std_text = f"{standard.get('code', '')}: {standard.get('title', '')}"
                if standard.get('description'):
                    desc = standard['description']
                    if len(desc) > 300:
                        # Smart truncation as done in the actual code
                        desc = desc[:200] + "..." + desc[-100:]
                    std_text += f" - {desc}"
                
                # Add grade and subject context
                std_text += f" [Grade context, Subject: {standard.get('subject', 'Unknown')}]"
                
            else:
                # Basic format
                std_text = f"{standard.get('code', '')}: {standard.get('title', '')}"
                if standard.get('description'):
                    std_text += f" - {standard['description'][:200]}..."
            
            total_tokens += self.count_tokens(std_text)
        
        # Add formatting overhead (numbering, bullets, etc.)
        formatting_overhead = len(standards) * 5  # ~5 tokens per standard for formatting
        total_tokens += formatting_overhead
        
        return total_tokens
    
    def calculate_optimal_chunk_size(self, 
                                    standards: List[Dict[str, Any]],
                                    topic_hierarchy: Optional[str] = None,
                                    mode: str = "enhanced") -> int:
        """
        Calculate the optimal chunk size for processing standards with aggressive utilization.
        
        Args:
            standards: List of all standards to process
            topic_hierarchy: Optional topic hierarchy for context
            mode: Processing mode ("enhanced" or "basic")
            
        Returns:
            Optimal chunk size (number of standards per chunk)
        """
        # Get model limits
        limits = self.MODEL_LIMITS.get(self.model_name, self.MODEL_LIMITS['gpt-4o-mini'])
        context_window = limits['context_window']
        max_output = limits['max_output']
        
        # Aggressive utilization - use 90-95% of context window for large models
        if self.model_name in ['gpt-4.1', 'gpt-4.1-mini']:
            utilization_rate = 0.95  # 95% utilization for massive context models
        elif self.model_name in ['gpt-5', 'gpt-5-mini']:
            utilization_rate = 0.92  # 92% utilization for large context models
        else:
            utilization_rate = 0.85  # 85% utilization for standard models
        
        # Calculate accurate prompt overhead using dynamic measurement
        prompt_overhead = self._calculate_accurate_prompt_overhead(topic_hierarchy, mode)
        
        # Calculate available tokens for standards
        available_for_standards = int((context_window - max_output - prompt_overhead) * utilization_rate)
        
        # Process all standards if they fit (preferred for large context models)
        if len(standards) <= 50:
            # For small datasets, always try to process all at once
            total_tokens = self._estimate_actual_tokens_for_standards(standards, mode)
            if total_tokens <= available_for_standards:
                logger.info(f"🚀 Processing ALL {len(standards)} standards in single chunk (fits in {total_tokens:,} tokens)")
                return len(standards)
        
        # Use representative sampling from beginning, middle, and end
        sample_standards = self._get_representative_sample(standards, sample_size=30)
        avg_tokens_per_standard = self._estimate_actual_tokens_for_standards(sample_standards, mode) / len(sample_standards)
        
        # Calculate optimal chunk size
        optimal_chunk = int(available_for_standards / avg_tokens_per_standard)
        
        # Apply model-specific bounds
        min_chunk = 50  # Minimum for meaningful categorization
        
        if self.model_name in ['gpt-4.1', 'gpt-4.1-mini']:
            max_chunk = len(standards)  # No artificial limits for massive context
        elif self.model_name in ['gpt-5', 'gpt-5-mini']:
            max_chunk = min(3000, len(standards))  # Up to 3000 for large context
        else:
            max_chunk = min(1000, len(standards))  # Up to 1000 for standard models
        
        optimal_chunk = max(min_chunk, min(optimal_chunk, max_chunk))
        
        # For GPT-4.1, prefer fewer, larger chunks
        if self.model_name in ['gpt-4.1', 'gpt-4.1-mini'] and optimal_chunk < len(standards) / 2:
            optimal_chunk = max(optimal_chunk, len(standards) // 2)  # At most 2 chunks
        
        # Comprehensive logging
        logger.info(f"🧠 Aggressive token calculation for {self.model_name} ({mode} mode):")
        logger.info(f"  📊 Context window: {context_window:,} tokens")
        logger.info(f"  📤 Max output: {max_output:,} tokens")
        logger.info(f"  🔧 Prompt overhead: {prompt_overhead:,} tokens")
        logger.info(f"  ⚡ Available for standards: {available_for_standards:,} tokens")
        logger.info(f"  📏 Avg tokens per standard: {avg_tokens_per_standard:.1f}")
        logger.info(f"  🎯 Utilization rate: {utilization_rate*100:.1f}%")
        logger.info(f"  📈 Calculated optimal chunk: {optimal_chunk:,} standards")
        logger.info(f"  🔄 Total chunks needed: {(len(standards) + optimal_chunk - 1) // optimal_chunk}")
        
        return optimal_chunk
    
    def validate_chunk(self, 
                      standards_chunk: List[Dict[str, Any]],
                      topic_hierarchy: Optional[str] = None) -> bool:
        """
        Validate that a chunk of standards fits within token limits.
        
        Args:
            standards_chunk: Chunk of standards to validate
            topic_hierarchy: Optional topic hierarchy for context
            
        Returns:
            True if chunk fits within limits, False otherwise
        """
        # Get model limits - prefer GPT-4.1 as fallback since it's our new default
        limits = self.MODEL_LIMITS.get(self.model_name, self.MODEL_LIMITS.get('gpt-4.1', self.MODEL_LIMITS['gpt-4o-mini']))
        context_window = limits['context_window']
        max_output = limits['max_output']
        
        # Calculate total input tokens
        input_tokens = (self.estimate_prompt_overhead(topic_hierarchy) + 
                       self.estimate_standards_batch_tokens(standards_chunk))
        
        # Calculate expected output tokens
        output_tokens = self.estimate_response_tokens(len(standards_chunk))
        
        # Check if it fits
        total_tokens = input_tokens + output_tokens
        fits = total_tokens < context_window
        
        if not fits:
            logger.warning(f"Chunk validation failed: {total_tokens} tokens > {context_window} limit")
        
        return fits
    
    def calculate_optimal_standards_for_hierarchy(self, 
                                                 standards: List[Dict[str, Any]],
                                                 system_prompt: str = "",
                                                 user_prompt_template: str = "",
                                                 safety_margin: Optional[float] = None) -> int:
        """
        Calculate the optimal number of standards to include in hierarchy generation with aggressive utilization.
        
        Args:
            standards: List of all standards to potentially include
            system_prompt: System message content
            user_prompt_template: User prompt template (without standards)
            safety_margin: Optional safety margin override
            
        Returns:
            Optimal number of standards to include
        """
        # Get model limits
        limits = self.MODEL_LIMITS.get(self.model_name, self.MODEL_LIMITS['gpt-4o-mini'])
        context_window = limits['context_window']
        
        # Aggressive utilization for hierarchy generation
        if safety_margin is None:
            if self.model_name in ['gpt-4.1', 'gpt-4.1-mini']:
                utilization_rate = 0.95  # Use 95% of massive context
            elif self.model_name in ['gpt-5', 'gpt-5-mini']:
                utilization_rate = 0.92  # Use 92% of large context
            else:
                utilization_rate = 0.85  # Use 85% of standard context
        else:
            utilization_rate = 1 - safety_margin
        
        # Calculate fixed prompt overhead using actual token counting
        system_tokens = self.count_tokens(system_prompt) if system_prompt else 0
        user_template_tokens = self.count_tokens(user_prompt_template) if user_prompt_template else 0
        
        # Estimate hierarchy response tokens
        response_tokens = self.estimate_hierarchy_response_tokens()
        
        # Calculate available tokens for standards with aggressive utilization
        available_for_standards = int((context_window - system_tokens - user_template_tokens - response_tokens) * utilization_rate)
        
        # For massive context models, try to use ALL standards if they fit
        if self.model_name in ['gpt-4.1', 'gpt-4.1-mini']:
            # Test if all standards fit
            all_standards_tokens = self._estimate_hierarchy_tokens_for_standards(standards)
            if all_standards_tokens <= available_for_standards:
                logger.info(f"🚀 Using ALL {len(standards)} standards for hierarchy generation (fits in {all_standards_tokens:,} tokens)")
                return len(standards)
        
        # Use representative sampling for token estimation
        sample_standards = self._get_representative_sample(standards, sample_size=50)
        total_sample_tokens = self._estimate_hierarchy_tokens_for_standards(sample_standards)
        avg_tokens_per_standard = total_sample_tokens / len(sample_standards)
        
        # Calculate optimal standards count
        optimal_count = int(available_for_standards / avg_tokens_per_standard)
        
        # Apply bounds based on model capabilities
        min_count = 50  # Minimum for meaningful hierarchy
        max_count = len(standards)  # Don't exceed total standards
        
        # For massive context models, prefer larger samples
        if self.model_name in ['gpt-4.1', 'gpt-4.1-mini']:
            min_count = min(500, len(standards))  # Use at least 500 standards if available
        elif self.model_name in ['gpt-5', 'gpt-5-mini']:
            min_count = min(200, len(standards))  # Use at least 200 standards if available
        
        optimal_count = max(min_count, min(optimal_count, max_count))
        
        # Comprehensive logging
        logger.info(f"🧠 Hierarchy generation token calculation for {self.model_name}:")
        logger.info(f"  📊 Context window: {context_window:,} tokens")
        logger.info(f"  🔧 System prompt tokens: {system_tokens:,}")
        logger.info(f"  📝 User template tokens: {user_template_tokens:,}")
        logger.info(f"  📤 Response tokens: {response_tokens:,}")
        logger.info(f"  ⚡ Available for standards: {available_for_standards:,} tokens")
        logger.info(f"  📏 Avg tokens per standard: {avg_tokens_per_standard:.1f}")
        logger.info(f"  📚 Total standards available: {len(standards):,}")
        logger.info(f"  🎯 Optimal standards count: {optimal_count:,}")
        logger.info(f"  📈 Utilization rate: {utilization_rate*100:.1f}%")
        
        return optimal_count
    
    def _estimate_hierarchy_tokens_for_standards(self, standards: List[Dict[str, Any]]) -> int:
        """
        Estimate tokens needed for standards in hierarchy generation context.
        
        Args:
            standards: List of standards to estimate
            
        Returns:
            Total estimated tokens for these standards in hierarchy context
        """
        total_tokens = 0
        
        for standard in standards:
            # Format as used in hierarchy generation: "- CODE: Title - Description..."
            std_text = f"- {standard.get('code', '')}: {standard.get('title', '')}"
            if standard.get('description'):
                # Truncate description as done in the actual code
                desc = standard['description']
                if len(desc) > 150:
                    desc = desc[:150] + "..."
                std_text += f" - {desc}"
            
            total_tokens += self.count_tokens(std_text)
        
        return total_tokens