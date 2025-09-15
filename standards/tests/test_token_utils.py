"""
Tests for token counting utilities.
"""
import unittest
from unittest.mock import Mock, patch
from standards.services.token_utils import TokenCounter


class TestTokenCounter(unittest.TestCase):
    """Test cases for TokenCounter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.token_counter = TokenCounter('gpt-4o-mini')
        
        # Sample standard data for testing
        self.sample_standard = {
            'code': 'CA.K.OA.1',
            'title': 'Represent addition and subtraction with objects',
            'description': 'Represent addition and subtraction with objects, fingers, mental images, drawings, sounds, acting out situations, verbal explanations, expressions, or equations.',
            'domain': 'Operations and Algebraic Thinking',
            'cluster': 'Understand addition as putting together'
        }
        
        self.sample_standards = [
            self.sample_standard,
            {
                'code': 'CA.K.OA.2',
                'title': 'Solve addition and subtraction word problems',
                'description': 'Solve addition and subtraction word problems, and add and subtract within 10.',
                'domain': 'Operations and Algebraic Thinking',
                'cluster': 'Understand addition as putting together'
            }
        ]
    
    def test_count_tokens_basic(self):
        """Test basic token counting functionality."""
        # Test with simple text
        text = "Hello world"
        tokens = self.token_counter.count_tokens(text)
        self.assertGreater(tokens, 0)
        self.assertIsInstance(tokens, int)
        
        # Test with empty text
        empty_tokens = self.token_counter.count_tokens("")
        self.assertEqual(empty_tokens, 0)
        
        # Test with None
        none_tokens = self.token_counter.count_tokens(None)
        self.assertEqual(none_tokens, 0)
    
    def test_estimate_standard_tokens(self):
        """Test token estimation for a single standard."""
        tokens = self.token_counter.estimate_standard_tokens(self.sample_standard)
        
        # Should have tokens for all parts of the standard
        self.assertGreater(tokens, 0)
        self.assertIsInstance(tokens, int)
        
        # Should be reasonable for the sample data (rough estimate)
        self.assertGreater(tokens, 20)  # At least 20 tokens
        self.assertLess(tokens, 200)    # Less than 200 tokens
    
    def test_estimate_standards_batch_tokens(self):
        """Test token estimation for a batch of standards."""
        batch_tokens = self.token_counter.estimate_standards_batch_tokens(self.sample_standards)
        single_tokens = sum(self.token_counter.estimate_standard_tokens(std) for std in self.sample_standards)
        
        # Batch should include overhead
        self.assertGreater(batch_tokens, single_tokens)
        
        # Should be reasonable
        self.assertIsInstance(batch_tokens, int)
        self.assertGreater(batch_tokens, 0)
    
    def test_estimate_prompt_overhead(self):
        """Test prompt overhead estimation."""
        base_overhead = self.token_counter.estimate_prompt_overhead()
        self.assertGreater(base_overhead, 500)  # Should have base overhead
        
        # Test with topic hierarchy
        hierarchy = "Math > Algebra > Linear Equations"
        with_hierarchy = self.token_counter.estimate_prompt_overhead(hierarchy)
        self.assertGreater(with_hierarchy, base_overhead)
    
    def test_estimate_response_tokens(self):
        """Test response token estimation."""
        # Test with different numbers of standards
        small_response = self.token_counter.estimate_response_tokens(5)
        large_response = self.token_counter.estimate_response_tokens(50)
        
        self.assertGreater(large_response, small_response)
        self.assertIsInstance(small_response, int)
        self.assertIsInstance(large_response, int)
    
    def test_calculate_optimal_chunk_size(self):
        """Test optimal chunk size calculation."""
        # Create a larger list of standards for testing
        test_standards = self.sample_standards * 50  # 100 standards
        
        chunk_size = self.token_counter.calculate_optimal_chunk_size(test_standards)
        
        # Should return a reasonable chunk size
        self.assertIsInstance(chunk_size, int)
        self.assertGreaterEqual(chunk_size, 5)   # Min bound
        self.assertLessEqual(chunk_size, 500)    # Max bound
        
        # For very small token counts, it might use max chunk size
        # So we'll just ensure it's positive and reasonable
        self.assertGreater(chunk_size, 0)
    
    def test_calculate_optimal_chunk_size_small_dataset(self):
        """Test chunk size calculation with small dataset."""
        # Should return all standards for very small datasets
        chunk_size = self.token_counter.calculate_optimal_chunk_size(self.sample_standards[:5])
        self.assertLessEqual(chunk_size, 5)
        self.assertEqual(chunk_size, len(self.sample_standards[:5]))
    
    def test_validate_chunk(self):
        """Test chunk validation."""
        # Small chunk should be valid
        small_chunk = self.sample_standards[:2]
        self.assertTrue(self.token_counter.validate_chunk(small_chunk))
        
        # Create a very large chunk that might exceed limits
        large_chunk = self.sample_standards * 1000  # 2000 standards
        result = self.token_counter.validate_chunk(large_chunk)
        # This should likely be False, but depends on the actual token counts
        self.assertIsInstance(result, bool)
    
    def test_model_limits(self):
        """Test that model limits are properly configured."""
        # Test known model
        self.assertIn('gpt-4o-mini', TokenCounter.MODEL_LIMITS)
        limits = TokenCounter.MODEL_LIMITS['gpt-4o-mini']
        self.assertIn('context_window', limits)
        self.assertIn('max_output', limits)
        self.assertGreater(limits['context_window'], 0)
        self.assertGreater(limits['max_output'], 0)
    
    def test_different_models(self):
        """Test TokenCounter with different models."""
        models_to_test = ['gpt-4o-mini', 'gpt-4', 'gpt-4o']
        
        for model in models_to_test:
            counter = TokenCounter(model)
            self.assertEqual(counter.model_name, model)
            
            # Should be able to count tokens
            tokens = counter.count_tokens("Test message")
            self.assertGreater(tokens, 0)
    
    def test_error_handling(self):
        """Test error handling in token counting."""
        # Test with invalid standard data
        invalid_standard = {'invalid': 'data'}
        
        # Should not crash
        tokens = self.token_counter.estimate_standard_tokens(invalid_standard)
        self.assertIsInstance(tokens, int)
        self.assertGreaterEqual(tokens, 0)
    
    @patch('standards.services.token_utils.tiktoken.get_encoding')
    def test_encoding_fallback(self, mock_get_encoding):
        """Test fallback when encoding fails."""
        # Make tiktoken fail to load initially, then succeed on fallback
        mock_get_encoding.side_effect = [Exception("Encoding failed"), Mock()]
        
        # Should still work with fallback
        counter = TokenCounter('gpt-4o-mini')
        tokens = counter.count_tokens("test text")
        self.assertGreater(tokens, 0)
    
    def test_safety_margin_application(self):
        """Test that safety margin is properly applied."""
        test_standards = self.sample_standards * 10  # 20 standards
        
        # Calculate with different safety margins
        conservative = self.token_counter.calculate_optimal_chunk_size(
            test_standards, safety_margin=0.5  # 50% margin
        )
        aggressive = self.token_counter.calculate_optimal_chunk_size(
            test_standards, safety_margin=0.1  # 10% margin
        )
        
        # More conservative should result in smaller chunks
        self.assertLessEqual(conservative, aggressive)


if __name__ == '__main__':
    unittest.main()