#!/usr/bin/env python
"""
Quick test script to validate the aggressive token calculation improvements.
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'state_standards_project.settings')
django.setup()

from standards.services.token_utils import TokenCounter
from standards.services.topic_categorization import TopicCategorizationService

def test_token_optimization():
    """Test the aggressive token calculation improvements."""
    
    print("🧪 Testing Token Calculation Optimization")
    print("=" * 60)
    
    # Test different models
    models_to_test = ['gpt-4.1', 'gpt-5', 'gpt-4o']
    
    # Create sample standards data
    sample_standards = []
    for i in range(1000):  # Test with 1000 standards
        sample_standards.append({
            'code': f'TEST.{i+1}',
            'title': f'Sample Standard {i+1}',
            'description': f'This is a sample educational standard for testing purposes. Standard {i+1} covers important learning objectives and assessment criteria for students.',
            'subject': 'Mathematics'
        })
    
    for model in models_to_test:
        print(f"\n🧠 Testing model: {model}")
        print("-" * 40)
        
        counter = TokenCounter(model)
        service = TopicCategorizationService()
        service.token_counter = counter
        
        # Test hierarchy generation optimization
        optimal_hierarchy_count = counter.calculate_optimal_standards_for_hierarchy(
            sample_standards[:500],  # Test with 500 standards
            system_prompt="You are an education curriculum expert...",
            user_prompt_template="Standards:\n"
        )
        
        # Test chunk size calculation
        optimal_chunk_size = counter.calculate_optimal_chunk_size(
            sample_standards,
            mode="enhanced"
        )
        
        print(f"  📊 Hierarchy generation:")
        print(f"     Optimal standards count: {optimal_chunk_size:,}")
        print(f"     Percentage of total: {optimal_hierarchy_count/500*100:.1f}%")
        
        print(f"  🎯 Categorization chunking:")
        print(f"     Optimal chunk size: {optimal_chunk_size:,} standards")
        print(f"     Expected chunks for 1000 standards: {(1000 + optimal_chunk_size - 1) // optimal_chunk_size}")
        
        # Calculate context utilization
        if hasattr(counter, 'MODEL_LIMITS') and model in counter.MODEL_LIMITS:
            context_window = counter.MODEL_LIMITS[model]['context_window']
            utilization_estimate = (optimal_chunk_size * 100) / context_window  # Rough estimate
            print(f"     Context utilization estimate: {utilization_estimate:.1f}%")
        
        print(f"  🚀 Performance improvement:")
        old_chunk_size = 25  # Previous conservative size
        improvement_factor = optimal_chunk_size / old_chunk_size
        print(f"     Improvement factor: {improvement_factor:.1f}x")
        print(f"     Processing speed improvement: ~{improvement_factor:.1f}x faster")

if __name__ == "__main__":
    test_token_optimization()
    print("\n✅ Token optimization test completed!")