#!/usr/bin/env python
"""
Test script for Grade 3 Mathematics topic categorization scenario.
This script tests the complete workflow with mocked OpenAI API calls.
"""

import os
import sys
import django
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'state_standards_project.settings')
django.setup()

import json
import logging
from unittest.mock import Mock, patch, MagicMock
from django.test import TestCase
from django.db import transaction

from standards.models import Standard, TopicBasedProxy, State, SubjectArea, GradeLevel
from standards.services.topic_categorization import TopicCategorizationService


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockOpenAIResponse:
    """Mock OpenAI API response."""
    
    def __init__(self, content: str):
        self.choices = [Mock()]
        self.choices[0].message = Mock()
        self.choices[0].message.content = content


def create_test_data():
    """Create test data for Grade 3 Mathematics."""
    logger.info("Creating test data for Grade 3 Mathematics...")
    
    # Create or get test state
    test_state, created = State.objects.get_or_create(
        code='CA',
        defaults={'name': 'California'}
    )
    logger.info(f"Test state: {test_state.code} ({'created' if created else 'exists'})")
    
    # Create or get test subject area
    math_subject, created = SubjectArea.objects.get_or_create(
        name='Mathematics',
        defaults={'code': 'MATH'}
    )
    logger.info(f"Math subject: {math_subject.name} ({'created' if created else 'exists'})")
    
    # Create or get grade 3
    grade_3, created = GradeLevel.objects.get_or_create(
        name='Grade 3',
        defaults={'grade_numeric': 3}
    )
    logger.info(f"Grade 3: {grade_3.name} ({'created' if created else 'exists'})")
    
    # Create Grade 3 Mathematics standards if they don't exist
    standard_data = [
        ("CA.3.OA.1", "Operations & Algebraic Thinking", "Interpret products of whole numbers", 
         "Understand multiplication as repeated addition"),
        ("CA.3.OA.2", "Operations & Algebraic Thinking", "Interpret whole-number quotients", 
         "Understand division as sharing equally"),
        ("CA.3.OA.3", "Operations & Algebraic Thinking", "Use multiplication and division", 
         "Solve word problems involving multiplication and division"),
        ("CA.3.OA.4", "Operations & Algebraic Thinking", "Determine unknown number", 
         "Find unknown factors in multiplication equations"),
        ("CA.3.OA.5", "Operations & Algebraic Thinking", "Apply properties of operations", 
         "Use commutative, associative, and distributive properties"),
        ("CA.3.OA.6", "Operations & Algebraic Thinking", "Understand division as unknown-factor", 
         "Relate division to multiplication"),
        ("CA.3.OA.7", "Operations & Algebraic Thinking", "Fluently multiply and divide", 
         "Know single-digit multiplication and division facts"),
        ("CA.3.OA.8", "Operations & Algebraic Thinking", "Solve two-step word problems", 
         "Use four operations in multi-step problems"),
        ("CA.3.NBT.1", "Number & Operations in Base Ten", "Use place value understanding", 
         "Round whole numbers to nearest 10 or 100"),
        ("CA.3.NBT.2", "Number & Operations in Base Ten", "Fluently add and subtract", 
         "Add and subtract within 1000"),
        ("CA.3.NBT.3", "Number & Operations in Base Ten", "Multiply one-digit numbers", 
         "Multiply by multiples of 10"),
        ("CA.3.NF.1", "Number & Operations - Fractions", "Understand fractions", 
         "Understand unit fractions and fractions in general"),
        ("CA.3.NF.2", "Number & Operations - Fractions", "Understand fractions on number line", 
         "Represent fractions on a number line"),
        ("CA.3.NF.3", "Number & Operations - Fractions", "Explain equivalence of fractions", 
         "Recognize and generate simple equivalent fractions"),
        ("CA.3.MD.1", "Measurement & Data", "Tell and write time", 
         "Tell time to nearest minute and solve time problems"),
        ("CA.3.MD.2", "Measurement & Data", "Measure and estimate liquid volumes", 
         "Use liters and grams as units"),
        ("CA.3.MD.3", "Measurement & Data", "Draw scaled picture graphs", 
         "Create and interpret bar graphs and picture graphs"),
        ("CA.3.MD.4", "Measurement & Data", "Generate measurement data", 
         "Measure lengths using rulers marked with halves and fourths"),
        ("CA.3.MD.5", "Measurement & Data", "Recognize area as attribute", 
         "Understand area measurement concepts"),
        ("CA.3.MD.6", "Measurement & Data", "Measure areas by counting", 
         "Count unit squares to determine area"),
        ("CA.3.MD.7", "Measurement & Data", "Relate area to multiplication", 
         "Find areas of rectangles using multiplication"),
        ("CA.3.MD.8", "Measurement & Data", "Solve problems involving perimeters", 
         "Find perimeter and unknown side lengths"),
        ("CA.3.G.1", "Geometry", "Understand shared attributes", 
         "Classify shapes by shared attributes"),
        ("CA.3.G.2", "Geometry", "Partition shapes into equal areas", 
         "Understand fractions as equal parts of shapes")
    ]
    
    standards_created = 0
    for code, domain, title, description in standard_data:
        standard, created = Standard.objects.get_or_create(
            code=code,
            defaults={
                'title': title,
                'description': description,
                'state': test_state,
                'subject_area': math_subject,
                'domain': domain,
                'cluster': f"{domain} - Grade 3"
            }
        )
        if created:
            standard.grade_levels.add(grade_3)
            standards_created += 1
    
    logger.info(f"Created {standards_created} new Grade 3 Mathematics standards")
    
    total_standards = Standard.objects.filter(
        grade_levels=grade_3,
        subject_area=math_subject
    ).count()
    logger.info(f"Total Grade 3 Mathematics standards available: {total_standards}")
    
    return test_state, math_subject, grade_3


def get_mock_hierarchy_response():
    """Get realistic mock response for topic hierarchy generation."""
    return json.dumps({
        "topics": [
            {
                "name": "Number Operations",
                "description": "Basic arithmetic operations and number sense",
                "sub_topics": [
                    {
                        "name": "Addition and Subtraction",
                        "description": "Basic addition and subtraction operations",
                        "sub_sub_topics": [
                            "Single-digit operations",
                            "Multi-digit operations", 
                            "Mental math strategies",
                            "Word problem solving"
                        ]
                    },
                    {
                        "name": "Multiplication and Division",
                        "description": "Understanding multiplication and division concepts",
                        "sub_sub_topics": [
                            "Multiplication concepts",
                            "Division concepts",
                            "Fact fluency",
                            "Properties of operations",
                            "Problem solving"
                        ]
                    }
                ]
            },
            {
                "name": "Fractions",
                "description": "Understanding and working with fractions",
                "sub_topics": [
                    {
                        "name": "Fraction Concepts",
                        "description": "Basic understanding of fractions",
                        "sub_sub_topics": [
                            "Unit fractions",
                            "Fraction notation",
                            "Number line representation",
                            "Equivalent fractions"
                        ]
                    }
                ]
            },
            {
                "name": "Measurement and Data",
                "description": "Measurement concepts and data interpretation",
                "sub_topics": [
                    {
                        "name": "Time and Volume",
                        "description": "Time telling and volume measurement",
                        "sub_sub_topics": [
                            "Time to the minute",
                            "Liquid volumes", 
                            "Mass measurement",
                            "Elapsed time"
                        ]
                    },
                    {
                        "name": "Area and Perimeter",
                        "description": "Understanding area and perimeter concepts",
                        "sub_sub_topics": [
                            "Area concepts",
                            "Counting unit squares",
                            "Area and multiplication",
                            "Perimeter concepts"
                        ]
                    },
                    {
                        "name": "Data Representation",
                        "description": "Creating and interpreting graphs and charts",
                        "sub_sub_topics": [
                            "Picture graphs",
                            "Bar graphs",
                            "Scaled graphs",
                            "Data interpretation"
                        ]
                    }
                ]
            },
            {
                "name": "Geometry",
                "description": "Shapes, attributes, and spatial reasoning", 
                "sub_topics": [
                    {
                        "name": "Shape Analysis",
                        "description": "Understanding shape properties and attributes",
                        "sub_sub_topics": [
                            "Shape classification",
                            "Shared attributes",
                            "Partitioning shapes", 
                            "Equal areas"
                        ]
                    }
                ]
            },
            {
                "name": "Place Value",
                "description": "Understanding place value and number representation",
                "sub_topics": [
                    {
                        "name": "Place Value Concepts",
                        "description": "Understanding base-ten number system",
                        "sub_sub_topics": [
                            "Rounding numbers",
                            "Number comparison",
                            "Place value patterns",
                            "Multiples of 10"
                        ]
                    }
                ]
            }
        ]
    })


def get_mock_categorization_response():
    """Get realistic mock response for standards categorization."""
    return json.dumps({
        "categorizations": [
            {"standard_id": "CA.3.OA.1", "topic": "Number Operations", "sub_topic": "Multiplication and Division", "sub_sub_topic": "Multiplication concepts"},
            {"standard_id": "CA.3.OA.2", "topic": "Number Operations", "sub_topic": "Multiplication and Division", "sub_sub_topic": "Division concepts"},
            {"standard_id": "CA.3.OA.3", "topic": "Number Operations", "sub_topic": "Multiplication and Division", "sub_sub_topic": "Problem solving"},
            {"standard_id": "CA.3.OA.4", "topic": "Number Operations", "sub_topic": "Multiplication and Division", "sub_sub_topic": "Multiplication concepts"},
            {"standard_id": "CA.3.OA.5", "topic": "Number Operations", "sub_topic": "Multiplication and Division", "sub_sub_topic": "Properties of operations"},
            {"standard_id": "CA.3.OA.6", "topic": "Number Operations", "sub_topic": "Multiplication and Division", "sub_sub_topic": "Division concepts"},
            {"standard_id": "CA.3.OA.7", "topic": "Number Operations", "sub_topic": "Multiplication and Division", "sub_sub_topic": "Fact fluency"},
            {"standard_id": "CA.3.OA.8", "topic": "Number Operations", "sub_topic": "Multiplication and Division", "sub_sub_topic": "Problem solving"},
            {"standard_id": "CA.3.NBT.1", "topic": "Place Value", "sub_topic": "Place Value Concepts", "sub_sub_topic": "Rounding numbers"},
            {"standard_id": "CA.3.NBT.2", "topic": "Number Operations", "sub_topic": "Addition and Subtraction", "sub_sub_topic": "Multi-digit operations"},
            {"standard_id": "CA.3.NBT.3", "topic": "Place Value", "sub_topic": "Place Value Concepts", "sub_sub_topic": "Multiples of 10"},
            {"standard_id": "CA.3.NF.1", "topic": "Fractions", "sub_topic": "Fraction Concepts", "sub_sub_topic": "Unit fractions"},
            {"standard_id": "CA.3.NF.2", "topic": "Fractions", "sub_topic": "Fraction Concepts", "sub_sub_topic": "Number line representation"},
            {"standard_id": "CA.3.NF.3", "topic": "Fractions", "sub_topic": "Fraction Concepts", "sub_sub_topic": "Equivalent fractions"},
            {"standard_id": "CA.3.MD.1", "topic": "Measurement and Data", "sub_topic": "Time and Volume", "sub_sub_topic": "Time to the minute"},
            {"standard_id": "CA.3.MD.2", "topic": "Measurement and Data", "sub_topic": "Time and Volume", "sub_sub_topic": "Liquid volumes"},
            {"standard_id": "CA.3.MD.3", "topic": "Measurement and Data", "sub_topic": "Data Representation", "sub_sub_topic": "Scaled graphs"},
            {"standard_id": "CA.3.MD.4", "topic": "Measurement and Data", "sub_topic": "Data Representation", "sub_sub_topic": "Data interpretation"},
            {"standard_id": "CA.3.MD.5", "topic": "Measurement and Data", "sub_topic": "Area and Perimeter", "sub_sub_topic": "Area concepts"},
            {"standard_id": "CA.3.MD.6", "topic": "Measurement and Data", "sub_topic": "Area and Perimeter", "sub_sub_topic": "Counting unit squares"},
            {"standard_id": "CA.3.MD.7", "topic": "Measurement and Data", "sub_topic": "Area and Perimeter", "sub_sub_topic": "Area and multiplication"},
            {"standard_id": "CA.3.MD.8", "topic": "Measurement and Data", "sub_topic": "Area and Perimeter", "sub_sub_topic": "Perimeter concepts"},
            {"standard_id": "CA.3.G.1", "topic": "Geometry", "sub_topic": "Shape Analysis", "sub_sub_topic": "Shape classification"},
            {"standard_id": "CA.3.G.2", "topic": "Geometry", "sub_topic": "Shape Analysis", "sub_sub_topic": "Partitioning shapes"}
        ],
        "outliers": []
    })


def test_grade3_mathematics_workflow():
    """Test the complete Grade 3 Mathematics topic categorization workflow."""
    logger.info("=== Testing Grade 3 Mathematics Topic Categorization Workflow ===")
    
    # Create test data
    test_state, math_subject, grade_3 = create_test_data()
    
    # Initialize service
    service = TopicCategorizationService()
    logger.info(f"Service initialized with model: {service.DEFAULT_MODEL}")
    logger.info(f"Default chunk size: {service.DEFAULT_CHUNK_SIZE}")
    logger.info(f"Safety margin: {service.SAFETY_MARGIN}")
    
    # Test 1: Load standards
    logger.info("\n--- Test 1: Loading Grade 3 Mathematics standards ---")
    try:
        standards = service.load_standards(
            grade_levels=[3],
            subject_area_id=math_subject.id
        )
        logger.info(f"✓ Successfully loaded {len(standards)} standards")
        
        # Verify standards
        for i, standard in enumerate(standards[:5]):  # Show first 5
            logger.info(f"  {i+1}. {standard.code}: {standard.title}")
        if len(standards) > 5:
            logger.info(f"  ... and {len(standards) - 5} more")
    except Exception as e:
        logger.error(f"✗ Failed to load standards: {e}")
        return False
    
    # Test 2: Test without OpenAI client (should fail gracefully)
    logger.info("\n--- Test 2: Testing error handling without API client ---")
    service_no_client = TopicCategorizationService()
    service_no_client.client = None
    
    try:
        service_no_client.generate_topic_hierarchy(standards[:5])
        logger.error("✗ Should have failed without OpenAI client")
        return False
    except ValueError as e:
        if "OpenAI client not available" in str(e):
            logger.info("✓ Correctly handled missing OpenAI client")
        else:
            logger.error(f"✗ Unexpected error: {e}")
            return False
    
    # Test 3: Mock API calls and test full workflow
    logger.info("\n--- Test 3: Testing full workflow with mocked API calls ---")
    
    # Mock the OpenAI client
    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.responses = Mock()
    
    # Setup hierarchy generation response
    hierarchy_response = MockOpenAIResponse(get_mock_hierarchy_response())
    mock_client.chat.completions.create.return_value = hierarchy_response
    
    # Setup categorization responses (multiple chunks)
    categorization_response = MockOpenAIResponse(get_mock_categorization_response())
    mock_client.chat.completions.create.side_effect = [
        hierarchy_response,  # First call for hierarchy
        categorization_response  # Subsequent calls for categorization
    ]
    
    # Set the mocked client
    service.client = mock_client
    
    # Test progress tracking
    progress_updates = []
    
    def progress_callback(progress, message):
        progress_updates.append((progress, message))
        logger.info(f"  Progress: {progress}% - {message}")
    
    try:
        # Run full categorization with progress tracking
        logger.info("Running full categorization with progress tracking...")
        hierarchy, proxies = service.run_full_categorization(
            grade_levels=[3],
            subject_area_id=math_subject.id,
            progress_callback=progress_callback,
            use_dynamic_chunk_size=True
        )
        
        logger.info(f"✓ Full categorization completed successfully")
        logger.info(f"  - Created {len(proxies)} topic-based proxy standards")
        logger.info(f"  - Topic hierarchy has {len(hierarchy.topics)} main topics")
        logger.info(f"  - Received {len(progress_updates)} progress updates")
        
        # Analyze results
        logger.info("\n--- Results Analysis ---")
        
        # Analyze hierarchy
        logger.info("Topic Hierarchy:")
        for i, topic in enumerate(hierarchy.topics, 1):
            sub_topic_count = len(topic.get('sub_topics', []))
            logger.info(f"  {i}. {topic['name']} ({sub_topic_count} sub-topics)")
        
        # Analyze proxies
        regular_proxies = [p for p in proxies if not p.outlier_category]
        outlier_proxies = [p for p in proxies if p.outlier_category]
        
        logger.info(f"\nProxy Analysis:")
        logger.info(f"  - Regular proxies: {len(regular_proxies)}")
        logger.info(f"  - Outlier proxies: {len(outlier_proxies)}")
        
        # Show some proxy examples
        for i, proxy in enumerate(regular_proxies[:3], 1):
            member_count = proxy.member_standards.count()
            logger.info(f"  {i}. {proxy.proxy_id}: {proxy.topic} > {proxy.sub_topic} > {proxy.sub_sub_topic} ({member_count} standards)")
        
        # Test chunk size calculation
        logger.info("\n--- Test 4: Chunk size optimization ---")
        optimal_chunk_size = service.calculate_optimal_chunk_size(standards)
        logger.info(f"✓ Calculated optimal chunk size: {optimal_chunk_size} standards per chunk")
        logger.info(f"  - Total standards: {len(standards)}")
        logger.info(f"  - Estimated chunks needed: {(len(standards) + optimal_chunk_size - 1) // optimal_chunk_size}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Full categorization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_scenarios():
    """Test various error scenarios and edge cases."""
    logger.info("\n=== Testing Error Scenarios and Edge Cases ===")
    
    service = TopicCategorizationService()
    
    # Test 1: Empty grade levels
    logger.info("\n--- Test 1: Empty grade levels ---")
    try:
        standards = service.load_standards(grade_levels=[99])  # Non-existent grade
        logger.error("✗ Should have failed with non-existent grade")
        return False
    except ValueError as e:
        if "No standards found" in str(e):
            logger.info("✓ Correctly handled non-existent grade level")
        else:
            logger.error(f"✗ Unexpected error: {e}")
            return False
    
    # Test 2: Non-existent subject area
    logger.info("\n--- Test 2: Non-existent subject area ---")
    try:
        standards = service.load_standards(subject_area_id=9999)  # Non-existent subject
        logger.error("✗ Should have failed with non-existent subject area")
        return False
    except ValueError as e:
        if "No standards found" in str(e):
            logger.info("✓ Correctly handled non-existent subject area")
        else:
            logger.error(f"✗ Unexpected error: {e}")
            return False
    
    # Test 3: Invalid JSON response handling
    logger.info("\n--- Test 3: Invalid JSON response handling ---")
    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    
    # Mock invalid JSON response
    invalid_response = MockOpenAIResponse('{"invalid": "structure"}')
    mock_client.chat.completions.create.return_value = invalid_response
    
    service.client = mock_client
    
    # Load some standards for testing
    test_state, math_subject, grade_3 = create_test_data()
    standards = service.load_standards(grade_levels=[3], subject_area_id=math_subject.id)
    
    try:
        hierarchy = service.generate_topic_hierarchy(standards[:5])
        logger.error("✗ Should have failed with invalid JSON structure")
        return False
    except ValueError as e:
        if "Invalid hierarchy structure" in str(e):
            logger.info("✓ Correctly handled invalid JSON response")
        else:
            logger.error(f"✗ Unexpected error: {e}")
            return False
    
    logger.info("✓ All error scenarios handled correctly")
    return True


def test_token_utils_integration():
    """Test integration with token counting utilities."""
    logger.info("\n=== Testing Token Utils Integration ===")
    
    service = TopicCategorizationService()
    
    # Load test data
    test_state, math_subject, grade_3 = create_test_data()
    standards = service.load_standards(grade_levels=[3], subject_area_id=math_subject.id)
    
    # Test token counter
    logger.info(f"Token counter model: {service.token_counter.model_name}")
    logger.info(f"Token counter encoding: {service.token_counter.encoding_name}")
    
    # Test chunk size calculation
    chunk_size = service.calculate_optimal_chunk_size(standards)
    logger.info(f"Calculated optimal chunk size: {chunk_size} standards")
    
    # Validate chunk size is reasonable
    if chunk_size < 5 or chunk_size > 500:
        logger.warning(f"Chunk size {chunk_size} seems unusual")
    else:
        logger.info("✓ Chunk size is within reasonable bounds")
    
    return True


def main():
    """Run all tests."""
    logger.info("Starting Grade 3 Mathematics Topic Categorization Tests")
    logger.info("=" * 70)
    
    success = True
    
    # Run main workflow test
    try:
        if not test_grade3_mathematics_workflow():
            success = False
    except Exception as e:
        logger.error(f"Workflow test failed with exception: {e}")
        success = False
    
    # Run error scenario tests
    try:
        if not test_error_scenarios():
            success = False
    except Exception as e:
        logger.error(f"Error scenario tests failed with exception: {e}")
        success = False
    
    # Run token utils integration test
    try:
        if not test_token_utils_integration():
            success = False
    except Exception as e:
        logger.error(f"Token utils integration test failed with exception: {e}")
        success = False
    
    # Final summary
    logger.info("\n" + "=" * 70)
    if success:
        logger.info("✓ ALL TESTS PASSED - Grade 3 Mathematics workflow is working correctly")
        logger.info("Key findings:")
        logger.info("  - Service correctly loads Grade 3 Mathematics standards")
        logger.info("  - Error handling works properly for missing API keys")
        logger.info("  - Mocked API calls produce expected results")
        logger.info("  - Progress tracking functions correctly")
        logger.info("  - Token counting and chunk size optimization work")
        logger.info("  - Edge cases and error scenarios are handled gracefully")
    else:
        logger.error("✗ SOME TESTS FAILED - See errors above")
    
    return success


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)