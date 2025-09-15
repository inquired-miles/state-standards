#!/usr/bin/env python
"""
Test script for the EdTech RAG system
Run this after setting up the environment and database
"""
import os
import django
import json

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'state_standards_project.settings')
django.setup()

from standards.models import State, SubjectArea, GradeLevel, Standard
from standards.services import EmbeddingService, BellCurveAnalysisService, StorylineDiscoveryService


def test_basic_functionality():
    """Test basic functionality of the RAG system"""
    print("🚀 Testing EdTech RAG System")
    print("=" * 50)
    
    # Test 1: Check database setup
    print("\n1. Testing Database Setup...")
    states = State.objects.all()
    subjects = SubjectArea.objects.all()
    standards = Standard.objects.all()
    
    print(f"   States in database: {states.count()}")
    print(f"   Subject areas: {subjects.count()}")
    print(f"   Standards loaded: {standards.count()}")
    
    if states.count() == 0:
        print("   ⚠️  No states found. Run management commands to load data.")
        return False
    
    # Test 2: Test embedding service
    print("\n2. Testing Embedding Service...")
    embedding_service = EmbeddingService()
    
    test_text = "Students will learn about fraction operations and mathematical reasoning"
    embedding = embedding_service.generate_embedding(test_text)
    
    if embedding:
        print(f"   ✅ Generated embedding with {len(embedding)} dimensions")
    else:
        print("   ❌ Failed to generate embedding. Check OpenAI API key.")
        return False
    
    # Test 3: Test standards with embeddings
    print("\n3. Testing Standards with Embeddings...")
    standards_with_embeddings = Standard.objects.filter(embedding__isnull=False)
    print(f"   Standards with embeddings: {standards_with_embeddings.count()}")
    
    if standards_with_embeddings.count() == 0:
        print("   ⚠️  No standards have embeddings. Run generate_embeddings command.")
        print("   Attempting to generate embedding for one standard...")
        
        first_standard = standards.first()
        if first_standard:
            embedding = embedding_service.generate_standard_embedding(first_standard)
            if embedding:
                first_standard.embedding = embedding
                first_standard.save()
                print("   ✅ Generated embedding for test standard")
            else:
                print("   ❌ Failed to generate embedding for standard")
                return False
    
    # Test 4: Test Bell Curve Analysis Service
    print("\n4. Testing Bell Curve Analysis Service...")
    bell_curve_service = BellCurveAnalysisService()
    
    try:
        # Test coverage distribution
        distribution = bell_curve_service.analyze_coverage_distribution()
        print("   ✅ Coverage distribution analysis completed")
        print(f"   Found distribution data: {len(distribution.get('distribution', {})) > 0}")
        
        # Test MVC calculation
        mvc_result = bell_curve_service.find_minimum_viable_coverage(
            target_coverage_percentage=70.0,
            max_concepts=5
        )
        print("   ✅ Minimum Viable Coverage calculation completed")
        print(f"   MVC efficiency score: {mvc_result.get('efficiency_score', 'N/A')}")
        
    except Exception as e:
        print(f"   ❌ Bell curve analysis failed: {e}")
        return False
    
    # Test 5: Test Storyline Discovery Service
    print("\n5. Testing Storyline Discovery Service...")
    storyline_service = StorylineDiscoveryService()
    
    try:
        # Test common threads discovery
        common_threads = storyline_service.find_common_threads(
            min_state_coverage=2  # Lower threshold for testing
        )
        print("   ✅ Common threads discovery completed")
        print(f"   Found {len(common_threads)} common threads")
        
        # Test regional patterns
        regional_patterns = storyline_service.analyze_regional_patterns()
        print("   ✅ Regional patterns analysis completed")
        print(f"   Analyzed regions: {len(regional_patterns.get('regional_analysis', {}))}")
        
    except Exception as e:
        print(f"   ❌ Storyline discovery failed: {e}")
        return False
    
    # Test 6: Test content analysis simulation
    print("\n6. Testing Content Analysis...")
    try:
        # Simulate content analysis
        test_content = "This lesson teaches students about multiplication using arrays and equal groups. Students will solve word problems involving multiplication and division."
        
        content_embedding = embedding_service.generate_embedding(test_content)
        if content_embedding:
            print("   ✅ Generated embedding for test content")
            
            # Find similar standards (if any exist with embeddings)
            if standards_with_embeddings.count() > 0:
                from pgvector.django import CosineDistance
                
                similar_standards = Standard.objects.filter(
                    embedding__isnull=False
                ).annotate(
                    distance=CosineDistance('embedding', content_embedding)
                ).order_by('distance')[:5]
                
                print(f"   ✅ Found {similar_standards.count()} similar standards")
                for i, standard in enumerate(similar_standards, 1):
                    similarity = 1 - standard.distance
                    print(f"      {i}. {standard.state.code} - {standard.code} (similarity: {similarity:.3f})")
            else:
                print("   ⚠️  No standards with embeddings for similarity comparison")
                
        else:
            print("   ❌ Failed to generate embedding for test content")
            return False
            
    except Exception as e:
        print(f"   ❌ Content analysis failed: {e}")
        return False
    
    # Test 7: Database integrity check
    print("\n7. Database Integrity Check...")
    try:
        # Check for pgvector extension
        from django.db import connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            has_pgvector = cursor.fetchone() is not None
            
        if has_pgvector:
            print("   ✅ pgvector extension is installed")
        else:
            print("   ❌ pgvector extension not found")
            print("   Install with: CREATE EXTENSION vector;")
            return False
            
    except Exception as e:
        print(f"   ⚠️  Could not check pgvector extension: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! RAG system is ready to use.")
    print("\nNext steps:")
    print("1. Load more standards data: python manage.py bulk_import_standards sample_standards.json")
    print("2. Generate embeddings: python manage.py generate_embeddings")
    print("3. Start the server: python manage.py runserver")
    print("4. Test API endpoints at http://localhost:8000/api/rag/")
    
    return True


def print_api_examples():
    """Print example API calls"""
    print("\n📡 Example API Calls:")
    print("-" * 30)
    
    examples = [
        {
            "name": "Bell Curve Analysis",
            "endpoint": "POST /api/rag/bell-curve/",
            "body": {
                "concepts": ["fraction operations", "multiplication", "place value"],
                "subject_area": 1
            }
        },
        {
            "name": "Minimum Viable Coverage",
            "endpoint": "POST /api/rag/minimum-viable-coverage/",
            "body": {
                "target_coverage_percentage": 80.0,
                "subject_area": 1
            }
        },
        {
            "name": "Content Analysis",
            "endpoint": "POST /api/rag/analyze-content/",
            "body": {
                "content_title": "Fraction Lesson",
                "content_text": "Students will learn to add and subtract fractions with unlike denominators"
            }
        },
        {
            "name": "Common Threads",
            "endpoint": "GET /api/rag/common-threads/?min_state_coverage=25",
            "body": None
        }
    ]
    
    for example in examples:
        print(f"\n{example['name']}:")
        print(f"  {example['endpoint']}")
        if example['body']:
            print(f"  Body: {json.dumps(example['body'], indent=2)}")


if __name__ == "__main__":
    try:
        success = test_basic_functionality()
        if success:
            print_api_examples()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("Make sure you have:")
        print("1. PostgreSQL running with pgvector extension")
        print("2. Django migrations applied")
        print("3. OpenAI API key configured")
        print("4. Sample data loaded")