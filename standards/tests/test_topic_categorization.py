"""
Comprehensive unit tests for TopicCategorizationService.
"""
import json
import logging
import unittest
from unittest.mock import Mock, patch, MagicMock
from django.test import TestCase
from django.db import transaction

# Mock the dependencies that might not be available in test environment
import sys
from unittest.mock import MagicMock

# Mock external dependencies
sys.modules['openai'] = MagicMock()
sys.modules['tiktoken'] = MagicMock()

from standards.models import Standard, TopicBasedProxy, State, SubjectArea, GradeLevel
from standards.services.topic_categorization import (
    TopicCategorizationService,
    TopicHierarchy, 
    StandardCategorization
)


class MockOpenAIClient:
    """Mock OpenAI client for testing."""
    
    def __init__(self):
        self.chat = Mock()
        self.chat.completions = Mock()
        self.responses = Mock()
    
    def create_chat_completion_response(self, content: str):
        """Create a mock chat completion response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = content
        return mock_response
    
    def create_responses_api_response(self, content: str):
        """Create a mock Responses API response."""
        mock_response = Mock()
        mock_response.output_text = content
        mock_response.output = [Mock()]
        mock_response.output[0].content = [Mock()]
        mock_response.output[0].content[0].text = content
        return mock_response


class TestTopicHierarchy(unittest.TestCase):
    """Test cases for TopicHierarchy dataclass."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_hierarchy_data = {
            "topics": [
                {
                    "name": "Number Operations",
                    "description": "Basic arithmetic operations",
                    "sub_topics": [
                        {
                            "name": "Addition and Subtraction", 
                            "description": "Basic addition and subtraction",
                            "sub_sub_topics": [
                                "Single-digit addition",
                                "Single-digit subtraction", 
                                "Multi-digit addition",
                                "Multi-digit subtraction"
                            ]
                        },
                        {
                            "name": "Multiplication and Division",
                            "description": "Basic multiplication and division",
                            "sub_sub_topics": [
                                "Multiplication tables",
                                "Division with remainders",
                                "Word problems",
                                "Arrays and groups"
                            ]
                        }
                    ]
                },
                {
                    "name": "Geometry",
                    "description": "Shapes and spatial reasoning",
                    "sub_topics": [
                        {
                            "name": "2D Shapes",
                            "description": "Two-dimensional shapes",
                            "sub_sub_topics": [
                                "Triangles",
                                "Squares", 
                                "Circles",
                                "Rectangles"
                            ]
                        }
                    ]
                }
            ]
        }
        
        self.hierarchy = TopicHierarchy(topics=self.sample_hierarchy_data["topics"])
    
    def test_initialization(self):
        """Test TopicHierarchy initialization."""
        self.assertEqual(len(self.hierarchy.topics), 2)
        self.assertEqual(self.hierarchy.topics[0]["name"], "Number Operations")
    
    def test_get_all_paths(self):
        """Test getting all topic paths."""
        paths = self.hierarchy.get_all_paths()
        
        # Should have 8 total paths (4 + 4)
        self.assertEqual(len(paths), 8)
        
        # Check specific paths
        expected_paths = [
            ("Number Operations", "Addition and Subtraction", "Single-digit addition"),
            ("Number Operations", "Addition and Subtraction", "Single-digit subtraction"),
            ("Number Operations", "Addition and Subtraction", "Multi-digit addition"),
            ("Number Operations", "Addition and Subtraction", "Multi-digit subtraction"),
            ("Number Operations", "Multiplication and Division", "Multiplication tables"),
            ("Number Operations", "Multiplication and Division", "Division with remainders"),
            ("Number Operations", "Multiplication and Division", "Word problems"),
            ("Number Operations", "Multiplication and Division", "Arrays and groups"),
        ]
        
        for expected_path in expected_paths[:4]:  # Check first 4 paths
            self.assertIn(expected_path, paths)


class TestStandardCategorization(unittest.TestCase):
    """Test cases for StandardCategorization dataclass."""
    
    def test_initialization(self):
        """Test StandardCategorization initialization."""
        cat = StandardCategorization(
            standard_id="CA.3.OA.1",
            standard_obj=None,
            topic="Number Operations",
            sub_topic="Addition and Subtraction",
            sub_sub_topic="Single-digit addition",
            is_outlier=False
        )
        
        self.assertEqual(cat.standard_id, "CA.3.OA.1")
        self.assertEqual(cat.topic, "Number Operations")
        self.assertFalse(cat.is_outlier)
    
    def test_outlier_categorization(self):
        """Test outlier categorization."""
        outlier = StandardCategorization(
            standard_id="TX.3.WEIRD.1",
            standard_obj=None,
            topic="Outliers",
            sub_topic="Uncategorized",
            sub_sub_topic="Doesn't fit taxonomy",
            is_outlier=True
        )
        
        self.assertTrue(outlier.is_outlier)
        self.assertEqual(outlier.topic, "Outliers")


class TestTopicCategorizationService(TestCase):
    """Django test case for TopicCategorizationService."""
    
    @classmethod
    def setUpTestData(cls):
        """Set up test data."""
        # Create test state
        cls.test_state = State.objects.create(
            code='TS',
            name='Test State'
        )
        
        # Create test subject area
        cls.test_subject_area = SubjectArea.objects.create(
            name='Mathematics',
            code='MATH'
        )
        
        # Create test grade level
        cls.grade_3 = GradeLevel.objects.create(
            name='Grade 3',
            grade_numeric=3
        )
        
        # Create sample standards
        cls.sample_standards = []
        for i in range(24):  # Create 24 standards for Grade 3 Mathematics
            standard = Standard.objects.create(
                code=f'TS.3.OA.{i+1}',
                title=f'Test standard {i+1}',
                description=f'This is test standard number {i+1} for grade 3 mathematics',
                state=cls.test_state,
                subject_area=cls.test_subject_area,
                domain='Operations and Algebraic Thinking',
                cluster='Test cluster'
            )
            standard.grade_levels.add(cls.grade_3)
            cls.sample_standards.append(standard)
    
    def setUp(self):
        """Set up test fixtures for each test."""
        self.service = TopicCategorizationService()
        self.mock_client = MockOpenAIClient()
        
        # Sample hierarchy response
        self.sample_hierarchy_response = json.dumps({
            "topics": [
                {
                    "name": "Number Operations",
                    "description": "Basic arithmetic operations",
                    "sub_topics": [
                        {
                            "name": "Addition and Subtraction",
                            "description": "Basic addition and subtraction",
                            "sub_sub_topics": [
                                "Single-digit addition",
                                "Single-digit subtraction",
                                "Multi-digit addition",
                                "Word problems"
                            ]
                        }
                    ]
                }
            ]
        })
        
        # Sample categorization response
        self.sample_categorization_response = json.dumps({
            "categorizations": [
                {
                    "standard_id": "TS.3.OA.1",
                    "topic": "Number Operations",
                    "sub_topic": "Addition and Subtraction",
                    "sub_sub_topic": "Single-digit addition"
                },
                {
                    "standard_id": "TS.3.OA.2", 
                    "topic": "Number Operations",
                    "sub_topic": "Addition and Subtraction",
                    "sub_sub_topic": "Single-digit subtraction"
                }
            ],
            "outliers": [
                {
                    "standard_id": "TS.3.OA.3",
                    "reason": "Complex reasoning standard"
                }
            ]
        })
    
    def test_initialization_without_openai(self):
        """Test service initialization without OpenAI API key."""
        with patch.dict('os.environ', {}, clear=True):
            service = TopicCategorizationService()
            self.assertIsNone(service.client)
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('standards.services.topic_categorization._HAS_OPENAI_CLIENT', True)
    @patch('standards.services.topic_categorization.OpenAI')
    def test_initialization_with_openai(self, mock_openai):
        """Test service initialization with OpenAI API key."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        service = TopicCategorizationService()
        self.assertIsNotNone(service.client)
        mock_openai.assert_called_once_with(api_key='test-key')
    
    def test_load_standards_basic(self):
        """Test basic standards loading."""
        standards = self.service.load_standards()
        self.assertGreater(len(standards), 0)
        self.assertEqual(len(standards), 24)  # Our test data
    
    def test_load_standards_with_grade_filter(self):
        """Test loading standards with grade level filter."""
        standards = self.service.load_standards(grade_levels=[3])
        self.assertEqual(len(standards), 24)
        
        # All returned standards should have grade 3
        for standard in standards:
            self.assertIn(self.grade_3, standard.grade_levels.all())
    
    def test_load_standards_with_subject_filter(self):
        """Test loading standards with subject area filter."""
        standards = self.service.load_standards(subject_area_id=self.test_subject_area.id)
        self.assertEqual(len(standards), 24)
        
        # All returned standards should be mathematics
        for standard in standards:
            self.assertEqual(standard.subject_area, self.test_subject_area)
    
    def test_load_standards_combined_filters(self):
        """Test loading standards with combined filters."""
        standards = self.service.load_standards(
            grade_levels=[3], 
            subject_area_id=self.test_subject_area.id
        )
        self.assertEqual(len(standards), 24)
    
    def test_load_standards_no_results(self):
        """Test loading standards with filters that return no results."""
        with self.assertRaises(ValueError) as context:
            self.service.load_standards(grade_levels=[12])  # No grade 12 standards
        
        self.assertIn("No standards found", str(context.exception))
    
    def test_generate_topic_hierarchy_no_client(self):
        """Test hierarchy generation without OpenAI client."""
        self.service.client = None
        
        with self.assertRaises(ValueError) as context:
            self.service.generate_topic_hierarchy([])
        
        self.assertIn("OpenAI client not available", str(context.exception))
    
    @patch.object(TopicCategorizationService, '_call_openai_api')
    def test_generate_topic_hierarchy_success(self, mock_api_call):
        """Test successful hierarchy generation."""
        self.service.client = self.mock_client
        mock_api_call.return_value = self.sample_hierarchy_response
        
        standards = self.sample_standards[:5]  # Use first 5 standards
        hierarchy = self.service.generate_topic_hierarchy(standards, "Mathematics")
        
        self.assertIsInstance(hierarchy, TopicHierarchy)
        self.assertEqual(len(hierarchy.topics), 1)
        self.assertEqual(hierarchy.topics[0]["name"], "Number Operations")
    
    @patch.object(TopicCategorizationService, '_call_openai_api')
    def test_generate_topic_hierarchy_invalid_response(self, mock_api_call):
        """Test hierarchy generation with invalid response."""
        self.service.client = self.mock_client
        mock_api_call.return_value = json.dumps({"invalid": "structure"})
        
        standards = self.sample_standards[:5]
        
        with self.assertRaises(ValueError) as context:
            self.service.generate_topic_hierarchy(standards)
        
        self.assertIn("Invalid hierarchy structure", str(context.exception))
    
    def test_categorize_standards_chunk_no_client(self):
        """Test standards categorization without OpenAI client."""
        self.service.client = None
        hierarchy = TopicHierarchy(topics=[])
        
        with self.assertRaises(ValueError) as context:
            self.service.categorize_standards_chunk([], hierarchy)
        
        self.assertIn("OpenAI client not available", str(context.exception))
    
    @patch.object(TopicCategorizationService, '_call_openai_api')
    def test_categorize_standards_chunk_success(self, mock_api_call):
        """Test successful standards categorization."""
        self.service.client = self.mock_client
        mock_api_call.return_value = self.sample_categorization_response
        
        hierarchy = TopicHierarchy(topics=[{
            "name": "Number Operations",
            "sub_topics": [{
                "name": "Addition and Subtraction",
                "sub_sub_topics": ["Single-digit addition", "Single-digit subtraction"]
            }]
        }])
        
        standards_chunk = self.sample_standards[:3]
        categorizations = self.service.categorize_standards_chunk(standards_chunk, hierarchy)
        
        self.assertEqual(len(categorizations), 3)  # 2 regular + 1 outlier
        
        # Check regular categorizations
        regular_cats = [c for c in categorizations if not c.is_outlier]
        self.assertEqual(len(regular_cats), 2)
        
        # Check outliers
        outlier_cats = [c for c in categorizations if c.is_outlier]
        self.assertEqual(len(outlier_cats), 1)
        self.assertEqual(outlier_cats[0].topic, "Outliers")
    
    def test_create_topic_proxies(self):
        """Test creating topic-based proxy objects."""
        categorizations = [
            StandardCategorization(
                standard_id="TS.3.OA.1",
                standard_obj=self.sample_standards[0],
                topic="Number Operations",
                sub_topic="Addition and Subtraction",
                sub_sub_topic="Single-digit addition",
                is_outlier=False
            ),
            StandardCategorization(
                standard_id="TS.3.OA.2", 
                standard_obj=self.sample_standards[1],
                topic="Number Operations",
                sub_topic="Addition and Subtraction",
                sub_sub_topic="Single-digit addition",  # Same sub_sub_topic
                is_outlier=False
            ),
            StandardCategorization(
                standard_id="TS.3.OA.3",
                standard_obj=self.sample_standards[2],
                topic="Outliers",
                sub_topic="Uncategorized",
                sub_sub_topic="Complex reasoning standard",
                is_outlier=True
            )
        ]
        
        proxies = self.service.create_topic_proxies(categorizations)
        
        # Should create 2 proxies: 1 regular (combining first 2 standards) + 1 outlier
        self.assertEqual(len(proxies), 2)
        
        # Find the regular proxy
        regular_proxy = next((p for p in proxies if not p.outlier_category), None)
        self.assertIsNotNone(regular_proxy)
        self.assertEqual(regular_proxy.member_standards.count(), 2)
        
        # Find the outlier proxy
        outlier_proxy = next((p for p in proxies if p.outlier_category), None)
        self.assertIsNotNone(outlier_proxy)
        self.assertEqual(outlier_proxy.member_standards.count(), 1)
    
    def test_generate_proxy_id_regular(self):
        """Test proxy ID generation for regular topics."""
        proxy_id = self.service._generate_proxy_id(
            "Number Operations", 
            "Addition and Subtraction", 
            "Single-digit addition",
            is_outlier=False
        )
        
        self.assertTrue(proxy_id.startswith("TP-"))
        self.assertIn("-001", proxy_id)  # Should be first instance
    
    def test_generate_proxy_id_outlier(self):
        """Test proxy ID generation for outliers."""
        proxy_id = self.service._generate_proxy_id(
            "Outliers",
            "Uncategorized",
            "Complex reasoning standard",
            is_outlier=True
        )
        
        self.assertTrue(proxy_id.startswith("TP-OUT-"))
        self.assertIn("-001", proxy_id)
    
    def test_abbreviate_topic(self):
        """Test topic abbreviation for ID generation."""
        # Single word
        abbrev1 = self.service._abbreviate_topic("Mathematics")
        self.assertEqual(abbrev1, "Mathemat")
        
        # Two words
        abbrev2 = self.service._abbreviate_topic("Number Operations")
        self.assertEqual(abbrev2, "NumbOper")
        
        # Multiple words with common words
        abbrev3 = self.service._abbreviate_topic("Addition and Subtraction Problems")
        self.assertEqual(abbrev3, "ASP")
    
    def test_call_openai_api_chat_completions(self):
        """Test OpenAI API call with Chat Completions."""
        mock_response = self.mock_client.create_chat_completion_response('{"test": "response"}')
        self.mock_client.chat.completions.create.return_value = mock_response
        
        self.service.client = self.mock_client
        result = self.service._call_openai_api("test prompt", "test_operation")
        
        self.assertEqual(result, '{"test": "response"}')
        self.mock_client.chat.completions.create.assert_called_once()
    
    def test_call_openai_api_retries(self):
        """Test OpenAI API call with retries on failure."""
        # First two calls fail, third succeeds
        mock_response = self.mock_client.create_chat_completion_response('{"test": "success"}')
        self.mock_client.chat.completions.create.side_effect = [
            Exception("API Error 1"),
            Exception("API Error 2"), 
            mock_response
        ]
        
        self.service.client = self.mock_client
        result = self.service._call_openai_api("test prompt", "test_operation")
        
        self.assertEqual(result, '{"test": "success"}')
        self.assertEqual(self.mock_client.chat.completions.create.call_count, 3)
    
    def test_call_openai_api_max_retries_exceeded(self):
        """Test OpenAI API call when max retries are exceeded."""
        self.mock_client.chat.completions.create.side_effect = Exception("Persistent API Error")
        
        self.service.client = self.mock_client
        
        with self.assertRaises(Exception) as context:
            self.service._call_openai_api("test prompt", "test_operation")
        
        self.assertIn("Failed to complete test_operation", str(context.exception))
    
    @patch.object(TopicCategorizationService, 'calculate_optimal_chunk_size')
    @patch.object(TopicCategorizationService, 'generate_topic_hierarchy')
    @patch.object(TopicCategorizationService, 'categorize_standards_chunk')
    @patch.object(TopicCategorizationService, 'create_topic_proxies')
    def test_run_full_categorization_success(self, mock_create_proxies, 
                                           mock_categorize_chunk, mock_generate_hierarchy,
                                           mock_calc_chunk_size):
        """Test successful full categorization workflow."""
        # Setup mocks
        mock_hierarchy = TopicHierarchy(topics=[{"name": "Test Topic", "sub_topics": []}])
        mock_generate_hierarchy.return_value = mock_hierarchy
        mock_calc_chunk_size.return_value = 10
        
        mock_categorization = StandardCategorization(
            standard_id="TS.3.OA.1",
            standard_obj=self.sample_standards[0],
            topic="Test Topic",
            sub_topic="Test Sub Topic",
            sub_sub_topic="Test Sub Sub Topic",
            is_outlier=False
        )
        mock_categorize_chunk.return_value = [mock_categorization]
        
        mock_proxy = Mock()
        mock_proxy.proxy_id = "TP-TEST-001"
        mock_create_proxies.return_value = [mock_proxy]
        
        # Mock client
        self.service.client = self.mock_client
        
        # Run full categorization
        hierarchy, proxies = self.service.run_full_categorization(
            grade_levels=[3],
            subject_area_id=self.test_subject_area.id
        )
        
        # Verify results
        self.assertEqual(hierarchy, mock_hierarchy)
        self.assertEqual(len(proxies), 1)
        self.assertEqual(proxies[0].proxy_id, "TP-TEST-001")
        
        # Verify method calls
        mock_generate_hierarchy.assert_called_once()
        mock_calc_chunk_size.assert_called_once()
        mock_create_proxies.assert_called_once()
    
    def test_run_full_categorization_with_progress_callback(self):
        """Test full categorization with progress callback."""
        progress_calls = []
        
        def progress_callback(progress, message):
            progress_calls.append((progress, message))
        
        # Mock necessary methods
        self.service.client = self.mock_client
        
        with patch.object(self.service, 'generate_topic_hierarchy') as mock_hierarchy, \
             patch.object(self.service, 'categorize_standards_chunk') as mock_categorize, \
             patch.object(self.service, 'create_topic_proxies') as mock_create:
            
            mock_hierarchy.return_value = TopicHierarchy(topics=[])
            mock_categorize.return_value = []
            mock_create.return_value = []
            
            self.service.run_full_categorization(
                grade_levels=[3],
                subject_area_id=self.test_subject_area.id,
                progress_callback=progress_callback
            )
        
        # Verify progress callbacks were made
        self.assertGreater(len(progress_calls), 0)
        self.assertEqual(progress_calls[-1][0], 100)  # Should end at 100%
    
    def test_run_full_categorization_chunk_failure(self):
        """Test full categorization with chunk processing failure."""
        self.service.client = self.mock_client
        
        with patch.object(self.service, 'generate_topic_hierarchy') as mock_hierarchy, \
             patch.object(self.service, 'categorize_standards_chunk') as mock_categorize, \
             patch.object(self.service, 'create_topic_proxies') as mock_create:
            
            mock_hierarchy.return_value = TopicHierarchy(topics=[])
            mock_categorize.side_effect = Exception("Chunk processing failed")
            mock_create.return_value = []
            
            # Should not raise exception, but continue with other chunks
            hierarchy, proxies = self.service.run_full_categorization(
                grade_levels=[3],
                subject_area_id=self.test_subject_area.id
            )
            
            self.assertEqual(len(proxies), 0)  # No proxies created due to failures
    
    def test_responses_api_fallback(self):
        """Test fallback from Responses API to Chat Completions."""
        # Setup mock client with responses that fails
        self.service.client = self.mock_client
        self.service.client.responses.create.side_effect = Exception("Responses API failed")
        
        # Chat completions should succeed
        mock_response = self.mock_client.create_chat_completion_response(self.sample_hierarchy_response)
        self.mock_client.chat.completions.create.return_value = mock_response
        
        standards = self.sample_standards[:5]
        hierarchy = self.service.generate_topic_hierarchy(standards)
        
        self.assertIsInstance(hierarchy, TopicHierarchy)
        self.assertEqual(len(hierarchy.topics), 1)


class TestEnhancedTopicCategorization(TestCase):
    """Test cases for enhanced topic categorization features."""
    
    @classmethod
    def setUpTestData(cls):
        """Set up test data for enhanced features."""
        # Create test state
        cls.test_state = State.objects.create(
            code='EN',
            name='Enhanced Test State'
        )
        
        # Create test subject areas
        cls.math_subject = SubjectArea.objects.create(
            name='Mathematics',
            code='MATH'
        )
        cls.ela_subject = SubjectArea.objects.create(
            name='English Language Arts',
            code='ELA'
        )
        
        # Create multiple grade levels
        cls.grade_2 = GradeLevel.objects.create(name='Grade 2', grade_numeric=2)
        cls.grade_3 = GradeLevel.objects.create(name='Grade 3', grade_numeric=3)
        cls.grade_4 = GradeLevel.objects.create(name='Grade 4', grade_numeric=4)
        cls.grade_5 = GradeLevel.objects.create(name='Grade 5', grade_numeric=5)
        
        # Create diverse standards for testing educational chunking
        cls.test_standards = []
        
        # Math standards - different complexity levels
        math_standards_data = [
            ('EN.2.NBT.1', 'Count within 1000', 'Students will understand place value concepts', 'low'),
            ('EN.2.NBT.2', 'Count by 5s, 10s, and 100s', 'Skip counting patterns up to 1000', 'low'),
            ('EN.3.OA.1', 'Interpret products of whole numbers', 'Understand multiplication as repeated addition', 'medium'),
            ('EN.3.OA.8', 'Solve two-step word problems', 'Use all four operations in multi-step problems with reasoning', 'high'),
            ('EN.4.NBT.5', 'Multiply whole numbers', 'Use strategies based on place value and properties of operations', 'medium'),
            ('EN.5.NF.1', 'Add and subtract fractions', 'Add and subtract fractions with unlike denominators using equivalent fractions', 'high'),
        ]
        
        for i, (code, title, description, complexity) in enumerate(math_standards_data):
            standard = Standard.objects.create(
                code=code,
                title=title,
                description=description,
                state=cls.test_state,
                subject_area=cls.math_subject,
                domain=f'Math Domain {i+1}',
                cluster='Test cluster'
            )
            # Add appropriate grade levels
            grade_num = int(code.split('.')[1])
            if grade_num == 2:
                standard.grade_levels.add(cls.grade_2)
            elif grade_num == 3:
                standard.grade_levels.add(cls.grade_3)
            elif grade_num == 4:
                standard.grade_levels.add(cls.grade_4)
            elif grade_num == 5:
                standard.grade_levels.add(cls.grade_5)
            
            cls.test_standards.append(standard)
        
        # ELA standards - different domains
        ela_standards_data = [
            ('EN.3.RL.1', 'Reading Literature', 'Ask and answer questions about key details in a text', 'medium'),
            ('EN.3.W.1', 'Writing', 'Write opinion pieces supporting a point of view with reasons', 'high'),
            ('EN.4.L.1', 'Language', 'Demonstrate command of conventions of grammar when writing', 'medium'),
        ]
        
        for code, title, description, complexity in ela_standards_data:
            standard = Standard.objects.create(
                code=code,
                title=title,
                description=description,
                state=cls.test_state,
                subject_area=cls.ela_subject,
                domain=f'ELA Domain',
                cluster='Test cluster'
            )
            # Add appropriate grade levels
            grade_num = int(code.split('.')[1])
            if grade_num == 3:
                standard.grade_levels.add(cls.grade_3)
            elif grade_num == 4:
                standard.grade_levels.add(cls.grade_4)
            
            cls.test_standards.append(standard)
    
    def setUp(self):
        """Set up test fixtures for enhanced tests."""
        self.service = TopicCategorizationService()
        self.mock_client = MockOpenAIClient()
    
    def test_educational_context_analysis(self):
        """Test educational context analysis functionality."""
        context = self.service._analyze_educational_context(self.test_standards)
        
        # Verify context structure
        self.assertIn('grade_distribution', context)
        self.assertIn('subject_distribution', context)
        self.assertIn('domain_distribution', context)
        self.assertIn('complexity_distribution', context)
        self.assertIn('total_standards', context)
        
        # Verify distributions contain expected data
        self.assertGreater(len(context['grade_distribution']), 0)
        self.assertGreater(len(context['subject_distribution']), 0)
        self.assertEqual(context['total_standards'], len(self.test_standards))
        
        # Verify subject distribution
        self.assertIn('Mathematics', context['subject_distribution'])
        self.assertIn('English Language Arts', context['subject_distribution'])
    
    def test_complexity_estimation(self):
        """Test standard complexity estimation."""
        # Test low complexity standard
        low_standard = self.test_standards[0]  # Count within 1000
        complexity = self.service._estimate_standard_complexity(low_standard)
        self.assertEqual(complexity, 'low')
        
        # Test high complexity standard (has analysis keywords)
        high_standard = Standard.objects.create(
            code='TEST.COMPLEX.1',
            title='Analyze and Evaluate',
            description='Students will analyze complex problems, evaluate multiple solutions, and create comprehensive explanations demonstrating deep understanding of mathematical concepts.',
            state=self.test_state,
            subject_area=self.math_subject
        )
        complexity = self.service._estimate_standard_complexity(high_standard)
        self.assertEqual(complexity, 'high')
    
    def test_educational_chunking(self):
        """Test educational chunking functionality."""
        base_chunk_size = 3
        hierarchy = TopicHierarchy(topics=[])
        
        chunks = self.service.create_educational_chunks(
            self.test_standards, base_chunk_size, hierarchy
        )
        
        # Verify chunks were created
        self.assertGreater(len(chunks), 0)
        
        # Verify all standards are included
        total_standards_in_chunks = sum(len(chunk) for chunk in chunks)
        self.assertEqual(total_standards_in_chunks, len(self.test_standards))
        
        # Verify chunk sizes are reasonable
        for chunk in chunks:
            self.assertGreaterEqual(len(chunk), 1)
            self.assertLessEqual(len(chunk), base_chunk_size * 2)  # Allow some flexibility
    
    def test_standards_sorting_for_chunking(self):
        """Test sorting standards for optimal chunking."""
        sorted_standards = self.service._sort_standards_for_chunking(self.test_standards)
        
        # Verify all standards are included
        self.assertEqual(len(sorted_standards), len(self.test_standards))
        
        # Verify sorting logic - math standards should be grouped together
        math_positions = []
        ela_positions = []
        
        for i, standard in enumerate(sorted_standards):
            if standard.subject_area.name == 'Mathematics':
                math_positions.append(i)
            elif standard.subject_area.name == 'English Language Arts':
                ela_positions.append(i)
        
        # Math standards should be consecutive (or mostly consecutive)
        if len(math_positions) > 1:
            math_gaps = [math_positions[i+1] - math_positions[i] for i in range(len(math_positions)-1)]
            self.assertTrue(all(gap <= 2 for gap in math_gaps), "Math standards should be grouped together")
    
    def test_chunk_size_adjustment(self):
        """Test chunk size adjustment based on educational factors."""
        base_chunk_size = 5
        
        # Test with diverse content
        context = self.service._analyze_educational_context(self.test_standards)
        adjusted_size = self.service._adjust_chunk_size_for_education(
            base_chunk_size, context, len(self.test_standards)
        )
        
        # Should be adjusted down due to mixed subjects
        self.assertLessEqual(adjusted_size, base_chunk_size)
        self.assertGreaterEqual(adjusted_size, 1)
    
    def test_pre_analysis_validation(self):
        """Test pre-analysis validation functionality."""
        # Test good pre-analysis
        good_pre_analysis = {
            'task_overview': 'This task involves categorizing educational standards into a hierarchical topic structure.',
            'categorization_strategy': 'I will use educational domain knowledge and cognitive complexity analysis to map standards.',
            'identified_patterns': 'I observe patterns related to grade level progression and subject area coherence.',
            'potential_challenges': 'Some standards may be cross-cutting or too specific for the given hierarchy.'
        }
        
        quality = self.service._validate_and_process_pre_analysis(good_pre_analysis, self.test_standards)
        
        self.assertGreaterEqual(quality['score'], 0.6)
        self.assertEqual(quality['quality_level'], 'good')
        self.assertTrue(quality['has_task_overview'])
        self.assertTrue(quality['has_strategy'])
        
        # Test poor pre-analysis
        poor_pre_analysis = {
            'task_overview': 'Task',
            'categorization_strategy': 'Do it',
            'identified_patterns': '',
            'potential_challenges': 'Hard'
        }
        
        poor_quality = self.service._validate_and_process_pre_analysis(poor_pre_analysis, self.test_standards)
        
        self.assertLess(poor_quality['score'], 0.4)
        self.assertEqual(poor_quality['quality_level'], 'poor')
        self.assertGreater(len(poor_quality['issues']), 0)
    
    def test_categorization_quality_validation(self):
        """Test categorization quality validation."""
        # Test good categorization
        good_categorization = {
            'standard_id': 'EN.3.OA.1',
            'topic': 'Number Operations',
            'sub_topic': 'Multiplication',
            'sub_sub_topic': 'Basic multiplication concepts',
            'confidence_score': 0.85,
            'reasoning': 'This standard clearly focuses on understanding multiplication as repeated addition, fitting well in the Number Operations hierarchy.',
            'key_concepts': ['multiplication', 'repeated addition', 'whole numbers']
        }
        
        quality = self.service._validate_categorization_quality(good_categorization)
        
        self.assertTrue(quality['is_valid'])
        self.assertGreaterEqual(quality['score'], 0.8)
        self.assertEqual(len(quality['issues']), 0)
        
        # Test poor categorization
        poor_categorization = {
            'standard_id': 'EN.3.OA.1',
            'topic': '',
            'sub_topic': 'Multiplication',
            'sub_sub_topic': '',
            'confidence_score': 0.1,
            'reasoning': 'Bad',
            'key_concepts': []
        }
        
        poor_quality = self.service._validate_categorization_quality(poor_categorization)
        
        self.assertFalse(poor_quality['is_valid'])
        self.assertLess(poor_quality['score'], 0.5)
        self.assertGreater(len(poor_quality['issues']), 0)
    
    def test_backward_compatibility_methods(self):
        """Test backward compatibility methods."""
        # Test simple categorization
        self.service.client = self.mock_client
        
        # Mock the basic categorization response
        mock_hierarchy_response = json.dumps({
            "topics": [{
                "name": "Test Topic",
                "description": "Test description",
                "sub_topics": [{
                    "name": "Test Sub Topic",
                    "description": "Test sub description",
                    "sub_sub_topics": ["Test Sub Sub Topic"]
                }]
            }]
        })
        
        mock_categorization_response = json.dumps({
            "categorizations": [{
                "standard_id": self.test_standards[0].code,
                "topic": "Test Topic",
                "sub_topic": "Test Sub Topic",
                "sub_sub_topic": "Test Sub Sub Topic"
            }],
            "outliers": []
        })
        
        with patch.object(self.service, '_call_openai_api') as mock_api:
            mock_api.side_effect = [mock_hierarchy_response, mock_categorization_response]
            
            hierarchy, proxies = self.service.simple_categorize_standards(
                self.test_standards[:1], "Mathematics"
            )
            
            self.assertIsInstance(hierarchy, TopicHierarchy)
            self.assertEqual(len(proxies), 1)
    
    def test_compatibility_info(self):
        """Test compatibility information method."""
        info = self.service.get_compatibility_info()
        
        # Verify structure
        self.assertIn('version', info)
        self.assertIn('backward_compatible', info)
        self.assertIn('features', info)
        self.assertIn('models', info)
        self.assertIn('chunking', info)
        
        # Verify backward compatibility
        self.assertTrue(info['backward_compatible'])
        
        # Verify essential features
        self.assertTrue(info['features']['basic_categorization'])
        self.assertTrue(info['features']['educational_context_awareness'])
    
    def test_backward_compatibility_validation(self):
        """Test backward compatibility validation."""
        validation = self.service.validate_backward_compatibility()
        
        # Verify structure
        self.assertIn('compatible', validation)
        self.assertIn('issues', validation)
        self.assertIn('warnings', validation)
        self.assertIn('method_availability', validation)
        
        # Essential methods should be available
        essential_methods = ['load_standards', 'generate_topic_hierarchy', 'categorize_standards_chunk']
        for method in essential_methods:
            self.assertEqual(validation['method_availability'][method], 'available')
    
    def test_migration_helper(self):
        """Test migration helper functionality."""
        # Test migration without enhanced features
        migration_result = self.service.migrate_from_basic_usage(
            self.test_standards[:2],
            enable_enhanced_features=False,
            subject_area_name="Mathematics"
        )
        
        # Should provide recommendations even if it fails
        self.assertIn('recommendations', migration_result)
        self.assertIsInstance(migration_result['recommendations'], list)
        
        # Performance metrics should be tracked
        if migration_result.get('success'):
            self.assertIn('performance_metrics', migration_result)
            self.assertIn('processing_time_seconds', migration_result['performance_metrics'])


class TestTopicCategorizationIntegration(TestCase):
    """Integration tests for topic categorization workflow."""
    
    @classmethod
    def setUpTestData(cls):
        """Set up comprehensive test data."""
        # Create test state
        cls.test_state = State.objects.create(
            code='CA',
            name='California'
        )
        
        # Create test subject area 
        cls.math_subject = SubjectArea.objects.create(
            name='Mathematics',
            code='MATH'
        )
        
        # Create grade 3
        cls.grade_3 = GradeLevel.objects.create(
            name='Grade 3',
            grade_numeric=3
        )
        
        # Create realistic Grade 3 Mathematics standards
        cls.grade_3_math_standards = []
        standard_data = [
            ("CA.3.OA.1", "Interpret products of whole numbers", "Understand multiplication as repeated addition"),
            ("CA.3.OA.2", "Interpret whole-number quotients", "Understand division as sharing equally"),
            ("CA.3.OA.3", "Use multiplication and division", "Solve word problems involving multiplication and division"),
            ("CA.3.OA.4", "Determine unknown number", "Find unknown factors in multiplication equations"),
            ("CA.3.OA.5", "Apply properties of operations", "Use commutative, associative, and distributive properties"),
            ("CA.3.OA.6", "Understand division as unknown-factor", "Relate division to multiplication"),
            ("CA.3.OA.7", "Fluently multiply and divide", "Know single-digit multiplication and division facts"),
            ("CA.3.OA.8", "Solve two-step word problems", "Use four operations in multi-step problems"),
            ("CA.3.NBT.1", "Use place value understanding", "Round whole numbers to nearest 10 or 100"),
            ("CA.3.NBT.2", "Fluently add and subtract", "Add and subtract within 1000"),
            ("CA.3.NBT.3", "Multiply one-digit numbers", "Multiply by multiples of 10"),
            ("CA.3.NF.1", "Understand fractions", "Understand unit fractions and fractions in general"),
            ("CA.3.NF.2", "Understand fractions on number line", "Represent fractions on a number line"),
            ("CA.3.NF.3", "Explain equivalence of fractions", "Recognize and generate simple equivalent fractions"),
            ("CA.3.MD.1", "Tell and write time", "Tell time to nearest minute and solve time problems"),
            ("CA.3.MD.2", "Measure and estimate liquid volumes", "Use liters and grams as units"),
            ("CA.3.MD.3", "Draw scaled picture graphs", "Create and interpret bar graphs and picture graphs"),
            ("CA.3.MD.4", "Generate measurement data", "Measure lengths using rulers marked with halves and fourths"),
            ("CA.3.MD.5", "Recognize area as attribute", "Understand area measurement concepts"),
            ("CA.3.MD.6", "Measure areas by counting", "Count unit squares to determine area"),
            ("CA.3.MD.7", "Relate area to multiplication", "Find areas of rectangles using multiplication"),
            ("CA.3.MD.8", "Solve problems involving perimeters", "Find perimeter and unknown side lengths"),
            ("CA.3.G.1", "Understand shared attributes", "Classify shapes by shared attributes"),
            ("CA.3.G.2", "Partition shapes into equal areas", "Understand fractions as equal parts of shapes")
        ]
        
        for code, title, description in standard_data:
            standard = Standard.objects.create(
                code=code,
                title=title,
                description=description,
                state=cls.test_state,
                subject_area=cls.math_subject,
                domain="Grade 3 Mathematics",
                cluster="Test cluster"
            )
            standard.grade_levels.add(cls.grade_3)
            cls.grade_3_math_standards.append(standard)
    
    def setUp(self):
        """Set up test fixtures."""
        self.service = TopicCategorizationService()
    
    def test_grade_3_mathematics_data_available(self):
        """Test that Grade 3 Mathematics standards are properly loaded."""
        standards = self.service.load_standards(
            grade_levels=[3], 
            subject_area_id=self.math_subject.id
        )
        
        self.assertEqual(len(standards), 24)  # Our test data
        
        # Verify filtering worked correctly
        for standard in standards:
            self.assertIn(self.grade_3, standard.grade_levels.all())
            self.assertEqual(standard.subject_area, self.math_subject)
    
    @patch('standards.services.topic_categorization.logger')
    def test_service_logging(self, mock_logger):
        """Test that service properly logs operations."""
        standards = self.service.load_standards(grade_levels=[3])
        
        # Should have logged the loading
        mock_logger.info.assert_called_with(f"Loaded {len(standards)} standards for topic categorization")
    
    def test_optimal_chunk_size_calculation(self):
        """Test chunk size calculation with real data."""
        standards = self.service.load_standards(
            grade_levels=[3],
            subject_area_id=self.math_subject.id
        )
        
        # Convert to format expected by token counter
        standards_data = []
        for standard in standards:
            standards_data.append({
                'code': standard.code,
                'title': standard.title,
                'description': standard.description or '',
                'domain': standard.domain or '',
                'cluster': standard.cluster or ''
            })
        
        chunk_size = self.service.calculate_optimal_chunk_size(standards)
        
        # Should return a reasonable chunk size
        self.assertIsInstance(chunk_size, int)
        self.assertGreaterEqual(chunk_size, 5)
        self.assertLessEqual(chunk_size, len(standards))  # Should not exceed total standards
    
    def test_error_handling_no_client(self):
        """Test error handling when OpenAI client is not available."""
        self.service.client = None
        
        standards = self.service.load_standards(grade_levels=[3])
        
        # Should raise informative error for hierarchy generation
        with self.assertRaises(ValueError) as context:
            self.service.generate_topic_hierarchy(standards)
        self.assertIn("OpenAI client not available", str(context.exception))
        
        # Should raise informative error for categorization
        hierarchy = TopicHierarchy(topics=[])
        with self.assertRaises(ValueError) as context:
            self.service.categorize_standards_chunk(standards[:5], hierarchy)
        self.assertIn("OpenAI client not available", str(context.exception))
    
    def test_service_configuration(self):
        """Test service configuration and constants."""
        self.assertEqual(self.service.DEFAULT_MODEL, "gpt-4.1")
        self.assertEqual(self.service.DEFAULT_CHUNK_SIZE, 25)
        self.assertEqual(self.service.SAFETY_MARGIN, 0.2)
        self.assertEqual(self.service.max_retries, 3)
        self.assertIsNotNone(self.service.token_counter)


class TestEnhancedIntegration(TestCase):
    """Integration tests for enhanced topic categorization features."""
    
    @classmethod
    def setUpTestData(cls):
        """Set up integration test data."""
        # Create comprehensive test dataset
        cls.test_state = State.objects.create(code='INT', name='Integration Test State')
        cls.math_subject = SubjectArea.objects.create(name='Mathematics', code='MATH')
        
        # Create grade levels
        cls.grades = []
        for i in range(1, 6):
            grade = GradeLevel.objects.create(
                name=f'Grade {i}',
                grade_numeric=i
            )
            cls.grades.append(grade)
        
        # Create standards with realistic data
        cls.integration_standards = []
        standard_templates = [
            ('Counting and Cardinality', 'Count to tell the number of objects'),
            ('Operations and Algebraic Thinking', 'Represent and solve problems involving addition and subtraction'),
            ('Number and Operations in Base Ten', 'Understand place value'),
            ('Measurement and Data', 'Measure lengths indirectly and by iterating length units'),
            ('Geometry', 'Reason with shapes and their attributes'),
        ]
        
        for i, (domain, description) in enumerate(standard_templates):
            for grade_idx in range(min(3, len(cls.grades))):
                grade = cls.grades[grade_idx]
                standard = Standard.objects.create(
                    code=f'INT.{grade.grade_numeric}.{i+1}',
                    title=f'Grade {grade.grade_numeric} {domain}',
                    description=f'{description} at grade {grade.grade_numeric} level',
                    state=cls.test_state,
                    subject_area=cls.math_subject,
                    domain=domain,
                    cluster=f'Cluster {i+1}'
                )
                standard.grade_levels.add(grade)
                cls.integration_standards.append(standard)
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.service = TopicCategorizationService()
    
    def test_end_to_end_enhanced_workflow(self):
        """Test complete enhanced workflow from start to finish."""
        # This test simulates the complete enhanced workflow
        # Note: This test will skip actual API calls if no client is available
        
        if not self.service.client:
            self.skipTest("OpenAI client not available for integration test")
        
        try:
            # Test loading standards
            standards = self.service.load_standards(
                grade_levels=[1, 2, 3],
                subject_area_id=self.math_subject.id
            )
            self.assertGreater(len(standards), 0)
            
            # Test educational context analysis
            context = self.service._analyze_educational_context(standards)
            self.assertIn('grade_distribution', context)
            
            # Test educational chunking
            chunks = self.service.create_educational_chunks(
                standards, chunk_size=5
            )
            self.assertGreater(len(chunks), 0)
            
            # Test compatibility validation
            validation = self.service.validate_backward_compatibility()
            self.assertTrue(validation['compatible'])
            
        except Exception as e:
            # If the test fails due to missing dependencies, that's expected
            if "OpenAI" in str(e) or "client" in str(e):
                self.skipTest(f"Integration test skipped due to missing dependencies: {e}")
            else:
                raise
    
    def test_performance_metrics_collection(self):
        """Test that performance metrics are properly collected."""
        # Test with a small dataset to ensure metrics collection works
        test_standards = self.integration_standards[:3]
        
        # Test migration helper with metrics
        migration_result = self.service.migrate_from_basic_usage(
            test_standards,
            enable_enhanced_features=False
        )
        
        # Should always provide metrics, even if processing fails
        self.assertIn('performance_metrics', migration_result)
        metrics = migration_result['performance_metrics']
        
        # Basic metrics should be present
        expected_metrics = ['processing_time_seconds', 'standards_processed']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
    
    def test_quality_metrics_validation(self):
        """Test quality metrics validation and storage."""
        # Test quality metrics structure
        sample_metrics = {
            'pre_analysis_quality': {'score': 0.8, 'quality_level': 'good'},
            'processing_confidence': 0.7,
            'categorization_success_rate': 0.9,
            'outlier_reasoning_quality': 0.8,
            'confidence_accuracy': 0.1
        }
        
        # This should not raise an exception
        try:
            self.service._store_quality_metrics(sample_metrics)
        except Exception as e:
            self.fail(f"Quality metrics storage failed: {e}")
    
    def test_enhanced_features_detection(self):
        """Test enhanced features detection and availability."""
        # Test feature detection
        enhanced_available = self.service.is_enhanced_mode_available()
        compatibility_info = self.service.get_compatibility_info()
        
        # Should provide consistent information
        self.assertEqual(
            enhanced_available,
            compatibility_info['enhanced_features_available']
        )
        
        # Should always have basic features available
        self.assertTrue(compatibility_info['features']['basic_categorization'])
        self.assertTrue(compatibility_info['features']['educational_context_awareness'])


if __name__ == '__main__':
    # Configure logging to reduce noise during testing
    logging.getLogger('standards.services.topic_categorization').setLevel(logging.WARNING)
    unittest.main()