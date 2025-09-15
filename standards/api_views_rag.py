"""
RAG-specific API views for bell curve analysis and storyline discovery
"""
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes

from .models import SubjectArea, GradeLevel, State
from .services import BellCurveAnalysisService, StorylineDiscoveryService
from .services.coverage_v2 import CoverageAnalyzer


@extend_schema(
    description="""Calculate bell curve distribution for concept coverage across states.
    
    This endpoint answers: 'If we cover X concepts, how many states will we cover?'
    Helps identify the optimal set of concepts that provide maximum state coverage.""",
    request={
        'application/json': {
            'type': 'object',
            'properties': {
                'concepts': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': 'List of educational concepts to analyze'
                },
                'subject_area': {'type': 'integer', 'description': 'Subject area ID (optional)'},
                'grade_levels': {
                    'type': 'array',
                    'items': {'type': 'integer'},
                    'description': 'Grade level IDs to filter by (optional)'
                },
                'target_states': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': 'Target state codes to analyze (optional)'
                }
            },
            'required': ['concepts']
        }
    },
    tags=['RAG System']
)
@api_view(['POST'])
@permission_classes([IsAuthenticatedOrReadOnly])
def bell_curve_analysis(request):
    """Calculate bell curve distribution for concept coverage across states"""
    concepts = request.data.get('concepts', [])
    
    if not concepts:
        return Response(
            {'error': 'Concepts list is required'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Get optional filters
    subject_area = None
    if 'subject_area' in request.data:
        subject_area = get_object_or_404(SubjectArea, id=request.data['subject_area'])
    
    grade_levels = None
    if 'grade_levels' in request.data:
        grade_levels = GradeLevel.objects.filter(id__in=request.data['grade_levels'])
    
    target_states = request.data.get('target_states')
    
    # Perform bell curve analysis
    service = BellCurveAnalysisService()
    analysis = service.calculate_bell_curve(
        concepts=concepts,
        subject_area=subject_area,
        grade_levels=grade_levels,
        target_states=target_states
    )
    
    return Response(analysis)


@extend_schema(
    description="""Find the minimum set of concepts needed to achieve target coverage percentage.
    
    This endpoint uses optimization algorithms to identify the smallest set of concepts
    that will cover the desired percentage of state standards.""",
    request={
        'application/json': {
            'type': 'object',
            'properties': {
                'target_coverage_percentage': {
                    'type': 'number',
                    'description': 'Target coverage percentage (0-100)',
                    'minimum': 0,
                    'maximum': 100
                },
                'subject_area': {'type': 'integer', 'description': 'Subject area ID (optional)'},
                'grade_levels': {
                    'type': 'array',
                    'items': {'type': 'integer'},
                    'description': 'Grade level IDs to filter by (optional)'
                },
                'max_concepts': {
                    'type': 'integer',
                    'description': 'Maximum number of concepts to return (default: 20)'
                }
            },
            'required': ['target_coverage_percentage']
        }
    },
    tags=['RAG System']
)
@api_view(['POST'])
@permission_classes([IsAuthenticatedOrReadOnly])
def minimum_viable_coverage(request):
    """Find the minimum set of concepts needed to achieve target coverage"""
    target_coverage = request.data.get('target_coverage_percentage', 80.0)
    
    if not 0 < target_coverage <= 100:
        return Response(
            {'error': 'Target coverage must be between 0 and 100'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Get optional filters
    subject_area = None
    if 'subject_area' in request.data:
        subject_area = get_object_or_404(SubjectArea, id=request.data['subject_area'])
    
    grade_levels = None
    if 'grade_levels' in request.data:
        grade_levels = GradeLevel.objects.filter(id__in=request.data['grade_levels'])
    
    max_concepts = request.data.get('max_concepts', 20)
    
    # Find minimum viable coverage
    service = BellCurveAnalysisService()
    mvc_analysis = service.find_minimum_viable_coverage(
        target_coverage_percentage=target_coverage,
        subject_area=subject_area,
        grade_levels=grade_levels,
        max_concepts=max_concepts
    )
    
    return Response(mvc_analysis)


@extend_schema(
    description="Analyze the distribution of concept coverage across all states",
    parameters=[
        OpenApiParameter(
            name='subject_area',
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            description='Subject area ID to filter by'
        ),
        OpenApiParameter(
            name='grade_levels',
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            description='Comma-separated grade level IDs'
        )
    ],
    tags=['RAG System']
)
@api_view(['GET'])
@permission_classes([IsAuthenticatedOrReadOnly])
def coverage_distribution(request):
    """Analyze the distribution of concept coverage across all states"""
    subject_area = None
    if 'subject_area' in request.query_params:
        subject_area = get_object_or_404(SubjectArea, id=request.query_params['subject_area'])
    
    grade_levels = None
    if 'grade_levels' in request.query_params:
        grade_ids = [int(x) for x in request.query_params['grade_levels'].split(',')]
        grade_levels = GradeLevel.objects.filter(id__in=grade_ids)
    
    service = BellCurveAnalysisService()
    distribution = service.analyze_coverage_distribution(
        subject_area=subject_area,
        grade_levels=grade_levels
    )
    
    return Response(distribution)


@extend_schema(
    description="""Discover educational storylines that progress through grade levels.
    
    Identifies common learning progressions and educational threads that span multiple grades.""",
    request={
        'application/json': {
            'type': 'object',
            'properties': {
                'subject_area': {'type': 'integer', 'description': 'Subject area ID'},
                'start_grade': {'type': 'integer', 'description': 'Starting grade (0 for K)', 'minimum': 0, 'maximum': 12},
                'end_grade': {'type': 'integer', 'description': 'Ending grade', 'minimum': 0, 'maximum': 12}
            },
            'required': ['subject_area']
        }
    },
    tags=['RAG System']
)
@api_view(['POST'])
@permission_classes([IsAuthenticatedOrReadOnly])
def discover_storylines(request):
    """Discover educational storylines that progress through grade levels"""
    if 'subject_area' not in request.data:
        return Response(
            {'error': 'Subject area is required'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    subject_area = get_object_or_404(SubjectArea, id=request.data['subject_area'])
    start_grade = request.data.get('start_grade', 0)
    end_grade = request.data.get('end_grade', 12)
    
    service = StorylineDiscoveryService()
    storylines = service.discover_storylines(
        subject_area=subject_area,
        start_grade=start_grade,
        end_grade=end_grade
    )
    
    return Response(storylines)


@extend_schema(
    description="Find educational concepts that appear across many states, identifying nationally common standards",
    parameters=[
        OpenApiParameter(
            name='min_state_coverage',
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            description='Minimum number of states for a concept to be considered common (default: 30)'
        ),
        OpenApiParameter(
            name='subject_area',
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            description='Subject area ID to filter by'
        )
    ],
    tags=['RAG System']
)
@api_view(['GET'])
@permission_classes([IsAuthenticatedOrReadOnly])
def find_common_threads(request):
    """Find educational concepts that appear across many states"""
    min_state_coverage = int(request.query_params.get('min_state_coverage', 30))
    
    subject_area = None
    if 'subject_area' in request.query_params:
        subject_area = get_object_or_404(SubjectArea, id=request.query_params['subject_area'])
    
    service = StorylineDiscoveryService()
    common_threads = service.find_common_threads(
        min_state_coverage=min_state_coverage,
        subject_area=subject_area
    )
    
    return Response(common_threads)


@extend_schema(
    description="Analyze regional patterns in educational standards to identify geographic trends and similarities",
    tags=['RAG System']
)
@api_view(['GET'])
@permission_classes([IsAuthenticatedOrReadOnly])
def analyze_regional_patterns(request):
    """Analyze regional patterns in educational standards"""
    service = StorylineDiscoveryService()
    regional_analysis = service.analyze_regional_patterns()
    
    return Response(regional_analysis)


@extend_schema(
    description="""Create learning pathways that lead to a target concept.
    
    Generates prerequisite chains and learning progressions for complex concepts.""",
    request={
        'application/json': {
            'type': 'object',
            'properties': {
                'target_concept': {'type': 'string', 'description': 'The target concept to build pathways toward'},
                'subject_area': {'type': 'integer', 'description': 'Subject area ID'},
                'max_grade_span': {'type': 'integer', 'description': 'Maximum number of grades in pathway (default: 3)'}
            },
            'required': ['target_concept', 'subject_area']
        }
    },
    tags=['RAG System']
)
@api_view(['POST'])
@permission_classes([IsAuthenticatedOrReadOnly])
def create_learning_pathways(request):
    """Create learning pathways that lead to a target concept"""
    target_concept = request.data.get('target_concept')
    if not target_concept:
        return Response(
            {'error': 'Target concept is required'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    if 'subject_area' not in request.data:
        return Response(
            {'error': 'Subject area is required'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    subject_area = get_object_or_404(SubjectArea, id=request.data['subject_area'])
    max_grade_span = request.data.get('max_grade_span', 3)
    
    service = StorylineDiscoveryService()
    pathways = service.create_learning_pathways(
        target_concept=target_concept,
        subject_area=subject_area,
        max_grade_span=max_grade_span
    )
    
    return Response(pathways)


@extend_schema(
    description="""Analyze curriculum coverage using atom-level similarity and rollups.
    
    Evaluates how well educational content aligns with state standards across different regions.""",
    request={
        'application/json': {
            'type': 'object',
            'properties': {
                'content': {'type': 'string', 'description': 'Text content to analyze'},
                'curriculum_id': {'type': 'integer', 'description': 'ID of existing curriculum document'}
            }
        }
    },
    tags=['RAG System']
)
@api_view(['POST'])
@permission_classes([IsAuthenticatedOrReadOnly])
def analyze_content_coverage(request):
    """Analyze curriculum coverage using atom-level similarity and rollups"""
    content = request.data.get('content')
    curriculum_id = request.data.get('curriculum_id')
    if not content and not curriculum_id:
        return Response({'error': 'Provide either content or curriculum_id'}, status=status.HTTP_400_BAD_REQUEST)

    from .models import CurriculumDocument
    if curriculum_id:
        curriculum = get_object_or_404(CurriculumDocument, id=curriculum_id)
    else:
        curriculum = CurriculumDocument.objects.create(name='Ad-hoc Curriculum', content=content)

    analyzer = CoverageAnalyzer()
    report = analyzer.analyze_curriculum(curriculum)
    return Response(report)