"""
API views for the EdTech Standards Alignment System
"""
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter
from django.db.models import Q, Count, Avg
from django.shortcuts import get_object_or_404
from drf_spectacular.utils import extend_schema, extend_schema_view, OpenApiParameter
from drf_spectacular.types import OpenApiTypes

from .models import (
    State, SubjectArea, GradeLevel, Standard, StandardCorrelation,
    Concept, TopicCluster, CoverageAnalysis, ContentAlignment, StrategicPlan
)
from .serializers import (
    StateSerializer, SubjectAreaSerializer, GradeLevelSerializer,
    StandardSerializer, StandardDetailSerializer, StandardCorrelationSerializer,
    ConceptSerializer, TopicClusterSerializer, CoverageAnalysisSerializer,
    ContentAlignmentSerializer, ContentAlignmentCreateSerializer,
    StrategicPlanSerializer, SimilaritySearchResultSerializer,
    CoverageCalculatorSerializer, CoverageCalculatorResponseSerializer
)
from .services import (
    EmbeddingService, CoverageAnalysisService, SearchService
)


@extend_schema_view(
    list=extend_schema(
        description="List all US states with their 2-letter codes",
        tags=['Meta']
    ),
    retrieve=extend_schema(
        description="Retrieve a specific state by its 2-letter code",
        tags=['Meta']
    )
)
class StateViewSet(viewsets.ReadOnlyModelViewSet):
    """API viewset for State model - US states with educational standards"""
    queryset = State.objects.all()
    serializer_class = StateSerializer
    lookup_field = 'code'
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['code', 'name']
    ordering_fields = ['name', 'created_at']
    ordering = ['name']
    
    @extend_schema(
        description="Get all educational standards for a specific state",
        parameters=[
            OpenApiParameter(
                name='subject',
                type=OpenApiTypes.INT,
                location=OpenApiParameter.QUERY,
                description='Filter by subject area ID'
            ),
            OpenApiParameter(
                name='grade',
                type=OpenApiTypes.INT,
                location=OpenApiParameter.QUERY,
                description='Filter by grade level ID'
            )
        ],
        tags=['Standards']
    )
    @action(detail=True, methods=['get'])
    def standards(self, request, code=None):
        """Get all standards for a specific state"""
        state = self.get_object()
        standards = Standard.objects.filter(state=state).select_related(
            'subject_area'
        ).prefetch_related('grade_levels')
        
        # Apply filters
        subject_id = request.query_params.get('subject')
        if subject_id:
            standards = standards.filter(subject_area_id=subject_id)
        
        grade_id = request.query_params.get('grade')
        if grade_id:
            standards = standards.filter(grade_levels__id=grade_id)
        
        page = self.paginate_queryset(standards)
        if page is not None:
            serializer = StandardSerializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        
        serializer = StandardSerializer(standards, many=True)
        return Response(serializer.data)
    
    @extend_schema(
        description="Get coverage statistics for a specific state including total standards, subjects covered, and grades covered",
        tags=['Meta']
    )
    @action(detail=True, methods=['get'])
    def coverage_stats(self, request, code=None):
        """Get coverage statistics for a specific state"""
        state = self.get_object()
        
        stats = {
            'total_standards': Standard.objects.filter(state=state).count(),
            'subjects_covered': Standard.objects.filter(state=state).values('subject_area').distinct().count(),
            'grades_covered': Standard.objects.filter(
                state=state
            ).values('grade_levels').distinct().count(),
            'standards_by_subject': list(
                Standard.objects.filter(state=state).values('subject_area__name').annotate(
                    count=Count('id')
                ).order_by('-count')
            ),
            'standards_by_grade': list(
                Standard.objects.filter(state=state).values('grade_levels__grade').annotate(
                    count=Count('id')
                ).order_by('grade_levels__grade_numeric')
            )
        }
        
        return Response(stats)


@extend_schema_view(
    list=extend_schema(
        description="List all subject areas (Math, ELA, Science, Social Studies, etc.)",
        tags=['Meta']
    ),
    retrieve=extend_schema(
        description="Retrieve a specific subject area by ID",
        tags=['Meta']
    )
)
class SubjectAreaViewSet(viewsets.ReadOnlyModelViewSet):
    """API viewset for SubjectArea model - Academic subject areas"""
    queryset = SubjectArea.objects.all()
    serializer_class = SubjectAreaSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['name', 'description']
    ordering_fields = ['name', 'created_at']
    ordering = ['name']


@extend_schema_view(
    list=extend_schema(
        description="List all grade levels from K-12",
        tags=['Meta']
    ),
    retrieve=extend_schema(
        description="Retrieve a specific grade level by ID",
        tags=['Meta']
    )
)
class GradeLevelViewSet(viewsets.ReadOnlyModelViewSet):
    """API viewset for GradeLevel model - K-12 grade levels"""
    queryset = GradeLevel.objects.all()
    serializer_class = GradeLevelSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [OrderingFilter]
    ordering_fields = ['grade_numeric', 'created_at']
    ordering = ['grade_numeric']


@extend_schema_view(
    list=extend_schema(
        description="List all educational standards with filtering and search capabilities",
        tags=['Standards']
    ),
    retrieve=extend_schema(
        description="Retrieve detailed information about a specific educational standard",
        tags=['Standards']
    )
)
class StandardViewSet(viewsets.ReadOnlyModelViewSet):
    """API viewset for Standard model - Educational standards with vector embeddings"""
    queryset = Standard.objects.select_related(
        'state', 'subject_area'
    ).prefetch_related('grade_levels')
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['state__code', 'subject_area', 'grade_levels', 'domain']
    search_fields = ['code', 'title', 'description', 'keywords', 'skills']
    ordering_fields = ['code', 'title', 'created_at']
    ordering = ['state__name', 'code']
    
    def get_serializer_class(self):
        if self.action == 'retrieve':
            return StandardDetailSerializer
        return StandardSerializer
    
    @extend_schema(
        description="Find educational standards similar to the current one using vector similarity search",
        parameters=[
            OpenApiParameter(
                name='threshold',
                type=OpenApiTypes.FLOAT,
                location=OpenApiParameter.QUERY,
                description='Similarity threshold (0.0-1.0, default 0.8)'
            ),
            OpenApiParameter(
                name='limit',
                type=OpenApiTypes.INT,
                location=OpenApiParameter.QUERY,
                description='Maximum number of results to return (default 10)'
            )
        ],
        tags=['Search']
    )
    @action(detail=True, methods=['get'])
    def similar_standards(self, request, pk=None):
        """Find standards similar to the current one"""
        standard = self.get_object()
        
        # Get similarity threshold from query params
        threshold = float(request.query_params.get('threshold', 0.8))
        limit = int(request.query_params.get('limit', 10))
        
        # Use search service to find similar standards
        search_service = SearchService()
        similar_standards = search_service.find_similar_standards(
            standard, threshold=threshold, limit=limit
        )
        
        serializer = SimilaritySearchResultSerializer(similar_standards, many=True)
        return Response(serializer.data)
    
    @extend_schema(
        description="Perform semantic search on educational standards using natural language queries",
        request={
            'application/json': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string', 'description': 'Natural language search query'},
                    'state': {'type': 'string', 'description': 'Filter by state code'},
                    'subject': {'type': 'integer', 'description': 'Filter by subject area ID'},
                    'grade': {'type': 'integer', 'description': 'Filter by grade level ID'},
                    'threshold': {'type': 'number', 'description': 'Similarity threshold (0.0-1.0)'},
                    'limit': {'type': 'integer', 'description': 'Maximum results to return'}
                },
                'required': ['query']
            }
        },
        tags=['Search']
    )
    @action(detail=False, methods=['post'])
    def semantic_search(self, request):
        """Perform semantic search on standards"""
        query = request.data.get('query', '')
        if not query:
            return Response(
                {'error': 'Query parameter is required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Optional filters
        filters = {
            'state_code': request.data.get('state'),
            'subject_id': request.data.get('subject'),
            'grade_id': request.data.get('grade'),
            'threshold': float(request.data.get('threshold', 0.7)),
            'limit': int(request.data.get('limit', 20))
        }
        
        search_service = SearchService()
        results = search_service.semantic_search(query, **filters)
        
        serializer = SimilaritySearchResultSerializer(results, many=True)
        return Response(serializer.data)


@extend_schema_view(
    list=extend_schema(
        description="List cross-state standard correlations based on vector similarity",
        tags=['Standards']
    ),
    retrieve=extend_schema(
        description="Retrieve details of a specific standard correlation",
        tags=['Standards']
    )
)
class StandardCorrelationViewSet(viewsets.ReadOnlyModelViewSet):
    """API viewset for StandardCorrelation model - Cross-state standard alignments"""
    queryset = StandardCorrelation.objects.select_related(
        'standard_1__state', 'standard_1__subject_area',
        'standard_2__state', 'standard_2__subject_area'
    )
    serializer_class = StandardCorrelationSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend, OrderingFilter]
    filterset_fields = [
        'correlation_type', 'verified', 'standard_1__state__code', 
        'standard_2__state__code', 'standard_1__subject_area'
    ]
    ordering_fields = ['similarity_score', 'created_at']
    ordering = ['-similarity_score']
    
    @action(detail=False, methods=['get'])
    def cross_state_analysis(self, request):
        """Analyze correlations across different states"""
        state1 = request.query_params.get('state1')
        state2 = request.query_params.get('state2')
        
        if not (state1 and state2):
            return Response(
                {'error': 'Both state1 and state2 parameters are required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        correlations = self.queryset.filter(
            standard_1__state__code=state1,
            standard_2__state__code=state2
        )
        
        analysis = {
            'total_correlations': correlations.count(),
            'average_similarity': correlations.aggregate(
                avg_similarity=Avg('similarity_score')
            )['avg_similarity'] or 0,
            'correlation_types': dict(
                correlations.values('correlation_type').annotate(
                    count=Count('id')
                ).values_list('correlation_type', 'count')
            ),
            'subject_breakdown': list(
                correlations.values('standard_1__subject_area__name').annotate(
                    count=Count('id'),
                    avg_similarity=Avg('similarity_score')
                ).order_by('-count')
            )
        }
        
        return Response(analysis)


class ConceptViewSet(viewsets.ReadOnlyModelViewSet):
    """API viewset for Concept model"""
    queryset = Concept.objects.prefetch_related('subject_areas', 'grade_levels')
    serializer_class = ConceptSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['subject_areas', 'grade_levels']
    search_fields = ['name', 'description', 'keywords']
    ordering_fields = ['coverage_percentage', 'states_covered', 'name', 'created_at']
    ordering = ['-coverage_percentage']
    
    @action(detail=False, methods=['get'])
    def coverage_distribution(self, request):
        """Get distribution of concept coverage across states"""
        concepts = self.get_queryset()
        
        # Apply filters
        subject_id = request.query_params.get('subject')
        if subject_id:
            concepts = concepts.filter(subject_areas__id=subject_id)
        
        grade_id = request.query_params.get('grade')
        if grade_id:
            concepts = concepts.filter(grade_levels__id=grade_id)
        
        # Generate distribution data
        coverage_ranges = [
            (0, 20, 'Very Low (0-20%)'),
            (21, 40, 'Low (21-40%)'),
            (41, 60, 'Medium (41-60%)'),
            (61, 80, 'High (61-80%)'),
            (81, 100, 'Very High (81-100%)')
        ]
        
        distribution = []
        for min_val, max_val, label in coverage_ranges:
            count = concepts.filter(
                coverage_percentage__gte=min_val,
                coverage_percentage__lte=max_val
            ).count()
            
            distribution.append({
                'range': label,
                'count': count,
                'percentage': (count / concepts.count() * 100) if concepts.count() > 0 else 0
            })
        
        return Response({
            'total_concepts': concepts.count(),
            'distribution': distribution,
            'average_coverage': concepts.aggregate(
                avg=Avg('coverage_percentage')
            )['avg'] or 0
        })


class TopicClusterViewSet(viewsets.ReadOnlyModelViewSet):
    """API viewset for TopicCluster model"""
    queryset = TopicCluster.objects.select_related('subject_area').prefetch_related('grade_levels')
    serializer_class = TopicClusterSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['subject_area', 'grade_levels']
    search_fields = ['name', 'description', 'common_terms']
    ordering_fields = ['silhouette_score', 'standards_count', 'name', 'created_at']
    ordering = ['-silhouette_score']


class CoverageAnalysisViewSet(viewsets.ReadOnlyModelViewSet):
    """API viewset for CoverageAnalysis model"""
    queryset = CoverageAnalysis.objects.select_related('state', 'subject_area', 'grade_level')
    serializer_class = CoverageAnalysisSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend, OrderingFilter]
    filterset_fields = ['analysis_type', 'state', 'subject_area', 'grade_level']
    ordering_fields = ['coverage_percentage', 'created_at']
    ordering = ['-created_at']
    
    @action(detail=False, methods=['post'])
    def generate_analysis(self, request):
        """Generate a new coverage analysis"""
        state_code = request.data.get('state')
        subject_id = request.data.get('subject')
        grade_id = request.data.get('grade')
        
        # Get objects if IDs provided
        state = None
        if state_code:
            state = get_object_or_404(State, code=state_code)
        
        subject_area = None
        if subject_id:
            subject_area = get_object_or_404(SubjectArea, id=subject_id)
        
        grade_level = None
        if grade_id:
            grade_level = get_object_or_404(GradeLevel, id=grade_id)
        
        # Generate analysis
        coverage_service = CoverageAnalysisService()
        analysis = coverage_service.generate_coverage_report(
            state=state,
            subject_area=subject_area,
            grade_level=grade_level
        )
        
        serializer = CoverageAnalysisSerializer(analysis)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    
    @action(detail=False, methods=['post'])
    def calculate_coverage(self, request):
        """Calculate coverage for a set of concepts"""
        serializer = CoverageCalculatorSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        # This would integrate with coverage calculation logic
        # For now, return a mock response
        response_data = {
            'total_states_covered': 42,
            'coverage_percentage': 84.0,
            'missing_states': ['AK', 'HI', 'MT', 'ND', 'VT', 'WY', 'DE', 'RI'],
            'coverage_by_concept': {
                concept: {'states_covered': 35 + (hash(concept) % 15), 'percentage': 70 + (hash(concept) % 30)}
                for concept in serializer.validated_data['concepts']
            },
            'recommendations': [
                'Consider adding content for fraction operations to reach more states',
                'Focus on elementary math concepts for broader coverage'
            ]
        }
        
        response_serializer = CoverageCalculatorResponseSerializer(response_data)
        return Response(response_serializer.data)


class ContentAlignmentViewSet(viewsets.ModelViewSet):
    """API viewset for ContentAlignment model"""
    queryset = ContentAlignment.objects.prefetch_related('matched_standards')
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['file_type', 'overall_alignment_score']
    search_fields = ['content_title', 'original_filename']
    ordering_fields = ['overall_alignment_score', 'created_at']
    ordering = ['-created_at']
    
    def get_serializer_class(self):
        if self.action == 'create':
            return ContentAlignmentCreateSerializer
        return ContentAlignmentSerializer
    
    @action(detail=True, methods=['get'])
    def state_alignment_breakdown(self, request, pk=None):
        """Get detailed alignment breakdown by state"""
        content_alignment = self.get_object()
        
        # This would calculate state-by-state alignment
        # For now, return mock data
        alignment_breakdown = {
            'full_alignment_states': list(content_alignment.states_with_full_alignment.values('code', 'name')),
            'partial_alignment_states': list(content_alignment.states_with_partial_alignment.values('code', 'name')),
            'no_alignment_states': [],  # Would be calculated from remaining states
            'alignment_scores_by_state': {}  # Would contain detailed scores
        }
        
        return Response(alignment_breakdown)


class StrategicPlanViewSet(viewsets.ModelViewSet):
    """API viewset for StrategicPlan model"""
    queryset = StrategicPlan.objects.prefetch_related(
        'target_states', 'target_subjects', 'target_grades', 'high_priority_concepts'
    )
    serializer_class = StrategicPlanSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['status', 'implementation_difficulty', 'target_coverage_percentage']
    search_fields = ['name', 'description', 'created_by']
    ordering_fields = ['target_coverage_percentage', 'created_at']
    ordering = ['-created_at']