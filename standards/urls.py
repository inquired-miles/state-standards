"""
URL configuration for the standards app API
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter

app_name = 'standards'
from .api_views import (
    StateViewSet, SubjectAreaViewSet, GradeLevelViewSet,
    StandardViewSet, StandardCorrelationViewSet, ConceptViewSet,
    TopicClusterViewSet, CoverageAnalysisViewSet, ContentAlignmentViewSet,
    StrategicPlanViewSet
)
from .api_views_rag import (
    bell_curve_analysis, minimum_viable_coverage, coverage_distribution,
    discover_storylines, find_common_threads, analyze_regional_patterns,
    create_learning_pathways, analyze_content_coverage
)
from . import admin_views

# Create router and register viewsets
router = DefaultRouter()
router.register(r'states', StateViewSet)
router.register(r'subjects', SubjectAreaViewSet)
router.register(r'grades', GradeLevelViewSet)
router.register(r'standards', StandardViewSet)
router.register(r'correlations', StandardCorrelationViewSet)
router.register(r'concepts', ConceptViewSet)
router.register(r'clusters', TopicClusterViewSet)
router.register(r'coverage', CoverageAnalysisViewSet)
router.register(r'content', ContentAlignmentViewSet)
router.register(r'plans', StrategicPlanViewSet)

urlpatterns = [
    # Standard REST API endpoints
    path('api/', include(router.urls)),
    path('api-auth/', include('rest_framework.urls')),
    
    # RAG-specific endpoints
    path('api/rag/bell-curve/', bell_curve_analysis, name='bell_curve_analysis'),
    path('api/rag/minimum-viable-coverage/', minimum_viable_coverage, name='minimum_viable_coverage'),
    path('api/rag/coverage-distribution/', coverage_distribution, name='coverage_distribution'),
    path('api/rag/discover-storylines/', discover_storylines, name='discover_storylines'),
    path('api/rag/common-threads/', find_common_threads, name='find_common_threads'),
    path('api/rag/regional-patterns/', analyze_regional_patterns, name='analyze_regional_patterns'),
    path('api/rag/learning-pathways/', create_learning_pathways, name='create_learning_pathways'),
    path('api/rag/analyze-content/', analyze_content_coverage, name='analyze_content_coverage'),
    
]