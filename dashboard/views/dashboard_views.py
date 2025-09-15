"""
Core dashboard views for the main interface pages
"""
from django.shortcuts import render
from django.contrib.admin.views.decorators import staff_member_required
from django.core.cache import cache
from django.db.models import Count
from typing import Dict, Any

from standards.models import (
    Standard, State, SubjectArea, GradeLevel, 
    StandardCorrelation, TopicCluster, ProxyRun
)
from ..forms import ConceptForm, TopicAnalysisForm, TopicSearchForm
from .utils import calculate_state_groupings, calculate_topic_prevalence


@staff_member_required
def dashboard_home(request):
    """Main dashboard landing page with overview statistics"""
    # Use cached stats for performance
    cache_key = 'dashboard_home_stats'
    context = cache.get(cache_key)
    
    if not context:
        context = {
            'total_standards': Standard.objects.count(),
            'total_states': State.objects.count(),
            'total_correlations': StandardCorrelation.objects.count(),
            'total_subject_areas': SubjectArea.objects.count(),
            'subject_areas': SubjectArea.objects.all()[:10],  # Limit for performance
        }
        # Cache for 5 minutes
        cache.set(cache_key, context, 300)
    
    return render(request, 'dashboard/home.html', context)


@staff_member_required
def coverage_dashboard(request):
    """Coverage Map Dashboard showing state coverage analysis"""
    context = {
        'title': 'Coverage Map Dashboard',
        'subtitle': 'Analyze curriculum coverage across all 50 states',
        'total_states': State.objects.count(),
        'subject_areas': SubjectArea.objects.all(),
        'grade_levels': GradeLevel.objects.all().order_by('grade_numeric'),
    }
    return render(request, 'dashboard/coverage.html', context)


@staff_member_required
def groupings_dashboard(request):
    """State Groupings Discovery Dashboard"""
    
    # Calculate state similarity clusters with caching
    cache_key = 'state_groupings_data_v2'
    groupings_data = cache.get(cache_key)
    
    if not groupings_data:
        try:
            groupings_data = calculate_state_groupings()
            # Cache for 1 hour
            cache.set(cache_key, groupings_data, 3600)
        except Exception as e:
            # Fallback to empty data if calculation fails
            groupings_data = {'groupings': [], 'unique_states': []}
    
    context = {
        'title': 'State Groupings Discovery',
        'subtitle': 'Identify similar state clusters for efficient curriculum deployment',
        'groupings': groupings_data.get('groupings', []),
        'unique_states': groupings_data.get('unique_states', []),
        'total_groups': len(groupings_data.get('groupings', [])),
    }
    return render(request, 'dashboard/groupings.html', context)


@staff_member_required
def topics_dashboard(request):
    """Topic Intelligence Dashboard"""
    
    # Get topic prevalence data with caching
    cache_key = 'topic_intelligence_data_v2'
    topic_data = cache.get(cache_key)
    
    if not topic_data:
        try:
            topic_data = calculate_topic_prevalence()
            # Cache for 1 hour
            cache.set(cache_key, topic_data, 3600)
        except Exception as e:
            # Fallback to empty data if calculation fails
            topic_data = {'topics': []}
    
    # Initialize forms
    concept_form = ConceptForm()
    analysis_form = TopicAnalysisForm()
    search_form = TopicSearchForm(request.GET or None)
    
    context = {
        'title': 'Topic Intelligence',
        'subtitle': 'Analyze topic prevalence and importance across all states',
        'topics': topic_data.get('topics', []),
        'subject_areas': SubjectArea.objects.all(),
        'total_topics': len(topic_data.get('topics', [])),
        'concept_form': concept_form,
        'analysis_form': analysis_form,
        'search_form': search_form,
    }
    
    return render(request, 'dashboard/topics.html', context)


@staff_member_required
def proxies_dashboard(request):
    """Proxy Standards generation UI"""
    # Only include subject areas that have standards for better UX
    subject_areas_with_standards = SubjectArea.objects.filter(
        standards__isnull=False
    ).annotate(
        standards_count=Count('standards')
    ).distinct().order_by('name')
    
    context = {
        'title': 'Proxy Standards',
        'subtitle': 'Cluster atoms into cross-state proxies (UMAP + HDBSCAN)',
        'subject_areas': subject_areas_with_standards,
    }
    return render(request, 'dashboard/proxies.html', context)


@staff_member_required
def embeddings_dashboard(request):
    """Standards Embeddings Visualization Dashboard"""
    context = {
        'title': 'Standards Embeddings Visualization',
        'subtitle': 'Explore state philosophical alignments and semantic relationships',
        'subject_areas': SubjectArea.objects.all(),
        'grade_levels': GradeLevel.objects.all().order_by('grade_numeric'),
    }
    return render(request, 'dashboard/embeddings.html', context)


@staff_member_required
def custom_clusters_dashboard(request):
    """Dashboard for managing user-created custom clusters and comparison reports"""
    context = {
        'title': 'Custom Clusters',
        'subtitle': 'Curate semantic groups of standards and compare coverage across states',
        'subject_areas': SubjectArea.objects.all(),
        'grade_levels': GradeLevel.objects.all().order_by('grade_numeric'),
    }
    return render(request, 'dashboard/custom_clusters.html', context)


@staff_member_required
def proxy_runs_dashboard(request):
    """Proxy Runs Analysis Dashboard for viewing and comparing proxy generation results"""
    from standards.services.proxy_run_analyzer import ProxyRunAnalyzer
    
    # Get filter parameters with validation
    run_type_filter = request.GET.get('run_type', '')
    status_filter = request.GET.get('status', 'completed')
    
    # Validate filters
    valid_run_types = [choice[0] for choice in ProxyRun.RUN_TYPES]
    valid_statuses = [choice[0] for choice in ProxyRun.STATUS_CHOICES]
    
    if run_type_filter and run_type_filter not in valid_run_types:
        run_type_filter = ''
    if status_filter not in valid_statuses:
        status_filter = 'completed'
    
    # Build queryset with proper prefetching
    runs_query = ProxyRun.objects.select_related('report').order_by('-started_at')
    
    if run_type_filter:
        runs_query = runs_query.filter(run_type=run_type_filter)
    if status_filter:
        runs_query = runs_query.filter(status=status_filter)
    
    # Limit to recent runs for performance
    runs = runs_query[:20]
    
    # Get selected run for detailed view
    selected_run_id = request.GET.get('run_id')
    selected_run = None
    report_data = None
    chart_data = None
    
    if selected_run_id:
        try:
            selected_run = ProxyRun.objects.select_related('report').get(run_id=selected_run_id)
            
            # Initialize analyzer
            analyzer = ProxyRunAnalyzer()
            
            # Generate or get report
            if hasattr(selected_run, 'report') and selected_run.report:
                report = selected_run.report
            else:
                report = analyzer.analyze_run(selected_run)
            
            # Prepare data for visualization
            if selected_run.run_type == 'topics':
                chart_data = analyzer.get_topic_prevalence_chart_data(report)
            
            # Safely extract report data
            report_data = {
                'state_coverage': getattr(report, 'state_coverage', {}),
                'topic_prevalence': getattr(report, 'topic_prevalence', {}),
                'coverage_distribution': getattr(report, 'coverage_distribution', {}),
                'outlier_analysis': getattr(report, 'outlier_analysis', {}),
                'hierarchy_stats': getattr(report, 'topic_hierarchy_stats', {}),
                'cross_state_commonality': getattr(report, 'cross_state_commonality', {}),
                'must_have_topics': getattr(report, 'must_have_topics', []),
                'important_topics': getattr(report, 'important_topics', []),
                'regional_topics': getattr(report, 'regional_topics', []),
            }
            
        except ProxyRun.DoesNotExist:
            selected_run = None
        except Exception as e:
            # Log error but don't crash the page
            import logging
            logging.getLogger(__name__).error(f"Error loading proxy run {selected_run_id}: {e}")
            selected_run = None
    
    context = {
        'title': 'Proxy Run Analysis',
        'subtitle': 'Analyze and compare proxy generation results',
        'runs': runs,
        'selected_run': selected_run,
        'report_data': report_data,
        'chart_data': chart_data,
        'run_type_filter': run_type_filter,
        'status_filter': status_filter,
        'run_type_choices': ProxyRun.RUN_TYPES,
        'status_choices': ProxyRun.STATUS_CHOICES,
    }
    return render(request, 'dashboard/proxy_runs.html', context)
