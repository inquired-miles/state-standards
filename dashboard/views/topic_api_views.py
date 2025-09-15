"""
Secure API views for topic categorization operations
"""
import uuid
import json
import time
import threading
from typing import Dict, List, Optional, Any
from django.http import JsonResponse
from django.core.exceptions import ValidationError
from django.db import transaction
from django.core.cache import cache
from django.utils import timezone
import logging

from standards.models import (
    Standard, SubjectArea, ProxyRun, Concept
)
from standards.services.topic_categorization import TopicCategorizationService
from standards.services.discovery import TopicDiscoveryService
from standards.services.bell_curve import BellCurveAnalysisService
from standards.services.proxy_run_analyzer import ProxyRunAnalyzer
from ..forms import ConceptForm, TopicAnalysisForm
from .base import BaseAPIView, api_endpoint
from .utils import extract_concepts_from_content

logger = logging.getLogger(__name__)


class TopicAPIView(BaseAPIView):
    """Base class for topic-related API views"""
    
    def validate_subject_area(self, subject_area_id: Any) -> Optional[SubjectArea]:
        """Validate and return subject area object"""
        if not subject_area_id:
            return None
            
        try:
            subject_id = int(subject_area_id)
            return SubjectArea.objects.get(id=subject_id)
        except (ValueError, TypeError, SubjectArea.DoesNotExist):
            raise ValidationError(f"Invalid subject area ID: {subject_area_id}")


@api_endpoint(['GET'])
def validate_topic_criteria(request):
    """Validate if standards exist for given criteria with comprehensive checking"""
    view = TopicAPIView()
    
    try:
        # Get and validate parameters
        grade_levels_param = request.GET.get('grade_levels')
        subject_area_id = request.GET.get('subject_area_id')
        
        if not grade_levels_param:
            return view.error_response('Grade levels are required', status=400)
        
        # Parse and validate grade levels
        try:
            grade_levels = view.validate_list_of_integers(grade_levels_param, field_name="grade_levels")
            # Validate grade range
            for grade in grade_levels:
                if not (0 <= grade <= 12):
                    raise ValidationError(f"Grade {grade} must be between 0 (K) and 12")
        except ValidationError as e:
            return view.validation_error_response(str(e))
        
        # Validate subject area if provided
        subject_area = None
        if subject_area_id:
            try:
                subject_area = view.validate_subject_area(subject_area_id)
            except ValidationError as e:
                return view.validation_error_response(str(e))
        
        # Build query with proper filtering
        standards_query = Standard.objects.all()
        
        if grade_levels:
            standards_query = standards_query.filter(
                grade_levels__grade_numeric__in=grade_levels
            ).distinct()
        
        if subject_area:
            standards_query = standards_query.filter(subject_area=subject_area)
        
        standards_count = standards_query.count()
        
        if standards_count == 0:
            # Provide helpful error message with suggestions
            error_msg = "No standards found for the specified criteria. "
            
            if subject_area:
                # Check what combinations do exist
                available_subjects = SubjectArea.objects.filter(
                    standards__grade_levels__grade_numeric__in=grade_levels
                ).distinct()
                
                if available_subjects.exists():
                    subject_names = [s.name for s in available_subjects]
                    error_msg += f"Available subjects for grades {grade_levels}: {', '.join(subject_names)}. "
                else:
                    # Check available grades for this subject
                    available_grades = Standard.objects.filter(
                        subject_area=subject_area
                    ).values_list('grade_levels__grade_numeric', flat=True).distinct()
                    
                    if available_grades:
                        sorted_grades = sorted(set(g for g in available_grades if g is not None))
                        error_msg += f"Available grades for {subject_area.name}: {sorted_grades}. "
            else:
                error_msg += f"Try selecting different grade levels or a specific subject area."
            
            return view.success_response({
                'valid': False,
                'error': error_msg,
                'standards_count': 0,
                'suggestions': {
                    'available_subjects': list(
                        SubjectArea.objects.filter(
                            standards__grade_levels__grade_numeric__in=grade_levels
                        ).distinct().values_list('name', flat=True)
                    ) if grade_levels else [],
                }
            })
        
        return view.success_response({
            'valid': True,
            'standards_count': standards_count,
            'message': f'Found {standards_count} standards for the specified criteria',
            'breakdown': {
                'grade_levels': grade_levels,
                'subject_area': subject_area.name if subject_area else 'All subjects'
            }
        })
        
    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['POST'])
def generate_topic_proxies_api(request):
    """Generate topic-based proxy standards using LLM categorization"""
    view = TopicAPIView()
    
    try:
        data = view.parse_json_body(request)
        
        # Parse and validate grade level selection
        grade_levels = None
        grade_selection = data.get('grade_selection', {})
        
        if grade_selection.get('type') == 'specific':
            selected_grades = grade_selection.get('grades', [])
            if selected_grades:
                grade_levels = [view.validate_integer(g, 0, 12, f"grade {g}") for g in selected_grades]
        elif grade_selection.get('type') == 'range':
            min_grade = grade_selection.get('min_grade')
            max_grade = grade_selection.get('max_grade')
            if min_grade is not None and max_grade is not None:
                min_g = view.validate_integer(min_grade, 0, 12, "min_grade")
                max_g = view.validate_integer(max_grade, 0, 12, "max_grade")
                if min_g > max_g:
                    raise ValidationError("min_grade must be <= max_grade")
                grade_levels = list(range(min_g, max_g + 1))
        
        # Validate additional parameters
        subject_area_id = data.get('subject_area_id')
        subject_area = view.validate_subject_area(subject_area_id) if subject_area_id else None
        
        use_dynamic_chunk = bool(data.get('use_dynamic_chunk', True))
        chunk_size = None
        if not use_dynamic_chunk:
            chunk_size = view.validate_integer(data.get('chunk_size', 25), 5, 100, "chunk_size")
        
        include_outliers = bool(data.get('include_outliers', True))
        
        # Validate that standards exist for the criteria
        standards_query = Standard.objects.all()
        if grade_levels:
            standards_query = standards_query.filter(
                grade_levels__grade_numeric__in=grade_levels
            ).distinct()
        if subject_area:
            standards_query = standards_query.filter(subject_area=subject_area)
        
        if standards_query.count() == 0:
            return view.error_response(
                "No standards found for the specified criteria",
                status=400,
                error_code='NO_STANDARDS'
            )
        
        # Generate job ID and start background task
        job_id = str(uuid.uuid4())
        cache_key = f'topic_proxy_job_{job_id}'
        
        cache.set(cache_key, {
            'status': 'queued',
            'progress': 0,
            'message': 'Topic categorization job queued'
        }, 3600)
        
        # Start background thread
        thread = threading.Thread(
            target=_run_topic_proxy_job_secure,
            args=(job_id, grade_levels, subject_area_id, chunk_size, include_outliers, use_dynamic_chunk, data)
        )
        thread.daemon = True
        thread.start()
        
        return view.success_response({
            'job_id': job_id,
            'status': 'queued',
            'parameters': {
                'grade_levels': grade_levels,
                'subject_area': subject_area.name if subject_area else None,
                'use_dynamic_chunk': use_dynamic_chunk,
                'chunk_size': chunk_size,
                'include_outliers': include_outliers
            }
        })
        
    except ValidationError as e:
        return view.validation_error_response(str(e))
    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['GET'])
def topic_proxy_job_status_api(request, job_id: str):
    """Get topic categorization job status"""
    view = TopicAPIView()
    
    try:
        # Validate job_id format
        try:
            uuid.UUID(job_id)
        except ValueError:
            return view.error_response('Invalid job ID format', status=400)
        
        cache_key = f'topic_proxy_job_{job_id}'
        data = cache.get(cache_key)
        
        if not data:
            return view.error_response('Job not found', status=404)
        
        return view.success_response(data)
        
    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['POST'])
def create_topic_api(request):
    """Create new topics/concepts with validation"""
    view = TopicAPIView()
    
    try:
        data = view.parse_json_body(request)
        
        form = ConceptForm(data)
        if form.is_valid():
            concept = form.save()
            return view.success_response({
                'concept': {
                    'id': concept.id,
                    'name': concept.name,
                    'description': concept.description,
                }
            }, message="Concept created successfully")
        else:
            return view.validation_error_response(form.errors)
            
    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['POST'])
def run_topic_analysis_api(request):
    """Run topic analysis with comprehensive validation"""
    view = TopicAPIView()
    
    try:
        data = view.parse_json_body(request)
        
        form = TopicAnalysisForm(data)
        if form.is_valid():
            job_id = str(uuid.uuid4())
            analysis_config = form.cleaned_data
            
            cache_key = f'topic_analysis_job_{job_id}'
            cache.set(cache_key, {
                'status': 'queued',
                'progress': 0,
                'message': 'Analysis queued'
            }, 3600)
            
            # Start background job
            thread = threading.Thread(
                target=_run_topic_analysis_job_secure,
                args=(job_id, analysis_config)
            )
            thread.daemon = True
            thread.start()
            
            return view.success_response({
                'job_id': job_id,
                'config': analysis_config
            })
        else:
            return view.validation_error_response(form.errors)
            
    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['GET'])
def topic_analysis_status_api(request, job_id: str):
    """Get topic analysis job status"""
    view = TopicAPIView()
    
    try:
        # Validate job_id format
        try:
            uuid.UUID(job_id)
        except ValueError:
            return view.error_response('Invalid job ID format', status=400)
        
        cache_key = f'topic_analysis_job_{job_id}'
        data = cache.get(cache_key)
        
        if not data:
            return view.error_response('Job not found', status=404)
        
        return view.success_response(data)
        
    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['POST'])
def analyze_coverage_api(request):
    """Real-time coverage analysis with input validation"""
    view = TopicAPIView()
    
    try:
        data = view.parse_json_body(request)
        
        # Validate required fields
        content = view.validate_string(
            data.get('content', ''),
            min_length=10,
            max_length=10000,
            field_name="content"
        )
        
        subject_area_id = data.get('subject_area')
        subject_area = view.validate_subject_area(subject_area_id) if subject_area_id else None
        
        # Use bell curve analysis service
        bell_curve_service = BellCurveAnalysisService()
        
        # Extract concepts from content
        concepts = extract_concepts_from_content(content)
        
        if not concepts:
            return view.error_response(
                "No educational concepts detected in the content",
                status=400,
                error_code='NO_CONCEPTS'
            )
        
        # Calculate coverage
        coverage_result = bell_curve_service.calculate_bell_curve(
            concepts=concepts,
            subject_area=subject_area
        )
        
        # Format response with safe defaults
        response_data = {
            'total_states': coverage_result.get('total_states', 50),
            'fully_covered': coverage_result.get('fully_covered_states', 0),
            'partially_covered': coverage_result.get('partially_covered_states', 0),
            'not_covered': coverage_result.get('not_covered_states', 0),
            'coverage_percentage': round(coverage_result.get('coverage_percentage', 0), 2),
            'quick_wins': coverage_result.get('quick_wins', []),
            'concepts_analyzed': concepts,
            'analysis_metadata': {
                'content_length': len(content),
                'concepts_found': len(concepts),
                'subject_area': subject_area.name if subject_area else 'All subjects'
            }
        }
        
        return view.success_response(response_data)
        
    except ValidationError as e:
        return view.validation_error_response(str(e))
    except Exception as e:
        return view.handle_exception(request, e)


def _run_topic_proxy_job_secure(job_id: str, grade_levels: Optional[List[int]] = None,
                               subject_area_id: Optional[int] = None,
                               chunk_size: Optional[int] = None,
                               include_outliers: bool = True,
                               use_dynamic_chunk: bool = True,
                               original_payload: Dict = None):
    """Secure background job for topic-based categorization"""
    cache_key = f'topic_proxy_job_{job_id}'
    
    def update_progress(progress: int, message: str):
        cache.set(cache_key, {
            'status': 'running',
            'progress': progress,
            'message': message
        }, 3600)
    
    start_time = time.time()
    proxy_run = None
    
    try:
        # Create run name
        run_name = f"Topic Categorization {timezone.now().strftime('%Y-%m-%d %H:%M')}"
        if grade_levels:
            if len(grade_levels) == 1:
                run_name += f" - Grade {grade_levels[0]}"
            else:
                run_name += f" - Grades {min(grade_levels)}-{max(grade_levels)}"
        
        # Create ProxyRun record with transaction safety
        with transaction.atomic():
            proxy_run = ProxyRun.objects.create(
                run_id=f"topic-{job_id}",
                name=run_name,
                run_type='topics',
                filter_parameters={
                    'grade_selection': {
                        'type': 'specific' if grade_levels else 'all',
                        'grades': grade_levels
                    } if grade_levels else {},
                    'subject_area_id': subject_area_id
                },
                algorithm_parameters={
                    'use_dynamic_chunk': use_dynamic_chunk,
                    'chunk_size': chunk_size,
                    'include_outliers': include_outliers
                },
                job_id=job_id,
                status='running'
            )
        
        grade_msg = f" for grades {grade_levels}" if grade_levels else ""
        subject_msg = f" in subject area {subject_area_id}" if subject_area_id else ""
        update_progress(5, f'Starting topic categorization{grade_msg}{subject_msg}...')
        
        # Count input standards for tracking
        standards_query = Standard.objects.all()
        if grade_levels:
            standards_query = standards_query.filter(
                grade_levels__grade_numeric__in=grade_levels
            ).distinct()
        if subject_area_id:
            standards_query = standards_query.filter(subject_area_id=subject_area_id)
        
        total_standards = standards_query.count()
        proxy_run.total_input_standards = total_standards
        proxy_run.save(update_fields=['total_input_standards'])
        
        # Initialize service
        svc = TopicCategorizationService()
        
        # Run full categorization with progress updates
        hierarchy, proxies = svc.run_full_categorization(
            grade_levels=grade_levels,
            subject_area_id=subject_area_id,
            progress_callback=update_progress,
            use_dynamic_chunk_size=use_dynamic_chunk,
            override_chunk_size=chunk_size
        )
        
        # Store actual chunk size used if dynamic
        if use_dynamic_chunk and hasattr(svc, 'chunk_size'):
            proxy_run.algorithm_parameters['actual_chunk_size'] = svc.chunk_size
            proxy_run.save(update_fields=['algorithm_parameters'])
        
        # Generate summary statistics
        total_proxies = len(proxies)
        outlier_proxies = len([p for p in proxies if getattr(p, 'outlier_category', None)])
        regular_proxies = total_proxies - outlier_proxies
        
        # Count topics and sub-topics
        topics = set(getattr(p, 'topic', '') for p in proxies if not getattr(p, 'outlier_category', None))
        sub_topics = set(
            f"{getattr(p, 'topic', '')} > {getattr(p, 'sub_topic', '')}"
            for p in proxies
            if not getattr(p, 'outlier_category', None)
        )
        
        summary_msg = (f'Created {total_proxies} topic-based proxies: '
                      f'{regular_proxies} in {len(topics)} topics/{len(sub_topics)} sub-topics')
        if outlier_proxies > 0:
            summary_msg += f', {outlier_proxies} outliers'
        
        # Update ProxyRun with completion status
        end_time = time.time()
        proxy_run.status = 'completed'
        proxy_run.completed_at = timezone.now()
        proxy_run.duration_seconds = int(end_time - start_time)
        proxy_run.total_proxies_created = total_proxies
        proxy_run.outlier_proxies_count = outlier_proxies
        proxy_run.save(update_fields=[
            'status', 'completed_at', 'duration_seconds',
            'total_proxies_created', 'outlier_proxies_count'
        ])
        
        # Calculate coverage statistics
        proxy_run.calculate_coverage_stats()
        
        # Generate analysis report
        update_progress(95, 'Generating analysis report...')
        analyzer = ProxyRunAnalyzer()
        analyzer.analyze_run(proxy_run)
        
        cache.set(cache_key, {
            'status': 'completed',
            'progress': 100,
            'message': summary_msg,
            'results': {
                'total_proxies': total_proxies,
                'regular_proxies': regular_proxies,
                'outlier_proxies': outlier_proxies,
                'topics_count': len(topics),
                'sub_topics_count': len(sub_topics),
                'run_id': proxy_run.run_id,
                'duration_seconds': int(end_time - start_time),
                'hierarchy': {
                    'topics': getattr(hierarchy, 'topics', []) if hierarchy else []
                }
            }
        }, 3600)
        
    except Exception as e:
        logger.error(f"Topic categorization job {job_id} failed: {str(e)}", exc_info=True)
        
        # Update ProxyRun with failure status
        if proxy_run:
            proxy_run.status = 'failed'
            proxy_run.error_message = str(e)
            proxy_run.completed_at = timezone.now()
            proxy_run.duration_seconds = int(time.time() - start_time)
            proxy_run.save(update_fields=['status', 'error_message', 'completed_at', 'duration_seconds'])
        
        cache.set(cache_key, {
            'status': 'failed',
            'progress': 0,
            'message': f'Job failed: {str(e)}'
        }, 3600)


def _run_topic_analysis_job_secure(job_id: str, analysis_config: Dict):
    """Secure background job for topic analysis"""
    cache_key = f'topic_analysis_job_{job_id}'
    
    def update_progress(progress: int, message: str):
        cache.set(cache_key, {
            'status': 'running',
            'progress': progress,
            'message': message
        }, 3600)
    
    try:
        update_progress(5, 'Starting topic discovery...')
        
        service = TopicDiscoveryService()
        
        # Extract and validate parameters
        subject_area_id = analysis_config.get('subject_area')
        grade_levels = analysis_config.get('grade_levels', [])
        min_standards = analysis_config.get('min_standards_per_topic', 5)
        analysis_type = analysis_config.get('analysis_type', 'discover')
        
        # Get subject area object if specified
        subject_area = None
        if subject_area_id:
            try:
                subject_area = SubjectArea.objects.get(id=subject_area_id)
            except SubjectArea.DoesNotExist:
                logger.warning(f"Subject area {subject_area_id} not found")
        
        update_progress(30, f'Running {analysis_type} analysis...')
        
        # Run appropriate analysis type
        if analysis_type == 'discover':
            results = service.discover_topics(
                subject_area=subject_area,
                min_standards=min_standards
            )
        elif analysis_type == 'update_coverage':
            results = service.update_coverage_statistics()
        elif analysis_type == 'cluster_refinement':
            results = service.refine_clusters()
        else:  # full_analysis
            results = service.run_full_analysis(
                subject_area=subject_area,
                min_standards=min_standards
            )
        
        update_progress(80, 'Processing results...')
        
        # Process and save results with transaction safety
        topics_created = 0
        with transaction.atomic():
            for topic_data in results:
                # Create or update TopicCluster
                from standards.models import TopicCluster
                cluster, created = TopicCluster.objects.get_or_create(
                    name=topic_data['name'],
                    defaults={
                        'description': topic_data.get('description', ''),
                        'subject_area': subject_area,
                        'standards_count': topic_data.get('standards_count', 0),
                        'states_represented': topic_data.get('states_represented', 0),
                        'silhouette_score': topic_data.get('silhouette_score', 0.0),
                    }
                )
                if created:
                    topics_created += 1
        
        cache.set(cache_key, {
            'status': 'completed',
            'progress': 100,
            'message': f'Analysis complete. {topics_created} topics discovered.',
            'results': {
                'topics_created': topics_created,
                'total_topics': len(results),
                'analysis_type': analysis_type
            }
        }, 3600)
        
    except Exception as e:
        logger.error(f"Topic analysis job {job_id} failed: {str(e)}", exc_info=True)
        
        cache.set(cache_key, {
            'status': 'failed',
            'progress': 0,
            'message': f'Analysis failed: {str(e)}'
        }, 3600)