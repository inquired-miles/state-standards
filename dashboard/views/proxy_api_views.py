"""
Secure API views for proxy standard operations with proper validation and error handling
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
    ProxyRun, ProxyStandard, Standard, StandardAtom, GradeLevel
)
from standards.services.clustering import ClusteringService, StandardClusteringService
from standards.services.naming import ProxyNamingService
from standards.services.proxy_run_analyzer import ProxyRunAnalyzer
from .base import BaseAPIView, api_endpoint

logger = logging.getLogger(__name__)


class ProxyAPIView(BaseAPIView):
    """Base class for proxy-related API views"""
    
    def validate_grade_levels(self, grade_levels: List[int]) -> List[int]:
        """Validate grade levels are within valid range"""
        valid_grades = []
        for grade in grade_levels:
            try:
                grade_int = int(grade)
                if 0 <= grade_int <= 12:  # K=0, 1-12
                    valid_grades.append(grade_int)
                else:
                    raise ValidationError(f"Grade level {grade} must be between 0 (K) and 12")
            except (ValueError, TypeError):
                raise ValidationError(f"Invalid grade level: {grade}")
        
        return valid_grades
    
    def parse_grade_selection(self, data: Dict) -> Optional[List[int]]:
        """Parse grade selection from request data"""
        grade_selection = data.get('grade_selection', {})
        
        if not grade_selection:
            return None
            
        selection_type = grade_selection.get('type')
        
        if selection_type == 'specific':
            selected_grades = grade_selection.get('grades', [])
            if selected_grades:
                return self.validate_grade_levels(selected_grades)
        elif selection_type == 'range':
            min_grade = grade_selection.get('min_grade')
            max_grade = grade_selection.get('max_grade')
            if min_grade is not None and max_grade is not None:
                min_g = self.validate_integer(min_grade, 0, 12, "min_grade")
                max_g = self.validate_integer(max_grade, 0, 12, "max_grade")
                if min_g > max_g:
                    raise ValidationError("min_grade must be <= max_grade")
                return list(range(min_g, max_g + 1))
        
        return None


@api_endpoint(['GET'])
def proxy_runs_list_api(request):
    """List completed proxy runs with optional filters (run_type, grade, subject)."""
    view = ProxyAPIView()
    try:
        run_type = request.GET.get('run_type')
        status = request.GET.get('status', 'completed')
        subject_area_id = request.GET.get('subject_area_id')
        grades_param = request.GET.get('grades')
        q = (request.GET.get('q') or '').strip().lower()

        grade_levels = None
        if grades_param:
            grade_levels = view.validate_list_of_integers(grades_param, field_name="grades")
            grade_levels = view.validate_grade_levels(grade_levels)

        queryset = ProxyRun.objects.all().order_by('-started_at')
        if status:
            queryset = queryset.filter(status=status)
        if run_type:
            queryset = queryset.filter(run_type=run_type)

        runs = []
        for run in queryset[:100]:
            # Subject filter check (post-filter on JSON)
            if subject_area_id:
                try:
                    subj = int(subject_area_id)
                except (TypeError, ValueError):
                    return view.validation_error_response('subject_area_id must be an integer')
                if (run.filter_parameters or {}).get('subject_area_id') != subj:
                    continue

            # Grade filter check
            if grade_levels:
                grades_ok = False
                sel = (run.filter_parameters or {}).get('grade_selection') or {}
                if sel.get('type') == 'specific':
                    run_grades = set(int(g) for g in sel.get('grades', []) if isinstance(g, (int, str)))
                    if run_grades & set(grade_levels):
                        grades_ok = True
                elif sel.get('type') == 'range':
                    mn = sel.get('min_grade')
                    mx = sel.get('max_grade')
                    try:
                        rng = set(range(int(mn), int(mx) + 1))
                    except Exception:
                        rng = set()
                    if rng & set(grade_levels):
                        grades_ok = True
                else:
                    # No grade filter on run; include
                    grades_ok = True
                if not grades_ok:
                    continue

            # Text search
            if q:
                blob = f"{run.name} {run.description} {run.filter_summary} {run.algorithm_summary}".lower()
                if q not in blob:
                    continue

            runs.append({
                'run_id': run.run_id,
                'name': run.name,
                'run_type': run.run_type,
                'status': run.status,
                'started_at': run.started_at,
                'completed_at': run.completed_at,
                'duration_seconds': run.duration_seconds,
                'total_input_standards': run.total_input_standards,
                'total_proxies_created': run.total_proxies_created,
                'outlier_proxies_count': run.outlier_proxies_count,
                'filter_summary': run.filter_summary,
                'algorithm_summary': run.algorithm_summary,
            })

        return view.success_response({'runs': runs})
    except ValidationError as e:
        return view.validation_error_response(str(e))
    except Exception as e:
        return view.handle_exception(request, e)

@api_endpoint(['GET'])
def proxy_run_coverage_api(request):
    """Return per-state coverage for a given run with validation and pagination"""
    view = ProxyAPIView()
    
    try:
        # Validate required parameters
        run_id = view.validate_string(
            request.GET.get('run_id'), 
            min_length=1, 
            max_length=100, 
            field_name="run_id"
        )
        
        # Optional parameters with validation
        state_code = request.GET.get('state')
        if state_code:
            state_code = view.validate_string(state_code, max_length=2, field_name="state").upper()
        
        # Parse grade levels
        grades_param = request.GET.get('grades')
        grade_levels = None
        if grades_param:
            grade_levels = view.validate_list_of_integers(grades_param, field_name="grades")
            grade_levels = view.validate_grade_levels(grade_levels)
        
        # Get proxy run with proper error handling
        try:
            run = ProxyRun.objects.get(run_id=run_id, status='completed')
        except ProxyRun.DoesNotExist:
            return view.error_response('Run not found', status=404)
        
        # Get associated proxies
        proxies = run.get_associated_proxies()
        
        # Apply grade level filtering
        if grade_levels:
            if run.run_type == 'atoms':
                proxies = proxies.filter(
                    member_atoms__standard__grade_levels__grade_numeric__in=grade_levels
                ).distinct()
            elif run.run_type in ['standards', 'topics']:
                proxies = proxies.filter(
                    member_standards__grade_levels__grade_numeric__in=grade_levels
                ).distinct()
        
        # Build per-state breakdown
        result = []
        
        # Determine states to analyze
        if state_code:
            states = [state_code]
        else:
            # Get all states represented in the proxies
            states = set()
            for proxy in proxies:
                if hasattr(proxy, 'member_atoms') and proxy.member_atoms.exists():
                    member_states = set(
                        proxy.member_atoms.values_list('standard__state__code', flat=True)
                    )
                    states.update(member_states)
                elif hasattr(proxy, 'member_standards') and proxy.member_standards.exists():
                    member_states = set(
                        proxy.member_standards.values_list('state__code', flat=True)
                    )
                    states.update(member_states)
            states = [s for s in states if s]
        
        # Precompute mapping: standard_id -> proxy titles that cover it
        std_to_proxy_titles = {}
        for proxy in proxies:
            title = getattr(proxy, 'title', '') or getattr(proxy, 'proxy_id', str(proxy.id))
            
            if hasattr(proxy, 'member_atoms') and proxy.member_atoms.exists():
                member_query = proxy.member_atoms.all()
                if state_code:
                    member_query = member_query.filter(standard__state__code=state_code)
                if grade_levels:
                    member_query = member_query.filter(
                        standard__grade_levels__grade_numeric__in=grade_levels
                    ).distinct()
                
                for sid in member_query.values_list('standard_id', flat=True):
                    std_to_proxy_titles.setdefault(sid, set()).add(title)
                    
            elif hasattr(proxy, 'member_standards') and proxy.member_standards.exists():
                member_query = proxy.member_standards.all()
                if state_code:
                    member_query = member_query.filter(state__code=state_code)
                if grade_levels:
                    member_query = member_query.filter(
                        grade_levels__grade_numeric__in=grade_levels
                    ).distinct()
                
                for sid in member_query.values_list('id', flat=True):
                    std_to_proxy_titles.setdefault(sid, set()).add(title)
        
        # Process each state
        for code in states:
            # Get all standards in this state within the filtered scope
            state_standards_qs = Standard.objects.filter(state__code=code)
            if grade_levels:
                state_standards_qs = state_standards_qs.filter(
                    grade_levels__grade_numeric__in=grade_levels
                ).distinct()
            
            all_ids = set(state_standards_qs.values_list('id', flat=True))
            
            # Find covered standards by proxies
            covered_ids = set()
            for proxy in proxies:
                if hasattr(proxy, 'member_atoms') and proxy.member_atoms.exists():
                    proxy_covered = set(
                        proxy.member_atoms.filter(
                            standard__state__code=code
                        ).values_list('standard_id', flat=True)
                    )
                    covered_ids.update(proxy_covered)
                elif hasattr(proxy, 'member_standards') and proxy.member_standards.exists():
                    proxy_covered = set(
                        proxy.member_standards.filter(
                            state__code=code
                        ).values_list('id', flat=True)
                    )
                    covered_ids.update(proxy_covered)
            
            covered_ids = covered_ids & all_ids
            not_covered_ids = all_ids - covered_ids
            
            # Build detailed results
            covered_standards = []
            for standard in Standard.objects.filter(id__in=covered_ids):
                covered_standards.append({
                    'id': str(standard.id),
                    'code': standard.code,
                    'title': standard.title or '',
                    'description': standard.description or '',
                    'state__code': standard.state.code if standard.state else '',
                    'proxy_titles': sorted(list(std_to_proxy_titles.get(standard.id, [])))
                })
            
            not_covered_standards = []
            for standard in Standard.objects.filter(id__in=not_covered_ids):
                not_covered_standards.append({
                    'id': str(standard.id),
                    'code': standard.code,
                    'title': standard.title or '',
                    'description': standard.description or '',
                    'state__code': standard.state.code if standard.state else '',
                })
            
            coverage_percentage = (len(covered_ids) / len(all_ids) * 100) if all_ids else 0
            
            result.append({
                'state': code,
                'covered_count': len(covered_ids),
                'total_count': len(all_ids),
                'coverage_percentage': round(coverage_percentage, 2),
                'covered': covered_standards,
                'not_covered': not_covered_standards,
            })
        
        # Sort by state code for consistency
        result.sort(key=lambda x: x['state'])
        
        return view.success_response({
            'run_id': run_id,
            'states': result,
            'total_states': len(result),
            'parameters': {
                'grade_levels': grade_levels,
                'state_filter': state_code
            }
        })
        
    except ValidationError as e:
        return view.validation_error_response(str(e))
    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['GET'])
def proxy_run_proxies_api(request):
    """Return standards in scope and proxy standards with covered standards"""
    view = ProxyAPIView()
    
    try:
        # Validate required parameters
        run_id = view.validate_string(
            request.GET.get('run_id'),
            min_length=1,
            max_length=100,
            field_name="run_id"
        )
        
        # Optional parameters
        state_code = request.GET.get('state')
        if state_code:
            state_code = view.validate_string(state_code, max_length=2, field_name="state").upper()
        
        grades_param = request.GET.get('grades')
        grade_levels = None
        if grades_param:
            grade_levels = view.validate_list_of_integers(grades_param, field_name="grades")
            grade_levels = view.validate_grade_levels(grade_levels)
        subject_area_id = request.GET.get('subject_area_id')
        subject_area_id = int(subject_area_id) if subject_area_id else None
        
        # Get proxy run
        try:
            run = ProxyRun.objects.get(run_id=run_id, status='completed')
        except ProxyRun.DoesNotExist:
            return view.error_response('Run not found', status=404)
        
        proxies = run.get_associated_proxies()
        
        # Build standards in scope (filtered universe)
        standards_qs = Standard.objects.all()
        if state_code:
            standards_qs = standards_qs.filter(state__code=state_code)
        if grade_levels:
            standards_qs = standards_qs.filter(
                grade_levels__grade_numeric__in=grade_levels
            ).distinct()
        
        # Use pagination for large datasets
        paginated_standards = view.paginate_queryset(standards_qs, request, page_size=100)
        standards_in_scope = list(
            standards_qs.values('id', 'code', 'title', 'description', 'state__code')
        )
        scope_ids = set(standards_qs.values_list('id', flat=True))
        
        # Process each proxy
        proxy_items = []
        for proxy in proxies:
            covered_ids = set()
            
            # Determine covered standards for this proxy within scope
            if hasattr(proxy, 'member_atoms') and proxy.member_atoms.exists():
                member_query = proxy.member_atoms.all()
                if state_code:
                    member_query = member_query.filter(standard__state__code=state_code)
                if grade_levels:
                    member_query = member_query.filter(
                        standard__grade_levels__grade_numeric__in=grade_levels
                    ).distinct()
                if subject_area_id:
                    member_query = member_query.filter(standard__subject_area_id=subject_area_id)
                covered_ids.update(member_query.values_list('standard_id', flat=True))
                
            elif hasattr(proxy, 'member_standards') and proxy.member_standards.exists():
                member_query = proxy.member_standards.all()
                if state_code:
                    member_query = member_query.filter(state__code=state_code)
                if grade_levels:
                    member_query = member_query.filter(
                        grade_levels__grade_numeric__in=grade_levels
                    ).distinct()
                if subject_area_id:
                    member_query = member_query.filter(subject_area_id=subject_area_id)
                covered_ids.update(member_query.values_list('id', flat=True))
            
            covered_ids = list(covered_ids & scope_ids)
            covered_standards = list(
                Standard.objects.filter(id__in=covered_ids).values(
                    'id', 'code', 'title', 'description', 'state__code'
                )
            ) or []
            
            # Calculate unique states covered in current scope
            unique_states_in_scope = len(set(
                s['state__code'] for s in covered_standards if s['state__code']
            ))
            
            # Determine proxy type badge
            if hasattr(proxy, 'member_standards') and not hasattr(proxy, 'member_atoms'):
                proxy_type = 'topics' if run.run_type == 'topics' else 'standards'
            else:
                proxy_type = getattr(proxy, 'source_type', 'atoms') or 'atoms'

            proxy_items.append({
                'proxy_id': getattr(proxy, 'proxy_id', str(proxy.id)),
                'title': getattr(proxy, 'title', '') or '',
                'description': getattr(proxy, 'description', '') or '',
                'coverage_count': getattr(proxy, 'coverage_count', 0),
                'states_in_scope': unique_states_in_scope,
                'covered_count': len(covered_standards),
                'covered': covered_standards or [],
                'proxy_type': proxy_type,
            })
        
        # Calculate overall not covered standards
        covered_all_ids = set()
        for item in proxy_items:
            covered_all_ids.update(s['id'] for s in item['covered'])
        
        not_covered_ids = list(scope_ids - covered_all_ids)
        not_covered_standards = list(
            Standard.objects.filter(id__in=not_covered_ids).values(
                'id', 'code', 'title', 'description', 'state__code'
            )
        ) or []
        
        # Sort proxies by states covered in scope, then by standards covered
        proxy_items.sort(key=lambda x: (x['states_in_scope'], x['covered_count']), reverse=True)
        
        return view.success_response({
            'run_id': run_id,
            'standards_in_scope_count': len(standards_in_scope),
            'standards_in_scope': standards_in_scope,
            'proxies': proxy_items,
            'not_covered': not_covered_standards,
            'parameters': {
                'grade_levels': grade_levels,
                'state_filter': state_code
            }
        })
        
    except ValidationError as e:
        return view.validation_error_response(str(e))
    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['POST'])
def generate_proxies_api(request):
    """Generate proxy standards with comprehensive validation"""
    view = ProxyAPIView()
    
    try:
        data = view.parse_json_body(request)
        
        # Validate parameters
        min_cluster = view.validate_integer(data.get('min_cluster', 8), 2, 100, "min_cluster")
        epsilon = view.validate_float(data.get('epsilon', 0.15), 0.01, 1.0, "epsilon")
        name_proxies = bool(data.get('name_proxies', False))
        
        # Parse grade levels
        grade_levels = view.parse_grade_selection(data)
        
        # Generate job ID and start background task
        job_id = str(uuid.uuid4())
        cache_key = f'proxy_job_{job_id}'
        
        cache.set(cache_key, {
            'status': 'queued',
            'progress': 0,
            'message': 'Job queued for processing'
        }, 3600)  # 1 hour TTL
        
        # Start background thread
        thread = threading.Thread(
            target=_run_proxy_job_secure,
            args=(job_id, min_cluster, epsilon, name_proxies, grade_levels)
        )
        thread.daemon = True
        thread.start()
        
        return view.success_response({
            'job_id': job_id,
            'status': 'queued',
            'parameters': {
                'min_cluster': min_cluster,
                'epsilon': epsilon,
                'name_proxies': name_proxies,
                'grade_levels': grade_levels
            }
        })
        
    except ValidationError as e:
        return view.validation_error_response(str(e))
    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['GET'])
def proxy_job_status_api(request, job_id: str):
    """Get proxy job status with validation"""
    view = ProxyAPIView()
    
    try:
        # Validate job_id format
        try:
            uuid.UUID(job_id)
        except ValueError:
            return view.error_response('Invalid job ID format', status=400)
        
        cache_key = f'proxy_job_{job_id}'
        data = cache.get(cache_key)
        
        if not data:
            return view.error_response('Job not found', status=404)
        
        return view.success_response(data)
        
    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['POST'])
def generate_standard_proxies_api(request):
    """Generate standard-level proxy standards"""
    view = ProxyAPIView()
    
    try:
        data = view.parse_json_body(request)
        
        # Validate parameters
        min_cluster = view.validate_integer(data.get('min_cluster', 8), 2, 100, "min_cluster")
        epsilon = view.validate_float(data.get('epsilon', 0.15), 0.01, 1.0, "epsilon")
        name_proxies = bool(data.get('name_proxies', False))
        
        # Parse grade levels
        grade_levels = view.parse_grade_selection(data)
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        cache_key = f'standard_proxy_job_{job_id}'
        
        cache.set(cache_key, {
            'status': 'queued',
            'progress': 0,
            'message': 'Standard clustering job queued'
        }, 3600)
        
        # Start background thread
        thread = threading.Thread(
            target=_run_standard_proxy_job_secure,
            args=(job_id, min_cluster, epsilon, name_proxies, grade_levels)
        )
        thread.daemon = True
        thread.start()
        
        return view.success_response({
            'job_id': job_id,
            'status': 'queued'
        })
        
    except ValidationError as e:
        return view.validation_error_response(str(e))
    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['GET'])
def standard_proxy_job_status_api(request, job_id: str):
    """Get standard proxy job status"""
    view = ProxyAPIView()
    
    try:
        # Validate job_id format
        try:
            uuid.UUID(job_id)
        except ValueError:
            return view.error_response('Invalid job ID format', status=400)
        
        cache_key = f'standard_proxy_job_{job_id}'
        data = cache.get(cache_key)
        
        if not data:
            return view.error_response('Job not found', status=404)
        
        return view.success_response(data)
        
    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['GET'])
def proxy_detail_api(request, proxy_id):
    """Get proxy standard details with validation"""
    view = ProxyAPIView()
    
    try:
        # Validate proxy_id
        try:
            uuid.UUID(proxy_id)
        except ValueError:
            return view.error_response('Invalid proxy ID format', status=400)
        
        try:
            proxy = ProxyStandard.objects.select_related('medoid_atom').prefetch_related(
                'member_atoms__standard__state',
                'member_atoms__standard__subject_area',
                'grade_levels'
            ).get(id=proxy_id)
        except ProxyStandard.DoesNotExist:
            return view.error_response('Proxy standard not found', status=404)
        
        # Prepare atom data
        atoms_data = []
        for atom in proxy.member_atoms.all():
            atoms_data.append({
                'id': str(atom.id),
                'text': atom.text,
                'atom_code': atom.atom_code,
                'standard': {
                    'code': atom.standard.code if atom.standard else None,
                    'title': atom.standard.title if atom.standard else None,
                    'state': atom.standard.state.code if atom.standard and atom.standard.state else None,
                    'subject_area': atom.standard.subject_area.name if atom.standard and atom.standard.subject_area else None
                }
            })
        
        response_data = {
            'id': str(proxy.id),
            'proxy_id': proxy.proxy_id,
            'title': proxy.title,
            'description': proxy.description,
            'grade_range': proxy.grade_range_display,
            'min_grade': proxy.min_grade,
            'max_grade': proxy.max_grade,
            'grade_levels': [
                {'grade_numeric': gl.grade_numeric, 'grade': gl.grade}
                for gl in proxy.grade_levels.all()
            ],
            'coverage_count': proxy.coverage_count,
            'avg_similarity': proxy.avg_similarity,
            'atom_count': len(atoms_data),
            'atoms': atoms_data,
            'created_at': proxy.created_at.isoformat(),
        }
        
        return view.success_response({'proxy': response_data})
        
    except ValidationError as e:
        return view.validation_error_response(str(e))
    except Exception as e:
        return view.handle_exception(request, e)


def _run_proxy_job_secure(job_id: str, min_cluster: int, epsilon: float, 
                         name_proxies: bool, grade_levels: Optional[List[int]] = None):
    """Secure background job for proxy clustering with comprehensive error handling"""
    cache_key = f'proxy_job_{job_id}'
    
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
        run_name = f"Atom Clustering {timezone.now().strftime('%Y-%m-%d %H:%M')}"
        if grade_levels:
            if len(grade_levels) == 1:
                run_name += f" - Grade {grade_levels[0]}"
            else:
                run_name += f" - Grades {min(grade_levels)}-{max(grade_levels)}"
        
        # Create ProxyRun record with transaction safety
        with transaction.atomic():
            proxy_run = ProxyRun.objects.create(
                run_id=f"atoms-{job_id}",
                name=run_name,
                run_type='atoms',
                filter_parameters={
                    'grade_selection': {
                        'type': 'specific' if grade_levels else 'all',
                        'grades': grade_levels
                    } if grade_levels else {}
                },
                algorithm_parameters={
                    'min_cluster': min_cluster,
                    'epsilon': epsilon,
                    'name_proxies': name_proxies
                },
                job_id=job_id,
                status='running'
            )
        
        update_progress(5, f'Starting clustering for {run_name}...')
        
        # Count input atoms for tracking
        atoms_query = StandardAtom.objects.filter(embedding__isnull=False)
        if grade_levels:
            atoms_query = atoms_query.filter(
                standard__grade_levels__grade_numeric__in=grade_levels
            ).distinct()
        
        total_atoms = atoms_query.count()
        proxy_run.total_input_standards = total_atoms
        proxy_run.save(update_fields=['total_input_standards'])
        
        update_progress(15, f'Processing {total_atoms} atoms...')
        
        # Run clustering
        svc = ClusteringService()
        results = svc.run_full(
            min_cluster_size=min_cluster,
            epsilon=epsilon,
            grade_levels=grade_levels
        )
        
        update_progress(70, f'Creating {results["n_clusters"]} clusters...')
        created = svc.persist_proxies(results)
        
        # Name proxies if requested
        if name_proxies:
            update_progress(85, 'Naming proxies...')
            namer = ProxyNamingService()
            to_name = ProxyStandard.objects.filter(title="", source_type="atoms")[:500]
            
            named = 0
            for proxy in to_name:
                try:
                    meta = namer.name_proxy(proxy)
                    proxy.title = meta.get('title', proxy.title)
                    proxy.description = meta.get('description', proxy.description)
                    proxy.save(update_fields=['title', 'description'])
                    named += 1
                    
                    if named % 25 == 0:
                        update_progress(90, f'Named {named} proxies...')
                except Exception as e:
                    logger.warning(f"Failed to name proxy {proxy.id}: {e}")
        
        # Update completion status
        end_time = time.time()
        proxy_run.status = 'completed'
        proxy_run.completed_at = timezone.now()
        proxy_run.duration_seconds = int(end_time - start_time)
        proxy_run.total_proxies_created = created
        proxy_run.save(update_fields=['status', 'completed_at', 'duration_seconds', 'total_proxies_created'])
        
        # Calculate coverage statistics
        proxy_run.calculate_coverage_stats()
        
        # Generate analysis report
        update_progress(95, 'Generating analysis report...')
        analyzer = ProxyRunAnalyzer()
        analyzer.analyze_run(proxy_run)
        
        cache.set(cache_key, {
            'status': 'completed',
            'progress': 100,
            'message': f'Successfully created {created} proxies',
            'run_id': proxy_run.run_id,
            'results': {
                'total_proxies': created,
                'input_atoms': total_atoms,
                'duration_seconds': int(end_time - start_time)
            }
        }, 3600)
        
    except Exception as e:
        logger.error(f"Atom clustering job {job_id} failed: {str(e)}", exc_info=True)
        
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


def _run_standard_proxy_job_secure(job_id: str, min_cluster: int, epsilon: float,
                                  name_proxies: bool, grade_levels: Optional[List[int]] = None):
    """Secure background job for standard-level proxy clustering"""
    cache_key = f'standard_proxy_job_{job_id}'
    
    def update_progress(progress: int, message: str):
        cache.set(cache_key, {
            'status': 'running',
            'progress': progress,
            'message': message
        }, 3600)
    
    start_time = time.time()
    proxy_run = None
    
    try:
        # Create ProxyRun record
        with transaction.atomic():
            proxy_run = ProxyRun.objects.create(
                run_id=job_id,
                run_type='standards',
                name='Standard Clustering Run',
                description=f'Standards clustering - min_cluster={min_cluster}, epsilon={epsilon}',
                status='running',
                filter_parameters={'grade_levels': grade_levels},
                algorithm_parameters={
                    'min_cluster': min_cluster,
                    'epsilon': epsilon,
                    'name_proxies': name_proxies
                }
            )
        
        # Count input standards
        input_queryset = Standard.objects.all()
        if grade_levels:
            input_queryset = input_queryset.filter(
                grade_levels__grade_numeric__in=grade_levels
            ).distinct()
        
        input_count = input_queryset.count()
        proxy_run.total_input_standards = input_count
        proxy_run.save(update_fields=['total_input_standards'])
        
        grade_msg = f" for grades {grade_levels}" if grade_levels else ""
        update_progress(5, f'Starting standard clustering{grade_msg}...')
        
        # Run clustering
        svc = StandardClusteringService()
        results = svc.run_full(
            min_cluster_size=min_cluster,
            epsilon=epsilon,
            grade_levels=grade_levels
        )
        
        update_progress(70, f'Persisting {results["n_clusters"]} clusters...')
        created = svc.persist_standard_proxies(results)
        
        proxy_run.total_proxies_created = created
        proxy_run.save(update_fields=['total_proxies_created'])
        
        # Name proxies if requested
        if name_proxies:
            update_progress(85, 'Naming standard proxies...')
            namer = ProxyNamingService()
            to_name = ProxyStandard.objects.filter(title="", source_type="standards")[:500]
            
            named = 0
            for proxy in to_name:
                try:
                    meta = namer.name_proxy(proxy)
                    proxy.title = meta.get('title', proxy.title)
                    proxy.description = meta.get('description', proxy.description)
                    proxy.save(update_fields=['title', 'description'])
                    named += 1
                    
                    if named % 25 == 0:
                        update_progress(90, f'Named {named} standard proxies...')
                except Exception as e:
                    logger.warning(f"Failed to name proxy {proxy.id}: {e}")
        
        # Generate analysis report
        update_progress(95, 'Generating analysis report...')
        analyzer = ProxyRunAnalyzer()
        analyzer.analyze_run(proxy_run)
        
        # Update completion status
        proxy_run.status = 'completed'
        proxy_run.completed_at = timezone.now()
        proxy_run.duration_seconds = int(time.time() - start_time)
        proxy_run.save(update_fields=['status', 'completed_at', 'duration_seconds'])
        
        cache.set(cache_key, {
            'status': 'completed',
            'progress': 100,
            'message': f'Created {created} standard-level proxies',
            'run_id': job_id,
            'results': {
                'total_proxies': created,
                'input_standards': input_count
            }
        }, 3600)
        
    except Exception as e:
        logger.error(f"Standard proxy job {job_id} failed: {str(e)}", exc_info=True)
        
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
