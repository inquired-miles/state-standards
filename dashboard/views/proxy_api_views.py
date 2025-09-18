"""
Secure API views for proxy standard operations with proper validation and error handling
"""
import uuid
import json
import time
import threading
from typing import Dict, List, Optional, Any, Set
from django.http import JsonResponse
from django.core.exceptions import ValidationError, PermissionDenied
from django.db import transaction
from django.db.models import Q, Count
from django.core.cache import cache
from django.utils import timezone
import logging

from standards.models import (
    ProxyRun, ProxyStandard, Standard, StandardAtom, GradeLevel, ClusterReport
)
from standards.services.clustering import ClusteringService, StandardClusteringService
from standards.services.naming import ProxyNamingService
from standards.services.proxy_run_analyzer import ProxyRunAnalyzer
from standards.services.discovery import CustomClusterService
from .base import BaseAPIView, api_endpoint

logger = logging.getLogger(__name__)


def _compute_cluster_coverage(report_scope: Optional[Dict[str, Any]], covered_ids: Optional[Set[uuid.UUID]]) -> Optional[List[Dict[str, Any]]]:
    """Return per-cluster coverage metrics for a scoped report."""
    if not report_scope:
        return None

    covered_ids = covered_ids or set()
    clusters_with_coverage: List[Dict[str, Any]] = []

    for cluster in report_scope.get('clusters', []):
        cluster_standard_ids = [sid for sid in cluster.get('standard_ids', []) if sid]
        cluster_ids: Set[uuid.UUID] = set()
        for sid in cluster_standard_ids:
            try:
                cluster_ids.add(uuid.UUID(str(sid)))
            except (ValueError, TypeError, AttributeError):
                continue
        covered_in_cluster = cluster_ids & covered_ids
        uncovered_in_cluster = cluster_ids - covered_ids
        clusters_with_coverage.append({
            'cluster_id': cluster.get('cluster_id'),
            'cluster_name': cluster.get('cluster_name'),
            'selection_order': cluster.get('selection_order'),
            'notes': cluster.get('notes'),
            'cluster_description': cluster.get('cluster_description'),
            'states_breakdown': cluster.get('states_breakdown', {}),
            'standards_count': cluster.get('standards_count', 0),
            'covered_count': len(covered_in_cluster),
            'not_covered_count': len(uncovered_in_cluster),
            'standard_ids': cluster_standard_ids,
            'covered_standard_ids': [str(sid) for sid in covered_in_cluster],
            'not_covered_standard_ids': [str(sid) for sid in uncovered_in_cluster],
        })

    clusters_with_coverage.sort(key=lambda c: c.get('selection_order', 0))
    return clusters_with_coverage


def _build_report_metadata(report_scope: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not report_scope:
        return None

    updated_at = report_scope.get('updated_at')
    return {
        'id': report_scope.get('report_id'),
        'title': report_scope.get('title'),
        'description': report_scope.get('description'),
        'is_shared': report_scope.get('is_shared'),
        'total_clusters': report_scope.get('total_clusters', 0),
        'standard_count': len(report_scope.get('standard_ids', []) or []),
        'updated_at': updated_at.isoformat() if updated_at else None,
    }


def _build_run_state_coverage(
    *,
    proxies,
    state_code: Optional[str],
    grade_levels: Optional[List[int]],
    scope_standard_ids: Optional[Set[uuid.UUID]],
):
    """Compute state coverage details for a proxy run."""

    # Determine states to analyze
    if state_code:
        states = [state_code]
    elif scope_standard_ids is not None:
        scoped_states = (
            Standard.objects.filter(id__in=scope_standard_ids)
            .exclude(state__code__isnull=True)
            .values_list('state__code', flat=True)
            .distinct()
        )
        states = sorted(code for code in scoped_states if code)
    else:
        states_set = set()
        for proxy in proxies:
            if hasattr(proxy, 'member_atoms') and proxy.member_atoms.exists():
                member_states = proxy.member_atoms.values_list('standard__state__code', flat=True)
                states_set.update(code for code in member_states if code)
            elif hasattr(proxy, 'member_standards') and proxy.member_standards.exists():
                member_states = proxy.member_standards.values_list('state__code', flat=True)
                states_set.update(code for code in member_states if code)
        states = sorted(states_set)

    std_to_proxy_titles: Dict[uuid.UUID, Set[str]] = {}
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
            if scope_standard_ids is not None:
                member_query = member_query.filter(standard_id__in=scope_standard_ids)

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
            if scope_standard_ids is not None:
                member_query = member_query.filter(id__in=scope_standard_ids)

            for sid in member_query.values_list('id', flat=True):
                std_to_proxy_titles.setdefault(sid, set()).add(title)

    result = []
    global_covered_ids: Set[uuid.UUID] = set()

    for code in states:
        state_standards_qs = Standard.objects.filter(state__code=code)
        if scope_standard_ids is not None:
            state_standards_qs = state_standards_qs.filter(id__in=scope_standard_ids)
        if grade_levels:
            state_standards_qs = state_standards_qs.filter(
                grade_levels__grade_numeric__in=grade_levels
            ).distinct()

        all_ids = set(state_standards_qs.values_list('id', flat=True))

        covered_ids: Set[uuid.UUID] = set()
        for proxy in proxies:
            if hasattr(proxy, 'member_atoms') and proxy.member_atoms.exists():
                proxy_covered = set(
                    proxy.member_atoms.filter(
                        standard__state__code=code
                    ).values_list('standard_id', flat=True)
                )
                if scope_standard_ids is not None:
                    proxy_covered &= scope_standard_ids
                covered_ids.update(proxy_covered)
            elif hasattr(proxy, 'member_standards') and proxy.member_standards.exists():
                proxy_covered = set(
                    proxy.member_standards.filter(
                        state__code=code
                    ).values_list('id', flat=True)
                )
                if scope_standard_ids is not None:
                    proxy_covered &= scope_standard_ids
                covered_ids.update(proxy_covered)

        covered_ids = covered_ids & all_ids
        not_covered_ids = all_ids - covered_ids
        global_covered_ids.update(covered_ids)

        covered_standards = [
            {
                'id': str(standard.id),
                'code': standard.code,
                'title': standard.title or '',
                'description': standard.description or '',
                'state__code': standard.state.code if standard.state else '',
                'proxy_titles': sorted(list(std_to_proxy_titles.get(standard.id, []))),
            }
            for standard in Standard.objects.filter(id__in=covered_ids)
        ]

        not_covered_standards = [
            {
                'id': str(standard.id),
                'code': standard.code,
                'title': standard.title or '',
                'description': standard.description or '',
                'state__code': standard.state.code if standard.state else '',
            }
            for standard in Standard.objects.filter(id__in=not_covered_ids)
        ]

        coverage_percentage = (len(covered_ids) / len(all_ids) * 100) if all_ids else 0

        result.append({
            'state': code,
            'covered_count': len(covered_ids),
            'total_count': len(all_ids),
            'coverage_percentage': round(coverage_percentage, 2),
            'covered': covered_standards,
            'not_covered': not_covered_standards,
        })

    result.sort(key=lambda x: x['state'])
    return result, global_covered_ids


def _load_report_scope(view: 'ProxyAPIView', request, report_id_param: Optional[str]):
    if not report_id_param:
        return None, None, None, None

    try:
        report_uuid = uuid.UUID(report_id_param)
    except ValueError:
        raise ValidationError('report_id must be a valid UUID')

    try:
        report = ClusterReport.objects.select_related('created_by').get(id=report_uuid)
    except ClusterReport.DoesNotExist:
        raise ValidationError('Coverage report not found')

    user = request.user
    can_view_report = (
        report.created_by == user
        or report.is_shared
        or getattr(user, 'is_staff', False)
        or getattr(user, 'is_superuser', False)
    )
    if not can_view_report:
        raise PermissionDenied('You do not have access to this coverage report')

    cluster_service = CustomClusterService()
    report_scope = cluster_service.build_report_scope(report)
    scope_standard_ids: Set[uuid.UUID] = set()
    for sid in report_scope.get('standard_ids', []) or []:
        try:
            scope_standard_ids.add(uuid.UUID(str(sid)))
        except (ValueError, TypeError, AttributeError):
            continue

    report_metadata = _build_report_metadata(report_scope)
    if report_metadata is not None:
        report_metadata['standard_count'] = len(scope_standard_ids)

    return report_scope, scope_standard_ids, report_metadata, report


def _build_report_cluster_items(
    report_scope: Dict[str, Any],
    *,
    grade_levels: Optional[List[int]],
    state_code: Optional[str],
    subject_area_id: Optional[int],
):
    clusters = report_scope.get('clusters', []) or []
    standards_in_scope_map: Dict[str, Dict[str, Any]] = {}
    proxy_items: List[Dict[str, Any]] = []
    covered_all_ids: Set[uuid.UUID] = set()

    for cluster in clusters:
        standard_ids = cluster.get('standard_ids', []) or []
        standard_qs = Standard.objects.filter(id__in=standard_ids)
        if grade_levels:
            standard_qs = standard_qs.filter(grade_levels__grade_numeric__in=grade_levels).distinct()
        if state_code:
            standard_qs = standard_qs.filter(state__code=state_code)
        if subject_area_id:
            standard_qs = standard_qs.filter(subject_area_id=subject_area_id)

        standards_list = list(
            standard_qs.values('id', 'code', 'title', 'description', 'state__code')
        )

        for item in standards_list:
            standards_in_scope_map[str(item['id'])] = {
                'id': str(item['id']),
                'code': item['code'],
                'title': item['title'] or '',
                'description': item['description'] or '',
                'state__code': item['state__code'] or '',
            }
            covered_all_ids.add(uuid.UUID(str(item['id'])))

        states_in_scope = len({item['state__code'] for item in standards_list if item['state__code']})
        coverage_count = len(cluster.get('states_breakdown', {}) or {})

        proxy_items.append({
            'proxy_id': cluster.get('cluster_id') or cluster.get('cluster_name'),
            'title': cluster.get('cluster_name') or 'Cluster',
            'description': cluster.get('cluster_description') or '',
            'coverage_count': coverage_count,
            'states_in_scope': states_in_scope,
            'covered_count': len(standards_list),
            'covered': [
                {
                    'id': str(item['id']),
                    'code': item['code'],
                    'title': item['title'] or '',
                    'description': item['description'] or '',
                    'state__code': item['state__code'] or '',
                }
                for item in standards_list
            ],
            'proxy_type': 'custom_cluster',
        })

    standards_in_scope = sorted(standards_in_scope_map.values(), key=lambda x: x['code'])
    return standards_in_scope, proxy_items, covered_all_ids


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
def proxy_coverage_reports_api(request):
    """List coverage reports accessible to the current staff user."""
    view = ProxyAPIView()
    user = request.user

    try:
        include_shared = request.GET.get('include_shared', 'true').lower() == 'true'
        queryset = ClusterReport.objects.select_related('created_by')
        if include_shared:
            queryset = queryset.filter(Q(created_by=user) | Q(is_shared=True))
        else:
            queryset = queryset.filter(created_by=user)

        queryset = queryset.annotate(cluster_count=Count('clusterreportentry')).order_by('-updated_at')[:100]

        reports = [
            {
                'id': str(report.id),
                'title': report.title,
                'description': report.description,
                'updated_at': report.updated_at,
                'is_shared': report.is_shared,
                'cluster_count': report.cluster_count,
            }
            for report in queryset
        ]

        return view.success_response({'reports': reports})
    except Exception as e:
        return view.handle_exception(request, e)

@api_endpoint(['GET'])
def proxy_run_coverage_api(request):
    """Return per-state coverage for a proxy run or a saved coverage report."""
    view = ProxyAPIView()

    try:
        run_id_param = request.GET.get('run_id')
        report_id_param = request.GET.get('report_id')

        if not run_id_param and not report_id_param:
            return view.validation_error_response('run_id or report_id is required')

        run_id = None
        if run_id_param:
            run_id = view.validate_string(run_id_param, min_length=1, max_length=100, field_name='run_id')

        state_code = request.GET.get('state')
        if state_code:
            state_code = view.validate_string(state_code, max_length=2, field_name="state").upper()

        grades_param = request.GET.get('grades')
        grade_levels: Optional[List[int]] = None
        if grades_param:
            grade_levels = view.validate_list_of_integers(grades_param, field_name="grades")
            grade_levels = view.validate_grade_levels(grade_levels)

        try:
            report_scope, scope_standard_ids, report_metadata, _ = _load_report_scope(view, request, report_id_param)
        except PermissionDenied as exc:
            return view.error_response(str(exc), status=403)
        except ValidationError as exc:
            message = str(exc)
            if message == 'Coverage report not found':
                return view.error_response(message, status=404)
            return view.validation_error_response(message)

        if run_id:
            try:
                run = ProxyRun.objects.get(run_id=run_id, status='completed')
            except ProxyRun.DoesNotExist:
                return view.error_response('Run not found', status=404)

            proxies = run.get_associated_proxies()
            if grade_levels:
                if run.run_type == 'atoms':
                    proxies = proxies.filter(
                        member_atoms__standard__grade_levels__grade_numeric__in=grade_levels
                    ).distinct()
                elif run.run_type in ['standards', 'topics']:
                    proxies = proxies.filter(
                        member_standards__grade_levels__grade_numeric__in=grade_levels
                    ).distinct()

            result, global_covered_ids = _build_run_state_coverage(
                proxies=proxies,
                state_code=state_code,
                grade_levels=grade_levels,
                scope_standard_ids=scope_standard_ids,
            )

            if report_scope is not None and report_metadata is not None:
                clusters_with_coverage = _compute_cluster_coverage(report_scope, global_covered_ids) or []
                report_metadata['clusters'] = clusters_with_coverage

            return view.success_response({
                'run_id': run_id,
                'states': result,
                'total_states': len(result),
                'parameters': {
                    'grade_levels': grade_levels,
                    'state_filter': state_code,
                    'report_id': report_metadata['id'] if report_metadata else None,
                },
                'report': report_metadata,
            })

        # Report-only analysis
        if not report_scope or not scope_standard_ids:
            return view.validation_error_response('report_id is required when run_id is not provided')

        result = []
        covered_ids_set = set(scope_standard_ids)

        report_standard_qs = Standard.objects.filter(id__in=scope_standard_ids)
        if grade_levels:
            report_standard_qs = report_standard_qs.filter(
                grade_levels__grade_numeric__in=grade_levels
            ).distinct()

        if state_code:
            report_standard_qs = report_standard_qs.filter(state__code=state_code)

        states = sorted({code for code in report_standard_qs.values_list('state__code', flat=True) if code})

        for code in states:
            state_qs = report_standard_qs.filter(state__code=code)
            standards_list = list(state_qs.values('id', 'code', 'title', 'description', 'state__code'))
            result.append({
                'state': code,
                'covered_count': len(standards_list),
                'total_count': len(standards_list),
                'coverage_percentage': 100.0 if standards_list else 0.0,
                'covered': [
                    {
                        'id': str(item['id']),
                        'code': item['code'],
                        'title': item['title'] or '',
                        'description': item['description'] or '',
                        'state__code': item['state__code'] or '',
                        'proxy_titles': [],
                    }
                    for item in standards_list
                ],
                'not_covered': [],
            })

        result.sort(key=lambda x: x['state'])

        clusters_with_coverage = _compute_cluster_coverage(report_scope, covered_ids_set) if report_scope else []
        if report_metadata is not None:
            report_metadata['clusters'] = clusters_with_coverage or []

        return view.success_response({
            'run_id': None,
            'states': result,
            'total_states': len(result),
            'parameters': {
                'grade_levels': grade_levels,
                'state_filter': state_code,
                'report_id': report_metadata['id'] if report_metadata else None,
            },
            'report': report_metadata,
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
        run_id_param = request.GET.get('run_id')
        report_id_param = request.GET.get('report_id')

        if not run_id_param and not report_id_param:
            return view.validation_error_response('run_id or report_id is required')

        run_id = None
        if run_id_param:
            run_id = view.validate_string(run_id_param, min_length=1, max_length=100, field_name='run_id')

        state_code = request.GET.get('state')
        if state_code:
            state_code = view.validate_string(state_code, max_length=2, field_name='state').upper()

        grades_param = request.GET.get('grades')
        grade_levels = None
        if grades_param:
            grade_levels = view.validate_list_of_integers(grades_param, field_name='grades')
            grade_levels = view.validate_grade_levels(grade_levels)

        subject_area_id_param = request.GET.get('subject_area_id')
        subject_area_id = int(subject_area_id_param) if subject_area_id_param else None

        try:
            report_scope, scope_standard_ids, report_metadata, _ = _load_report_scope(view, request, report_id_param)
        except PermissionDenied as exc:
            return view.error_response(str(exc), status=403)
        except ValidationError as exc:
            message = str(exc)
            if message == 'Coverage report not found':
                return view.error_response(message, status=404)
            return view.validation_error_response(message)

        if run_id:
            try:
                run = ProxyRun.objects.get(run_id=run_id, status='completed')
            except ProxyRun.DoesNotExist:
                return view.error_response('Run not found', status=404)

            proxies = run.get_associated_proxies()
            if grade_levels:
                if run.run_type == 'atoms':
                    proxies = proxies.filter(
                        member_atoms__standard__grade_levels__grade_numeric__in=grade_levels
                    ).distinct()
                elif run.run_type in ['standards', 'topics']:
                    proxies = proxies.filter(
                        member_standards__grade_levels__grade_numeric__in=grade_levels
                    ).distinct()

            standards_qs = Standard.objects.all()
            if state_code:
                standards_qs = standards_qs.filter(state__code=state_code)
            if grade_levels:
                standards_qs = standards_qs.filter(grade_levels__grade_numeric__in=grade_levels)
            if subject_area_id is not None:
                standards_qs = standards_qs.filter(subject_area_id=subject_area_id)
            if scope_standard_ids is not None:
                standards_qs = standards_qs.filter(id__in=scope_standard_ids)

            standards_qs = standards_qs.distinct()
            standards_in_scope = list(
                standards_qs.values('id', 'code', 'title', 'description', 'state__code')
            )
            scope_ids = set(standards_qs.values_list('id', flat=True))

            proxy_items = []
            covered_all_ids: Set[uuid.UUID] = set()
            for proxy in proxies:
                covered_ids: Set[uuid.UUID] = set()

                if hasattr(proxy, 'member_atoms') and proxy.member_atoms.exists():
                    member_query = proxy.member_atoms.all()
                    if state_code:
                        member_query = member_query.filter(standard__state__code=state_code)
                    if grade_levels:
                        member_query = member_query.filter(
                            standard__grade_levels__grade_numeric__in=grade_levels
                        ).distinct()
                    if subject_area_id is not None:
                        member_query = member_query.filter(standard__subject_area_id=subject_area_id)
                    if scope_standard_ids is not None:
                        member_query = member_query.filter(standard_id__in=scope_standard_ids)
                    covered_ids.update(member_query.values_list('standard_id', flat=True))

                elif hasattr(proxy, 'member_standards') and proxy.member_standards.exists():
                    member_query = proxy.member_standards.all()
                    if state_code:
                        member_query = member_query.filter(state__code=state_code)
                    if grade_levels:
                        member_query = member_query.filter(
                            grade_levels__grade_numeric__in=grade_levels
                        ).distinct()
                    if subject_area_id is not None:
                        member_query = member_query.filter(subject_area_id=subject_area_id)
                    if scope_standard_ids is not None:
                        member_query = member_query.filter(id__in=scope_standard_ids)
                    covered_ids.update(member_query.values_list('id', flat=True))

                covered_ids = covered_ids & scope_ids
                covered_all_ids.update(covered_ids)
                covered_standards = list(
                    Standard.objects.filter(id__in=covered_ids).values(
                        'id', 'code', 'title', 'description', 'state__code'
                    )
                ) or []

                unique_states_in_scope = len({
                    s['state__code'] for s in covered_standards if s['state__code']
                })

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

            not_covered_ids = list(scope_ids - covered_all_ids)
            not_covered_standards = list(
                Standard.objects.filter(id__in=not_covered_ids).values(
                    'id', 'code', 'title', 'description', 'state__code'
                )
            ) or []

            proxy_items.sort(key=lambda x: (x['states_in_scope'], x['covered_count']), reverse=True)

            if report_scope is not None and report_metadata is not None:
                clusters_with_coverage = _compute_cluster_coverage(report_scope, covered_all_ids) or []
                report_metadata['clusters'] = clusters_with_coverage

            return view.success_response({
                'run_id': run_id,
                'standards_in_scope_count': len(standards_in_scope),
                'standards_in_scope': standards_in_scope,
                'proxies': proxy_items,
                'not_covered': not_covered_standards,
                'parameters': {
                    'grade_levels': grade_levels,
                    'state_filter': state_code,
                    'report_id': report_metadata['id'] if report_metadata else None,
                },
                'report': report_metadata,
            })

        if not report_scope:
            return view.validation_error_response('report_id is required when run_id is not provided')

        standards_in_scope, proxy_items, covered_all_ids = _build_report_cluster_items(
            report_scope,
            grade_levels=grade_levels,
            state_code=state_code,
            subject_area_id=subject_area_id,
        )

        if report_metadata is not None:
            report_metadata['clusters'] = _compute_cluster_coverage(report_scope, covered_all_ids) or []

        return view.success_response({
            'run_id': None,
            'standards_in_scope_count': len(standards_in_scope),
            'standards_in_scope': standards_in_scope,
            'proxies': proxy_items,
            'not_covered': [],
            'parameters': {
                'grade_levels': grade_levels,
                'state_filter': state_code,
                'report_id': report_metadata['id'] if report_metadata else None,
            },
            'report': report_metadata,
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
