"""API views for managing custom topic clusters and comparison reports"""
import logging
from typing import Any, Dict, List
from uuid import UUID

from django.core.exceptions import ValidationError, PermissionDenied
from django.db.models import Q

from standards.models import TopicCluster, ClusterReport
from standards.services.discovery import CustomClusterService
from .base import BaseAPIView, api_endpoint

logger = logging.getLogger(__name__)


class CustomClusterAPIView(BaseAPIView):
    """Shared helpers for custom cluster endpoints"""

    def __init__(self):
        super().__init__()
        self.service = CustomClusterService()

    def parse_uuid(self, value: str, field_name: str = 'id') -> UUID:
        try:
            return UUID(str(value))
        except (TypeError, ValueError):
            raise ValidationError(f"Invalid {field_name}")

    def serialize_cluster(
        self,
        cluster: TopicCluster,
        include_members: bool = False,
        current_user=None
    ) -> Dict[str, Any]:
        can_edit = bool(
            current_user
            and cluster.origin == 'custom'
            and cluster.created_by == current_user
        )
        data = {
            'id': str(cluster.id),
            'name': cluster.name,
            'description': cluster.description,
            'origin': cluster.origin,
            'created_by': cluster.created_by.email if cluster.created_by else None,
            'is_shared': cluster.is_shared,
            'standards_count': cluster.standards_count,
            'states_represented': cluster.states_represented,
            'subject_area': cluster.subject_area.name if cluster.subject_area else None,
            'updated_at': cluster.updated_at,
            'search_context': cluster.search_context or {},
            'can_edit': can_edit,
        }
        if include_members:
            members = []
            qs = cluster.clustermembership_set.select_related('standard').order_by('selection_order')
            for membership in qs:
                std = membership.standard
                members.append({
                    'id': str(std.id),
                    'code': std.code,
                    'title': std.title,
                    'state': std.state.code if std.state else None,
                    'similarity_score': membership.similarity_score,
                    'selection_order': membership.selection_order,
                })
            data['members'] = members
            data['coverage_summary'] = self.service.summarize_cluster(cluster)
        return data

    def serialize_report(self, report: ClusterReport) -> Dict[str, Any]:
        return {
            'id': str(report.id),
            'title': report.title,
            'description': report.description,
            'is_shared': report.is_shared,
            'created_by': report.created_by.email if report.created_by else None,
            'updated_at': report.updated_at,
        }

    def ensure_cluster_owner(self, cluster: TopicCluster, user) -> None:
        if cluster.origin != 'custom' or cluster.created_by != user:
            raise PermissionDenied("You do not have permission to modify this cluster")

    def ensure_report_owner(self, report: ClusterReport, user) -> None:
        if report.created_by != user:
            raise PermissionDenied("You do not have permission to modify this report")


@api_endpoint(['GET', 'POST'])
def custom_clusters_api(request):
    view = CustomClusterAPIView()
    user = request.user

    try:
        if request.method == 'GET':
            include_shared = request.GET.get('include_shared', 'true').lower() == 'true'
            queryset = TopicCluster.objects.filter(origin='custom')
            if include_shared:
                queryset = queryset.filter(Q(created_by=user) | Q(is_shared=True))
            else:
                queryset = queryset.filter(created_by=user)
            queryset = queryset.select_related('subject_area', 'created_by').order_by('-updated_at')
            clusters = [view.serialize_cluster(cluster, current_user=user) for cluster in queryset]
            return view.success_response({'clusters': clusters})

        data = view.parse_json_body(request)
        title = view.validate_string(data.get('title'), min_length=3, max_length=200, field_name='title')
        standard_ids = data.get('standard_ids') or []
        if not isinstance(standard_ids, list) or not standard_ids:
            return view.validation_error_response('standard_ids must be a non-empty list')

        description = data.get('description', '')
        is_shared = bool(data.get('is_shared', False))
        search_context = data.get('search_context') or {}
        similarity_map = data.get('similarity_map') or {}

        cluster = view.service.create_custom_cluster(
            owner=user,
            title=title,
            standard_ids=standard_ids,
            description=description,
            is_shared=is_shared,
            search_context=search_context,
            similarity_map=similarity_map,
        )

        return view.success_response(
            view.serialize_cluster(cluster, include_members=True, current_user=user),
            status=201
        )

    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['GET', 'PATCH', 'DELETE'])
def custom_cluster_detail_api(request, cluster_id: str):
    view = CustomClusterAPIView()
    user = request.user

    try:
        cluster_uuid = view.parse_uuid(cluster_id)
        cluster = TopicCluster.objects.select_related('subject_area', 'created_by').get(id=cluster_uuid)

        if request.method == 'GET':
            if cluster.origin == 'custom' and cluster.created_by != user and not cluster.is_shared:
                raise PermissionDenied("Cluster is private")
            return view.success_response(
                view.serialize_cluster(cluster, include_members=True, current_user=user)
            )

        if request.method == 'PATCH':
            view.ensure_cluster_owner(cluster, user)
            data = view.parse_json_body(request)
            title = data.get('title')
            standard_ids = data.get('standard_ids')
            description = data.get('description')
            is_shared = data.get('is_shared')
            search_context = data.get('search_context')
            similarity_map = data.get('similarity_map') or {}

            if title is not None:
                title = view.validate_string(title, min_length=3, max_length=200, field_name='title')

            if standard_ids:
                cluster = view.service.update_custom_cluster(
                    cluster,
                    standard_ids=standard_ids,
                    title=title,
                    description=description,
                    is_shared=is_shared,
                    search_context=search_context,
                    similarity_map=similarity_map,
                    acting_user=user,
                )
            else:
                update_fields = []
                if title is not None:
                    cluster.name = title
                    update_fields.append('name')
                if description is not None:
                    cluster.description = description
                    update_fields.append('description')
                if is_shared is not None:
                    cluster.is_shared = bool(is_shared)
                    update_fields.append('is_shared')
                if search_context is not None:
                    cluster.search_context = search_context
                    update_fields.append('search_context')
                if update_fields:
                    cluster.save(update_fields=update_fields)

            cluster.refresh_from_db()
            return view.success_response(
                view.serialize_cluster(cluster, include_members=True, current_user=user)
            )

        # DELETE
        view.ensure_cluster_owner(cluster, user)
        cluster.delete()
        return view.success_response(message='Cluster deleted')

    except TopicCluster.DoesNotExist:
        return view.error_response('Cluster not found', status=404)
    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['GET', 'POST'])
def cluster_reports_api(request):
    view = CustomClusterAPIView()
    user = request.user

    try:
        if request.method == 'GET':
            include_shared = request.GET.get('include_shared', 'true').lower() == 'true'
            queryset = ClusterReport.objects.all()
            if include_shared:
                queryset = queryset.filter(Q(created_by=user) | Q(is_shared=True))
            else:
                queryset = queryset.filter(created_by=user)
            queryset = queryset.select_related('created_by').order_by('-updated_at')
            reports = [view.serialize_report(report) for report in queryset]
            return view.success_response({'reports': reports})

        data = view.parse_json_body(request)
        title = view.validate_string(data.get('title'), min_length=3, max_length=200, field_name='title')
        cluster_ids = data.get('cluster_ids') or []
        if not isinstance(cluster_ids, list) or not cluster_ids:
            return view.validation_error_response('cluster_ids must be a non-empty list')

        description = data.get('description', '')
        is_shared = bool(data.get('is_shared', False))
        notes = data.get('notes') or {}

        report = view.service.create_report(
            owner=user,
            title=title,
            cluster_ids=cluster_ids,
            description=description,
            is_shared=is_shared,
            notes=notes,
        )

        return view.success_response(view.serialize_report(report), status=201)

    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['GET', 'PATCH', 'DELETE'])
def cluster_report_detail_api(request, report_id: str):
    view = CustomClusterAPIView()
    user = request.user

    try:
        report_uuid = view.parse_uuid(report_id)
        report = ClusterReport.objects.select_related('created_by').get(id=report_uuid)

        if request.method == 'GET':
            if report.created_by != user and not report.is_shared:
                raise PermissionDenied("Report is private")
            summary = view.service.summarize_report(report)
            summary.update(view.serialize_report(report))
            return view.success_response(summary)

        if request.method == 'PATCH':
            view.ensure_report_owner(report, user)
            data = view.parse_json_body(request)
            if 'title' in data:
                report.title = view.validate_string(data['title'], min_length=3, max_length=200, field_name='title')
            if 'description' in data:
                report.description = data.get('description') or ''
            if 'is_shared' in data:
                report.is_shared = bool(data['is_shared'])
            report.save(update_fields=['title', 'description', 'is_shared', 'updated_at'])
            summary = view.service.summarize_report(report)
            summary.update(view.serialize_report(report))
            return view.success_response(summary)

        # DELETE
        view.ensure_report_owner(report, user)
        report.delete()
        return view.success_response(message='Report deleted')

    except ClusterReport.DoesNotExist:
        return view.error_response('Report not found', status=404)
    except Exception as e:
        return view.handle_exception(request, e)
