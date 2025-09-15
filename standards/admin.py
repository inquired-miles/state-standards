import csv
import io
from datetime import datetime

from django.contrib import admin
from django.urls import path, reverse
from django.utils.html import format_html
from django.utils import timezone
from django.http import HttpResponseRedirect, HttpResponse
from .models import (
    State, SubjectArea, GradeLevel, Standard, StandardCorrelation,
    Concept, TopicCluster, ClusterMembership, CoverageAnalysis,
    ContentAlignment, ContentStandardMatch, StrategicPlan, CacheEntry,
    UploadJob, CorrelationJob, ProxyStandard, ProxyStateCoverage, StandardAtom,
    TopicBasedProxy, ProxyRun, ProxyRunReport
)
from .services import EmbeddingService
from .services.proxy_run_analyzer import ProxyRunAnalyzer


@admin.register(State)
class StateAdmin(admin.ModelAdmin):
    list_display = ('code', 'name', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('code', 'name')
    ordering = ('name',)


@admin.register(SubjectArea)
class SubjectAreaAdmin(admin.ModelAdmin):
    list_display = ('name', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('name', 'description')
    ordering = ('name',)


@admin.register(GradeLevel)
class GradeLevelAdmin(admin.ModelAdmin):
    list_display = ('grade', 'grade_numeric', 'created_at')
    list_filter = ('grade_numeric',)
    search_fields = ('grade',)
    ordering = ('grade_numeric',)


@admin.register(Standard)
class StandardAdmin(admin.ModelAdmin):
    list_display = ('code', 'display_name_admin', 'state', 'subject_area', 'domain', 'has_embedding', 'created_at')
    list_filter = ('state', 'subject_area', 'domain', 'created_at')
    search_fields = ('code', 'title', 'description', 'keywords', 'skills')
    filter_horizontal = ('grade_levels',)
    readonly_fields = ('id', 'created_at', 'updated_at')
    actions = ['generate_embeddings_action', 'export_to_csv_action', 'validate_data_action']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'state', 'subject_area', 'grade_levels')
        }),
        ('Standard Details', {
            'fields': ('code', 'title', 'description')
        }),
        ('Categorization', {
            'fields': ('domain', 'cluster')
        }),
        ('Vector & Metadata', {
            'fields': ('embedding', 'keywords', 'skills'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def get_queryset(self, request):
        """Optimize queryset to reduce database queries"""
        return super().get_queryset(request).select_related('state', 'subject_area').prefetch_related('grade_levels')
    
    def display_name_admin(self, obj):
        """Display name for admin interface - uses title if available, otherwise fallback"""
        return obj.display_name
    display_name_admin.short_description = 'Title/Name'
    
    def has_embedding(self, obj):
        """Display whether the standard has an embedding"""
        return obj.embedding is not None
    has_embedding.boolean = True
    has_embedding.short_description = 'Has Embedding'
    
    def changelist_view(self, request, extra_context=None):
        """Add bulk upload button to the changelist view"""
        extra_context = extra_context or {}
        extra_context['bulk_upload_url'] = reverse('admin_bulk_upload')
        return super().changelist_view(request, extra_context=extra_context)
    
    def generate_embeddings_action(self, request, queryset):
        """Admin action to generate embeddings for selected standards"""
        embedding_service = EmbeddingService()
        updated_count = 0
        
        for standard in queryset:
            embedding = embedding_service.generate_standard_embedding(standard)
            if embedding:
                standard.embedding = embedding
                standard.save(update_fields=['embedding'])
                updated_count += 1
        
        self.message_user(
            request,
            f"Generated embeddings for {updated_count} standards."
        )
    generate_embeddings_action.short_description = "Generate embeddings for selected standards"
    
    def export_to_csv_action(self, request, queryset):
        """Admin action to export selected standards to CSV"""
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="standards_export.csv"'
        
        writer = csv.writer(response)
        writer.writerow([
            'state', 'subject', 'grade', 'code', 'title', 'description',
            'domain', 'cluster', 'keywords', 'skills'
        ])
        
        for standard in queryset:
            grades = ','.join([g.grade for g in standard.grade_levels.all()])
            keywords = ','.join(standard.keywords) if standard.keywords else ''
            skills = ','.join(standard.skills) if standard.skills else ''
            
            writer.writerow([
                standard.state.code,
                standard.subject_area.name,
                grades,
                standard.code,
                standard.title,
                standard.description,
                standard.domain,
                standard.cluster,
                keywords,
                skills
            ])
        
        return response
    export_to_csv_action.short_description = "Export selected standards to CSV"
    
    def validate_data_action(self, request, queryset):
        """Admin action to validate selected standards"""
        issues = []
        
        for standard in queryset:
            standard_issues = []
            
            # Check required fields
            if not standard.code:
                standard_issues.append("Missing code")
            if not standard.title:
                standard_issues.append("Missing title")
            if not standard.description:
                standard_issues.append("Missing description")
            
            # Check state code format
            if standard.state and len(standard.state.code) != 2:
                standard_issues.append("Invalid state code format")
            
            # Check for duplicate codes within same state
            duplicates = Standard.objects.filter(
                state=standard.state,
                code=standard.code
            ).exclude(id=standard.id).count()
            if duplicates > 0:
                standard_issues.append("Duplicate code within state")
            
            if standard_issues:
                issues.append(f"{standard.code}: {', '.join(standard_issues)}")
        
        if issues:
            self.message_user(
                request,
                f"Found {len(issues)} issues: {'; '.join(issues[:5])}{'...' if len(issues) > 5 else ''}",
                level='WARNING'
            )
        else:
            self.message_user(request, "All selected standards validated successfully.")
    validate_data_action.short_description = "Validate selected standards"


@admin.register(StandardCorrelation)
class StandardCorrelationAdmin(admin.ModelAdmin):
    list_display = ('get_correlation_display', 'similarity_score', 'correlation_type', 'verified', 'created_at')
    list_filter = ('correlation_type', 'verified', 'similarity_score', 'created_at')
    search_fields = ('standard_1__code', 'standard_1__title', 'standard_2__code', 'standard_2__title', 'notes')
    readonly_fields = ('id', 'get_standard_1_details', 'get_standard_2_details', 'created_at', 'updated_at')
    raw_id_fields = ('standard_1', 'standard_2')
    actions = ['verify_correlations', 'unverify_correlations']
    
    fieldsets = (
        ('Correlation Information', {
            'fields': ('id', 'standard_1', 'standard_2')
        }),
        ('Standard 1 Details', {
            'fields': ('get_standard_1_details',),
            'classes': ('wide',)
        }),
        ('Standard 2 Details', {
            'fields': ('get_standard_2_details',),
            'classes': ('wide',)
        }),
        ('Metrics', {
            'fields': ('similarity_score', 'correlation_type', 'verified')
        }),
        ('Additional Information', {
            'fields': ('notes',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def get_correlation_display(self, obj):
        """Display correlation in a readable format"""
        return f"{obj.standard_1.state.code} {obj.standard_1.code} ↔ {obj.standard_2.state.code} {obj.standard_2.code}"
    get_correlation_display.short_description = 'Correlation'
    
    def get_standard_1_details(self, obj):
        """Display detailed information for standard 1"""
        if not obj.standard_1:
            return "No standard selected"
        
        std = obj.standard_1
        details = []
        
        # Header with state and code
        details.append(f"<h3 style='color: #0066cc; margin: 0 0 10px 0;'>{std.state.code} - {std.code}</h3>")
        
        # Title
        if std.title:
            details.append(f"<p><strong>Title:</strong> {std.title}</p>")
        
        # Description
        if std.description:
            details.append(f"<p><strong>Description:</strong> {std.description}</p>")
        
        # Subject and grade
        subject = std.subject_area.name if std.subject_area else "N/A"
        grades = ", ".join([gl.grade for gl in std.grade_levels.all()]) if std.grade_levels.exists() else "N/A"
        details.append(f"<p><strong>Subject:</strong> {subject} | <strong>Grade(s):</strong> {grades}</p>")
        
        # Domain and cluster
        if std.domain or std.cluster:
            domain = std.domain or "N/A"
            cluster = std.cluster or "N/A"
            details.append(f"<p><strong>Domain:</strong> {domain} | <strong>Cluster:</strong> {cluster}</p>")
        
        # Keywords and skills
        if std.keywords:
            keywords = ", ".join(std.keywords)
            details.append(f"<p><strong>Keywords:</strong> {keywords}</p>")
        
        if std.skills:
            skills = ", ".join(std.skills)
            details.append(f"<p><strong>Skills:</strong> {skills}</p>")
        
        return format_html("".join(details))
    get_standard_1_details.short_description = "Standard 1 Details"
    
    def get_standard_2_details(self, obj):
        """Display detailed information for standard 2"""
        if not obj.standard_2:
            return "No standard selected"
        
        std = obj.standard_2
        details = []
        
        # Header with state and code
        details.append(f"<h3 style='color: #0066cc; margin: 0 0 10px 0;'>{std.state.code} - {std.code}</h3>")
        
        # Title
        if std.title:
            details.append(f"<p><strong>Title:</strong> {std.title}</p>")
        
        # Description
        if std.description:
            details.append(f"<p><strong>Description:</strong> {std.description}</p>")
        
        # Subject and grade
        subject = std.subject_area.name if std.subject_area else "N/A"
        grades = ", ".join([gl.grade for gl in std.grade_levels.all()]) if std.grade_levels.exists() else "N/A"
        details.append(f"<p><strong>Subject:</strong> {subject} | <strong>Grade(s):</strong> {grades}</p>")
        
        # Domain and cluster
        if std.domain or std.cluster:
            domain = std.domain or "N/A"
            cluster = std.cluster or "N/A"
            details.append(f"<p><strong>Domain:</strong> {domain} | <strong>Cluster:</strong> {cluster}</p>")
        
        # Keywords and skills
        if std.keywords:
            keywords = ", ".join(std.keywords)
            details.append(f"<p><strong>Keywords:</strong> {keywords}</p>")
        
        if std.skills:
            skills = ", ".join(std.skills)
            details.append(f"<p><strong>Skills:</strong> {skills}</p>")
        
        return format_html("".join(details))
    get_standard_2_details.short_description = "Standard 2 Details"
    
    def changelist_view(self, request, extra_context=None):
        """Add correlation generation button to the changelist view"""
        extra_context = extra_context or {}
        extra_context['generate_correlations_url'] = reverse('admin_generate_correlations')
        extra_context['correlation_analysis_url'] = reverse('admin_correlation_analysis')
        return super().changelist_view(request, extra_context=extra_context)
    
    def verify_correlations(self, request, queryset):
        """Admin action to mark correlations as verified"""
        updated = queryset.update(verified=True)
        self.message_user(
            request,
            f"Successfully verified {updated} correlations."
        )
    verify_correlations.short_description = "Mark selected correlations as verified"
    
    def unverify_correlations(self, request, queryset):
        """Admin action to mark correlations as unverified"""
        updated = queryset.update(verified=False)
        self.message_user(
            request,
            f"Successfully unverified {updated} correlations."
        )
    unverify_correlations.short_description = "Mark selected correlations as unverified"
    
    def get_queryset(self, request):
        """Optimize queryset to reduce database queries"""
        return super().get_queryset(request).select_related(
            'standard_1__state', 
            'standard_1__subject_area',
            'standard_2__state', 
            'standard_2__subject_area'
        )


@admin.register(Concept)
class ConceptAdmin(admin.ModelAdmin):
    list_display = ('name', 'states_covered', 'coverage_percentage', 'complexity_score', 'created_at')
    list_filter = ('subject_areas', 'grade_levels', 'states_covered', 'complexity_score')
    search_fields = ('name', 'description', 'keywords')
    filter_horizontal = ('grade_levels', 'subject_areas')
    readonly_fields = ('id', 'states_covered', 'coverage_percentage', 'created_at', 'updated_at')
    
    fieldsets = (
        ('Concept Information', {
            'fields': ('id', 'name', 'description', 'keywords')
        }),
        ('Classification', {
            'fields': ('subject_areas', 'grade_levels', 'complexity_score')
        }),
        ('Coverage Statistics', {
            'fields': ('states_covered', 'coverage_percentage'),
            'classes': ('collapse',)
        }),
        ('Vector Data', {
            'fields': ('embedding',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(TopicCluster)
class TopicClusterAdmin(admin.ModelAdmin):
    list_display = ('name', 'subject_area', 'standards_count', 'states_represented', 'silhouette_score', 'created_at')
    list_filter = ('subject_area', 'grade_levels', 'silhouette_score', 'states_represented')
    search_fields = ('name', 'description', 'common_terms')
    filter_horizontal = ('grade_levels',)
    readonly_fields = ('id', 'standards_count', 'states_represented', 'created_at', 'updated_at')
    
    fieldsets = (
        ('Cluster Information', {
            'fields': ('id', 'name', 'description', 'subject_area', 'grade_levels')
        }),
        ('Quality Metrics', {
            'fields': ('silhouette_score', 'cohesion_score', 'standards_count', 'states_represented')
        }),
        ('Analysis Results', {
            'fields': ('common_terms', 'regional_pattern'),
            'classes': ('collapse',)
        }),
        ('Vector Data', {
            'fields': ('embedding',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


class ClusterMembershipInline(admin.TabularInline):
    model = ClusterMembership
    extra = 0
    readonly_fields = ('created_at',)


@admin.register(CoverageAnalysis)
class CoverageAnalysisAdmin(admin.ModelAdmin):
    list_display = ('get_analysis_display', 'analysis_type', 'coverage_percentage', 'total_standards', 'created_at')
    list_filter = ('analysis_type', 'state', 'subject_area', 'grade_level', 'coverage_percentage')
    search_fields = ('state__name', 'subject_area__name', 'grade_level__grade')
    readonly_fields = ('id', 'created_at', 'updated_at')
    
    fieldsets = (
        ('Analysis Parameters', {
            'fields': ('id', 'analysis_type', 'state', 'subject_area', 'grade_level')
        }),
        ('Results', {
            'fields': ('total_standards', 'covered_concepts', 'coverage_percentage')
        }),
        ('Analysis Data', {
            'fields': ('bell_curve_data', 'gap_analysis', 'benchmark_comparison'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def get_analysis_display(self, obj):
        """Display analysis scope in a readable format"""
        scope = obj.state.name if obj.state else "All States"
        grade = f" - Grade {obj.grade_level.grade}" if obj.grade_level else ""
        return f"{scope} - {obj.subject_area.name}{grade}"
    get_analysis_display.short_description = 'Analysis Scope'


@admin.register(ContentAlignment)
class ContentAlignmentAdmin(admin.ModelAdmin):
    list_display = ('content_title', 'file_type', 'overall_alignment_score', 'exact_matches', 'semantic_matches', 'created_at')
    list_filter = ('file_type', 'overall_alignment_score', 'created_at')
    search_fields = ('content_title', 'original_filename', 'content_text')
    readonly_fields = ('id', 'content_hash', 'total_standards_analyzed', 'created_at', 'updated_at')
    filter_horizontal = ('states_with_full_alignment', 'states_with_partial_alignment')
    
    fieldsets = (
        ('Content Information', {
            'fields': ('id', 'content_title', 'original_filename', 'file_type', 'content_hash')
        }),
        ('Content Text', {
            'fields': ('content_text',),
            'classes': ('collapse',)
        }),
        ('Alignment Results', {
            'fields': ('overall_alignment_score', 'total_standards_analyzed', 'exact_matches', 'semantic_matches', 'conceptual_matches')
        }),
        ('State Coverage', {
            'fields': ('states_with_full_alignment', 'states_with_partial_alignment'),
            'classes': ('collapse',)
        }),
        ('Analysis Data', {
            'fields': ('alignment_report', 'improvement_suggestions', 'gap_analysis'),
            'classes': ('collapse',)
        }),
        ('Vector Data', {
            'fields': ('content_embedding',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


class ContentStandardMatchInline(admin.TabularInline):
    model = ContentStandardMatch
    extra = 0
    readonly_fields = ('created_at',)
    fields = ('standard', 'match_type', 'confidence_score', 'similarity_score', 'created_at')


@admin.register(StrategicPlan)
class StrategicPlanAdmin(admin.ModelAdmin):
    list_display = ('name', 'status', 'target_coverage_percentage', 'implementation_difficulty', 'estimated_completion_months', 'created_at')
    list_filter = ('status', 'implementation_difficulty', 'target_coverage_percentage', 'created_at')
    search_fields = ('name', 'description', 'created_by')
    filter_horizontal = ('target_states', 'target_subjects', 'target_grades', 'high_priority_concepts')
    readonly_fields = ('id', 'created_at', 'updated_at')
    
    fieldsets = (
        ('Plan Information', {
            'fields': ('id', 'name', 'description', 'created_by', 'status')
        }),
        ('Target Scope', {
            'fields': ('target_states', 'target_subjects', 'target_grades', 'target_coverage_percentage')
        }),
        ('Analysis Results', {
            'fields': ('mvc_data', 'priority_matrix', 'roi_analysis'),
            'classes': ('collapse',)
        }),
        ('Implementation', {
            'fields': ('implementation_difficulty', 'estimated_completion_months', 'timeline_data', 'risk_factors')
        }),
        ('Financial', {
            'fields': ('estimated_development_cost', 'projected_market_reach'),
            'classes': ('collapse',)
        }),
        ('Strategic Concepts', {
            'fields': ('high_priority_concepts',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(CacheEntry)
class CacheEntryAdmin(admin.ModelAdmin):
    list_display = ('cache_key', 'cache_type', 'computation_time', 'expires_at', 'is_expired_display', 'created_at')
    list_filter = ('cache_type', 'expires_at', 'created_at')
    search_fields = ('cache_key', 'parameters_hash')
    readonly_fields = ('created_at', 'is_expired_display')
    
    fieldsets = (
        ('Cache Information', {
            'fields': ('cache_key', 'cache_type', 'parameters_hash')
        }),
        ('Performance', {
            'fields': ('computation_time', 'expires_at', 'is_expired_display')
        }),
        ('Data', {
            'fields': ('cache_value',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )
    
    def is_expired_display(self, obj):
        """Display cache expiry status"""
        return "Yes" if obj.is_expired else "No"
    is_expired_display.short_description = 'Expired'
    is_expired_display.boolean = True
    
    actions = ['delete_expired_entries']
    
    def delete_expired_entries(self, request, queryset):
        """Admin action to delete expired cache entries"""
        expired_count = queryset.filter(expires_at__lt=timezone.now()).count()
        queryset.filter(expires_at__lt=timezone.now()).delete()
        self.message_user(request, f"Deleted {expired_count} expired cache entries.")
    delete_expired_entries.short_description = "Delete expired cache entries"


@admin.register(UploadJob)
class UploadJobAdmin(admin.ModelAdmin):
    list_display = ('original_filename', 'status', 'progress_percentage', 'uploaded_by', 'total_records', 'successful_records', 'failed_records', 'created_at')
    list_filter = ('status', 'file_type', 'clear_existing', 'generate_embeddings', 'created_at')
    search_fields = ('original_filename', 'uploaded_by')
    readonly_fields = ('id', 'file_size', 'total_records', 'processed_records', 'successful_records', 'failed_records', 'progress_percentage', 'error_log', 'processing_summary', 'created_at', 'updated_at')
    
    fieldsets = (
        ('File Information', {
            'fields': ('id', 'original_filename', 'file_size', 'file_type', 'uploaded_by')
        }),
        ('Configuration', {
            'fields': ('clear_existing', 'generate_embeddings', 'batch_size')
        }),
        ('Status & Progress', {
            'fields': ('status', 'progress_percentage', 'total_records', 'processed_records', 'successful_records', 'failed_records')
        }),
        ('Timing', {
            'fields': ('started_at', 'completed_at', 'estimated_completion')
        }),
        ('Results', {
            'fields': ('processing_summary', 'error_log'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    actions = ['cancel_active_jobs', 'delete_failed_jobs']
    
    def cancel_active_jobs(self, request, queryset):
        """Cancel selected active jobs"""
        active_jobs = queryset.filter(status__in=['pending', 'validating', 'processing', 'generating_embeddings'])
        count = active_jobs.count()
        
        active_jobs.update(
            status='cancelled',
            completed_at=timezone.now()
        )
        
        self.message_user(request, f"Cancelled {count} active jobs.")
    cancel_active_jobs.short_description = "Cancel selected active jobs"
    
    def delete_failed_jobs(self, request, queryset):
        """Delete failed and cancelled jobs"""
        failed_jobs = queryset.filter(status__in=['failed', 'cancelled'])
        count = failed_jobs.count()
        failed_jobs.delete()
        
        self.message_user(request, f"Deleted {count} failed/cancelled jobs.")
    delete_failed_jobs.short_description = "Delete failed and cancelled jobs"
    
    def get_queryset(self, request):
        """Optimize queryset"""
        return super().get_queryset(request)


@admin.register(CorrelationJob)
class CorrelationJobAdmin(admin.ModelAdmin):
    list_display = ('get_job_display', 'similarity_threshold', 'status', 'progress_percentage', 'correlations_created', 'created_by', 'created_at')
    list_filter = ('status', 'similarity_threshold', 'subject_area_filter', 'clear_existing', 'created_at')
    search_fields = ('created_by', 'subject_area_filter__name')
    readonly_fields = ('id', 'total_standards', 'processed_standards', 'correlations_created', 'correlations_updated', 'correlations_skipped', 'progress_percentage', 'avg_processing_time_per_standard', 'estimated_completion', 'error_log', 'processing_summary', 'statistics', 'created_at', 'updated_at')
    filter_horizontal = ('state_filter',)
    
    fieldsets = (
        ('Job Configuration', {
            'fields': ('id', 'similarity_threshold', 'clear_existing', 'batch_size', 'created_by')
        }),
        ('Filtering Options', {
            'fields': ('subject_area_filter', 'state_filter'),
            'classes': ('collapse',)
        }),
        ('Status & Progress', {
            'fields': ('status', 'progress_percentage', 'total_standards', 'processed_standards')
        }),
        ('Results', {
            'fields': ('correlations_created', 'correlations_updated', 'correlations_skipped')
        }),
        ('Performance', {
            'fields': ('avg_processing_time_per_standard', 'estimated_completion'),
            'classes': ('collapse',)
        }),
        ('Timing', {
            'fields': ('started_at', 'completed_at')
        }),
        ('Analysis Results', {
            'fields': ('processing_summary', 'statistics', 'error_log'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    actions = ['cancel_active_jobs', 'delete_completed_jobs']
    
    def get_job_display(self, obj):
        """Display job scope and configuration"""
        scope = obj.subject_area_filter.name if obj.subject_area_filter else "All Subjects"
        state_count = obj.state_filter.count()
        state_info = f" ({state_count} states)" if state_count > 0 else " (All states)"
        return f"{scope}{state_info}"
    get_job_display.short_description = 'Job Scope'
    
    def cancel_active_jobs(self, request, queryset):
        """Cancel selected active jobs"""
        active_jobs = queryset.filter(status__in=['pending', 'analyzing', 'generating'])
        count = active_jobs.count()
        
        active_jobs.update(
            status='cancelled',
            completed_at=timezone.now()
        )
        
        self.message_user(request, f"Cancelled {count} active correlation jobs.")
    cancel_active_jobs.short_description = "Cancel selected active jobs"
    
    def delete_completed_jobs(self, request, queryset):
        """Delete completed and failed jobs"""
        completed_jobs = queryset.filter(status__in=['completed', 'failed', 'cancelled'])
        count = completed_jobs.count()
        completed_jobs.delete()
        
        self.message_user(request, f"Deleted {count} completed/failed jobs.")
    delete_completed_jobs.short_description = "Delete completed and failed jobs"
    
    def get_queryset(self, request):
        """Optimize queryset"""
        return super().get_queryset(request).select_related('subject_area_filter').prefetch_related('state_filter')


@admin.register(StandardAtom)
class StandardAtomAdmin(admin.ModelAdmin):
    list_display = ('atom_code', 'standard', 'text_preview', 'method', 'has_embedding', 'created_at')
    list_filter = ('method', 'created_at', 'standard__state', 'standard__subject_area')
    search_fields = ('atom_code', 'text', 'standard__code', 'standard__state__code')
    readonly_fields = ('atom_code', 'created_at')
    raw_id_fields = ('standard',)
    
    fieldsets = (
        ('Atom Info', {
            'fields': ('atom_code', 'standard', 'text', 'method')
        }),
        ('Embedding', {
            'fields': ('embedding', 'embedding_generated_at'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )
    
    def text_preview(self, obj):
        """Show first 100 chars of text"""
        return obj.text[:100] + "..." if len(obj.text) > 100 else obj.text
    text_preview.short_description = 'Text Preview'
    
    def has_embedding(self, obj):
        """Show if atom has embedding"""
        return obj.embedding is not None
    has_embedding.boolean = True
    has_embedding.short_description = 'Has Embedding'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('standard__state', 'standard__subject_area')


@admin.register(ProxyStandard)
class ProxyStandardAdmin(admin.ModelAdmin):
    list_display = ('proxy_id', 'title_short', 'source_type', 'grade_range_display', 'member_count_display', 'coverage_count', 'avg_similarity', 'medoid_preview', 'created_at')
    search_fields = ('proxy_id', 'title', 'description', 'member_atoms__atom_code', 'member_standards__code')
    list_filter = ('source_type', 'coverage_count', 'created_at', 'min_grade', 'max_grade', 'grade_levels')
    readonly_fields = ('proxy_id', 'cluster_id', 'created_at', 'grade_range_display', 'member_count_display', 'source_type_display')
    filter_horizontal = ('member_atoms', 'member_standards', 'grade_levels')
    raw_id_fields = ('medoid_atom', 'medoid_standard')
    actions = ['update_grade_ranges', 'export_to_csv']

    fieldsets = (
        ('Proxy Info', {
            'fields': ('proxy_id', 'title', 'description', 'source_type_display', 'cluster_id')
        }),
        ('Source Objects', {
            'fields': ('medoid_atom', 'medoid_standard'),
            'description': 'Medoid atom is used for atom-level clustering, medoid standard for standard-level clustering'
        }),
        ('Grade Information', {
            'fields': ('grade_range_display', 'min_grade', 'max_grade', 'grade_levels')
        }),
        ('Members', {
            'fields': ('member_atoms', 'member_standards', 'member_count_display'),
            'description': 'Either member_atoms OR member_standards will be populated based on source_type'
        }),
        ('Metrics', {
            'fields': ('coverage_count', 'avg_similarity', 'centroid_embedding')
        }),
        ('Timestamps', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related(
            'medoid_atom', 'medoid_standard'
        ).prefetch_related(
            'member_atoms', 'member_standards', 'grade_levels'
        )
    
    def title_short(self, obj):
        """Show truncated title"""
        return obj.title[:50] + '...' if len(obj.title) > 50 else obj.title
    title_short.short_description = 'Title'
    
    def medoid_preview(self, obj):
        """Show preview of medoid atom or standard"""
        if obj.source_type == 'atoms' and obj.medoid_atom:
            return f"[ATOM] {obj.medoid_atom.atom_code}: {obj.medoid_atom.text[:50]}..."
        elif obj.source_type == 'standards' and obj.medoid_standard:
            return f"[STD] {obj.medoid_standard.code}: {obj.medoid_standard.title[:50]}..."
        return "No medoid"
    medoid_preview.short_description = 'Medoid Preview'
    
    def member_count_display(self, obj):
        """Show count of member items"""
        return obj.member_count
    member_count_display.short_description = 'Members'
    
    def source_type_display(self, obj):
        """Show human-readable source type"""
        return obj.get_source_type_display()
    source_type_display.short_description = 'Source Type'
    
    def update_grade_ranges(self, request, queryset):
        """Update grade ranges for selected proxy standards"""
        updated_count = 0
        for proxy in queryset:
            proxy.update_grade_range_from_members()
            proxy.save()
            updated_count += 1
        
        self.message_user(
            request,
            f'Successfully updated grade ranges for {updated_count} proxy standards.'
        )
    update_grade_ranges.short_description = 'Update grade ranges from members'
    
    def export_to_csv(self, request, queryset):
        """Export selected proxy standards to CSV"""
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="proxy_standards.csv"'
        
        writer = csv.writer(response)
        writer.writerow([
            'Proxy ID', 'Title', 'Source Type', 'Grade Range', 'Min Grade', 'Max Grade',
            'Member Count', 'Coverage Count', 'Avg Similarity', 'Created At'
        ])
        
        for proxy in queryset:
            writer.writerow([
                proxy.proxy_id,
                proxy.title,
                proxy.get_source_type_display(),
                proxy.grade_range_display,
                proxy.min_grade,
                proxy.max_grade,
                proxy.member_count,
                proxy.coverage_count,
                proxy.avg_similarity,
                proxy.created_at.strftime('%Y-%m-%d %H:%M:%S')
            ])
        
        return response
    export_to_csv.short_description = 'Export selected to CSV'


@admin.register(ProxyStateCoverage)
class ProxyStateCoverageAdmin(admin.ModelAdmin):
    list_display = ('proxy', 'state', 'atom_count', 'avg_similarity')
    list_filter = ('state', 'atom_count')
    search_fields = ('proxy__proxy_id', 'state__code', 'state__name')
    raw_id_fields = ('proxy',)
    
    fieldsets = (
        ('Coverage Info', {
            'fields': ('proxy', 'state', 'atom_count', 'avg_similarity')
        }),
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('proxy', 'state')


@admin.register(TopicBasedProxy)
class TopicBasedProxyAdmin(admin.ModelAdmin):
    list_display = (
        'proxy_id', 'short_hierarchy_display', 'standards_count', 
        'states_covered', 'grade_range_display', 'outlier_category', 'created_at'
    )
    list_filter = (
        'topic', 'sub_topic', 'outlier_category', 'filter_subject_area',
        'min_grade', 'max_grade', 'created_at'
    )
    search_fields = (
        'proxy_id', 'topic', 'sub_topic', 'sub_sub_topic', 
        'title', 'description', 'member_standards__code'
    )
    readonly_fields = (
        'proxy_id', 'standards_count', 'states_covered', 'subject_areas',
        'min_grade', 'max_grade', 'filter_grade_levels', 'filter_subject_area', 
        'filter_criteria', 'filter_criteria_display_readonly', 'created_at', 'updated_at'
    )
    filter_horizontal = ('member_standards', 'grade_levels')
    
    fieldsets = (
        ('Identification', {
            'fields': ('proxy_id', 'outlier_category', 'created_at', 'updated_at')
        }),
        ('Topic Hierarchy', {
            'fields': ('topic', 'sub_topic', 'sub_sub_topic')
        }),
        ('Original Filter Criteria', {
            'fields': ('filter_criteria_display_readonly', 'filter_grade_levels', 'filter_subject_area', 'filter_criteria'),
            'description': 'These fields show the original filtering criteria used when this proxy was created.',
        }),
        ('Generated Content', {
            'fields': ('title', 'description'),
            'classes': ('collapse',)
        }),
        ('Member Standards', {
            'fields': ('member_standards',),
            'classes': ('collapse',)
        }),
        ('Derived Metadata (from Member Standards)', {
            'fields': (
                'standards_count', 'states_covered', 'subject_areas',
                'min_grade', 'max_grade', 'grade_levels'
            ),
            'description': 'These fields are automatically calculated from the member standards.',
            'classes': ('collapse',)
        }),
    )
    
    actions = ['export_to_csv', 'update_derived_fields', 'regenerate_descriptions']
    
    def short_hierarchy_display(self, obj):
        """Display abbreviated hierarchy for list view"""
        if obj.outlier_category:
            return f"🔸 Outlier: {obj.sub_sub_topic[:50]}..."
        return obj.short_hierarchy
    short_hierarchy_display.short_description = 'Topic Hierarchy'
    
    def grade_range_display(self, obj):
        """Display grade range"""
        if obj.min_grade is not None and obj.max_grade is not None:
            if obj.min_grade == obj.max_grade:
                grade_display = f"Grade {obj.min_grade}" if obj.min_grade > 0 else "Kindergarten"
            else:
                min_display = f"Grade {obj.min_grade}" if obj.min_grade > 0 else "K"
                max_display = f"Grade {obj.max_grade}" if obj.max_grade > 0 else "K"
                grade_display = f"{min_display}-{max_display}"
            return grade_display
        return "No grades"
    grade_range_display.short_description = 'Grade Range'
    
    def filter_criteria_display_readonly(self, obj):
        """Display original filter criteria in readable format"""
        return obj.filter_criteria_display
    filter_criteria_display_readonly.short_description = 'Original Selection Summary'
    
    def export_to_csv(self, request, queryset):
        """Export selected topic proxies to CSV"""
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="topic_proxies_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"'
        
        writer = csv.writer(response)
        writer.writerow([
            'Proxy ID', 'Topic', 'Sub-topic', 'Sub-sub-topic', 'Title', 'Description',
            'Standards Count', 'States Covered', 'Subject Areas', 'Min Grade', 'Max Grade',
            'Outlier Category', 'Created At'
        ])
        
        for proxy in queryset.select_related().prefetch_related('member_standards'):
            subject_areas_str = ', '.join(proxy.subject_areas) if proxy.subject_areas else ''
            writer.writerow([
                proxy.proxy_id,
                proxy.topic,
                proxy.sub_topic,
                proxy.sub_sub_topic,
                proxy.title,
                proxy.description,
                proxy.standards_count,
                proxy.states_covered,
                subject_areas_str,
                proxy.min_grade,
                proxy.max_grade,
                proxy.outlier_category,
                proxy.created_at.strftime('%Y-%m-%d %H:%M:%S') if proxy.created_at else ''
            ])
        
        return response
    export_to_csv.short_description = "Export selected topic proxies to CSV"
    
    def update_derived_fields(self, request, queryset):
        """Update derived fields for selected proxies"""
        count = 0
        for proxy in queryset:
            proxy.update_derived_fields()
            count += 1
        
        self.message_user(
            request, 
            f"Successfully updated derived fields for {count} topic-based proxies."
        )
    update_derived_fields.short_description = "Update derived fields (counts, grades, etc.)"
    
    def regenerate_descriptions(self, request, queryset):
        """Regenerate descriptions for selected proxies"""
        count = 0
        for proxy in queryset:
            # Force regeneration of description
            proxy.description = ""
            proxy.save()
            count += 1
        
        self.message_user(
            request,
            f"Successfully regenerated descriptions for {count} topic-based proxies."
        )
    regenerate_descriptions.short_description = "Regenerate hierarchical descriptions"
    
    def get_queryset(self, request):
        return super().get_queryset(request).prefetch_related('member_standards', 'grade_levels')


@admin.register(ProxyRun)
class ProxyRunAdmin(admin.ModelAdmin):
    list_display = (
        'name', 'run_type', 'status', 'total_proxies_created', 
        'coverage_percentage', 'duration_display', 'started_at'
    )
    list_filter = (
        'run_type', 'status', 'started_at', 'total_proxies_created'
    )
    search_fields = ('name', 'run_id', 'description', 'created_by')
    readonly_fields = (
        'run_id', 'started_at', 'completed_at', 'duration_seconds', 
        'total_input_standards', 'total_proxies_created', 'outlier_proxies_count',
        'coverage_percentage', 'job_id'
    )
    
    fieldsets = (
        ('Run Information', {
            'fields': ('run_id', 'name', 'description', 'run_type', 'status')
        }),
        ('Parameters', {
            'fields': ('filter_parameters', 'algorithm_parameters'),
            'classes': ('collapse',)
        }),
        ('Timing', {
            'fields': ('started_at', 'completed_at', 'duration_seconds')
        }),
        ('Results', {
            'fields': (
                'total_input_standards', 'total_proxies_created', 
                'outlier_proxies_count', 'coverage_percentage'
            )
        }),
        ('Metadata', {
            'fields': ('created_by', 'job_id'),
            'classes': ('collapse',)
        }),
    )
    
    actions = ['generate_reports', 'recalculate_coverage', 'export_run_summary']
    
    def generate_reports(self, request, queryset):
        """Generate analysis reports for selected runs"""
        analyzer = ProxyRunAnalyzer()
        count = 0
        
        for run in queryset.filter(status='completed'):
            try:
                analyzer.analyze_run(run, force_regenerate=True)
                count += 1
            except Exception as e:
                self.message_user(request, f"Failed to generate report for {run.name}: {e}", level='ERROR')
        
        self.message_user(request, f"Successfully generated reports for {count} runs.")
    generate_reports.short_description = "Generate analysis reports"
    
    def recalculate_coverage(self, request, queryset):
        """Recalculate coverage statistics for selected runs"""
        count = 0
        for run in queryset.filter(status='completed'):
            try:
                run.calculate_coverage_stats()
                count += 1
            except Exception as e:
                self.message_user(request, f"Failed to recalculate coverage for {run.name}: {e}", level='ERROR')
        
        self.message_user(request, f"Successfully recalculated coverage for {count} runs.")
    recalculate_coverage.short_description = "Recalculate coverage statistics"
    
    def export_run_summary(self, request, queryset):
        """Export run summary to CSV"""
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="proxy_runs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"'
        
        writer = csv.writer(response)
        writer.writerow([
            'Run ID', 'Name', 'Type', 'Status', 'Started', 'Duration (s)', 
            'Input Standards', 'Proxies Created', 'Outliers', 'Coverage %',
            'Filter Summary', 'Algorithm Summary'
        ])
        
        for run in queryset:
            writer.writerow([
                run.run_id,
                run.name,
                run.get_run_type_display(),
                run.get_status_display(),
                run.started_at.strftime('%Y-%m-%d %H:%M:%S') if run.started_at else '',
                run.duration_seconds or '',
                run.total_input_standards,
                run.total_proxies_created,
                run.outlier_proxies_count,
                run.coverage_percentage or '',
                run.filter_summary,
                run.algorithm_summary
            ])
        
        return response
    export_run_summary.short_description = "Export run summary to CSV"
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related()


@admin.register(ProxyRunReport)
class ProxyRunReportAdmin(admin.ModelAdmin):
    list_display = (
        'run_name', 'run_type', 'generated_at', 'generation_time_display',
        'topics_count', 'states_covered'
    )
    list_filter = ('generated_at', 'run__run_type')
    search_fields = ('run__name', 'run__run_id')
    readonly_fields = (
        'run', 'generated_at', 'generation_time_seconds',
        'state_coverage', 'topic_prevalence', 'coverage_distribution',
        'outlier_analysis', 'topic_hierarchy_stats', 'cross_state_commonality',
        'silhouette_scores', 'coverage_gaps'
    )
    
    fieldsets = (
        ('Report Information', {
            'fields': ('run', 'generated_at', 'generation_time_seconds')
        }),
        ('State Analysis', {
            'fields': ('state_coverage',),
            'classes': ('collapse',)
        }),
        ('Topic Analysis', {
            'fields': ('topic_prevalence', 'topic_hierarchy_stats', 'cross_state_commonality'),
            'classes': ('collapse',)
        }),
        ('Coverage Analysis', {
            'fields': ('coverage_distribution', 'coverage_gaps'),
            'classes': ('collapse',)
        }),
        ('Quality Metrics', {
            'fields': ('outlier_analysis', 'silhouette_scores'),
            'classes': ('collapse',)
        }),
    )
    
    def run_name(self, obj):
        return obj.run.name
    run_name.short_description = 'Run Name'
    run_name.admin_order_field = 'run__name'
    
    def run_type(self, obj):
        return obj.run.get_run_type_display()
    run_type.short_description = 'Run Type'
    run_type.admin_order_field = 'run__run_type'
    
    def generation_time_display(self, obj):
        if obj.generation_time_seconds:
            return f"{obj.generation_time_seconds:.2f}s"
        return "Unknown"
    generation_time_display.short_description = 'Generation Time'
    
    def topics_count(self, obj):
        if obj.topic_prevalence:
            return len(obj.topic_prevalence)
        return 0
    topics_count.short_description = 'Topics'
    
    def states_covered(self, obj):
        if obj.state_coverage:
            return len(obj.state_coverage)
        return 0
    states_covered.short_description = 'States'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('run')


# Note: Custom admin URLs are added through the main URL configuration
# to avoid replacing the default admin site and losing model registrations
