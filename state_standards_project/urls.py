"""
URL configuration for state_standards_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from standards import admin_views
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularRedocView,
    SpectacularSwaggerView
)

urlpatterns = [
    # Admin bulk upload endpoints (must be before admin/ to avoid catch-all)
    path('admin/bulk-upload/', admin_views.bulk_upload_view, name='admin_bulk_upload'),
    path('admin/confirm-upload/', admin_views.confirm_upload_view, name='admin_confirm_upload'),
    path('admin/upload-status/<uuid:job_id>/', admin_views.upload_status_view, name='admin_upload_status'),
    path('admin/upload-status-api/<uuid:job_id>/', admin_views.upload_status_api, name='admin_upload_status_api'),
    path('admin/download-error-report/<uuid:job_id>/', admin_views.download_error_report, name='admin_download_error_report'),
    path('admin/generate-template/', admin_views.generate_template_view, name='admin_generate_template'),
    path('admin/cancel-upload/<uuid:job_id>/', admin_views.cancel_upload_job, name='admin_cancel_upload_job'),
    path('admin/restart-upload/<uuid:job_id>/', admin_views.restart_upload_job, name='admin_restart_upload_job'),
    
    # Admin correlation generation endpoints
    path('admin/generate-correlations/', admin_views.generate_correlations_view, name='admin_generate_correlations'),
    path('admin/confirm-correlation-generation/', admin_views.confirm_correlation_generation, name='admin_confirm_correlation_generation'),
    path('admin/correlation-status/<uuid:job_id>/', admin_views.correlation_status_view, name='admin_correlation_status'),
    path('admin/correlation-status-api/<uuid:job_id>/', admin_views.correlation_status_api, name='admin_correlation_status_api'),
    path('admin/correlation-analysis/', admin_views.correlation_analysis_view, name='admin_correlation_analysis'),
    path('admin/correlation-preview-api/', admin_views.correlation_preview_api, name='admin_correlation_preview_api'),
    path('admin/cancel-correlation/<uuid:job_id>/', admin_views.cancel_correlation_job, name='admin_cancel_correlation_job'),
    
    # Django admin
    path('admin/', admin.site.urls),
    
    # API Documentation (drf-spectacular)
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('api/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),
    
    # Dashboard app
    path('dashboard/', include('dashboard.urls', namespace='dashboard')),
    
    # Standards app APIs
    path('', include('standards.urls', namespace='standards')),
]
