import csv
import json
import os
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib import messages
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.utils import timezone
from django.urls import reverse
from django.db import transaction
import pandas as pd
from .models import BulkUpload, Standard, State, SubjectArea, GradeLevel
from .tasks import process_bulk_upload_task, generate_embeddings_batch_task


@staff_member_required
def bulk_upload_view(request):
    """Main bulk upload view for file upload and initial processing"""
    if request.method == 'POST':
        file = request.FILES.get('file')
        if not file:
            messages.error(request, 'No file provided')
            return redirect('admin_bulk_upload')
        
        # Validate file type
        file_extension = file.name.split('.')[-1].lower()
        if file_extension not in ['csv', 'json', 'xlsx']:
            messages.error(request, f'Unsupported file type: {file_extension}')
            return redirect('admin_bulk_upload')
        
        # Create BulkUpload instance
        bulk_upload = BulkUpload.objects.create(
            file=file,
            original_filename=file.name,
            file_size=file.size,
            file_type=file_extension,
            uploaded_by=request.user.username,
            clear_existing=request.POST.get('clear_existing', False) == 'on',
            generate_embeddings=request.POST.get('generate_embeddings', False) == 'on',
            batch_size=int(request.POST.get('batch_size', 100)),
            status='uploading'
        )
        
        # Process file for preview
        try:
            preview_data = _process_file_preview(bulk_upload)
            bulk_upload.preview_data = preview_data[:10]  # Store first 10 records for preview
            bulk_upload.total_records = len(preview_data)
            bulk_upload.status = 'preview'
            bulk_upload.save()
            
            return redirect('admin_confirm_upload', upload_id=bulk_upload.id)
            
        except Exception as e:
            bulk_upload.status = 'failed'
            bulk_upload.add_error(f'File processing error: {str(e)}')
            bulk_upload.save()
            messages.error(request, f'Error processing file: {str(e)}')
            return redirect('admin_bulk_upload')
    
    # GET request - show upload form
    recent_uploads = BulkUpload.objects.filter(uploaded_by=request.user.username)[:5]
    context = {
        'recent_uploads': recent_uploads,
        'title': 'Bulk Upload Standards',
    }
    return render(request, 'admin/standards/bulk_upload.html', context)


@staff_member_required
def confirm_upload_view(request, upload_id):
    """Preview and confirm upload data before processing"""
    bulk_upload = get_object_or_404(BulkUpload, id=upload_id)
    
    if request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'confirm':
            # Update column mapping if provided
            column_mapping = {}
            for key in request.POST:
                if key.startswith('map_'):
                    original_col = key.replace('map_', '')
                    mapped_col = request.POST[key]
                    if mapped_col:
                        column_mapping[original_col] = mapped_col
            
            bulk_upload.column_mapping = column_mapping
            bulk_upload.status = 'confirmed'
            bulk_upload.save()
            
            # Launch async processing task
            from .tasks import process_bulk_upload_task
            task = process_bulk_upload_task.delay(bulk_upload.id)
            bulk_upload.task_id = task.id
            bulk_upload.save()
            
            messages.success(request, 'Upload confirmed and processing started')
            return redirect('admin_upload_status', job_id=bulk_upload.id)
            
        elif action == 'cancel':
            bulk_upload.status = 'cancelled'
            bulk_upload.save()
            messages.info(request, 'Upload cancelled')
            return redirect('admin_bulk_upload')
    
    # GET request - show preview
    context = {
        'bulk_upload': bulk_upload,
        'preview_data': bulk_upload.preview_data[:10],
        'column_fields': _get_standard_fields(),
        'title': f'Confirm Upload: {bulk_upload.original_filename}',
    }
    return render(request, 'admin/standards/confirm_upload.html', context)


@staff_member_required
def upload_status_view(request, upload_id):
    """Display upload progress and results"""
    bulk_upload = get_object_or_404(BulkUpload, id=upload_id)
    
    context = {
        'bulk_upload': bulk_upload,
        'title': f'Upload Status: {bulk_upload.original_filename}',
        'is_ajax': request.headers.get('X-Requested-With') == 'XMLHttpRequest',
    }
    
    if context['is_ajax']:
        # Return JSON for AJAX polling
        return JsonResponse({
            'status': bulk_upload.status,
            'progress': bulk_upload.progress_percentage,
            'processed': bulk_upload.processed_records,
            'successful': bulk_upload.successful_records,
            'failed': bulk_upload.failed_records,
            'total': bulk_upload.total_records,
            'errors': bulk_upload.get_error_summary(),
            'is_active': bulk_upload.is_active,
        })
    
    return render(request, 'admin/standards/upload_status.html', context)


@staff_member_required
@require_http_methods(['POST'])
def process_upload_view(request, upload_id):
    """Process confirmed upload (trigger async task)"""
    bulk_upload = get_object_or_404(BulkUpload, id=upload_id)
    
    if bulk_upload.status != 'confirmed':
        return JsonResponse({'error': 'Upload not confirmed'}, status=400)
    
    # Start processing
    bulk_upload.status = 'processing'
    bulk_upload.started_at = timezone.now()
    bulk_upload.save()
    
    # Launch async task
    from .tasks import process_bulk_upload_task
    task = process_bulk_upload_task.delay(bulk_upload.id)
    bulk_upload.task_id = task.id
    bulk_upload.save()
    
    return JsonResponse({
        'status': 'processing',
        'task_id': task.id,
        'message': 'Upload processing started',
    })


@staff_member_required
def download_errors_view(request, upload_id):
    """Download error report as CSV"""
    bulk_upload = get_object_or_404(BulkUpload, id=upload_id)
    
    if not bulk_upload.error_log:
        messages.info(request, 'No errors to download')
        return redirect('admin_upload_status', job_id=upload_id)
    
    # Create CSV response
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="errors_{upload_id}.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['Line Number', 'Error Message', 'Timestamp', 'Record Data'])
    
    for error in bulk_upload.error_log:
        writer.writerow([
            error.get('line_number', 'N/A'),
            error.get('message', ''),
            error.get('timestamp', ''),
            json.dumps(error.get('record_data', {})),
        ])
    
    return response


@staff_member_required
def download_template_view(request):
    """Download a sample CSV template for bulk upload"""
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="standards_template.csv"'
    
    writer = csv.writer(response)
    writer.writerow([
        'state_code', 'subject_area', 'grade_level', 'code', 'title',
        'description', 'domain', 'cluster', 'keywords', 'skills'
    ])
    
    # Add sample rows
    writer.writerow([
        'CA', 'Mathematics', '1', 'CA.MATH.1.OA.1', 'Addition and Subtraction',
        'Use addition and subtraction within 20 to solve word problems',
        'Operations and Algebraic Thinking', 'Represent and solve problems',
        'addition,subtraction,word problems', 'problem solving,computation'
    ])
    
    writer.writerow([
        'TX', 'English Language Arts', '3', 'TX.ELA.3.RI.1', 'Key Ideas and Details',
        'Ask and answer questions to demonstrate understanding of a text',
        'Reading Informational Text', 'Key Ideas and Details',
        'comprehension,questioning', 'reading comprehension,analysis'
    ])
    
    return response


# Helper functions

def _process_file_preview(bulk_upload):
    """Process uploaded file and extract data for preview"""
    file_path = bulk_upload.file.path
    file_type = bulk_upload.file_type
    
    if file_type == 'csv':
        df = pd.read_csv(file_path)
    elif file_type == 'json':
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.json_normalize(data)
    elif file_type == 'xlsx':
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f'Unsupported file type: {file_type}')
    
    # Convert DataFrame to list of dicts for preview
    preview_data = df.head(100).to_dict('records')
    return preview_data


def _get_standard_fields():
    """Get list of Standard model fields for mapping"""
    return [
        ('state_code', 'State Code (e.g., CA, TX)'),
        ('subject_area', 'Subject Area'),
        ('grade_level', 'Grade Level'),
        ('code', 'Standard Code'),
        ('title', 'Title (optional)'),
        ('description', 'Description'),
        ('domain', 'Domain/Strand'),
        ('cluster', 'Cluster/Topic'),
        ('keywords', 'Keywords (comma-separated)'),
        ('skills', 'Skills (comma-separated)'),
    ]


@staff_member_required  
def cancel_upload_view(request, upload_id):
    """Cancel an active upload job"""
    bulk_upload = get_object_or_404(BulkUpload, id=upload_id)
    
    if bulk_upload.is_active:
        # Cancel Celery task if running
        if bulk_upload.task_id:
            from celery import current_app
            current_app.control.revoke(bulk_upload.task_id, terminate=True)
        
        bulk_upload.status = 'cancelled'
        bulk_upload.completed_at = timezone.now()
        bulk_upload.save()
        
        messages.info(request, f'Upload job {bulk_upload.original_filename} has been cancelled')
    else:
        messages.warning(request, 'This upload job is not active')
    
    return redirect('admin:upload_status', upload_id=upload_id)