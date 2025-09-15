"""
Custom admin views for bulk upload functionality
"""
import json
import csv
import io
import threading
from typing import Dict, Any
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.utils import timezone
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from .models import Standard, State, SubjectArea, GradeLevel, UploadJob, CorrelationJob, StandardCorrelation
from .forms import BulkUploadForm, PreviewUploadForm, GenerateTemplateForm, GenerateCorrelationsForm, CorrelationPreviewForm, ThresholdAnalysisForm
from .services import EmbeddingService
# Import StandardCorrelationService from the root services.py file
import standards.services as legacy_services


@staff_member_required
def bulk_upload_view(request):
    """Main bulk upload view"""
    if request.method == 'POST':
        form = BulkUploadForm(request.POST, request.FILES, user=request.user)
        if form.is_valid():
            try:
                upload_job, uploaded_file = form.save()
                
                # Store uploaded file temporarily
                file_path = default_storage.save(
                    f'uploads/{upload_job.id}_{uploaded_file.name}',
                    ContentFile(uploaded_file.read())
                )
                
                # Store file path in session for processing
                request.session[f'upload_file_{upload_job.id}'] = file_path
                
                # Generate preview data
                preview_data = _generate_preview_data(uploaded_file, upload_job.file_type)
                
                return render(request, 'admin/standards/bulk_upload_preview.html', {
                    'upload_job': upload_job,
                    'preview_data': preview_data,
                    'form': PreviewUploadForm(initial={'upload_job_id': upload_job.id}),
                })
                
            except Exception as e:
                messages.error(request, f"Upload failed: {str(e)}")
                return redirect('admin_bulk_upload')
                
    else:
        form = BulkUploadForm(user=request.user)
    
    # Get recent upload jobs
    recent_jobs = UploadJob.objects.filter(
        uploaded_by=request.user.username
    ).order_by('-created_at')[:10]
    
    return render(request, 'admin/standards/bulk_upload.html', {
        'form': form,
        'recent_jobs': recent_jobs,
    })


@staff_member_required
def confirm_upload_view(request):
    """Confirm and process the upload"""
    if request.method == 'POST':
        form = PreviewUploadForm(request.POST)
        if form.is_valid():
            upload_job_id = form.cleaned_data['upload_job_id']
            upload_job = get_object_or_404(UploadJob, id=upload_job_id)
            
            # Get file path from session
            file_path = request.session.get(f'upload_file_{upload_job.id}')
            if not file_path:
                messages.error(request, "Upload session expired. Please try again.")
                return redirect('admin_bulk_upload')
            
            # Start processing in background
            try:
                # Mark as started to avoid 'Pending' state if thread takes time to spawn
                upload_job.update_progress(status='validating')
                thread = threading.Thread(
                    target=_process_upload_job,
                    args=(upload_job, file_path)
                )
                # Don't set daemon=True to prevent premature termination
                thread.start()
                
                # Log successful thread start (using record_data for context info)
                upload_job.add_error(f"Background processing thread started successfully", record_data={"context": "thread_start"})
                
            except Exception as e:
                # If threading fails, fall back to synchronous processing for small files
                upload_job.add_error(f"Threading failed: {str(e)}, attempting synchronous processing", record_data={"context": "thread_error"})
                
                # For small files (< 1MB), process synchronously
                if upload_job.file_size < 1024 * 1024:
                    try:
                        _process_upload_job(upload_job, file_path)
                        messages.success(request, f"Upload processed synchronously: {upload_job.original_filename}")
                    except Exception as sync_error:
                        upload_job.add_error(f"Synchronous processing also failed: {str(sync_error)}", record_data={"context": "sync_error"})
                        messages.error(request, f"Upload processing failed: {str(sync_error)}")
                else:
                    messages.error(request, f"Background processing failed and file too large for synchronous processing: {str(e)}")
                    upload_job.update_progress(status='failed')
            
            messages.success(request, f"Upload job started: {upload_job.original_filename}")
            return redirect('admin_upload_status', job_id=upload_job.id)
    
    return redirect('admin_bulk_upload')


@staff_member_required
def upload_status_view(request, job_id):
    """View upload job status and progress"""
    upload_job = get_object_or_404(UploadJob, id=job_id)
    
    return render(request, 'admin/standards/upload_status.html', {
        'upload_job': upload_job,
    })


@staff_member_required
def upload_status_api(request, job_id):
    """API endpoint for getting upload status"""
    try:
        upload_job = get_object_or_404(UploadJob, id=job_id)
        
        data = {
            'status': upload_job.status,
            'progress_percentage': upload_job.progress_percentage,
            'processed_records': upload_job.processed_records,
            'successful_records': upload_job.successful_records,
            'failed_records': upload_job.failed_records,
            'total_records': upload_job.total_records,
            'error_count': len(upload_job.error_log) if upload_job.error_log else 0,
            'is_active': upload_job.is_active,
            'duration': str(upload_job.duration) if upload_job.duration else None,
            'started_at': upload_job.started_at.isoformat() if upload_job.started_at else None,
            'completed_at': upload_job.completed_at.isoformat() if upload_job.completed_at else None,
        }
        
        return JsonResponse(data)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@staff_member_required
def download_error_report(request, job_id):
    """Download error report for upload job"""
    upload_job = get_object_or_404(UploadJob, id=job_id)
    
    if not upload_job.error_log:
        raise Http404("No errors to report")
    
    # Create CSV error report
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Line Number', 'Error Message', 'Timestamp', 'Record Data'])
    
    # Write error data
    for error in upload_job.error_log:
        writer.writerow([
            error.get('line_number', ''),
            error.get('message', ''),
            error.get('timestamp', ''),
            str(error.get('record_data', ''))[:100] + '...' if error.get('record_data') else ''
        ])
    
    response = HttpResponse(output.getvalue(), content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="errors_{upload_job.original_filename}_{job_id}.csv"'
    
    return response


@staff_member_required
def generate_template_view(request):
    """Generate sample template files"""
    if request.method == 'POST':
        form = GenerateTemplateForm(request.POST)
        if form.is_valid():
            file_format = form.cleaned_data['format']
            include_sample_data = form.cleaned_data['include_sample_data']
            states_count = form.cleaned_data['states_count']
            
            if file_format == 'csv':
                response = _generate_csv_template(include_sample_data, states_count)
            else:
                response = _generate_json_template(include_sample_data, states_count)
            
            return response
    else:
        form = GenerateTemplateForm()
    
    return render(request, 'admin/standards/generate_template.html', {
        'form': form,
    })


@staff_member_required
def cancel_upload_job(request, job_id):
    """Cancel an active upload job"""
    upload_job = get_object_or_404(UploadJob, id=job_id)
    
    if upload_job.is_active:
        upload_job.status = 'cancelled'
        upload_job.completed_at = timezone.now()
        upload_job.save()
        messages.success(request, f"Upload job cancelled: {upload_job.original_filename}")
    else:
        messages.warning(request, "Upload job is not active and cannot be cancelled.")
    
    return redirect('standards:upload_status', job_id=job_id)


def _generate_preview_data(uploaded_file, file_type):
    """Generate preview data from uploaded file"""
    uploaded_file.seek(0)
    preview_data = {
        'records': [],
        'total_estimated': 0,
        'file_type': file_type
    }
    
    try:
        if file_type == 'csv':
            content = uploaded_file.read().decode('utf-8', errors='ignore')
            lines = content.split('\n')
            
            # Estimate total records
            preview_data['total_estimated'] = max(0, len(lines) - 1)  # Subtract header
            
            reader = csv.DictReader(lines)
            for i, row in enumerate(reader):
                if i >= 5:  # Show only first 5 records
                    break
                preview_data['records'].append(dict(row))
                
        elif file_type == 'json':
            content = uploaded_file.read().decode('utf-8')
            data = json.loads(content)
            
            if isinstance(data, list):
                preview_data['total_estimated'] = len(data)
                preview_data['records'] = data[:5]  # Show first 5 records
            elif isinstance(data, dict):
                # Count all records in nested structure
                total = 0
                sample_records = []
                
                for state_code, state_data in data.items():
                    if isinstance(state_data, list):
                        total += len(state_data)
                        # Add state code to records if not present
                        for record in state_data[:2]:  # 2 per state max
                            if len(sample_records) >= 5:
                                break
                            record_copy = record.copy()
                            if 'state' not in record_copy:
                                record_copy['state'] = state_code
                            sample_records.append(record_copy)
                    elif isinstance(state_data, dict):
                        for subject, subject_data in state_data.items():
                            if isinstance(subject_data, list):
                                total += len(subject_data)
                                for record in subject_data[:1]:  # 1 per subject max
                                    if len(sample_records) >= 5:
                                        break
                                    record_copy = record.copy()
                                    if 'state' not in record_copy:
                                        record_copy['state'] = state_code
                                    if 'subject' not in record_copy:
                                        record_copy['subject'] = subject
                                    sample_records.append(record_copy)
                
                preview_data['total_estimated'] = total
                preview_data['records'] = sample_records
                
    except Exception as e:
        preview_data['error'] = str(e)
    
    return preview_data


def _process_upload_job(upload_job, file_path):
    """Process upload job in background"""
    try:
        upload_job.update_progress(status='validating')
        
        # Read the file
        try:
            with default_storage.open(file_path, 'r') as file:
                if upload_job.file_type == 'csv':
                    data = _load_csv_data(file)
                else:
                    data = _load_json_data(file)
        except Exception as e:
            upload_job.add_error(f"Failed to read file: {str(e)}")
            upload_job.update_progress(status='failed')
            return
        
        upload_job.total_records = len(data)
        upload_job.update_progress(status='processing')
        upload_job.add_error(f"Processing {len(data)} records from {file_path}")  # Info message
        
        # Clear existing standards if requested
        if upload_job.clear_existing:
            try:
                count_before = Standard.objects.count()
                Standard.objects.all().delete()
                upload_job.add_error(f"Cleared {count_before} existing standards")  # Info message
            except Exception as e:
                upload_job.add_error(f"Failed to clear existing standards: {str(e)}")
        
        # Process data in batches
        try:
            _process_standards_data(upload_job, data)
        except Exception as e:
            upload_job.add_error(f"Data processing failed: {str(e)}")
            upload_job.update_progress(status='failed')
            return
        
        # Generate embeddings if requested
        if upload_job.generate_embeddings and upload_job.successful_records > 0:
            upload_job.update_progress(status='generating_embeddings')
            _generate_embeddings_for_upload(upload_job)
        
        # Mark as completed
        upload_job.update_progress(status='completed')
        
        # Clean up uploaded file
        default_storage.delete(file_path)
        
    except Exception as e:
        upload_job.add_error(f"Processing failed: {str(e)}")
        upload_job.update_progress(status='failed')
        print(f"Upload job {upload_job.id} failed: {str(e)}")


def _find_upload_file_path(upload_job):
    """Attempt to locate the stored upload file for an UploadJob if session path is missing."""
    try:
        base = 'uploads'
        dirs, files = default_storage.listdir(base)
        prefix = f"{upload_job.id}_"
        for name in files:
            if name.startswith(prefix):
                return f"{base}/{name}"
    except Exception:
        return None
    return None


@staff_member_required
def restart_upload_job(request, job_id):
    """Manually restart a stuck upload job by locating its stored file and launching processing."""
    upload_job = get_object_or_404(UploadJob, id=job_id)
    try:
        # Try session path first
        file_path = request.session.get(f'upload_file_{upload_job.id}')
        if not file_path:
            # Attempt to discover the file in storage
            file_path = _find_upload_file_path(upload_job)
        if not file_path:
            messages.error(request, "Could not locate uploaded file for this job. Please re-upload.")
            return redirect('admin_upload_status', job_id=job_id)

        # Kick off background processing
        upload_job.update_progress(status='validating')
        try:
            thread = threading.Thread(target=_process_upload_job, args=(upload_job, file_path))
            thread.start()
            upload_job.add_error("Manual restart: background processing thread started", record_data={"context": "manual_restart"})
            messages.success(request, "Upload job restarted.")
        except Exception as e:
            # Fallback to synchronous for small files
            upload_job.add_error(f"Manual restart threading failed: {str(e)}; attempting synchronous.", record_data={"context": "manual_restart_error"})
            with default_storage.open(file_path, 'r') as file:
                content = file.read()
            if len(content) < 1024 * 1024:  # <1MB
                try:
                    _process_upload_job(upload_job, file_path)
                    messages.success(request, "Upload processed synchronously after restart.")
                except Exception as sync_error:
                    upload_job.add_error(f"Synchronous processing failed on restart: {str(sync_error)}", record_data={"context": "manual_restart_sync_error"})
                    upload_job.update_progress(status='failed')
                    messages.error(request, f"Restart failed: {str(sync_error)}")
            else:
                upload_job.update_progress(status='failed')
                messages.error(request, f"Restart failed: {str(e)}")
        return redirect('admin_upload_status', job_id=job_id)
    except Exception as e:
        upload_job.add_error(f"Restart error: {str(e)}", record_data={"context": "manual_restart"})
        messages.error(request, f"Restart error: {str(e)}")
        return redirect('admin_upload_status', job_id=job_id)


def _load_csv_data(file):
    """Load data from CSV file"""
    content = file.read()
    reader = csv.DictReader(content.splitlines())
    return list(reader)


def _load_json_data(file):
    """Load data from JSON file"""
    content = file.read()
    data = json.loads(content)
    
    # Convert to flat list format
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # Convert nested structure to flat list
        flat_data = []
        for state_code, state_data in data.items():
            if isinstance(state_data, list):
                for record in state_data:
                    if 'state' not in record:
                        record['state'] = state_code
                    flat_data.append(record)
            elif isinstance(state_data, dict):
                for subject, subject_data in state_data.items():
                    if isinstance(subject_data, list):
                        for record in subject_data:
                            if 'state' not in record:
                                record['state'] = state_code
                            if 'subject' not in record:
                                record['subject'] = subject
                            flat_data.append(record)
        return flat_data
    
    return []


def _process_standards_data(upload_job, data):
    """Process standards data in batches"""
    from .management.commands.bulk_import_standards import Command as BulkImportCommand
    
    # Create command instance and process data
    command = BulkImportCommand()
    
    try:
        upload_job.add_error(f"Starting bulk import of {len(data)} records")  # Info message
        
        # Import all data at once using the bulk import command
        imported_count = command.import_standards(data, batch_size=upload_job.batch_size)
        
        upload_job.add_error(f"Bulk import completed: {imported_count} records imported")  # Info message
        
        # Update final progress
        upload_job.update_progress(
            processed=len(data),
            successful=imported_count,
            failed=len(data) - imported_count
        )
        
    except Exception as e:
        upload_job.add_error(f"Bulk import failed: {str(e)}")
        upload_job.update_progress(
            processed=len(data),
            successful=0,
            failed=len(data)
        )
        raise


def _generate_embeddings_for_upload(upload_job):
    """Generate embeddings for uploaded standards"""
    embedding_service = EmbeddingService()
    
    # Get standards without embeddings
    standards_without_embeddings = Standard.objects.filter(embedding__isnull=True)
    total = standards_without_embeddings.count()
    
    for i, standard in enumerate(standards_without_embeddings):
        try:
            embedding = embedding_service.generate_standard_embedding(standard)
            if embedding:
                standard.embedding = embedding
                standard.save(update_fields=['embedding'])
        except Exception as e:
            upload_job.add_error(f"Embedding generation failed for {standard.code}: {str(e)}")
        
        # Update progress every 10 records
        if i % 10 == 0:
            progress = (i / total) * 100 if total > 0 else 100
            upload_job.progress_percentage = 90 + (progress * 0.1)  # 90-100% range
            upload_job.save(update_fields=['progress_percentage'])


def _generate_csv_template(include_sample_data, states_count):
    """Generate CSV template"""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    headers = [
        'state', 'subject', 'grade', 'code', 'title', 'description',
        'domain', 'cluster', 'keywords', 'skills'
    ]
    writer.writerow(headers)
    
    if include_sample_data:
        sample_states = ['CA', 'TX', 'NY', 'FL', 'IL', 'OH', 'MI', 'GA', 'NC', 'VA'][:states_count]
        
        # Mix of standards with and without titles to demonstrate optional nature
        sample_data = [
            ('3', 'Understanding Multiplication', 'Interpret products of whole numbers as equal groups.', 'Operations and Algebraic Thinking', 'Multiplication and Division'),
            ('4', '', 'Use place value understanding to round multi-digit whole numbers to any place.', 'Number and Operations in Base Ten', 'Generalize place value understanding'),
            ('5', 'Decimal Operations', 'Add, subtract, multiply, and divide decimals to hundredths.', 'Number and Operations in Base Ten', 'Perform operations with multi-digit whole numbers'),
            ('2', '', 'Use addition and subtraction within 100 to solve word problems.', 'Operations and Algebraic Thinking', 'Add and subtract within 20'),
        ]
        
        for i, state in enumerate(sample_states):
            data_idx = i % len(sample_data)
            grade, title, description, domain, cluster = sample_data[data_idx]
            
            writer.writerow([
                state,
                'Mathematics',
                grade,
                f'{state}.{grade}.OA.{i+1}',
                title,  # Sometimes empty to show optional nature
                description,
                domain,
                cluster,
                'multiplication,equal groups,arrays' if title else 'place value,rounding,estimation',
                'problem solving,mathematical reasoning'
            ])
    
    response = HttpResponse(output.getvalue(), content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="standards_template.csv"'
    return response


def _generate_json_template(include_sample_data, states_count):
    """Generate JSON template"""
    if include_sample_data:
        sample_states = ['CA', 'TX', 'NY', 'FL', 'IL', 'OH', 'MI', 'GA', 'NC', 'VA'][:states_count]
        
        # Mix of standards with and without titles to demonstrate optional nature
        sample_data = [
            ('3', 'Understanding Multiplication', 'Interpret products of whole numbers as equal groups.', 'Operations and Algebraic Thinking', 'Multiplication and Division', ['multiplication', 'equal groups', 'arrays']),
            ('4', None, 'Use place value understanding to round multi-digit whole numbers to any place.', 'Number and Operations in Base Ten', 'Generalize place value understanding', ['place value', 'rounding', 'estimation']),
            ('5', 'Decimal Operations', 'Add, subtract, multiply, and divide decimals to hundredths.', 'Number and Operations in Base Ten', 'Perform operations with multi-digit whole numbers', ['decimals', 'operations', 'computation']),
            ('2', None, 'Use addition and subtraction within 100 to solve word problems.', 'Operations and Algebraic Thinking', 'Add and subtract within 20', ['addition', 'subtraction', 'word problems']),
        ]
        
        data = []
        for i, state in enumerate(sample_states):
            data_idx = i % len(sample_data)
            grade, title, description, domain, cluster, keywords = sample_data[data_idx]
            
            standard = {
                'state': state,
                'subject': 'Mathematics',
                'grade': grade,
                'code': f'{state}.{grade}.OA.{i+1}',
                'description': description,
                'domain': domain,
                'cluster': cluster,
                'keywords': keywords,
                'skills': ['problem solving', 'mathematical reasoning']
            }
            
            # Only include title if it's not None/empty (demonstrates optional nature)
            if title:
                standard['title'] = title
                
            data.append(standard)
    else:
        data = [
            {
                'state': 'CA',
                'subject': 'Mathematics',
                'grade': '3',
                'code': 'CA.3.OA.1',
                'title': 'Standard title here (optional)',
                'description': 'Standard description here (required)',
                'domain': 'Domain name (optional)',
                'cluster': 'Cluster name (optional)',
                'keywords': ['keyword1', 'keyword2'],
                'skills': ['skill1', 'skill2']
            },
            {
                'state': 'TX',
                'subject': 'Mathematics',
                'grade': '4',
                'code': 'TX.4.NBT.3',
                'description': 'Example without title - system will generate display name from code + description',
                'domain': 'Number and Operations in Base Ten',
                'cluster': 'Generalize place value understanding',
                'keywords': ['place value', 'rounding'],
                'skills': ['computational fluency']
            }
        ]
    
    response = HttpResponse(
        json.dumps(data, indent=2),
        content_type='application/json'
    )
    response['Content-Disposition'] = 'attachment; filename="standards_template.json"'
    return response


@staff_member_required
def generate_correlations_view(request):
    """Main correlation generation view"""
    if request.method == 'POST':
        form = GenerateCorrelationsForm(request.POST, user=request.user)
        if form.is_valid():
            try:
                correlation_job = form.save()
                
                # Get preview information from form validation
                preview_info = form.cleaned_data.get('_preview_info', {})
                
                return render(request, 'admin/standards/correlation_preview.html', {
                    'correlation_job': correlation_job,
                    'preview_info': preview_info,
                    'form': CorrelationPreviewForm(initial={'correlation_job_id': correlation_job.id}),
                })
                
            except Exception as e:
                messages.error(request, f"Error creating correlation job: {str(e)}")
                
    else:
        form = GenerateCorrelationsForm(user=request.user)
    
    # Get current correlation statistics
    current_stats = _get_correlation_statistics()
    
    # Get recent correlation jobs
    recent_jobs = CorrelationJob.objects.filter(
        created_by=request.user.username
    ).order_by('-created_at')[:10]
    
    return render(request, 'admin/standards/generate_correlations.html', {
        'form': form,
        'current_stats': current_stats,
        'recent_jobs': recent_jobs,
    })


@staff_member_required
def confirm_correlation_generation(request):
    """Confirm and start correlation generation"""
    if request.method == 'POST':
        form = CorrelationPreviewForm(request.POST)
        if form.is_valid():
            correlation_job_id = form.cleaned_data['correlation_job_id']
            correlation_job = get_object_or_404(CorrelationJob, id=correlation_job_id)
            
            # Start processing in background
            try:
                thread = threading.Thread(
                    target=_process_correlation_job,
                    args=(correlation_job,)
                )
                # Don't set daemon=True to prevent premature termination
                thread.start()
                
                # Log successful thread start
                correlation_job.add_error(f"Background correlation thread started successfully", context="thread_start")
                
            except Exception as e:
                # Log threading failure
                correlation_job.add_error(f"Threading failed: {str(e)}", context="thread_error")
                correlation_job.update_progress(status='failed')
                messages.error(request, f"Background processing failed: {str(e)}")
                return redirect('admin_generate_correlations')
            
            messages.success(request, f"Correlation generation started: Threshold {correlation_job.similarity_threshold}")
            return redirect('admin_correlation_status', job_id=correlation_job.id)
    
    return redirect('admin_generate_correlations')


@staff_member_required
def correlation_status_view(request, job_id):
    """View correlation job status and progress"""
    correlation_job = get_object_or_404(CorrelationJob, id=job_id)
    
    return render(request, 'admin/standards/correlation_status.html', {
        'correlation_job': correlation_job,
    })


@staff_member_required
def correlation_status_api(request, job_id):
    """API endpoint for getting correlation status"""
    try:
        correlation_job = get_object_or_404(CorrelationJob, id=job_id)
        
        data = {
            'status': correlation_job.status,
            'progress_percentage': correlation_job.progress_percentage,
            'processed_standards': correlation_job.processed_standards,
            'total_standards': correlation_job.total_standards,
            'correlations_created': correlation_job.correlations_created,
            'correlations_updated': correlation_job.correlations_updated,
            'correlations_skipped': correlation_job.correlations_skipped,
            'error_count': len(correlation_job.error_log) if correlation_job.error_log else 0,
            'is_active': correlation_job.is_active,
            'duration': str(correlation_job.duration) if correlation_job.duration else None,
            'estimated_completion': correlation_job.estimated_completion.isoformat() if correlation_job.estimated_completion else None,
            'avg_processing_time': correlation_job.avg_processing_time_per_standard,
            'performance_summary': correlation_job.get_performance_summary(),
        }
        
        return JsonResponse(data)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@staff_member_required
def correlation_analysis_view(request):
    """Interactive correlation threshold analysis"""
    if request.method == 'POST':
        form = ThresholdAnalysisForm(request.POST)
        if form.is_valid():
            analysis_results = _perform_threshold_analysis(form.cleaned_data)
            return JsonResponse(analysis_results)
    else:
        form = ThresholdAnalysisForm()
    
    # Get current correlation distribution
    distribution_data = _get_correlation_distribution()
    
    return render(request, 'admin/standards/correlation_analysis.html', {
        'form': form,
        'distribution_data': distribution_data,
    })


@staff_member_required
def correlation_preview_api(request):
    """API endpoint for real-time correlation preview"""
    threshold = float(request.GET.get('threshold', 0.8))
    subject_id = request.GET.get('subject_id')
    state_ids = request.GET.getlist('state_ids')
    
    try:
        # Build queryset based on filters
        queryset = Standard.objects.filter(embedding__isnull=False)
        
        if subject_id:
            queryset = queryset.filter(subject_area_id=subject_id)
        
        if state_ids:
            queryset = queryset.filter(state_id__in=state_ids)
        
        standards_count = queryset.count()
        
        # Estimate correlations (sample-based for performance)
        sample_size = min(50, standards_count)
        if sample_size > 0:
            sample_standards = queryset.order_by('?')[:sample_size]
            
            correlation_service = legacy_services.StandardCorrelationService()
            total_similar = 0
            
            for standard in sample_standards:
                similar_standards = correlation_service.find_similar_standards(
                    standard, limit=20, threshold=threshold
                )
                total_similar += len(similar_standards)
            
            avg_similar_per_standard = total_similar / sample_size if sample_size > 0 else 0
            estimated_total_correlations = int(standards_count * avg_similar_per_standard)
        else:
            estimated_total_correlations = 0
        
        # Estimate processing time
        estimated_minutes = standards_count * 0.5  # Rough estimate: 0.5 minutes per standard
        
        data = {
            'standards_count': standards_count,
            'estimated_correlations': estimated_total_correlations,
            'estimated_processing_time_minutes': estimated_minutes,
            'threshold_info': _get_threshold_info(threshold)
        }
        
        return JsonResponse(data)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@staff_member_required
def cancel_correlation_job(request, job_id):
    """Cancel an active correlation job"""
    correlation_job = get_object_or_404(CorrelationJob, id=job_id)
    
    if correlation_job.is_active:
        correlation_job.status = 'cancelled'
        correlation_job.completed_at = timezone.now()
        correlation_job.save()
        messages.success(request, f"Correlation job cancelled")
    else:
        messages.warning(request, "Correlation job is not active and cannot be cancelled.")
    
    return redirect('admin_correlation_status', job_id=job_id)


def _get_correlation_statistics():
    """Get current correlation statistics"""
    from django.db import models
    
    total_correlations = StandardCorrelation.objects.count()
    
    # Group by correlation type
    type_distribution = StandardCorrelation.objects.values('correlation_type').annotate(
        count=models.Count('id')
    ).order_by('correlation_type')
    
    # Group by similarity score ranges
    score_ranges = [
        (0.95, 1.0, 'Exact (0.95-1.0)'),
        (0.85, 0.95, 'Similar (0.85-0.95)'),
        (0.75, 0.85, 'Related (0.75-0.85)'),
        (0.0, 0.75, 'Partial (0.0-0.75)'),
    ]
    
    score_distribution = []
    for min_score, max_score, label in score_ranges:
        count = StandardCorrelation.objects.filter(
            similarity_score__gte=min_score,
            similarity_score__lt=max_score
        ).count()
        score_distribution.append({
            'label': label,
            'count': count,
            'percentage': (count / total_correlations * 100) if total_correlations > 0 else 0
        })
    
    return {
        'total_correlations': total_correlations,
        'type_distribution': list(type_distribution),
        'score_distribution': score_distribution,
        'standards_with_embeddings': Standard.objects.filter(embedding__isnull=False).count(),
        'standards_without_embeddings': Standard.objects.filter(embedding__isnull=True).count(),
    }


def _get_correlation_distribution():
    """Get correlation distribution data for visualization"""
    correlations = StandardCorrelation.objects.values('similarity_score').order_by('similarity_score')
    scores = [c['similarity_score'] for c in correlations]
    
    # Create histogram data
    bins = [i * 0.05 for i in range(21)]  # 0.0 to 1.0 in 0.05 increments
    hist_data = []
    
    for i in range(len(bins) - 1):
        count = sum(1 for score in scores if bins[i] <= score < bins[i + 1])
        hist_data.append({
            'range': f"{bins[i]:.2f}-{bins[i+1]:.2f}",
            'count': count
        })
    
    return {
        'histogram': hist_data,
        'total_correlations': len(scores),
        'average_score': sum(scores) / len(scores) if scores else 0,
        'median_score': sorted(scores)[len(scores) // 2] if scores else 0,
    }


def _get_threshold_info(threshold):
    """Get information about a specific threshold"""
    descriptions = {
        (0.95, 1.0): "Exact matches - Nearly identical standards",
        (0.85, 0.95): "Very similar - Strong alignment with minor differences",
        (0.75, 0.85): "Similar - Good alignment with some differences",
        (0.65, 0.75): "Related - Moderate alignment, related concepts",
        (0.50, 0.65): "Loosely related - Some conceptual overlap",
        (0.0, 0.50): "Weak correlation - Minimal alignment"
    }
    
    for (min_thresh, max_thresh), description in descriptions.items():
        if min_thresh <= threshold < max_thresh:
            return {
                'description': description,
                'quality': 'high' if threshold >= 0.8 else 'medium' if threshold >= 0.6 else 'low'
            }
    
    return {
        'description': "Custom threshold",
        'quality': 'medium'
    }


def _perform_threshold_analysis(analysis_params):
    """Perform threshold analysis and return results"""
    min_threshold = analysis_params['min_threshold']
    max_threshold = analysis_params['max_threshold']
    step_size = analysis_params['step_size']
    sample_size = analysis_params['sample_size']
    
    # Get sample standards
    sample_standards = Standard.objects.filter(embedding__isnull=False).order_by('?')[:sample_size]
    
    results = []
    correlation_service = legacy_services.StandardCorrelationService()
    
    threshold = min_threshold
    while threshold <= max_threshold:
        total_correlations = 0
        
        for standard in sample_standards:
            similar_standards = correlation_service.find_similar_standards(
                standard, limit=20, threshold=threshold
            )
            total_correlations += len(similar_standards)
        
        avg_correlations_per_standard = total_correlations / len(sample_standards) if sample_standards else 0
        
        results.append({
            'threshold': round(threshold, 3),
            'avg_correlations_per_standard': round(avg_correlations_per_standard, 2),
            'estimated_total_correlations': int(Standard.objects.filter(embedding__isnull=False).count() * avg_correlations_per_standard),
            'quality_info': _get_threshold_info(threshold)
        })
        
        threshold += step_size
    
    return {
        'analysis_results': results,
        'sample_size': len(sample_standards),
        'total_standards': Standard.objects.filter(embedding__isnull=False).count()
    }


def _process_correlation_job(correlation_job):
    """Process correlation job in background"""
    try:
        correlation_job.update_progress(status='analyzing')
        
        # Build queryset based on job filters
        queryset = Standard.objects.filter(embedding__isnull=False)
        
        if correlation_job.subject_area_filter:
            queryset = queryset.filter(subject_area=correlation_job.subject_area_filter)
        
        state_filter = correlation_job.state_filter.all()
        if state_filter:
            queryset = queryset.filter(state__in=state_filter)
        
        standards = list(queryset)
        correlation_job.total_standards = len(standards)
        correlation_job.update_progress(status='generating')
        
        # Clear existing correlations if requested
        if correlation_job.clear_existing:
            StandardCorrelation.objects.all().delete()
            correlation_job.add_error("Cleared all existing correlations", context="clear_existing")
        
        # Initialize correlation service
        correlation_service = legacy_services.StandardCorrelationService()
        
        # Process standards in batches
        processed = 0
        created = 0
        updated = 0
        skipped = 0
        
        for i in range(0, len(standards), correlation_job.batch_size):
            batch = standards[i:i + correlation_job.batch_size]
            
            for standard in batch:
                try:
                    # Find similar standards
                    similar_standards = correlation_service.find_similar_standards(
                        standard, limit=20, threshold=correlation_job.similarity_threshold
                    )
                    
                    for similar_standard, similarity_score in similar_standards:
                        # Check if correlation already exists
                        existing_correlation = StandardCorrelation.objects.filter(
                            standard_1=standard,
                            standard_2=similar_standard
                        ).first()
                        
                        if existing_correlation:
                            # Update existing correlation
                            existing_correlation.similarity_score = similarity_score
                            existing_correlation.save()
                            updated += 1
                        else:
                            # Create new correlation
                            correlation_type = 'exact' if similarity_score >= 0.95 else \
                                             'similar' if similarity_score >= 0.85 else \
                                             'related' if similarity_score >= 0.75 else 'partial'
                            
                            StandardCorrelation.objects.create(
                                standard_1=standard,
                                standard_2=similar_standard,
                                similarity_score=similarity_score,
                                correlation_type=correlation_type
                            )
                            created += 1
                    
                    processed += 1
                    
                except Exception as e:
                    correlation_job.add_error(
                        f"Error processing standard {standard.code}: {str(e)}",
                        standard_code=standard.code
                    )
                    skipped += 1
                    processed += 1
            
            # Update progress
            correlation_job.update_progress(
                processed=processed,
                created=created,
                updated=updated,
                skipped=skipped
            )
        
        # Mark as completed
        correlation_job.update_progress(status='completed')
        
        # Store final statistics
        correlation_job.statistics = {
            'total_standards_processed': processed,
            'correlations_created': created,
            'correlations_updated': updated,
            'correlations_skipped': skipped,
            'threshold_used': correlation_job.similarity_threshold,
            'processing_duration': str(correlation_job.duration),
        }
        correlation_job.save(update_fields=['statistics'])
        
    except Exception as e:
        correlation_job.add_error(f"Job processing failed: {str(e)}")
        correlation_job.update_progress(status='failed')
        print(f"Correlation job {correlation_job.id} failed: {str(e)}")