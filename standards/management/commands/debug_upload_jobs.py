"""
Management command to debug and fix stuck upload jobs
"""
import json
import csv
import io
from django.core.management.base import BaseCommand, CommandError
from django.core.files.storage import default_storage
from django.utils import timezone
from standards.models import UploadJob, Standard
from standards.services import EmbeddingService


class Command(BaseCommand):
    help = 'Debug and fix stuck upload jobs'

    def add_arguments(self, parser):
        parser.add_argument(
            '--job-id',
            type=str,
            help='Specific job ID to debug'
        )
        parser.add_argument(
            '--list-stuck',
            action='store_true',
            help='List all stuck jobs'
        )
        parser.add_argument(
            '--process-stuck',
            action='store_true',
            help='Process all stuck jobs'
        )
        parser.add_argument(
            '--retry-job',
            type=str,
            help='Retry a specific job by ID'
        )
        parser.add_argument(
            '--check-files',
            action='store_true',
            help='Check if uploaded files still exist'
        )

    def handle(self, *args, **options):
        if options['list_stuck']:
            self.list_stuck_jobs()
        elif options['job_id']:
            self.debug_specific_job(options['job_id'])
        elif options['process_stuck']:
            self.process_stuck_jobs()
        elif options['retry_job']:
            self.retry_job(options['retry_job'])
        elif options['check_files']:
            self.check_upload_files()
        else:
            self.stdout.write(self.style.ERROR('Please specify an action: --list-stuck, --job-id, --process-stuck, --retry-job, or --check-files'))

    def list_stuck_jobs(self):
        """List all stuck upload jobs"""
        stuck_jobs = UploadJob.objects.filter(
            status__in=['pending', 'validating', 'processing', 'generating_embeddings']
        ).order_by('-created_at')
        
        if not stuck_jobs:
            self.stdout.write(self.style.SUCCESS('No stuck jobs found!'))
            return
        
        self.stdout.write(f'Found {stuck_jobs.count()} stuck jobs:')
        self.stdout.write('=' * 80)
        
        for job in stuck_jobs:
            age = timezone.now() - job.created_at
            self.stdout.write(f'ID: {job.id}')
            self.stdout.write(f'File: {job.original_filename}')
            self.stdout.write(f'Status: {job.status}')
            self.stdout.write(f'Records: {job.processed_records}/{job.total_records}')
            self.stdout.write(f'Age: {age}')
            self.stdout.write(f'Errors: {len(job.error_log) if job.error_log else 0}')
            self.stdout.write('-' * 40)

    def debug_specific_job(self, job_id):
        """Debug a specific upload job"""
        try:
            job = UploadJob.objects.get(id=job_id)
        except UploadJob.DoesNotExist:
            raise CommandError(f'Job {job_id} not found')
        
        self.stdout.write(f'Debugging job: {job.original_filename}')
        self.stdout.write('=' * 60)
        
        # Basic info
        self.stdout.write(f'ID: {job.id}')
        self.stdout.write(f'Status: {job.status}')
        self.stdout.write(f'File Type: {job.file_type}')
        self.stdout.write(f'File Size: {job.file_size} bytes')
        self.stdout.write(f'Created: {job.created_at}')
        self.stdout.write(f'Started: {job.started_at}')
        self.stdout.write(f'Total Records: {job.total_records}')
        self.stdout.write(f'Processed: {job.processed_records}')
        
        # Check for uploaded file
        self.stdout.write('\nChecking for uploaded file...')
        file_found = False
        potential_paths = [
            f'uploads/{job.id}_{job.original_filename}',
            f'uploads/{job.original_filename}',
            job.original_filename
        ]
        
        for path in potential_paths:
            if default_storage.exists(path):
                self.stdout.write(f'✓ Found file at: {path}')
                file_found = True
                
                # Try to read file preview
                try:
                    with default_storage.open(path, 'r') as f:
                        if job.file_type == 'csv':
                            preview = f.read(500)
                            lines = preview.split('\n')
                            self.stdout.write(f'  File has ~{len(lines)} lines (first 500 chars)')
                            self.stdout.write(f'  Header: {lines[0] if lines else "No header"}')
                        else:
                            content = f.read(200)
                            self.stdout.write(f'  JSON preview: {content}...')
                except Exception as e:
                    self.stdout.write(f'  ⚠ Error reading file: {e}')
                break
        
        if not file_found:
            self.stdout.write('✗ No uploaded file found')
            self.stdout.write('  Checked paths:')
            for path in potential_paths:
                self.stdout.write(f'    - {path}')
        
        # Check errors
        if job.error_log:
            self.stdout.write(f'\nErrors ({len(job.error_log)}):')
            for i, error in enumerate(job.error_log[-5:]):  # Last 5 errors
                self.stdout.write(f'  {i+1}. {error.get("message", "Unknown")}')
                if 'timestamp' in error:
                    self.stdout.write(f'     At: {error["timestamp"]}')
        
        # Check if job can be retried
        self.stdout.write('\nRecommendations:')
        if not file_found:
            self.stdout.write('  - File missing: Job cannot be retried automatically')
            self.stdout.write('  - Re-upload the file manually')
        elif job.status == 'pending' and job.total_records == 0:
            self.stdout.write('  - Job never started processing')
            self.stdout.write('  - Use --retry-job to manually process')
        elif job.is_active:
            self.stdout.write('  - Job appears active but may be stuck')
            self.stdout.write('  - Check if background process is running')

    def retry_job(self, job_id):
        """Retry a specific job"""
        try:
            job = UploadJob.objects.get(id=job_id)
        except UploadJob.DoesNotExist:
            raise CommandError(f'Job {job_id} not found')
        
        self.stdout.write(f'Retrying job: {job.original_filename}')
        
        # Find the uploaded file
        potential_paths = [
            f'uploads/{job.id}_{job.original_filename}',
            f'uploads/{job.original_filename}',
        ]
        
        file_path = None
        for path in potential_paths:
            if default_storage.exists(path):
                file_path = path
                break
        
        if not file_path:
            raise CommandError('Cannot find uploaded file for this job')
        
        self.stdout.write(f'Found file at: {file_path}')
        
        # Reset job status
        job.status = 'validating'
        job.started_at = timezone.now()
        job.total_records = 0
        job.processed_records = 0
        job.successful_records = 0
        job.failed_records = 0
        job.progress_percentage = 0.0
        job.error_log = []
        job.save()
        
        # Process the job
        try:
            self._process_upload_job_sync(job, file_path)
            self.stdout.write(self.style.SUCCESS(f'Job {job_id} completed successfully!'))
        except Exception as e:
            job.status = 'failed'
            job.completed_at = timezone.now()
            job.add_error(f'Manual retry failed: {str(e)}')
            job.save()
            raise CommandError(f'Job processing failed: {e}')

    def process_stuck_jobs(self):
        """Process all stuck jobs"""
        stuck_jobs = UploadJob.objects.filter(
            status__in=['pending', 'validating', 'processing']
        )
        
        if not stuck_jobs:
            self.stdout.write(self.style.SUCCESS('No stuck jobs to process'))
            return
        
        self.stdout.write(f'Processing {stuck_jobs.count()} stuck jobs...')
        
        for job in stuck_jobs:
            self.stdout.write(f'Processing {job.original_filename}...')
            try:
                self.retry_job(str(job.id))
                self.stdout.write(self.style.SUCCESS(f'✓ {job.original_filename} completed'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'✗ {job.original_filename} failed: {e}'))

    def check_upload_files(self):
        """Check if uploaded files still exist"""
        all_jobs = UploadJob.objects.all().order_by('-created_at')[:20]  # Last 20 jobs
        
        self.stdout.write('Checking file existence for recent jobs...')
        self.stdout.write('=' * 60)
        
        for job in all_jobs:
            potential_paths = [
                f'uploads/{job.id}_{job.original_filename}',
                f'uploads/{job.original_filename}',
            ]
            
            file_found = False
            for path in potential_paths:
                if default_storage.exists(path):
                    file_found = True
                    self.stdout.write(f'✓ {job.original_filename} - Found at {path}')
                    break
            
            if not file_found:
                self.stdout.write(f'✗ {job.original_filename} - Missing')

    def _process_upload_job_sync(self, upload_job, file_path):
        """Process upload job synchronously with better error handling"""
        try:
            upload_job.update_progress(status='validating')
            
            # Read the file
            with default_storage.open(file_path, 'r') as file:
                if upload_job.file_type == 'csv':
                    data = self._load_csv_data(file)
                else:
                    data = self._load_json_data(file)
            
            upload_job.total_records = len(data)
            upload_job.update_progress(status='processing')
            
            self.stdout.write(f'Processing {len(data)} records...')
            
            # Clear existing standards if requested
            if upload_job.clear_existing:
                self.stdout.write('Clearing existing standards...')
                Standard.objects.all().delete()
            
            # Process data using bulk import command
            from standards.management.commands.bulk_import_standards import Command as BulkImportCommand
            command = BulkImportCommand()
            
            successful = 0
            failed = 0
            
            try:
                # Use the actual import method from the bulk import command
                self.stdout.write('Calling bulk import method...')
                imported_count = command.import_standards(data, batch_size=upload_job.batch_size)
                successful = imported_count
                self.stdout.write(f'Bulk import returned: {imported_count} successfully imported')
                
                # Update progress
                upload_job.update_progress(
                    processed=len(data),
                    successful=successful,
                    failed=len(data) - successful
                )
                
            except Exception as e:
                failed = len(data)
                upload_job.add_error(f"Bulk import failed: {str(e)}")
                self.stdout.write(f'Bulk import failed: {e}')
                raise
            
            # Generate embeddings if requested
            if upload_job.generate_embeddings and successful > 0:
                self.stdout.write('Generating embeddings...')
                upload_job.update_progress(status='generating_embeddings')
                self._generate_embeddings_for_upload(upload_job)
            
            # Mark as completed
            upload_job.update_progress(status='completed')
            
        except Exception as e:
            upload_job.update_progress(status='failed')
            upload_job.add_error(f'Processing failed: {str(e)}')
            raise

    def _load_csv_data(self, file):
        """Load CSV data from file"""
        content = file.read()
        reader = csv.DictReader(content.splitlines())
        return list(reader)

    def _load_json_data(self, file):
        """Load JSON data from file"""
        content = file.read()
        data = json.loads(content)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Flatten nested structure
            records = []
            for state_code, state_data in data.items():
                if isinstance(state_data, list):
                    for record in state_data:
                        if 'state' not in record:
                            record['state'] = state_code
                        records.append(record)
            return records
        else:
            raise ValueError("JSON must be array or object with arrays")

    def _generate_embeddings_for_upload(self, upload_job):
        """Generate embeddings for uploaded standards"""
        embedding_service = EmbeddingService()
        
        # Get standards without embeddings
        standards_without_embeddings = Standard.objects.filter(embedding__isnull=True)
        total = standards_without_embeddings.count()
        
        self.stdout.write(f'Generating embeddings for {total} standards...')
        
        for i, standard in enumerate(standards_without_embeddings):
            try:
                embedding = embedding_service.generate_standard_embedding(standard)
                if embedding:
                    standard.embedding = embedding
                    standard.save(update_fields=['embedding'])
                    
                if (i + 1) % 10 == 0:
                    self.stdout.write(f'  Generated {i + 1}/{total} embeddings')
                    
            except Exception as e:
                upload_job.add_error(f"Embedding generation failed for {standard.code}: {str(e)}")
                self.stdout.write(f'  Failed to generate embedding for {standard.code}: {e}')