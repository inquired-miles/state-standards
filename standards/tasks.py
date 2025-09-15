"""
Celery tasks for async processing of bulk uploads and embedding generation
"""
import csv
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import pandas as pd
from celery import shared_task, group, chord
from django.db import transaction
from django.utils import timezone
from .models import Standard, State, SubjectArea, GradeLevel, BulkUpload, UploadJob
from .services import EmbeddingService

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def process_bulk_upload_task(self, upload_id):
    """
    Process bulk upload file asynchronously
    """
    try:
        if hasattr(BulkUpload, 'objects'):
            upload_job = BulkUpload.objects.get(id=upload_id)
        else:
            upload_job = UploadJob.objects.get(id=upload_id)
            
        upload_job.update_progress(status='processing')
        
        # Read and process file
        file_path = upload_job.file.path if hasattr(upload_job, 'file') else None
        if not file_path:
            raise ValueError("No file path available")
            
        # Parse file based on type
        if upload_job.file_type == 'csv':
            df = pd.read_csv(file_path)
        elif upload_job.file_type == 'json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                df = pd.DataFrame(data) if isinstance(data, list) else pd.json_normalize(data)
        else:
            raise ValueError(f"Unsupported file type: {upload_job.file_type}")
        
        total_records = len(df)
        upload_job.total_records = total_records
        upload_job.save()
        
        # Process records in batches
        batch_size = upload_job.batch_size if hasattr(upload_job, 'batch_size') else 100
        successful = 0
        failed = 0
        
        for i in range(0, total_records, batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_results = _process_batch(batch, upload_job)
            
            successful += batch_results['successful']
            failed += batch_results['failed']
            
            # Update progress
            upload_job.update_progress(
                processed=i + len(batch),
                successful=successful,
                failed=failed
            )
        
        # Generate embeddings if requested
        if upload_job.generate_embeddings:
            upload_job.update_progress(status='generating_embeddings')
            generate_embeddings_batch_task.delay(upload_id)
        else:
            upload_job.update_progress(status='completed')
            
        return {
            'status': 'success',
            'total': total_records,
            'successful': successful,
            'failed': failed
        }
        
    except Exception as e:
        logger.error(f"Error processing bulk upload {upload_id}: {str(e)}")
        upload_job.update_progress(status='failed')
        upload_job.add_error(str(e))
        raise self.retry(exc=e, countdown=60)


@shared_task(bind=True)
def generate_embeddings_batch_task(self, upload_id):
    """
    Generate embeddings for standards in parallel batches
    """
    try:
        if hasattr(BulkUpload, 'objects'):
            upload_job = BulkUpload.objects.get(id=upload_id)
        else:
            upload_job = UploadJob.objects.get(id=upload_id)
            
        # Get standards without embeddings
        standards = Standard.objects.filter(embedding__isnull=True)
        total_standards = standards.count()
        
        if total_standards == 0:
            upload_job.update_progress(status='completed')
            return {'status': 'success', 'message': 'No standards need embeddings'}
        
        logger.info(f"Generating embeddings for {total_standards} standards")
        
        # Process using batch API calls for maximum efficiency
        batch_size = 50  # Process 50 standards at a time (OpenAI batch limit is 100)
        embedding_service = EmbeddingService()
        processed = 0
        successful = 0
        failed = 0
        
        for i in range(0, total_standards, batch_size):
            batch = list(standards[i:i+batch_size])
            
            # Generate embeddings for entire batch in one API call
            results = embedding_service.generate_standards_embeddings_batch(batch)
            
            # Save embeddings to database
            embeddings_map = results['embeddings']
            for standard in batch:
                if standard.id in embeddings_map:
                    standard.embedding = embeddings_map[standard.id]
                    standard.save(update_fields=['embedding'])
                    successful += 1
                else:
                    failed += 1
                    logger.error(f"Failed to generate embedding for standard {standard.code}")
            
            processed += len(batch)
            
            # Update progress
            progress = (processed / total_standards) * 100
            logger.info(f"Embedding generation progress: {progress:.1f}% (Successful: {successful}, Failed: {failed})")
        
        upload_job.update_progress(status='completed')
        return {
            'status': 'success',
            'total_processed': processed,
            'total_standards': total_standards
        }
        
    except Exception as e:
        logger.error(f"Error in embedding generation task: {str(e)}")
        upload_job.update_progress(status='failed')
        raise


@shared_task
def generate_embeddings_chunk_task(standard_ids: List[str]):
    """
    Generate embeddings for a chunk of standards
    Used for distributed processing across multiple workers
    """
    embedding_service = EmbeddingService()
    successful = 0
    failed = 0
    
    for standard_id in standard_ids:
        try:
            standard = Standard.objects.get(id=standard_id)
            embedding = embedding_service.generate_embedding(
                f"{standard.code} {standard.title or ''} {standard.description}"
            )
            standard.embedding = embedding
            standard.save(update_fields=['embedding'])
            successful += 1
        except Exception as e:
            logger.error(f"Error generating embedding for standard {standard_id}: {str(e)}")
            failed += 1
    
    return {'successful': successful, 'failed': failed}


@shared_task
def generate_all_embeddings_distributed():
    """
    Generate embeddings for all standards using distributed Celery workers
    This is the most efficient method for large-scale embedding generation
    """
    # Get all standards without embeddings
    standards = Standard.objects.filter(embedding__isnull=True).values_list('id', flat=True)
    standard_ids = list(standards)
    total = len(standard_ids)
    
    if total == 0:
        return {'status': 'success', 'message': 'No standards need embeddings'}
    
    # Split into chunks for distributed processing
    chunk_size = 20  # Each worker processes 20 standards
    chunks = [standard_ids[i:i+chunk_size] for i in range(0, total, chunk_size)]
    
    # Create a group of tasks
    job = group(generate_embeddings_chunk_task.s(chunk) for chunk in chunks)
    
    # Execute all tasks in parallel across available workers
    result = job.apply_async()
    
    return {
        'status': 'started',
        'total_standards': total,
        'num_chunks': len(chunks),
        'task_id': result.id
    }


# Helper functions

def _process_batch(batch_df, upload_job):
    """Process a batch of records from the DataFrame"""
    successful = 0
    failed = 0
    
    for idx, row in batch_df.iterrows():
        try:
            # Get or create related objects
            state, _ = State.objects.get_or_create(
                code=row.get('state_code', '').upper(),
                defaults={'name': row.get('state_name', row.get('state_code', ''))}
            )
            
            subject_area, _ = SubjectArea.objects.get_or_create(
                name=row.get('subject_area', 'Unknown')
            )
            
            # Handle grade levels (could be comma-separated)
            grade_levels = []
            grade_str = str(row.get('grade_level', ''))
            for grade in grade_str.split(','):
                grade = grade.strip()
                if grade:
                    # Convert K to 0 for numeric representation
                    grade_numeric = 0 if grade.upper() == 'K' else int(grade) if grade.isdigit() else 0
                    grade_level, _ = GradeLevel.objects.get_or_create(
                        grade=grade,
                        defaults={'grade_numeric': grade_numeric}
                    )
                    grade_levels.append(grade_level)
            
            # Create or update standard
            with transaction.atomic():
                standard, created = Standard.objects.update_or_create(
                    state=state,
                    code=row.get('code', ''),
                    defaults={
                        'subject_area': subject_area,
                        'title': row.get('title', ''),
                        'description': row.get('description', ''),
                        'domain': row.get('domain', ''),
                        'cluster': row.get('cluster', ''),
                        'keywords': row.get('keywords', '').split(',') if row.get('keywords') else [],
                        'skills': row.get('skills', '').split(',') if row.get('skills') else [],
                    }
                )
                
                # Set grade levels
                if grade_levels:
                    standard.grade_levels.set(grade_levels)
                
                successful += 1
                
        except Exception as e:
            failed += 1
            upload_job.add_error(
                f"Error processing row {idx}: {str(e)}",
                line_number=idx + 2,  # +2 for header and 0-indexing
                record_data=row.to_dict()
            )
    
    return {'successful': successful, 'failed': failed}


def _generate_embedding_for_standard(standard, embedding_service):
    """Generate embedding for a single standard"""
    try:
        # Combine relevant text fields
        text_parts = [
            standard.code,
            standard.title or '',
            standard.description,
            standard.domain or '',
            standard.cluster or '',
        ]
        
        # Add keywords and skills if available
        if hasattr(standard, 'keywords') and standard.keywords:
            text_parts.append(' '.join(standard.keywords))
        if hasattr(standard, 'skills') and standard.skills:
            text_parts.append(' '.join(standard.skills))
        
        combined_text = ' '.join(filter(None, text_parts))
        
        # Generate embedding
        embedding = embedding_service.generate_embedding(combined_text)
        
        # Save to database
        standard.embedding = embedding
        standard.save(update_fields=['embedding'])
        
        return True
    except Exception as e:
        logger.error(f"Error generating embedding for standard {standard.code}: {str(e)}")
        raise