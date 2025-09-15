"""
Secure API views for atomization and embeddings generation with proper resource management
"""
import uuid
import time
import threading
from typing import Dict, List, Optional, Any
from django.http import JsonResponse
from django.core.exceptions import ValidationError
from django.db import transaction
from django.core.cache import cache
from django.utils import timezone
import logging

from standards.models import Standard, StandardAtom
from standards.services.atomizer import AtomizationService
from standards.services.embedding import EmbeddingService
from standards.services.clustering import ClusteringService
from standards.services.naming import ProxyNamingService
from standards.models import ProxyStandard
from .base import BaseAPIView, api_endpoint

logger = logging.getLogger(__name__)


class AtomizationAPIView(BaseAPIView):
    """Base class for atomization-related API views with resource management"""
    
    MAX_PROCESSING_LIMIT = 1000  # Maximum standards to process in one job
    DEFAULT_BATCH_SIZE = 50     # Conservative batch size for memory management
    
    def validate_processing_params(self, state_code: Optional[str], limit: int, 
                                 batch_size: int) -> tuple:
        """Validate parameters for processing operations"""
        if state_code:
            state_code = self.validate_string(state_code, max_length=2, field_name="state").upper()
        
        # Validate and cap limits for resource management
        limit = min(
            self.validate_integer(limit, 0, self.MAX_PROCESSING_LIMIT, "limit"),
            self.MAX_PROCESSING_LIMIT
        ) if limit > 0 else 0
        
        batch_size = min(
            self.validate_integer(batch_size, 10, 500, "batch_size"),
            200  # Cap batch size for memory management
        )
        
        return state_code, limit, batch_size


@api_endpoint(['POST'])
def atomize_standards_api(request):
    """Start atomization job with comprehensive validation and resource management"""
    view = AtomizationAPIView()
    
    try:
        data = view.parse_json_body(request)
        
        # Validate parameters
        state_code = data.get('state')
        use_gpt = bool(data.get('use_gpt', False))
        limit = int(data.get('limit', 0))
        
        state_code, limit, _ = view.validate_processing_params(state_code, limit, 50)
        
        # Validate scope - check if standards exist
        standards_query = Standard.objects.all().order_by('id')
        if state_code:
            standards_query = standards_query.filter(state__code=state_code)
        
        if limit > 0:
            standards_query = standards_query[:limit]
        
        total_standards = standards_query.count()
        
        if total_standards == 0:
            return view.error_response(
                f"No standards found for the specified criteria (state: {state_code or 'all'})",
                status=400,
                error_code='NO_STANDARDS'
            )
        
        # Generate job ID and start processing
        job_id = str(uuid.uuid4())
        cache_key = f'atomize_job_{job_id}'
        
        cache.set(cache_key, {
            'status': 'queued',
            'progress': 0,
            'message': f'Atomization job queued for {total_standards} standards'
        }, 3600)
        
        # Start background thread
        thread = threading.Thread(
            target=_run_atomize_job_secure,
            args=(job_id, state_code, use_gpt, limit)
        )
        thread.daemon = True
        thread.start()
        
        return view.success_response({
            'job_id': job_id,
            'status': 'queued',
            'parameters': {
                'state_code': state_code,
                'use_gpt': use_gpt,
                'limit': limit,
                'estimated_standards': total_standards
            }
        })
        
    except ValidationError as e:
        return view.validation_error_response(str(e))
    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['GET'])
def atomize_job_status_api(request, job_id: str):
    """Get atomization job status with validation"""
    view = AtomizationAPIView()
    
    try:
        # Validate job_id format
        try:
            uuid.UUID(job_id)
        except ValueError:
            return view.error_response('Invalid job ID format', status=400)
        
        cache_key = f'atomize_job_{job_id}'
        data = cache.get(cache_key)
        
        if not data:
            return view.error_response('Job not found', status=404)
        
        return view.success_response(data)
        
    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['POST'])
def generate_atom_embeddings_api(request):
    """Generate embeddings for atoms with batch processing and memory management"""
    view = AtomizationAPIView()
    
    try:
        data = view.parse_json_body(request)
        
        # Validate parameters
        state_code = data.get('state')
        batch_size = int(data.get('batch_size', view.DEFAULT_BATCH_SIZE))
        
        state_code, _, batch_size = view.validate_processing_params(state_code, 0, batch_size)
        
        # Check scope
        atoms_query = StandardAtom.objects.filter(embedding__isnull=True)
        if state_code:
            atoms_query = atoms_query.filter(standard__state__code=state_code)
        
        total_atoms = atoms_query.count()
        
        if total_atoms == 0:
            return view.success_response({
                'message': 'No atoms require embeddings',
                'total_atoms': 0
            })
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        cache_key = f'atom_embeddings_job_{job_id}'
        
        cache.set(cache_key, {
            'status': 'queued',
            'progress': 0,
            'message': f'Embedding generation queued for {total_atoms} atoms'
        }, 3600)
        
        # Start background thread
        thread = threading.Thread(
            target=_run_atom_embeddings_job_secure,
            args=(job_id, state_code, batch_size)
        )
        thread.daemon = True
        thread.start()
        
        return view.success_response({
            'job_id': job_id,
            'status': 'queued',
            'parameters': {
                'state_code': state_code,
                'batch_size': batch_size,
                'estimated_atoms': total_atoms
            }
        })
        
    except ValidationError as e:
        return view.validation_error_response(str(e))
    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['GET'])
def atom_embeddings_job_status_api(request, job_id: str):
    """Get embedding generation job status"""
    view = AtomizationAPIView()
    
    try:
        # Validate job_id format
        try:
            uuid.UUID(job_id)
        except ValueError:
            return view.error_response('Invalid job ID format', status=400)
        
        cache_key = f'atom_embeddings_job_{job_id}'
        data = cache.get(cache_key)
        
        if not data:
            return view.error_response('Job not found', status=404)
        
        return view.success_response(data)
        
    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['POST'])
def run_proxy_pipeline_api(request):
    """Run full pipeline: atomize -> embeddings -> clustering with validation"""
    view = AtomizationAPIView()
    
    try:
        data = view.parse_json_body(request)
        
        # Validate pipeline configuration
        state_code = data.get('state')
        use_gpt = bool(data.get('use_gpt', False))
        limit = int(data.get('limit', 0))
        batch_size = int(data.get('batch_size', view.DEFAULT_BATCH_SIZE))
        min_cluster = view.validate_integer(data.get('min_cluster', 8), 2, 100, "min_cluster")
        epsilon = view.validate_float(data.get('epsilon', 0.15), 0.01, 1.0, "epsilon")
        name_proxies = bool(data.get('name_proxies', False))
        
        # Validate processing parameters
        state_code, limit, batch_size = view.validate_processing_params(state_code, limit, batch_size)
        
        # Parse grade selection
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
        
        # Validate scope
        standards_query = Standard.objects.all()
        if state_code:
            standards_query = standards_query.filter(state__code=state_code)
        if limit > 0:
            standards_query = standards_query[:limit]
        
        total_standards = standards_query.count()
        
        if total_standards == 0:
            return view.error_response(
                "No standards found for pipeline processing",
                status=400,
                error_code='NO_STANDARDS'
            )
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        cache_key = f'proxy_pipeline_job_{job_id}'
        
        pipeline_config = {
            'state': state_code,
            'use_gpt': use_gpt,
            'limit': limit,
            'batch_size': batch_size,
            'min_cluster': min_cluster,
            'epsilon': epsilon,
            'name_proxies': name_proxies,
            'grade_selection': grade_selection
        }
        
        cache.set(cache_key, {
            'status': 'queued',
            'progress': 0,
            'message': f'Pipeline queued for {total_standards} standards'
        }, 3600)
        
        # Start background thread
        thread = threading.Thread(
            target=_run_proxy_pipeline_job_secure,
            args=(job_id, pipeline_config)
        )
        thread.daemon = True
        thread.start()
        
        return view.success_response({
            'job_id': job_id,
            'status': 'queued',
            'pipeline_config': pipeline_config,
            'estimated_standards': total_standards
        })
        
    except ValidationError as e:
        return view.validation_error_response(str(e))
    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['GET'])
def proxy_pipeline_status_api(request, job_id: str):
    """Get pipeline job status"""
    view = AtomizationAPIView()
    
    try:
        # Validate job_id format
        try:
            uuid.UUID(job_id)
        except ValueError:
            return view.error_response('Invalid job ID format', status=400)
        
        cache_key = f'proxy_pipeline_job_{job_id}'
        data = cache.get(cache_key)
        
        if not data:
            return view.error_response('Job not found', status=404)
        
        return view.success_response(data)
        
    except Exception as e:
        return view.handle_exception(request, e)


def _run_atomize_job_secure(job_id: str, state_code: Optional[str] = None,
                           use_gpt: bool = False, limit: int = 0):
    """Secure atomization job with comprehensive error handling and resource management"""
    cache_key = f'atomize_job_{job_id}'
    
    def update_progress(progress: int, message: str):
        cache.set(cache_key, {
            'status': 'running',
            'progress': progress,
            'message': message
        }, 3600)
    
    try:
        update_progress(5, 'Starting atomization process...')
        
        # Build query with proper filtering
        standards_query = Standard.objects.all().order_by('id')
        if state_code:
            standards_query = standards_query.filter(state__code=state_code)
        if limit and limit > 0:
            standards_query = standards_query[:limit]
        
        total = standards_query.count()
        
        if total == 0:
            cache.set(cache_key, {
                'status': 'completed',
                'progress': 100,
                'message': 'No standards found to atomize'
            }, 3600)
            return
        
        update_progress(10, f'Processing {total} standards...')
        
        service = AtomizationService()
        processed = 0
        failed = 0
        
        # Process in batches for memory management
        batch_size = 10
        for i in range(0, total, batch_size):
            batch_standards = standards_query[i:i + batch_size]
            
            for standard in batch_standards:
                try:
                    with transaction.atomic():
                        service.generate_atoms_for_standard(standard, use_gpt=use_gpt)
                    processed += 1
                except Exception as e:
                    logger.warning(f"Failed to atomize standard {standard.id}: {e}")
                    failed += 1
                
                # Update progress every 10 standards
                if (processed + failed) % 10 == 0:
                    progress = min(95, int(((processed + failed) / total) * 90) + 10)
                    update_progress(
                        progress,
                        f'Atomized {processed}/{total} standards ({failed} failed)'
                    )
        
        # Final update
        success_rate = (processed / total * 100) if total > 0 else 0
        message = f'Completed: {processed} atomized, {failed} failed (success rate: {success_rate:.1f}%)'
        
        cache.set(cache_key, {
            'status': 'completed',
            'progress': 100,
            'message': message,
            'results': {
                'total_standards': total,
                'processed': processed,
                'failed': failed,
                'success_rate': round(success_rate, 1)
            }
        }, 3600)
        
    except Exception as e:
        logger.error(f"Atomization job {job_id} failed: {str(e)}", exc_info=True)
        cache.set(cache_key, {
            'status': 'failed',
            'progress': 0,
            'message': f'Atomization failed: {str(e)}'
        }, 3600)


def _run_atom_embeddings_job_secure(job_id: str, state_code: Optional[str] = None,
                                   batch_size: int = 50):
    """Secure embedding generation with memory management and error handling"""
    cache_key = f'atom_embeddings_job_{job_id}'
    
    def update_progress(progress: int, message: str):
        cache.set(cache_key, {
            'status': 'running',
            'progress': progress,
            'message': message
        }, 3600)
    
    try:
        update_progress(5, 'Starting embedding generation...')
        
        # Get atoms that need embeddings
        atoms_query = StandardAtom.objects.filter(
            embedding__isnull=True
        ).select_related('standard__state')
        
        if state_code:
            atoms_query = atoms_query.filter(standard__state__code=state_code)
        
        # Filter out atoms with empty/blank text
        atoms = [atom for atom in atoms_query if atom.text and atom.text.strip()]
        total = len(atoms)
        
        if total == 0:
            cache.set(cache_key, {
                'status': 'completed',
                'progress': 100,
                'message': 'No atoms require embeddings'
            }, 3600)
            return
        
        update_progress(10, f'Generating embeddings for {total} atoms...')
        
        embedder = EmbeddingService()
        
        if not embedder.is_available():
            raise Exception("Embedding service is not available (check OpenAI API key)")
        
        updated = 0
        failed = 0
        
        # Process in smaller batches for memory management
        effective_batch_size = min(batch_size, 25)  # Cap at 25 for safety
        
        for i in range(0, total, effective_batch_size):
            batch_atoms = atoms[i:i + effective_batch_size]
            batch_texts = [atom.text for atom in batch_atoms]
            
            try:
                # Generate embeddings for batch
                embeddings = embedder.generate_batch_embeddings(batch_texts)
                
                # Update atoms with embeddings
                atoms_to_update = []
                for atom, embedding in zip(batch_atoms, embeddings):
                    if embedding:
                        atom.embedding = embedding
                        atoms_to_update.append(atom)
                        updated += 1
                    else:
                        failed += 1
                
                # Batch update for efficiency
                if atoms_to_update:
                    with transaction.atomic():
                        StandardAtom.objects.bulk_update(
                            atoms_to_update,
                            ['embedding'],
                            batch_size=effective_batch_size
                        )
                
                # Update progress
                progress = min(95, int((i + effective_batch_size) / total * 85) + 10)
                update_progress(
                    progress,
                    f'Generated embeddings: {updated}/{total} ({failed} failed)'
                )
                
            except Exception as e:
                logger.error(f"Batch embedding generation failed: {e}")
                failed += len(batch_atoms)
        
        # Final update
        success_rate = (updated / total * 100) if total > 0 else 0
        message = f'Completed: {updated} embeddings generated, {failed} failed (success rate: {success_rate:.1f}%)'
        
        cache.set(cache_key, {
            'status': 'completed',
            'progress': 100,
            'message': message,
            'results': {
                'total_atoms': total,
                'updated': updated,
                'failed': failed,
                'success_rate': round(success_rate, 1)
            }
        }, 3600)
        
    except Exception as e:
        logger.error(f"Atom embeddings job {job_id} failed: {str(e)}", exc_info=True)
        cache.set(cache_key, {
            'status': 'failed',
            'progress': 0,
            'message': f'Embedding generation failed: {str(e)}'
        }, 3600)


def _run_proxy_pipeline_job_secure(job_id: str, pipeline_cfg: Dict[str, Any]):
    """Secure full pipeline execution with comprehensive error handling"""
    cache_key = f'proxy_pipeline_job_{job_id}'
    
    def update_progress(progress: int, message: str):
        cache.set(cache_key, {
            'status': 'running',
            'progress': progress,
            'message': message
        }, 3600)
    
    try:
        # Step 1: Atomization
        update_progress(5, 'Step 1: Atomizing standards...')
        _run_atomize_job_secure(
            job_id,
            pipeline_cfg.get('state'),
            pipeline_cfg.get('use_gpt', False),
            int(pipeline_cfg.get('limit', 0))
        )
        
        # Step 2: Embedding generation
        update_progress(35, 'Step 2: Generating embeddings...')
        _run_atom_embeddings_job_secure(
            job_id,
            pipeline_cfg.get('state'),
            int(pipeline_cfg.get('batch_size', 50))
        )
        
        # Step 3: Clustering
        update_progress(65, 'Step 3: Clustering into proxies...')
        
        # Parse grade levels for clustering
        grade_levels = None
        grade_selection = pipeline_cfg.get('grade_selection', {})
        if grade_selection.get('type') == 'specific':
            grade_levels = grade_selection.get('grades', [])
        elif grade_selection.get('type') == 'range':
            min_grade = grade_selection.get('min_grade')
            max_grade = grade_selection.get('max_grade')
            if min_grade is not None and max_grade is not None:
                grade_levels = list(range(int(min_grade), int(max_grade) + 1))
        
        # Run clustering
        svc = ClusteringService()
        results = svc.run_full(
            min_cluster_size=int(pipeline_cfg.get('min_cluster', 8)),
            epsilon=float(pipeline_cfg.get('epsilon', 0.15)),
            grade_levels=grade_levels
        )
        created = svc.persist_proxies(results)
        
        # Step 4: Naming (optional)
        if bool(pipeline_cfg.get('name_proxies', False)):
            update_progress(85, 'Step 4: Naming proxies...')
            namer = ProxyNamingService()
            to_name = ProxyStandard.objects.filter(title="")[:500]
            
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
        
        # Final completion
        cache.set(cache_key, {
            'status': 'completed',
            'progress': 100,
            'message': f'Pipeline complete. Created {created} proxies.',
            'results': {
                'proxies_created': created,
                'clusters_found': results.get('n_clusters', 0)
            }
        }, 3600)
        
    except Exception as e:
        logger.error(f"Pipeline job {job_id} failed: {str(e)}", exc_info=True)
        cache.set(cache_key, {
            'status': 'failed',
            'progress': 0,
            'message': f'Pipeline failed: {str(e)}'
        }, 3600)