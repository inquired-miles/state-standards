"""
Management command to generate embeddings for all standards
"""
from django.core.management.base import BaseCommand
from django.db.models import Q
from standards.models import Standard
from standards.services import EmbeddingService
from standards.tasks import generate_all_embeddings_distributed, generate_embeddings_batch_task
import time


class Command(BaseCommand):
    help = 'Generate embeddings for all standards that do not have them'

    def add_arguments(self, parser):
        parser.add_argument(
            '--batch-size',
            type=int,
            default=50,
            help='Number of standards to process in each batch (default: 50)'
        )
        parser.add_argument(
            '--use-celery',
            action='store_true',
            help='Use Celery for distributed processing (requires Celery workers running)'
        )
        parser.add_argument(
            '--state',
            type=str,
            help='Only generate embeddings for standards from a specific state'
        )

    def handle(self, *args, **options):
        batch_size = options['batch_size']
        use_celery = options['use_celery']
        state_filter = options.get('state')
        
        # Get standards without embeddings
        # Note: pgvector doesn't allow comparison with empty array, so just check for null
        query = Standard.objects.filter(embedding__isnull=True)
        
        if state_filter:
            query = query.filter(state__code=state_filter.upper())
        
        total_standards = query.count()
        
        if total_standards == 0:
            self.stdout.write(
                self.style.SUCCESS('All standards already have embeddings!')
            )
            return
        
        self.stdout.write(
            self.style.SUCCESS(f'Found {total_standards} standards without embeddings')
        )
        
        if use_celery:
            # Use Celery for distributed processing
            self.stdout.write(
                self.style.SUCCESS('Starting distributed embedding generation with Celery...')
            )
            
            result = generate_all_embeddings_distributed()
            
            self.stdout.write(
                self.style.SUCCESS(
                    f"Distributed task started: {result['num_chunks']} chunks created for {result['total_standards']} standards"
                )
            )
            self.stdout.write(
                self.style.WARNING(
                    'Tasks are being processed by Celery workers. Check worker logs for progress.'
                )
            )
            
        else:
            # Use synchronous batch processing
            self.stdout.write(
                self.style.SUCCESS(f'Starting batch embedding generation (batch size: {batch_size})...')
            )
            
            service = EmbeddingService()
            processed = 0
            successful = 0
            failed = 0
            
            # Process in batches
            for i in range(0, total_standards, batch_size):
                batch = list(query[i:i+batch_size])
                
                self.stdout.write(f'Processing batch {i//batch_size + 1} ({len(batch)} standards)...')
                
                # Generate embeddings for batch
                start_time = time.time()
                results = service.generate_standards_embeddings_batch(batch)
                batch_time = time.time() - start_time
                
                # Save embeddings to database
                embeddings_map = results['embeddings']
                batch_successful = 0
                
                for standard in batch:
                    if standard.id in embeddings_map:
                        standard.embedding = embeddings_map[standard.id]
                        standard.save(update_fields=['embedding'])
                        batch_successful += 1
                        successful += 1
                    else:
                        failed += 1
                        self.stdout.write(
                            self.style.WARNING(f'  Failed: {standard.code}')
                        )
                
                processed += len(batch)
                
                # Progress update
                progress = (processed / total_standards) * 100
                self.stdout.write(
                    self.style.SUCCESS(
                        f'  Batch complete in {batch_time:.2f}s - Progress: {progress:.1f}% '
                        f'({successful} successful, {failed} failed)'
                    )
                )
                
                # Small delay to avoid rate limiting
                if i + batch_size < total_standards:
                    time.sleep(0.5)
            
            # Final summary
            self.stdout.write(
                self.style.SUCCESS(
                    f'\nEmbedding generation complete!\n'
                    f'  Total processed: {processed}\n'
                    f'  Successful: {successful}\n'
                    f'  Failed: {failed}'
                )
            )
            
            if failed > 0:
                self.stdout.write(
                    self.style.WARNING(
                        f'Some embeddings failed to generate. '
                        f'You may want to run this command again to retry failed items.'
                    )
                )