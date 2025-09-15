"""
Management command to regenerate embeddings with the new title-independent approach
"""
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from standards.models import Standard
from standards.services import EmbeddingService
import time


class Command(BaseCommand):
    help = 'Regenerate embeddings for all standards using the new title-independent approach'

    def add_arguments(self, parser):
        parser.add_argument(
            '--batch-size',
            type=int,
            default=50,
            help='Number of standards to process in each batch (default: 50)'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Preview the changes without actually updating embeddings'
        )
        parser.add_argument(
            '--compare',
            action='store_true',
            help='Compare old vs new embedding approach (requires existing embeddings)'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Regenerate embeddings even if they already exist'
        )

    def handle(self, *args, **options):
        batch_size = options['batch_size']
        dry_run = options['dry_run']
        compare = options['compare']
        force = options['force']

        self.stdout.write("Starting embedding regeneration with title-independent approach...")
        
        # Initialize services
        embedding_service = EmbeddingService()
        
        if not embedding_service.openai_client:
            raise CommandError("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")

        # Get standards to process
        if force:
            standards = Standard.objects.all()
        else:
            standards = Standard.objects.filter(embedding__isnull=True)
        
        total_count = standards.count()
        
        if total_count == 0:
            self.stdout.write(self.style.SUCCESS("No standards need embedding generation."))
            if not force:
                self.stdout.write("Use --force to regenerate existing embeddings.")
            return

        self.stdout.write(f"Found {total_count} standards to process")
        
        if dry_run:
            self.stdout.write(self.style.WARNING("DRY RUN MODE - No changes will be made"))
            self._preview_changes(standards[:10], embedding_service)
            return

        # Process standards in batches
        processed = 0
        errors = 0
        
        for i in range(0, total_count, batch_size):
            batch = standards[i:i + batch_size]
            
            self.stdout.write(f"Processing batch {i // batch_size + 1} ({i + 1}-{min(i + batch_size, total_count)} of {total_count})")
            
            with transaction.atomic():
                for standard in batch:
                    try:
                        if compare and standard.embedding:
                            self._compare_embeddings(standard, embedding_service)
                        
                        # Generate new embedding
                        new_embedding = embedding_service.generate_standard_embedding(standard)
                        
                        if new_embedding:
                            # Store the embedding directly in the model
                            standard.embedding = new_embedding
                            standard.save(update_fields=['embedding'])
                            processed += 1
                            
                            if processed % 10 == 0:
                                self.stdout.write(f"  Processed {processed}/{total_count} standards...")
                        else:
                            self.stdout.write(
                                self.style.WARNING(f"  Failed to generate embedding for {standard.code}")
                            )
                            errors += 1
                    
                    except Exception as e:
                        self.stdout.write(
                            self.style.ERROR(f"  Error processing {standard.code}: {str(e)}")
                        )
                        errors += 1
            
            # Small delay to avoid hitting API rate limits
            if i + batch_size < total_count:
                time.sleep(1)

        # Summary
        self.stdout.write(self.style.SUCCESS(f"\nCompleted embedding regeneration:"))
        self.stdout.write(f"  Successfully processed: {processed}")
        self.stdout.write(f"  Errors: {errors}")
        
        if errors > 0:
            self.stdout.write(
                self.style.WARNING(f"  {errors} standards failed to process. Check logs for details.")
            )

    def _preview_changes(self, standards_sample, embedding_service):
        """Preview what the new embedding approach would generate"""
        self.stdout.write("\nPreview of new embedding approach (first 10 standards):")
        self.stdout.write("-" * 80)
        
        for standard in standards_sample:
            self.stdout.write(f"\nCode: {standard.code}")
            self.stdout.write(f"State: {standard.state.code}")
            self.stdout.write(f"Title: {standard.title or '(None)'}")
            
            # Show what text would be used for embedding
            parts = []
            if standard.code:
                parts.append(f"Standard: {standard.code}")
            if standard.description:
                parts.append(standard.description[:100] + "..." if len(standard.description) > 100 else standard.description)
            if standard.domain:
                parts.append(f"Domain: {standard.domain}")
            if standard.cluster:
                parts.append(f"Cluster: {standard.cluster}")
            if standard.keywords:
                parts.append(f"Keywords: {', '.join(standard.keywords)}")
            if standard.title and standard.title.lower() not in standard.description.lower():
                parts.append(f"Also known as: {standard.title}")
            
            embedding_text = " | ".join(parts)
            self.stdout.write(f"Embedding text: {embedding_text[:200]}{'...' if len(embedding_text) > 200 else ''}")

    def _compare_embeddings(self, standard, embedding_service):
        """Compare old vs new embedding approach for analysis"""
        # This would be used to analyze the differences between title-dependent 
        # and title-independent embeddings
        pass