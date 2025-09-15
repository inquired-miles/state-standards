from django.core.management.base import BaseCommand
from django.db import transaction
from standards.models import StandardAtom
from standards.services.embedding import EmbeddingService


class Command(BaseCommand):
    help = 'Generate embeddings for StandardAtom records missing embeddings'

    def add_arguments(self, parser):
        parser.add_argument('--state', type=str, help='Filter atoms by Standard.state code')
        parser.add_argument('--batch-size', type=int, default=100, help='Batch size for API calls')

    def handle(self, *args, **options):
        state_code = options.get('state')
        batch_size = int(options.get('batch_size') or 100)

        qs = StandardAtom.objects.filter(embedding__isnull=True).select_related('standard__state')
        if state_code:
            qs = qs.filter(standard__state__code=state_code.upper())

        atoms = list(qs)
        total = len(atoms)
        if total == 0:
            self.stdout.write('No atoms require embeddings.')
            return

        self.stdout.write(f'Generating embeddings for {total} atoms...')
        embedder = EmbeddingService()

        texts = [a.text for a in atoms]
        embeddings = embedder.generate_batch_embeddings(texts)

        updated = 0
        for atom, emb in zip(atoms, embeddings):
            if emb:
                atom.embedding = emb
                atom.embedding_generated_at = None
                updated += 1

        # Bulk update for efficiency
        with transaction.atomic():
            StandardAtom.objects.bulk_update(atoms, ['embedding', 'embedding_generated_at'], batch_size=batch_size)

        self.stdout.write(self.style.SUCCESS(f'Updated {updated}/{total} atom embeddings'))



