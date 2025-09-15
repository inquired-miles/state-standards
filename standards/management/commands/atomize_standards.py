from django.core.management.base import BaseCommand
from standards.models import Standard, State
from standards.services.atomizer import AtomizationService


class Command(BaseCommand):
    help = 'Generate StandardAtom records from Standards (rule-based or GPT-assisted)'

    def add_arguments(self, parser):
        parser.add_argument('--state', type=str, help='Filter by two-letter state code')
        parser.add_argument('--use-gpt', action='store_true', help='Use GPT to assist atomization when available')
        parser.add_argument('--limit', type=int, default=0, help='Limit number of standards to process')

    def handle(self, *args, **options):
        state_code = options.get('state')
        use_gpt = options.get('use_gpt', False)
        limit = options.get('limit') or 0

        qs = Standard.objects.all().order_by('id')
        if state_code:
            qs = qs.filter(state__code=state_code.upper())

        if limit > 0:
            qs = qs[:limit]

        service = AtomizationService()
        total = 0
        created_atoms = 0

        for standard in qs.iterator():
            atoms = service.generate_atoms_for_standard(standard, use_gpt=use_gpt)
            total += 1
            created_atoms += len(atoms)
            if total % 25 == 0:
                self.stdout.write(f"Processed {total} standards; atoms so far: {created_atoms}")

        self.stdout.write(self.style.SUCCESS(f"Atomization complete. Standards: {total}, Atoms: {created_atoms}"))



