"""
Management command to create correlations between standards
"""
from django.core.management.base import BaseCommand
from standards.services import StandardCorrelationService


class Command(BaseCommand):
    help = 'Create correlations between standards based on similarity'

    def add_arguments(self, parser):
        parser.add_argument(
            '--threshold',
            type=float,
            default=0.8,
            help='Minimum similarity threshold for creating correlations (default: 0.8)'
        )
        parser.add_argument(
            '--standard-code',
            type=str,
            help='Process only a specific standard by its code'
        )

    def handle(self, *args, **options):
        threshold = options['threshold']
        standard_code = options.get('standard_code')
        
        self.stdout.write(
            self.style.SUCCESS(f'Starting correlation creation with threshold {threshold}...')
        )
        
        service = StandardCorrelationService()
        
        try:
            if standard_code:
                from standards.models import Standard
                try:
                    standard = Standard.objects.get(code=standard_code)
                    self.stdout.write(f'Processing single standard: {standard_code}')
                    service.create_correlations_for_standard(standard, threshold)
                except Standard.DoesNotExist:
                    self.stdout.write(
                        self.style.ERROR(f'Standard with code {standard_code} not found')
                    )
                    return
            else:
                service.batch_create_correlations(threshold)
            
            self.stdout.write(
                self.style.SUCCESS('Successfully created correlations')
            )
        except Exception as e:
            import traceback
            self.stdout.write(
                self.style.ERROR(f'Error creating correlations: {e}')
            )
            self.stdout.write(traceback.format_exc())