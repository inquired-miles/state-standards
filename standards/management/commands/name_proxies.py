from django.core.management.base import BaseCommand
from standards.models import ProxyStandard
from standards.services.naming import ProxyNamingService


class Command(BaseCommand):
    help = 'Generate names and descriptions for ProxyStandard records (uses OpenAI if configured)'

    def add_arguments(self, parser):
        parser.add_argument('--limit', type=int, default=0)
        parser.add_argument('--force', action='store_true', help='Rename even if a title already exists')
        parser.add_argument('--verbose', action='store_true', help='Print naming inputs/outputs for debugging')

    def handle(self, *args, **options):
        limit = int(options.get('limit') or 0)
        force = bool(options.get('force') or False)
        verbose = bool(options.get('verbose') or False)
        qs = ProxyStandard.objects.all().order_by('proxy_id') if force else ProxyStandard.objects.filter(title="").order_by('proxy_id')
        if limit > 0:
            qs = qs[:limit]
        service = ProxyNamingService()
        count = 0
        for proxy in qs:
            named = service.name_proxy(proxy)
            if verbose:
                self.stdout.write(f"{proxy.proxy_id}: {named}")
            proxy.title = named.get('title') or proxy.title
            proxy.description = named.get('description') or proxy.description
            proxy.save(update_fields=['title', 'description'])
            count += 1
        self.stdout.write(self.style.SUCCESS(f"Named {count} proxies"))



