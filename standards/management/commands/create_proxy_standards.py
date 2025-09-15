from django.core.management.base import BaseCommand
from standards.services.clustering import ClusteringService


class Command(BaseCommand):
    help = 'Cluster StandardAtom embeddings into ProxyStandards (UMAP + HDBSCAN)'

    def add_arguments(self, parser):
        parser.add_argument('--min-cluster', type=int, default=8)
        parser.add_argument('--epsilon', type=float, default=0.15)

    def handle(self, *args, **options):
        min_cluster = int(options.get('min_cluster') or 8)
        epsilon = float(options.get('epsilon') or 0.15)
        service = ClusteringService()
        results = service.run_full(min_cluster_size=min_cluster, epsilon=epsilon)
        created = service.persist_proxies(results)
        self.stdout.write(self.style.SUCCESS(
            f"Created {created} proxy standards (clusters={results['n_clusters']}, coverage={results['coverage']:.1%})"
        ))



