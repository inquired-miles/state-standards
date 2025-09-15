"""
Management command to discover topic clusters across states
"""
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from standards.models import SubjectArea, GradeLevel, TopicCluster
from standards.services.discovery import TopicDiscoveryService


class Command(BaseCommand):
    help = 'Discover topic clusters across educational standards'

    def add_arguments(self, parser):
        parser.add_argument(
            '--subject-area',
            type=str,
            help='Filter by subject area name (e.g., "Mathematics", "English Language Arts")'
        )
        parser.add_argument(
            '--grade-level',
            type=str,
            help='Filter by grade level name (e.g., "1st Grade", "High School")'
        )
        parser.add_argument(
            '--min-standards',
            type=int,
            default=5,
            help='Minimum number of standards required to form a topic cluster (default: 5)'
        )
        parser.add_argument(
            '--analysis-type',
            type=str,
            choices=['discover', 'update_coverage', 'cluster_refinement', 'full_analysis'],
            default='discover',
            help='Type of analysis to run (default: discover)'
        )
        parser.add_argument(
            '--save-results',
            action='store_true',
            help='Save discovered topics as TopicCluster objects in the database'
        )
        parser.add_argument(
            '--clear-existing',
            action='store_true',
            help='Clear existing TopicCluster objects before saving new results'
        )

    def handle(self, *args, **options):
        service = TopicDiscoveryService()
        
        # Get filters
        subject_area = None
        if options['subject_area']:
            try:
                subject_area = SubjectArea.objects.get(name__icontains=options['subject_area'])
                self.stdout.write(f"Filtering by subject area: {subject_area.name}")
            except SubjectArea.DoesNotExist:
                self.stderr.write(
                    self.style.ERROR(
                        f'Subject area "{options["subject_area"]}" not found. '
                        f'Available: {", ".join(SubjectArea.objects.values_list("name", flat=True))}'
                    )
                )
                return
        
        grade_level = None
        if options['grade_level']:
            try:
                grade_level = GradeLevel.objects.get(name__icontains=options['grade_level'])
                self.stdout.write(f"Filtering by grade level: {grade_level.name}")
            except GradeLevel.DoesNotExist:
                self.stderr.write(
                    self.style.ERROR(
                        f'Grade level "{options["grade_level"]}" not found. '
                        f'Available: {", ".join(GradeLevel.objects.values_list("name", flat=True))}'
                    )
                )
                return
        
        min_standards = options['min_standards']
        analysis_type = options['analysis_type']
        
        self.stdout.write(f"Running {analysis_type} analysis with min {min_standards} standards per cluster...")
        
        try:
            # Run the appropriate analysis
            if analysis_type == 'discover':
                results = service.discover_topics(
                    subject_area=subject_area,
                    grade_level=grade_level,
                    min_standards=min_standards
                )
            elif analysis_type == 'update_coverage':
                results = service.update_coverage_statistics()
            elif analysis_type == 'cluster_refinement':
                results = service.refine_clusters()
            else:  # full_analysis
                results = service.run_full_analysis(
                    subject_area=subject_area,
                    min_standards=min_standards
                )
            
            if not results:
                self.stdout.write(
                    self.style.WARNING(
                        f'No topics discovered. Try reducing min_standards or checking your filters.'
                    )
                )
                return
            
            # Display results
            self.stdout.write(
                self.style.SUCCESS(f'\nDiscovered {len(results)} topic clusters:')
            )
            
            for i, topic in enumerate(results, 1):
                self.stdout.write(f"\n{i}. {topic['name']}")
                self.stdout.write(f"   Description: {topic['description']}")
                self.stdout.write(f"   Standards: {topic['standards_count']}")
                self.stdout.write(f"   States: {topic['states_represented']}")
                self.stdout.write(f"   Similarity Score: {topic['silhouette_score']:.3f}")
                
                if topic.get('common_terms'):
                    self.stdout.write(f"   Common Terms: {', '.join(topic['common_terms'])}")
            
            # Save results if requested
            if options['save_results']:
                self._save_results(results, subject_area, options['clear_existing'])
                
        except Exception as e:
            raise CommandError(f'Topic discovery failed: {str(e)}')
    
    @transaction.atomic
    def _save_results(self, results, subject_area, clear_existing):
        """Save discovered topics to the database"""
        
        if clear_existing:
            deleted_count = TopicCluster.objects.all().delete()[0]
            self.stdout.write(f"Cleared {deleted_count} existing topic clusters")
        
        created_count = 0
        updated_count = 0
        
        for topic_data in results:
            cluster, created = TopicCluster.objects.update_or_create(
                name=topic_data['name'],
                defaults={
                    'description': topic_data['description'],
                    'subject_area': subject_area,
                    'standards_count': topic_data['standards_count'],
                    'states_represented': topic_data['states_represented'],
                    'silhouette_score': topic_data['silhouette_score'],
                }
            )
            
            if created:
                created_count += 1
            else:
                updated_count += 1
        
        self.stdout.write(
            self.style.SUCCESS(
                f'\nSaved results: {created_count} created, {updated_count} updated'
            )
        )