"""
Management command to load sample state standards data
"""
from django.core.management.base import BaseCommand
from standards.models import State, SubjectArea, GradeLevel, Standard


class Command(BaseCommand):
    help = 'Load sample state standards data for testing'

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('Loading sample state standards data...')
        )
        
        # Create sample states
        states_data = [
            ('CA', 'California'),
            ('TX', 'Texas'),
            ('NY', 'New York'),
            ('FL', 'Florida'),
        ]
        
        states = {}
        for code, name in states_data:
            state, created = State.objects.get_or_create(code=code, defaults={'name': name})
            states[code] = state
            if created:
                self.stdout.write(f'Created state: {name}')
        
        # Create subject areas
        subjects_data = [
            ('Mathematics', 'Mathematical concepts and problem-solving skills'),
            ('English Language Arts', 'Reading, writing, speaking, and listening skills'),
            ('Science', 'Scientific inquiry and understanding of natural phenomena'),
            ('Social Studies', 'History, geography, civics, and social sciences'),
        ]
        
        subjects = {}
        for name, description in subjects_data:
            subject, created = SubjectArea.objects.get_or_create(
                name=name, 
                defaults={'description': description}
            )
            subjects[name] = subject
            if created:
                self.stdout.write(f'Created subject: {name}')
        
        # Create grade levels
        grades_data = [
            ('K', 0), ('1', 1), ('2', 2), ('3', 3), ('4', 4), ('5', 5),
            ('6', 6), ('7', 7), ('8', 8), ('9', 9), ('10', 10), ('11', 11), ('12', 12)
        ]
        
        grade_levels = {}
        for grade, numeric in grades_data:
            grade_level, created = GradeLevel.objects.get_or_create(
                grade=grade, 
                defaults={'grade_numeric': numeric}
            )
            grade_levels[grade] = grade_level
            if created:
                self.stdout.write(f'Created grade level: {grade}')
        
        # Sample standards data
        sample_standards = [
            {
                'state': 'CA',
                'subject': 'Mathematics',
                'grades': ['1'],
                'code': 'CA.1.OA.1',
                'title': 'Addition and Subtraction within 20',
                'description': 'Use addition and subtraction within 20 to solve word problems involving situations of adding to, taking from, putting together, taking apart, and comparing.',
                'domain': 'Operations and Algebraic Thinking',
                'cluster': 'Represent and solve problems involving addition and subtraction',
                'keywords': ['addition', 'subtraction', 'word problems', 'within 20'],
                'skills': ['problem solving', 'arithmetic operations', 'mathematical reasoning']
            },
            {
                'state': 'TX',
                'subject': 'Mathematics',
                'grades': ['1'],
                'code': 'TX.1.3A',
                'title': 'Addition and Subtraction Problem Solving',
                'description': 'Apply mathematics to problems arising in everyday life, society, and the workplace using addition and subtraction within 20.',
                'domain': 'Number and Operations',
                'cluster': 'Addition and Subtraction',
                'keywords': ['addition', 'subtraction', 'problem solving', 'everyday life'],
                'skills': ['real-world application', 'arithmetic operations', 'critical thinking']
            },
            {
                'state': 'CA',
                'subject': 'English Language Arts',
                'grades': ['2'],
                'code': 'CA.2.RL.1',
                'title': 'Reading Comprehension',
                'description': 'Ask and answer such questions as who, what, where, when, why, and how to demonstrate understanding of key details in a text.',
                'domain': 'Reading Literature',
                'cluster': 'Key Ideas and Details',
                'keywords': ['reading comprehension', 'questions', 'key details', 'text analysis'],
                'skills': ['reading comprehension', 'critical thinking', 'text analysis']
            },
            {
                'state': 'NY',
                'subject': 'English Language Arts',
                'grades': ['2'],
                'code': 'NY.2.RL.1',
                'title': 'Understanding Key Details',
                'description': 'Ask and answer questions about key details in a text read aloud or information presented orally or through other media.',
                'domain': 'Reading Literature',
                'cluster': 'Key Ideas and Details',
                'keywords': ['key details', 'questions', 'text comprehension', 'oral information'],
                'skills': ['listening comprehension', 'reading comprehension', 'information processing']
            },
        ]
        
        # Create sample standards
        created_count = 0
        for std_data in sample_standards:
            standard, created = Standard.objects.get_or_create(
                state=states[std_data['state']],
                code=std_data['code'],
                defaults={
                    'subject_area': subjects[std_data['subject']],
                    'title': std_data['title'],
                    'description': std_data['description'],
                    'domain': std_data['domain'],
                    'cluster': std_data['cluster'],
                    'keywords': std_data['keywords'],
                    'skills': std_data['skills'],
                }
            )
            
            if created:
                # Add grade levels
                for grade in std_data['grades']:
                    standard.grade_levels.add(grade_levels[grade])
                
                created_count += 1
                self.stdout.write(f'Created standard: {std_data["code"]}')
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully loaded {created_count} sample standards')
        )