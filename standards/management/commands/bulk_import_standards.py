"""
Management command to bulk import educational standards from all 50 states
"""
import csv
import json
import os

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from standards.models import State, SubjectArea, GradeLevel, Standard, StateFormat
from standards.services import EmbeddingService
from standards.utils.normalization import FormatDetector, StandardParser, ParsedStandard


class Command(BaseCommand):
    help = 'Bulk import educational standards from CSV or JSON files for all 50 states'

    def add_arguments(self, parser):
        parser.add_argument(
            'file_path',
            type=str,
            help='Path to the CSV or JSON file containing standards data'
        )
        parser.add_argument(
            '--format',
            type=str,
            default='auto',
            choices=['csv', 'json', 'auto'],
            help='File format (auto-detect by default)'
        )
        parser.add_argument(
            '--generate-embeddings',
            action='store_true',
            help='Generate embeddings for imported standards immediately'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=100,
            help='Number of standards to process in each batch'
        )
        parser.add_argument(
            '--clear-existing',
            action='store_true',
            help='Clear existing standards before importing'
        )
        parser.add_argument(
            '--detect-format',
            action='store_true',
            help='Auto-detect state code format patterns from sample'
        )
        parser.add_argument(
            '--normalize-codes',
            action='store_true',
            help='Parse and store normalized_code and hierarchy metadata on Standard'
        )

    def handle(self, *args, **options):
        file_path = options['file_path']
        
        if not os.path.exists(file_path):
            raise CommandError(f'File {file_path} does not exist')
        
        # Auto-detect format if needed
        file_format = options['format']
        if file_format == 'auto':
            file_format = 'json' if file_path.endswith('.json') else 'csv'
        
        self.stdout.write(f'Importing standards from {file_path} (format: {file_format})')
        
        # Clear existing standards if requested
        if options['clear_existing']:
            self.stdout.write('Clearing existing standards...')
            Standard.objects.all().delete()
            self.stdout.write(self.style.SUCCESS('Existing standards cleared'))
        
        # Load and process the file
        if file_format == 'json':
            standards_data = self.load_json_file(file_path)
        else:
            standards_data = self.load_csv_file(file_path)
        
        # Optional: detect format on first N codes per state
        if options['detect_format']:
            try:
                detector = FormatDetector()
                # Group sample codes by state for better signals
                by_state = {}
                for row in standards_data[:200]:
                    code = (row.get('code') or '').strip()
                    st = (row.get('state') or '').upper()
                    if st and code:
                        by_state.setdefault(st, []).append(code)
                for st, codes in by_state.items():
                    parser_type = detector.detect_format(codes[:20])
                    state_obj = State.objects.filter(code=st).first()
                    if state_obj:
                        StateFormat.objects.update_or_create(
                            state=state_obj,
                            parser_type=parser_type,
                            defaults={'format_pattern': '', 'example': codes[0] if codes else ''}
                        )
                        self.stdout.write(f"Detected {parser_type} for {st}")
            except Exception as e:
                self.stdout.write(self.style.WARNING(f"Format detection skipped: {e}"))

        # Import standards in batches
        total_imported = self.import_standards(
            standards_data,
            batch_size=options['batch_size'],
            normalize_codes=options['normalize_codes']
        )
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully imported {total_imported} standards')
        )
        
        # Generate embeddings if requested
        if options['generate_embeddings']:
            self.stdout.write('Generating embeddings for imported standards...')
            self.generate_embeddings_for_standards()
    
    def load_json_file(self, file_path):
        """Load standards data from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict):
            # If it's a dict, might be organized by state
            standards_data = []
            for state_code, state_standards in data.items():
                if isinstance(state_standards, list):
                    for standard in state_standards:
                        standard['state'] = state_code
                        standards_data.append(standard)
                else:
                    # Might be nested further by subject/grade
                    for subject, subject_standards in state_standards.items():
                        if isinstance(subject_standards, list):
                            for standard in subject_standards:
                                standard['state'] = state_code
                                standard['subject'] = subject
                                standards_data.append(standard)
            return standards_data
        elif isinstance(data, list):
            return data
        else:
            raise CommandError('Invalid JSON format')
    
    def load_csv_file(self, file_path):
        """Load standards data from CSV file"""
        standards_data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                standards_data.append(row)
        
        return standards_data
    
    def import_standards(self, standards_data, batch_size=100, normalize_codes=False):
        """Import standards data in batches"""
        total_imported = 0
        batch = []
        
        # Create or get all states first
        self.ensure_all_states_exist()
        
        # Create or get subject areas and grade levels
        subject_cache = {}
        grade_cache = {}
        
        for standard_data in standards_data:
            try:
                # Get or create state
                state_code = standard_data.get('state', '').upper()
                if len(state_code) != 2:
                    self.stdout.write(
                        self.style.WARNING(f'Invalid state code: {state_code}, skipping')
                    )
                    continue
                
                state = State.objects.filter(code=state_code).first()
                if not state:
                    self.stdout.write(
                        self.style.WARNING(f'State {state_code} not found, skipping')
                    )
                    continue
                
                # Get or create subject area
                subject_name = standard_data.get('subject', 'General').strip()
                if subject_name not in subject_cache:
                    subject, _ = SubjectArea.objects.get_or_create(
                        name=subject_name,
                        defaults={'description': f'{subject_name} subject area'}
                    )
                    subject_cache[subject_name] = subject
                else:
                    subject = subject_cache[subject_name]
                
                # Parse grade levels
                grade_str = standard_data.get('grade', '').strip()
                grade_levels = self.parse_grade_levels(grade_str, grade_cache)
                
                # Create standard object
                standard = Standard(
                    state=state,
                    subject_area=subject,
                    code=standard_data.get('code', '').strip(),
                    title=standard_data.get('title', '').strip(),
                    description=standard_data.get('description', '').strip(),
                    domain=standard_data.get('domain', '').strip(),
                    cluster=standard_data.get('cluster', '').strip()
                )

                # Optional normalization
                if normalize_codes and standard.code:
                    try:
                        parser = StandardParser()
                        parsed = parser.parse(standard.code, standard.description, default_grade=self._infer_default_grade(grade_levels))
                        standard.normalized_code = self._build_normalized_code(state.code, parsed)
                        standard.hierarchy = {
                            'format_type': parsed.format_type,
                            'grade': parsed.grade,
                            'parts': parsed.parts,
                        }
                    except Exception as e:
                        self.stdout.write(self.style.WARNING(f"Normalization failed for {standard.code}: {e}"))
                
                # Parse keywords and skills if present
                if 'keywords' in standard_data:
                    keywords = standard_data['keywords']
                    if isinstance(keywords, str):
                        standard.keywords = [k.strip() for k in keywords.split(',') if k.strip()]
                    elif isinstance(keywords, list):
                        standard.keywords = keywords
                
                if 'skills' in standard_data:
                    skills = standard_data['skills']
                    if isinstance(skills, str):
                        standard.skills = [s.strip() for s in skills.split(',') if s.strip()]
                    elif isinstance(skills, list):
                        standard.skills = skills
                
                batch.append((standard, grade_levels))
                
                # Process batch when it reaches the specified size
                if len(batch) >= batch_size:
                    self.process_batch(batch)
                    total_imported += len(batch)
                    self.stdout.write(f'Imported {total_imported} standards...')
                    batch = []
                    
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Error processing standard: {e}')
                )
                continue
        
        # Process remaining standards
        if batch:
            self.process_batch(batch)
            total_imported += len(batch)
        
        return total_imported
    
    def process_batch(self, batch):
        """Process a batch of standards"""
        with transaction.atomic():
            for standard, grade_levels in batch:
                try:
                    # Check if standard already exists
                    existing = Standard.objects.filter(
                        state=standard.state,
                        code=standard.code
                    ).first()
                    
                    if existing:
                        # Update existing standard
                        existing.title = standard.title
                        existing.description = standard.description
                        existing.domain = standard.domain
                        existing.cluster = standard.cluster
                        existing.keywords = standard.keywords
                        existing.skills = standard.skills
                        existing.save()
                        standard = existing
                    else:
                        # Save new standard
                        standard.save()
                    
                    # Set grade levels
                    if grade_levels:
                        standard.grade_levels.set(grade_levels)
                        
                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f'Error saving standard {standard.code}: {e}')
                    )
    
    def parse_grade_levels(self, grade_str, grade_cache):
        """Parse grade level string and return GradeLevel objects"""
        grade_levels = []
        
        if not grade_str:
            return grade_levels
        
        # Handle different grade formats
        grade_str = grade_str.upper().strip()
        
        # Handle range (e.g., "3-5", "K-2")
        if '-' in grade_str:
            parts = grade_str.split('-')
            if len(parts) == 2:
                start = parts[0].strip()
                end = parts[1].strip()
                
                start_num = 0 if start == 'K' else int(start) if start.isdigit() else None
                end_num = 0 if end == 'K' else int(end) if end.isdigit() else None
                
                if start_num is not None and end_num is not None:
                    for grade_num in range(start_num, end_num + 1):
                        grade_name = 'K' if grade_num == 0 else str(grade_num)
                        if grade_name not in grade_cache:
                            grade, _ = GradeLevel.objects.get_or_create(
                                grade=grade_name,
                                defaults={'grade_numeric': grade_num}
                            )
                            grade_cache[grade_name] = grade
                        else:
                            grade = grade_cache[grade_name]
                        grade_levels.append(grade)
        
        # Handle comma-separated (e.g., "3,4,5" or "K,1,2")
        elif ',' in grade_str:
            for grade_part in grade_str.split(','):
                grade_part = grade_part.strip()
                grade_num = 0 if grade_part == 'K' else int(grade_part) if grade_part.isdigit() else None
                
                if grade_num is not None:
                    grade_name = 'K' if grade_num == 0 else str(grade_num)
                    if grade_name not in grade_cache:
                        grade, _ = GradeLevel.objects.get_or_create(
                            grade=grade_name,
                            defaults={'grade_numeric': grade_num}
                        )
                        grade_cache[grade_name] = grade
                    else:
                        grade = grade_cache[grade_name]
                    grade_levels.append(grade)
        
        # Handle single grade
        else:
            grade_num = 0 if grade_str == 'K' else int(grade_str) if grade_str.isdigit() else None
            
            if grade_num is not None:
                grade_name = 'K' if grade_num == 0 else str(grade_num)
                if grade_name not in grade_cache:
                    grade, _ = GradeLevel.objects.get_or_create(
                        grade=grade_name,
                        defaults={'grade_numeric': grade_num}
                    )
                    grade_cache[grade_name] = grade
                else:
                    grade = grade_cache[grade_name]
                grade_levels.append(grade)
        
        return grade_levels

    def _infer_default_grade(self, grade_levels):
        """Infer a sensible default grade number from parsed GradeLevel objects."""
        if not grade_levels:
            return 3  # default
        # Pick min numeric
        nums = []
        for g in grade_levels:
            try:
                nums.append(int(g.grade_numeric))
            except Exception:
                continue
        return min(nums) if nums else 3

    def _build_normalized_code(self, state_code: str, parsed) -> str:
        base = f"{state_code}-{parsed.grade:02d}"
        parts = list(parsed.parts) if getattr(parsed, 'parts', None) else []
        if parts:
            return base + '-' + '-'.join(str(p) for p in parts[:3])
        return base
    
    def ensure_all_states_exist(self):
        """Ensure all 50 US states exist in the database"""
        states = [
            ('AL', 'Alabama'), ('AK', 'Alaska'), ('AZ', 'Arizona'), ('AR', 'Arkansas'),
            ('CA', 'California'), ('CO', 'Colorado'), ('CT', 'Connecticut'),
            ('DE', 'Delaware'), ('FL', 'Florida'), ('GA', 'Georgia'),
            ('HI', 'Hawaii'), ('ID', 'Idaho'), ('IL', 'Illinois'), ('IN', 'Indiana'),
            ('IA', 'Iowa'), ('KS', 'Kansas'), ('KY', 'Kentucky'), ('LA', 'Louisiana'),
            ('ME', 'Maine'), ('MD', 'Maryland'), ('MA', 'Massachusetts'),
            ('MI', 'Michigan'), ('MN', 'Minnesota'), ('MS', 'Mississippi'),
            ('MO', 'Missouri'), ('MT', 'Montana'), ('NE', 'Nebraska'),
            ('NV', 'Nevada'), ('NH', 'New Hampshire'), ('NJ', 'New Jersey'),
            ('NM', 'New Mexico'), ('NY', 'New York'), ('NC', 'North Carolina'),
            ('ND', 'North Dakota'), ('OH', 'Ohio'), ('OK', 'Oklahoma'),
            ('OR', 'Oregon'), ('PA', 'Pennsylvania'), ('RI', 'Rhode Island'),
            ('SC', 'South Carolina'), ('SD', 'South Dakota'), ('TN', 'Tennessee'),
            ('TX', 'Texas'), ('UT', 'Utah'), ('VT', 'Vermont'), ('VA', 'Virginia'),
            ('WA', 'Washington'), ('WV', 'West Virginia'), ('WI', 'Wisconsin'),
            ('WY', 'Wyoming'), ('DC', 'District of Columbia')
        ]
        
        for code, name in states:
            State.objects.get_or_create(code=code, defaults={'name': name})
        
        self.stdout.write(self.style.SUCCESS('All states verified/created'))
    
    def generate_embeddings_for_standards(self):
        """Generate embeddings for all standards without embeddings"""
        embedding_service = EmbeddingService()
        
        standards_without_embeddings = Standard.objects.filter(embedding__isnull=True)
        total = standards_without_embeddings.count()
        
        self.stdout.write(f'Found {total} standards without embeddings')
        
        for i, standard in enumerate(standards_without_embeddings):
            if i % 10 == 0:
                self.stdout.write(f'Processing {i}/{total} standards...')
            
            embedding = embedding_service.generate_standard_embedding(standard)
            if embedding:
                standard.embedding = embedding
                standard.save(update_fields=['embedding'])
        
        self.stdout.write(
            self.style.SUCCESS(f'Generated embeddings for {total} standards')
        )