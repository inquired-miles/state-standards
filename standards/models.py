from django.db import models
from django.db.models import Q
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone
from django.conf import settings
import uuid

# Check if we're using PostgreSQL for advanced features
try:
    DATABASES = getattr(settings, 'DATABASES', {})
    is_postgres = any('postgresql' in db.get('ENGINE', '') for db in DATABASES.values())
    
    if is_postgres:
        from django.contrib.postgres.fields import ArrayField
        from pgvector.django import VectorField
    else:
        # SQLite fallbacks
        ArrayField = None
        VectorField = None
except ImportError:
    is_postgres = False
    ArrayField = None
    VectorField = None

# Define fallback fields for SQLite
def create_array_field(base_field, **kwargs):
    """Create an ArrayField for PostgreSQL or TextField fallback for SQLite"""
    if ArrayField:
        return ArrayField(base_field, **kwargs)
    else:
        # For SQLite, use TextField to store JSON-serialized array
        # Convert list default to JSON string default
        if 'default' in kwargs and kwargs['default'] == list:
            kwargs['default'] = '[]'
        elif 'default' not in kwargs:
            kwargs['default'] = '[]'
        return models.TextField(**kwargs)

def create_vector_field(**kwargs):
    """Create a VectorField for PostgreSQL or TextField fallback for SQLite"""
    if VectorField:
        return VectorField(**kwargs)
    else:
        # For SQLite, use TextField to store vector data
        # Extract non-applicable kwargs for TextField
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['dimensions']}
        return models.TextField(**clean_kwargs)


class State(models.Model):
    """Model representing a US state"""
    code = models.CharField(max_length=2, unique=True, help_text="Two-letter state code (e.g., CA, NY)")
    name = models.CharField(max_length=100, help_text="Full state name")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']
        verbose_name = "State"
        verbose_name_plural = "States"

    def __str__(self):
        return f"{self.name} ({self.code})"


class SubjectArea(models.Model):
    """Model representing subject areas (e.g., Mathematics, Science, ELA)"""
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']
        verbose_name = "Subject Area"
        verbose_name_plural = "Subject Areas"

    def __str__(self):
        return self.name


class GradeLevel(models.Model):
    """Model representing grade levels"""
    grade = models.CharField(max_length=20, help_text="Grade level (e.g., K, 1, 2, ..., 12)")
    grade_numeric = models.IntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(12)],
        help_text="Numeric representation of grade (K=0, 1=1, etc.)"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['grade_numeric']
        unique_together = ['grade']
        verbose_name = "Grade Level"
        verbose_name_plural = "Grade Levels"

    def __str__(self):
        return f"Grade {self.grade}"


class Standard(models.Model):
    """Model representing educational standards with vector embeddings"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Basic Information
    state = models.ForeignKey(State, on_delete=models.CASCADE, related_name='standards')
    subject_area = models.ForeignKey(SubjectArea, on_delete=models.CASCADE, related_name='standards')
    grade_levels = models.ManyToManyField(GradeLevel, related_name='standards')
    
    # Standard Details
    code = models.CharField(max_length=50, help_text="Standard code (e.g., CCSS.MATH.CONTENT.1.OA.A.1)")
    title = models.CharField(max_length=200, blank=True, null=True, help_text="Standard title/name (optional - varies by state)")
    description = models.TextField()
    
    # Categorization
    domain = models.CharField(max_length=200, blank=True, help_text="Domain or strand")
    cluster = models.CharField(max_length=200, blank=True, help_text="Cluster or topic")
    
    # Vector Embedding for Similarity Search
    embedding = create_vector_field(dimensions=1536, null=True, blank=True, help_text="Text embedding for similarity search")
    
    # Metadata
    keywords = create_array_field(models.CharField(max_length=100), blank=True, default=list)
    skills = create_array_field(models.CharField(max_length=100), blank=True, default=list)
    
    # Normalization metadata (optional MVP fields)
    normalized_code = models.CharField(
        max_length=100, null=True, blank=True, db_index=True,
        help_text="Parsed/normalized representation of the code"
    )
    hierarchy = models.JSONField(default=dict, blank=True, help_text="Parsed hierarchy parts/grade/format")
    is_atomic = models.BooleanField(default=False, help_text="True if already a single learning objective")

    # Tracking
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['state', 'subject_area', 'code']
        unique_together = [['state', 'code']]
        indexes = [
            models.Index(fields=['state', 'subject_area']),
            models.Index(fields=['domain']),
            models.Index(fields=['cluster']),
        ]
        verbose_name = "Standard"
        verbose_name_plural = "Standards"

    @property
    def display_name(self):
        """Return title if available, otherwise generate from code + description"""
        if self.title:
            return self.title
        # Generate display name from code and truncated description
        desc_preview = self.description[:100] + "..." if len(self.description) > 100 else self.description
        return f"{self.code}: {desc_preview}"
    
    def __str__(self):
        return f"{self.state.code} - {self.code}: {self.display_name}"


class StandardCorrelation(models.Model):
    """Model representing correlations between standards across states"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Standards being correlated
    standard_1 = models.ForeignKey(Standard, on_delete=models.CASCADE, related_name='correlations_as_standard_1')
    standard_2 = models.ForeignKey(Standard, on_delete=models.CASCADE, related_name='correlations_as_standard_2')
    
    # Correlation metrics
    similarity_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Cosine similarity score between standard embeddings (0-1)"
    )
    
    # Correlation type
    CORRELATION_TYPES = [
        ('exact', 'Exact Match'),
        ('similar', 'Similar'),
        ('related', 'Related'),
        ('partial', 'Partial Match'),
    ]
    correlation_type = models.CharField(max_length=20, choices=CORRELATION_TYPES, default='similar')
    
    # Additional metadata
    notes = models.TextField(blank=True)
    verified = models.BooleanField(default=False, help_text="Has this correlation been manually verified?")
    
    # Tracking
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-similarity_score']
        unique_together = [['standard_1', 'standard_2']]
        verbose_name = "Standard Correlation"
        verbose_name_plural = "Standard Correlations"

    def __str__(self):
        return f"{self.standard_1.state.code} {self.standard_1.code} <-> {self.standard_2.state.code} {self.standard_2.code} ({self.similarity_score:.2f})"

    def save(self, *args, **kwargs):
        # Ensure standard_1 and standard_2 are ordered consistently to prevent duplicates
        if self.standard_1.id > self.standard_2.id:
            self.standard_1, self.standard_2 = self.standard_2, self.standard_1
        super().save(*args, **kwargs)


class Concept(models.Model):
    """Model representing educational concepts that appear across multiple states"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Concept details
    name = models.CharField(max_length=200, help_text="Concept name (e.g., 'Fraction Operations')")
    description = models.TextField(blank=True)
    embedding = create_vector_field(dimensions=1536, null=True, blank=True)
    
    # Coverage data
    grade_levels = models.ManyToManyField(GradeLevel, related_name='concepts')
    subject_areas = models.ManyToManyField(SubjectArea, related_name='concepts')
    
    # Metadata
    keywords = create_array_field(models.CharField(max_length=100), blank=True, default=list)
    complexity_score = models.FloatField(null=True, blank=True, help_text="Computed complexity metric (0-1)")
    
    # Coverage statistics (cached)
    states_covered = models.IntegerField(default=0, help_text="Number of states covering this concept")
    coverage_percentage = models.FloatField(default=0.0, help_text="Percentage of states covering this concept")
    
    # Tracking
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-coverage_percentage', 'name']
        indexes = [
            models.Index(fields=['states_covered']),
            models.Index(fields=['coverage_percentage']),
            models.Index(fields=['complexity_score']),
        ]
        verbose_name = "Concept"
        verbose_name_plural = "Concepts"
    
    def __str__(self):
        return f"{self.name} ({self.states_covered}/50 states)"


class StateFormat(models.Model):
    """Track detected numbering format per state (optional)."""
    state = models.ForeignKey(State, on_delete=models.CASCADE, related_name='formats')
    parser_type = models.CharField(max_length=50, default='generic')
    format_pattern = models.CharField(max_length=200, blank=True)
    example = models.CharField(max_length=100, blank=True)
    detected_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [['state', 'parser_type']]
        indexes = [models.Index(fields=['state', 'parser_type'])]
        verbose_name = "State Format"
        verbose_name_plural = "State Formats"

    def __str__(self) -> str:
        return f"{self.state.code}: {self.parser_type}"


class StandardAtom(models.Model):
    """Atomic learning objective derived from a Standard."""
    standard = models.ForeignKey(Standard, on_delete=models.CASCADE, related_name='atoms')
    atom_code = models.CharField(max_length=50, unique=True, db_index=True, help_text="e.g., CA.SS.3.4.1-A")
    text = models.TextField()
    embedding = create_vector_field(dimensions=1536, null=True, blank=True)
    embedding_generated_at = models.DateTimeField(null=True, blank=True)
    method = models.CharField(max_length=20, choices=[
        ('structural', 'Structural'),
        ('content', 'Content'),
        ('gpt', 'GPT'),
        ('manual', 'Manual'),
    ], default='content')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=['standard'])]
        ordering = ['atom_code']
        verbose_name = "Standard Atom"
        verbose_name_plural = "Standard Atoms"

    def __str__(self) -> str:
        return f"{self.atom_code}: {self.text[:50]}"


class ProxyStandard(models.Model):
    """Cross-state proxy standard derived from clustering of StandardAtom or Standard embeddings."""
    proxy_id = models.CharField(max_length=30, unique=True, db_index=True)  # Increased for grade info
    title = models.CharField(max_length=200, blank=True)
    description = models.TextField(blank=True)

    # Source type - distinguishes atom-level vs standard-level clustering
    SOURCE_TYPES = [
        ('atoms', 'StandardAtom Clustering'),
        ('standards', 'Standard Clustering'),
    ]
    source_type = models.CharField(max_length=20, choices=SOURCE_TYPES, default='atoms')

    cluster_id = models.IntegerField(db_index=True)
    
    # For atom-level clustering (existing)
    medoid_atom = models.ForeignKey(StandardAtom, on_delete=models.PROTECT, related_name='medoid_for_proxy', null=True, blank=True)
    member_atoms = models.ManyToManyField(StandardAtom, related_name='proxy_memberships', blank=True)
    
    # For standard-level clustering (new)
    medoid_standard = models.ForeignKey('Standard', on_delete=models.PROTECT, related_name='medoid_for_proxy', null=True, blank=True)
    member_standards = models.ManyToManyField('Standard', related_name='proxy_memberships', blank=True)
    
    centroid_embedding = create_vector_field(dimensions=1536)

    # Grade level fields
    grade_levels = models.ManyToManyField(GradeLevel, related_name='proxy_standards', blank=True)
    min_grade = models.IntegerField(null=True, blank=True, help_text="Minimum grade level (0 for K)")
    max_grade = models.IntegerField(null=True, blank=True, help_text="Maximum grade level (12 for Grade 12)")

    coverage_count = models.IntegerField(default=0)
    avg_similarity = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-coverage_count', 'proxy_id']
        verbose_name = "Proxy Standard"
        verbose_name_plural = "Proxy Standards"

    def __str__(self):
        label = self.title or 'Unnamed'
        return f"{self.proxy_id}: {label} ({self.coverage_count} states)"

    @property
    def grade_range_display(self):
        """Human-readable grade range"""
        if self.min_grade is None and self.max_grade is None:
            return "All Grades"
        elif self.min_grade == self.max_grade:
            if self.min_grade == 0:
                return "Kindergarten"
            else:
                return f"Grade {self.min_grade}"
        else:
            min_display = "K" if self.min_grade == 0 else str(self.min_grade)
            max_display = "K" if self.max_grade == 0 else str(self.max_grade)
            return f"Grade {min_display}-{max_display}"

    def update_grade_range_from_members(self):
        """Update min/max grade based on associated atoms or standards"""
        grade_levels = None
        
        if self.source_type == 'atoms' and self.member_atoms.exists():
            grade_levels = GradeLevel.objects.filter(
                standards__atoms__in=self.member_atoms.all()
            ).distinct().order_by('grade_numeric')
        elif self.source_type == 'standards' and self.member_standards.exists():
            grade_levels = GradeLevel.objects.filter(
                standards__in=self.member_standards.all()
            ).distinct().order_by('grade_numeric')
            
        if grade_levels and grade_levels.exists():
            self.min_grade = grade_levels.first().grade_numeric
            self.max_grade = grade_levels.last().grade_numeric
            self.grade_levels.set(grade_levels)
    
    def update_grade_range_from_atoms(self):
        """Legacy method - update min/max grade based on associated atoms"""
        self.update_grade_range_from_members()
    
    @property
    def member_count(self):
        """Get count of member items (atoms or standards)"""
        if self.source_type == 'atoms':
            return self.member_atoms.count()
        elif self.source_type == 'standards':
            return self.member_standards.count()
        return 0
    
    @property
    def states_covered(self):
        """Get list of states covered by this proxy"""
        if self.source_type == 'atoms':
            return list(set(
                atom.standard.state.code 
                for atom in self.member_atoms.select_related('standard__state').all() 
                if atom.standard and atom.standard.state
            ))
        elif self.source_type == 'standards':
            return list(set(
                standard.state.code 
                for standard in self.member_standards.select_related('state').all() 
                if standard.state
            ))
        return []
    
    @property
    def subject_areas_covered(self):
        """Get list of subject areas covered by this proxy"""
        if self.source_type == 'atoms':
            return list(set(
                atom.standard.subject_area.name 
                for atom in self.member_atoms.select_related('standard__subject_area').all() 
                if atom.standard and atom.standard.subject_area
            ))
        elif self.source_type == 'standards':
            return list(set(
                standard.subject_area.name 
                for standard in self.member_standards.select_related('subject_area').all() 
                if standard.subject_area
            ))
        return []


class ProxyStateCoverage(models.Model):
    """Per-state coverage details for a ProxyStandard."""
    proxy = models.ForeignKey(ProxyStandard, on_delete=models.CASCADE, related_name='state_coverages')
    state = models.ForeignKey(State, on_delete=models.CASCADE)
    atom_count = models.IntegerField(default=0)
    avg_similarity = models.FloatField(default=0.0)

    class Meta:
        unique_together = [['proxy', 'state']]
        indexes = [models.Index(fields=['state'])]
        verbose_name = "Proxy State Coverage"
        verbose_name_plural = "Proxy State Coverages"


class TopicBasedProxy(models.Model):
    """LLM-categorized proxy standards organized by hierarchical topics."""
    proxy_id = models.CharField(max_length=50, unique=True, db_index=True, help_text="Format: TP-Geography-MapSkills-001")
    
    # Hierarchical topic structure
    topic = models.CharField(max_length=500, help_text="Main topic (e.g., 'Geography and Spatial Thinking')")
    sub_topic = models.CharField(max_length=500, help_text="Sub-topic (e.g., 'Map Skills and Location')")
    sub_sub_topic = models.CharField(max_length=1000, help_text="Sub-sub-topic (e.g., 'Cardinal and Intermediate Directions') or detailed outlier reasoning")
    
    # Generated descriptions
    title = models.CharField(max_length=200, blank=True, help_text="AI-generated friendly title")
    description = models.TextField(blank=True, help_text="Hierarchical description: Topic > Sub-topic > Sub-sub-topic")
    
    # Related standards
    member_standards = models.ManyToManyField('Standard', related_name='topic_proxy_memberships', blank=True)
    
    # Grade level information derived from member standards
    min_grade = models.IntegerField(null=True, blank=True, help_text="Minimum grade level from member standards")
    max_grade = models.IntegerField(null=True, blank=True, help_text="Maximum grade level from member standards")
    grade_levels = models.ManyToManyField('GradeLevel', blank=True, help_text="All grade levels represented")
    
    # Metadata derived from member standards
    outlier_category = models.BooleanField(default=False, help_text="True if this groups standards that didn't fit the main taxonomy")
    standards_count = models.IntegerField(default=0, help_text="Number of member standards")
    states_covered = models.IntegerField(default=0, help_text="Number of unique states represented")
    subject_areas = models.JSONField(default=list, blank=True, help_text="List of subject areas covered")
    
    # Original filtering criteria used during creation
    filter_grade_levels = models.JSONField(default=list, blank=True, help_text="Original grade levels used as filter criteria during creation")
    filter_subject_area = models.ForeignKey('SubjectArea', on_delete=models.SET_NULL, null=True, blank=True, 
                                           related_name='topic_proxies_filtered_for', 
                                           help_text="Original subject area filter used during creation")
    filter_criteria = models.JSONField(default=dict, blank=True, help_text="Complete original filtering criteria used during creation")
    
    # Tracking
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['topic', 'sub_topic', 'sub_sub_topic', 'proxy_id']
        indexes = [
            models.Index(fields=['topic']),
            models.Index(fields=['topic', 'sub_topic']),
            models.Index(fields=['topic', 'sub_topic', 'sub_sub_topic']),
            models.Index(fields=['outlier_category']),
            models.Index(fields=['created_at']),
        ]
        verbose_name = "Topic-Based Proxy Standard"
        verbose_name_plural = "Topic-Based Proxy Standards"
    
    def __str__(self):
        if self.outlier_category:
            return f"{self.proxy_id} (Outlier: {self.sub_sub_topic})"
        return f"{self.proxy_id}: {self.topic} > {self.sub_topic} > {self.sub_sub_topic}"
    
    def save(self, *args, **kwargs):
        # Auto-generate description if not provided
        if not self.description:
            if self.outlier_category:
                self.description = f"Outlier category: {self.sub_sub_topic}"
            else:
                self.description = f"{self.topic} > {self.sub_topic} > {self.sub_sub_topic}"
        
        super().save(*args, **kwargs)
        
        # Update derived fields after saving
        self.update_derived_fields()
    
    def update_derived_fields(self):
        """Update grade levels, counts, and other derived fields from member standards."""
        standards = self.member_standards.select_related('state', 'subject_area').prefetch_related('grade_levels').all()
        
        if standards:
            # Update counts
            self.standards_count = len(standards)
            self.states_covered = len(set(s.state.code for s in standards if s.state))
            
            # Update subject areas
            subject_areas = list(set(s.subject_area.name for s in standards if s.subject_area))
            self.subject_areas = subject_areas
            
            # Update grade levels
            all_grades = []
            for standard in standards:
                all_grades.extend(gl.grade_numeric for gl in standard.grade_levels.all())
            
            if all_grades:
                self.min_grade = min(all_grades)
                self.max_grade = max(all_grades)
                
                # Update grade_levels ManyToManyField
                grade_level_objects = GradeLevel.objects.filter(grade_numeric__in=set(all_grades))
                self.grade_levels.set(grade_level_objects)
            
            # Save changes (prevent recursion with update_fields)
            type(self).objects.filter(pk=self.pk).update(
                standards_count=self.standards_count,
                states_covered=self.states_covered,
                subject_areas=self.subject_areas,
                min_grade=self.min_grade,
                max_grade=self.max_grade,
                description=self.description
            )
    
    @property
    def hierarchy_path(self):
        """Return the full hierarchical path as a string."""
        return f"{self.topic} > {self.sub_topic} > {self.sub_sub_topic}"
    
    @property
    def short_hierarchy(self):
        """Return abbreviated hierarchy for display."""
        def truncate(text, length=20):
            return text[:length] + "..." if len(text) > length else text
        
        return f"{truncate(self.topic)} > {truncate(self.sub_topic)} > {truncate(self.sub_sub_topic)}"
    
    @property
    def filter_criteria_display(self):
        """Human-readable display of original filter criteria."""
        if not self.filter_criteria:
            return "No original criteria stored"
        
        parts = []
        
        # Grade selection display
        grade_selection = self.filter_criteria.get('grade_selection', {})
        if grade_selection.get('type') == 'specific':
            grades = grade_selection.get('grades', [])
            if grades:
                grade_names = []
                for grade in grades:
                    grade_names.append('K' if grade == 0 else f'Grade {grade}')
                parts.append(f"Grades: {', '.join(grade_names)}")
        elif grade_selection.get('type') == 'range':
            min_grade = grade_selection.get('min_grade')
            max_grade = grade_selection.get('max_grade')
            if min_grade is not None and max_grade is not None:
                min_name = 'K' if min_grade == 0 else f'Grade {min_grade}'
                max_name = 'K' if max_grade == 0 else f'Grade {max_grade}'
                parts.append(f"Grades: {min_name} to {max_name}")
        elif grade_selection.get('type') == 'all':
            parts.append("Grades: All")
        
        # Subject area display
        if self.filter_subject_area:
            parts.append(f"Subject: {self.filter_subject_area.name}")
        elif self.filter_criteria.get('subject_area_id'):
            parts.append(f"Subject ID: {self.filter_criteria['subject_area_id']}")
        
        # Additional settings
        if self.filter_criteria.get('use_dynamic_chunk'):
            parts.append("Dynamic chunking: Yes")
        if not self.filter_criteria.get('include_outliers', True):
            parts.append("Outliers: Excluded")
        
        return " | ".join(parts) if parts else "Default settings"
    
    @property
    def filter_grade_levels_display(self):
        """Human-readable display of filter grade levels."""
        if not self.filter_grade_levels:
            return "None specified"
        
        grade_names = []
        for grade in self.filter_grade_levels:
            grade_names.append('K' if grade == 0 else f'Grade {grade}')
        return ', '.join(grade_names)


class ProxyRun(models.Model):
    """Model to track individual proxy generation runs for comparison and analysis."""
    
    # Run identification
    run_id = models.CharField(max_length=50, unique=True, db_index=True, help_text="Unique identifier for this run")
    name = models.CharField(max_length=200, help_text="User-friendly name for this run")
    description = models.TextField(blank=True, help_text="Optional description of run parameters and goals")
    
    # Run type and parameters
    RUN_TYPES = [
        ('atoms', 'StandardAtom Clustering'),
        ('standards', 'Standard Clustering'), 
        ('topics', 'Topic Categorization'),
    ]
    run_type = models.CharField(max_length=20, choices=RUN_TYPES, help_text="Type of proxy generation method used")
    
    # Filtering parameters (JSON to store flexible filter criteria)
    filter_parameters = models.JSONField(default=dict, help_text="Grade levels, subject areas, and other filters used")
    
    # Algorithm parameters (JSON to store algorithm-specific settings)
    algorithm_parameters = models.JSONField(default=dict, help_text="Min cluster size, epsilon, chunk size, etc.")
    
    # Run status and timing
    STATUS_CHOICES = [
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='running')
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    duration_seconds = models.IntegerField(null=True, blank=True, help_text="Total run time in seconds")
    
    # Results summary (updated after run completion)
    total_input_standards = models.IntegerField(default=0, help_text="Number of standards processed")
    total_proxies_created = models.IntegerField(default=0, help_text="Number of proxy standards created")
    outlier_proxies_count = models.IntegerField(default=0, help_text="Number of outlier/uncategorized proxies")
    coverage_percentage = models.FloatField(null=True, blank=True, help_text="Overall coverage percentage")
    
    # Metadata
    created_by = models.CharField(max_length=100, blank=True, help_text="User who initiated this run")
    job_id = models.CharField(max_length=100, blank=True, help_text="Background job ID for tracking")
    
    class Meta:
        ordering = ['-started_at']
        indexes = [
            models.Index(fields=['run_type', 'status']),
            models.Index(fields=['started_at']),
            models.Index(fields=['coverage_percentage']),
        ]
        verbose_name = "Proxy Run"
        verbose_name_plural = "Proxy Runs"
    
    def __str__(self):
        status_emoji = {'running': '🔄', 'completed': '✅', 'failed': '❌', 'cancelled': '⏹️'}.get(self.status, '❓')
        return f"{status_emoji} {self.name} ({self.get_run_type_display()})"
    
    @property
    def is_completed(self):
        return self.status == 'completed'
    
    @property
    def duration_display(self):
        """Human-readable duration display"""
        if not self.duration_seconds:
            return "Unknown"
        
        minutes, seconds = divmod(self.duration_seconds, 60)
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"
    
    @property
    def filter_summary(self):
        """Human-readable summary of filters applied"""
        filters = []
        params = self.filter_parameters
        
        if params.get('grade_selection'):
            grade_sel = params['grade_selection']
            if grade_sel.get('type') == 'specific':
                grades = grade_sel.get('grades', [])
                filters.append(f"Grades: {', '.join(map(str, grades))}")
            elif grade_sel.get('type') == 'range':
                min_g, max_g = grade_sel.get('min_grade'), grade_sel.get('max_grade')
                filters.append(f"Grades: {min_g}-{max_g}")
        
        if params.get('subject_area_id'):
            try:
                subject = SubjectArea.objects.get(id=params['subject_area_id'])
                filters.append(f"Subject: {subject.name}")
            except SubjectArea.DoesNotExist:
                pass
        
        return "; ".join(filters) if filters else "No filters"
    
    @property 
    def algorithm_summary(self):
        """Human-readable summary of algorithm parameters"""
        params = self.algorithm_parameters
        summary = []
        
        if self.run_type in ['atoms', 'standards']:
            summary.append(f"Min cluster: {params.get('min_cluster', 'N/A')}")
            summary.append(f"Epsilon: {params.get('epsilon', 'N/A')}")
        elif self.run_type == 'topics':
            if params.get('use_dynamic_chunk'):
                actual_size = params.get('actual_chunk_size', 'calculating...')
                summary.append(f"Dynamic chunk sizing (actual: {actual_size})")
            else:
                summary.append(f"Fixed chunk size: {params.get('chunk_size', 'N/A')}")
            
        return "; ".join(summary) if summary else "Default parameters"
    
    def get_associated_proxies(self):
        """Get all proxy standards created during this run"""
        # This will be implemented based on the proxy type
        if self.run_type == 'topics':
            return TopicBasedProxy.objects.filter(created_at__gte=self.started_at, created_at__lte=self.completed_at or timezone.now())
        else:
            return ProxyStandard.objects.filter(created_at__gte=self.started_at, created_at__lte=self.completed_at or timezone.now())
    
    def calculate_coverage_stats(self):
        """Calculate and update coverage statistics for this run"""
        proxies = self.get_associated_proxies()
        
        if self.run_type == 'topics':
            # For topic-based proxies
            total_standards = sum(p.standards_count for p in proxies)
            outliers = proxies.filter(outlier_category=True).count()
        else:
            # For traditional proxies
            total_standards = sum(p.member_atoms.count() for p in proxies if hasattr(p, 'member_atoms'))
            outliers = 0  # Traditional clustering doesn't have outliers in the same way
        
        self.total_proxies_created = proxies.count()
        self.outlier_proxies_count = outliers
        
        # Calculate coverage percentage (compared to input standards)
        if self.total_input_standards > 0:
            self.coverage_percentage = (total_standards / self.total_input_standards) * 100
        
        self.save(update_fields=['total_proxies_created', 'outlier_proxies_count', 'coverage_percentage'])


class ProxyRunReport(models.Model):
    """Detailed analytics and insights for a completed proxy run."""
    
    run = models.OneToOneField(ProxyRun, on_delete=models.CASCADE, related_name='report')
    
    # State-by-state analysis
    state_coverage = models.JSONField(default=dict, help_text="Coverage statistics per state")
    topic_prevalence = models.JSONField(default=dict, help_text="Topic prevalence across states")
    
    # Coverage distribution analysis
    coverage_distribution = models.JSONField(default=dict, help_text="Bell curve data for coverage distribution")
    outlier_analysis = models.JSONField(default=dict, help_text="Analysis of outlier patterns")
    
    # Topic intelligence (for topic-based runs)
    topic_hierarchy_stats = models.JSONField(default=dict, help_text="Statistics about topic hierarchy usage")
    cross_state_commonality = models.JSONField(default=dict, help_text="Topics that appear across multiple states")
    
    # Quality metrics
    silhouette_scores = models.JSONField(default=dict, help_text="Clustering quality metrics")
    coverage_gaps = models.JSONField(default=dict, help_text="Areas with low coverage")
    
    # Report metadata
    generated_at = models.DateTimeField(auto_now_add=True)
    generation_time_seconds = models.FloatField(null=True, blank=True)
    
    class Meta:
        verbose_name = "Proxy Run Report"
        verbose_name_plural = "Proxy Run Reports"
    
    def __str__(self):
        return f"Report for {self.run.name}"
    
    @property
    def must_have_topics(self):
        """Topics that appear in 80%+ of states (must-have)"""
        prevalence = self.topic_prevalence
        total_states = len(self.state_coverage)
        threshold = total_states * 0.8
        
        return {
            topic: data for topic, data in prevalence.items() 
            if data.get('state_count', 0) >= threshold
        }
    
    @property
    def important_topics(self):
        """Topics that appear in 60-79% of states (important)"""
        prevalence = self.topic_prevalence
        total_states = len(self.state_coverage)
        min_threshold = total_states * 0.6
        max_threshold = total_states * 0.8
        
        return {
            topic: data for topic, data in prevalence.items() 
            if min_threshold <= data.get('state_count', 0) < max_threshold
        }
    
    @property
    def regional_topics(self):
        """Topics that appear in <60% of states (regional)"""
        prevalence = self.topic_prevalence
        total_states = len(self.state_coverage)
        threshold = total_states * 0.6
        
        return {
            topic: data for topic, data in prevalence.items() 
            if data.get('state_count', 0) < threshold
        }


class CurriculumDocument(models.Model):
    """Uploaded or provided curriculum document for coverage analysis."""
    name = models.CharField(max_length=255)
    content = models.TextField()
    file = models.FileField(upload_to='curricula/', null=True, blank=True)
    embedding = create_vector_field(dimensions=1536, null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)

    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = "Curriculum Document"
        verbose_name_plural = "Curriculum Documents"

    def __str__(self) -> str:
        return self.name


class CoverageDetail(models.Model):
    """Per-atom coverage details for a curriculum document."""
    curriculum = models.ForeignKey(CurriculumDocument, on_delete=models.CASCADE, related_name='atom_coverages')
    atom = models.ForeignKey(StandardAtom, on_delete=models.CASCADE)
    similarity_score = models.FloatField()
    is_covered = models.BooleanField(default=False)

    class Meta:
        unique_together = [['curriculum', 'atom']]
        indexes = [models.Index(fields=['curriculum', 'is_covered'])]
        verbose_name = "Coverage Detail"
        verbose_name_plural = "Coverage Details"


class StandardCoverage(models.Model):
    """Rollup coverage for a Standard based on its atoms."""
    curriculum = models.ForeignKey(CurriculumDocument, on_delete=models.CASCADE, related_name='standard_coverages')
    standard = models.ForeignKey(Standard, on_delete=models.CASCADE)
    total_atoms = models.IntegerField()
    covered_atoms = models.IntegerField()
    coverage_percentage = models.FloatField()
    status = models.CharField(max_length=10, choices=[
        ('FULL', 'Fully Covered (≥90%)'),
        ('PARTIAL', 'Partially Covered (70-89%)'),
        ('MINIMAL', 'Minimally Covered (40-69%)'),
        ('NONE', 'Not Covered (<40%)')
    ])

    class Meta:
        unique_together = [['curriculum', 'standard']]
        verbose_name = "Standard Coverage"
        verbose_name_plural = "Standard Coverages"


class StateCoverage(models.Model):
    """Rollup coverage at state level for a curriculum document."""
    curriculum = models.ForeignKey(CurriculumDocument, on_delete=models.CASCADE, related_name='state_coverages')
    state = models.ForeignKey(State, on_delete=models.CASCADE)

    total_standards = models.IntegerField()
    full_coverage_count = models.IntegerField()
    partial_coverage_count = models.IntegerField()
    minimal_coverage_count = models.IntegerField()
    none_coverage_count = models.IntegerField()
    overall_percentage = models.FloatField()
    is_marketable = models.BooleanField(default=False)

    class Meta:
        unique_together = [['curriculum', 'state']]
        ordering = ['-overall_percentage']
        verbose_name = "State Coverage"
        verbose_name_plural = "State Coverages"


class TopicCluster(models.Model):
    """Model representing discovered topic clusters across states"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Cluster details
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    embedding = create_vector_field(dimensions=1536, null=True, blank=True, help_text="Centroid embedding for cluster")
    origin = models.CharField(
        max_length=10,
        choices=[('auto', 'Discovered'), ('custom', 'User Cluster')],
        default='auto',
        help_text="Whether the cluster was system-discovered or user-created",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='custom_topic_clusters',
        help_text="Staff user who created the cluster when origin='custom'",
    )
    is_shared = models.BooleanField(
        default=False,
        help_text="Share custom cluster with other staff users",
    )
    search_context = models.JSONField(
        default=dict,
        blank=True,
        help_text="Serialized query/filters used when creating a custom cluster",
    )
    
    # Cluster metadata
    subject_area = models.ForeignKey(SubjectArea, on_delete=models.CASCADE, related_name='topic_clusters')
    grade_levels = models.ManyToManyField(GradeLevel, related_name='topic_clusters')
    standards = models.ManyToManyField(Standard, related_name='topic_clusters', through='ClusterMembership')
    
    # Cluster quality metrics
    silhouette_score = models.FloatField(null=True, blank=True, help_text="Cluster quality metric (-1 to 1)")
    cohesion_score = models.FloatField(null=True, blank=True, help_text="Internal cluster cohesion (0-1)")
    standards_count = models.IntegerField(default=0)
    states_represented = models.IntegerField(default=0)
    
    # Regional patterns
    regional_pattern = models.JSONField(default=dict, blank=True, help_text="Regional distribution data")
    common_terms = create_array_field(models.CharField(max_length=100), blank=True, default=list)
    
    # Tracking
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-silhouette_score', '-standards_count']
        indexes = [
            models.Index(fields=['subject_area', 'standards_count']),
            models.Index(fields=['silhouette_score']),
            models.Index(fields=['origin', 'created_by']),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=['created_by', 'name'],
                condition=Q(origin='custom'),
                name='unique_custom_cluster_per_owner_name'
            )
        ]
        verbose_name = "Topic Cluster"
        verbose_name_plural = "Topic Clusters"
    
    def __str__(self):
        return f"{self.name} ({self.standards_count} standards, {self.states_represented} states)"


class ClusterMembership(models.Model):
    """Through model for Standard-TopicCluster relationship with membership strength"""
    standard = models.ForeignKey(Standard, on_delete=models.CASCADE)
    cluster = models.ForeignKey(TopicCluster, on_delete=models.CASCADE)
    added_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='custom_cluster_memberships',
        help_text="Staff user who added the standard when origin='custom'",
    )
    selection_order = models.PositiveIntegerField(
        default=0,
        help_text="Preserve manual ordering of standards inside a custom cluster",
    )
    similarity_score = models.FloatField(
        null=True,
        blank=True,
        help_text="Semantic similarity score captured during selection",
    )
    membership_strength = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="How strongly this standard belongs to the cluster (0-1)"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['standard', 'cluster']
        ordering = ['selection_order', '-membership_strength']


class ClusterReport(models.Model):
    """Saved grouping of topic clusters for comparison/coverage reporting"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='cluster_reports'
    )
    is_shared = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    clusters = models.ManyToManyField(
        TopicCluster,
        through='ClusterReportEntry',
        related_name='cluster_reports'
    )

    class Meta:
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['created_by', 'is_shared'])
        ]
        constraints = [
            models.UniqueConstraint(
                fields=['created_by', 'title'],
                name='unique_cluster_report_per_owner_title'
            )
        ]

    def __str__(self):
        return self.title


class ClusterReportEntry(models.Model):
    """Through table linking reports to clusters with ordering metadata"""
    report = models.ForeignKey(ClusterReport, on_delete=models.CASCADE)
    cluster = models.ForeignKey(TopicCluster, on_delete=models.CASCADE)
    selection_order = models.PositiveIntegerField(default=0)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['report', 'cluster']
        ordering = ['selection_order']

class CoverageAnalysis(models.Model):
    """Model storing coverage analysis results"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Analysis parameters
    state = models.ForeignKey(State, on_delete=models.CASCADE, null=True, blank=True, related_name='coverage_analyses')
    subject_area = models.ForeignKey(SubjectArea, on_delete=models.CASCADE, related_name='coverage_analyses')
    grade_level = models.ForeignKey(GradeLevel, on_delete=models.CASCADE, null=True, blank=True, related_name='coverage_analyses')
    
    # Analysis results
    total_standards = models.IntegerField(default=0)
    covered_concepts = models.IntegerField(default=0)
    coverage_percentage = models.FloatField(default=0.0)
    
    # Bell curve data
    bell_curve_data = models.JSONField(default=dict, help_text="Distribution data for visualization")
    gap_analysis = models.JSONField(default=dict, help_text="Identified gaps and recommendations")
    benchmark_comparison = models.JSONField(default=dict, help_text="Comparison with other states")
    
    # Metadata
    analysis_type = models.CharField(max_length=50, choices=[
        ('state', 'State Analysis'),
        ('subject', 'Subject Analysis'),  
        ('grade', 'Grade Level Analysis'),
        ('comprehensive', 'Comprehensive Analysis'),
    ], default='state')
    
    # Tracking
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['state', 'subject_area', 'grade_level']),
            models.Index(fields=['analysis_type', 'created_at']),
        ]
        verbose_name = "Coverage Analysis"
        verbose_name_plural = "Coverage Analyses"
    
    def __str__(self):
        scope = self.state.name if self.state else "All States"
        return f"Coverage Analysis: {scope} - {self.subject_area.name} ({self.coverage_percentage:.1f}%)"


class ContentAlignment(models.Model):
    """Model storing content alignment analysis results"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Content details
    content_title = models.CharField(max_length=200)
    content_text = models.TextField()
    content_hash = models.CharField(max_length=64, unique=True, help_text="SHA-256 hash of content")
    content_embedding = create_vector_field(dimensions=1536, null=True, blank=True)
    
    # File information
    original_filename = models.CharField(max_length=255, blank=True)
    file_type = models.CharField(max_length=20, choices=[
        ('pdf', 'PDF Document'),
        ('docx', 'Word Document'),
        ('txt', 'Text File'),
        ('html', 'HTML Content'),
    ], blank=True)
    
    # Alignment results
    total_standards_analyzed = models.IntegerField(default=0)
    matched_standards = models.ManyToManyField(Standard, through='ContentStandardMatch')
    
    # Alignment metrics
    overall_alignment_score = models.FloatField(default=0.0, help_text="Overall alignment score (0-100)")
    exact_matches = models.IntegerField(default=0)
    semantic_matches = models.IntegerField(default=0)
    conceptual_matches = models.IntegerField(default=0)
    
    # State coverage
    states_with_full_alignment = models.ManyToManyField(State, related_name='fully_aligned_content', blank=True)
    states_with_partial_alignment = models.ManyToManyField(State, related_name='partially_aligned_content', blank=True)
    
    # Analysis results
    alignment_report = models.JSONField(default=dict, help_text="Detailed alignment analysis")
    improvement_suggestions = models.TextField(blank=True)
    gap_analysis = models.JSONField(default=dict, help_text="Missing elements for better alignment")
    
    # Tracking
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['content_hash']),
            models.Index(fields=['overall_alignment_score']),
            models.Index(fields=['created_at']),
        ]
        verbose_name = "Content Alignment"
        verbose_name_plural = "Content Alignments"
    
    def __str__(self):
        return f"{self.content_title} (Score: {self.overall_alignment_score:.1f}%)"


class ContentStandardMatch(models.Model):
    """Through model for ContentAlignment-Standard relationship with match details"""
    content_alignment = models.ForeignKey(ContentAlignment, on_delete=models.CASCADE)
    standard = models.ForeignKey(Standard, on_delete=models.CASCADE)
    
    # Match details
    match_type = models.CharField(max_length=20, choices=[
        ('exact', 'Exact Match'),
        ('semantic', 'Semantic Match'),
        ('conceptual', 'Conceptual Match'),
        ('skill', 'Skill-Based Match'),
    ])
    confidence_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Confidence in the match (0-1)"
    )
    similarity_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Similarity score (0-1)"
    )
    
    # Match context
    matched_text_snippet = models.TextField(blank=True, help_text="Snippet of content that matched")
    explanation = models.TextField(blank=True, help_text="Explanation of why this match was made")
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['content_alignment', 'standard']
        ordering = ['-confidence_score', '-similarity_score']


class StrategicPlan(models.Model):
    """Model storing strategic planning analysis results"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Plan details
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    target_states = models.ManyToManyField(State, related_name='strategic_plans')
    target_subjects = models.ManyToManyField(SubjectArea, related_name='strategic_plans')
    target_grades = models.ManyToManyField(GradeLevel, related_name='strategic_plans')
    
    # MVC Analysis
    mvc_data = models.JSONField(default=dict, help_text="Minimum Viable Coverage calculations")
    target_coverage_percentage = models.FloatField(default=80.0, help_text="Target coverage percentage")
    
    # Priority Matrix
    priority_matrix = models.JSONField(default=dict, help_text="Impact vs Effort analysis")
    high_priority_concepts = models.ManyToManyField(Concept, related_name='high_priority_plans', blank=True)
    
    # ROI Analysis
    roi_analysis = models.JSONField(default=dict, help_text="Return on investment calculations")
    estimated_development_cost = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    projected_market_reach = models.IntegerField(null=True, blank=True, help_text="Estimated student reach")
    
    # Implementation timeline
    timeline_data = models.JSONField(default=dict, help_text="Implementation timeline and milestones")
    estimated_completion_months = models.IntegerField(null=True, blank=True)
    
    # Risk assessment
    risk_factors = models.JSONField(default=dict, help_text="Identified risks and mitigation strategies")
    implementation_difficulty = models.CharField(max_length=20, choices=[
        ('low', 'Low Difficulty'),
        ('medium', 'Medium Difficulty'),
        ('high', 'High Difficulty'),
        ('very_high', 'Very High Difficulty'),
    ], default='medium')
    
    # Status tracking
    status = models.CharField(max_length=20, choices=[
        ('draft', 'Draft'),
        ('under_review', 'Under Review'),
        ('approved', 'Approved'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
    ], default='draft')
    
    # Tracking
    created_by = models.CharField(max_length=100, blank=True)  # Will integrate with User model later
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status', 'created_at']),
            models.Index(fields=['target_coverage_percentage']),
        ]
        verbose_name = "Strategic Plan"
        verbose_name_plural = "Strategic Plans"
    
    def __str__(self):
        return f"{self.name} (Target: {self.target_coverage_percentage}% coverage)"


class CacheEntry(models.Model):
    """Custom caching model for complex query results"""
    cache_key = models.CharField(max_length=255, unique=True, help_text="Unique cache key")
    cache_value = models.JSONField(help_text="Cached data")
    cache_type = models.CharField(max_length=50, choices=[
        ('coverage_analysis', 'Coverage Analysis'),
        ('topic_discovery', 'Topic Discovery'),
        ('similarity_search', 'Similarity Search'),
        ('strategic_analysis', 'Strategic Analysis'),
    ])
    
    # Cache metadata
    parameters_hash = models.CharField(max_length=64, help_text="Hash of query parameters")
    computation_time = models.FloatField(null=True, blank=True, help_text="Time taken to compute (seconds)")
    
    # Expiry
    expires_at = models.DateTimeField(help_text="Cache expiry time")
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['cache_key']),
            models.Index(fields=['cache_type', 'expires_at']),
            models.Index(fields=['parameters_hash']),
        ]
        verbose_name = "Cache Entry"  
        verbose_name_plural = "Cache Entries"
    
    def __str__(self):
        return f"{self.cache_type}: {self.cache_key}"
    
    @property
    def is_expired(self):
        return timezone.now() > self.expires_at


class BulkUpload(models.Model):
    """Model to track bulk upload jobs and their progress"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # File information
    file = models.FileField(upload_to='bulk_uploads/', max_length=500)
    original_filename = models.CharField(max_length=255)
    file_size = models.BigIntegerField(help_text="File size in bytes")
    file_type = models.CharField(max_length=10, choices=[
        ('csv', 'CSV File'),
        ('json', 'JSON File'),
        ('xlsx', 'Excel File'),
    ])
    
    # Upload configuration
    clear_existing = models.BooleanField(default=False, help_text="Clear existing standards before import")
    generate_embeddings = models.BooleanField(default=False, help_text="Generate embeddings after import")
    batch_size = models.IntegerField(default=100, help_text="Number of records to process per batch")
    
    # Job status
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('uploading', 'Uploading'),
        ('preview', 'Preview'),
        ('confirmed', 'Confirmed'),
        ('processing', 'Processing'),
        ('generating_embeddings', 'Generating Embeddings'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    status = models.CharField(max_length=25, choices=STATUS_CHOICES, default='pending')
    
    # Progress tracking
    total_records = models.IntegerField(default=0)
    processed_records = models.IntegerField(default=0)
    successful_records = models.IntegerField(default=0)
    failed_records = models.IntegerField(default=0)
    progress_percentage = models.FloatField(default=0.0)
    
    # Preview data
    preview_data = models.JSONField(default=list, blank=True, help_text="Sample records for preview")
    column_mapping = models.JSONField(default=dict, blank=True, help_text="Column mapping configuration")
    
    # Results and errors
    error_log = models.JSONField(default=list, blank=True, help_text="List of errors encountered during processing")
    processing_summary = models.JSONField(default=dict, blank=True, help_text="Summary of processing results")
    
    # Celery task tracking
    task_id = models.CharField(max_length=255, blank=True, help_text="Celery task ID for async processing")
    
    # Timing information
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    estimated_completion = models.DateTimeField(null=True, blank=True)
    
    # User information
    uploaded_by = models.CharField(max_length=150, help_text="Username of the user who uploaded the file")
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status', 'created_at']),
            models.Index(fields=['uploaded_by', 'created_at']),
            models.Index(fields=['task_id']),
        ]
        verbose_name = "Bulk Upload"
        verbose_name_plural = "Bulk Uploads"
    
    def __str__(self):
        return f"{self.original_filename} - {self.get_status_display()} ({self.progress_percentage:.1f}%)"
    
    @property
    def is_active(self):
        """Check if the job is currently active"""
        return self.status in ['pending', 'uploading', 'preview', 'confirmed', 'processing', 'generating_embeddings']
    
    @property
    def duration(self):
        """Get the duration of the job"""
        if self.started_at:
            end_time = self.completed_at if self.completed_at else timezone.now()
            return end_time - self.started_at
        return None
    
    def update_progress(self, processed=None, successful=None, failed=None, status=None):
        """Update job progress"""
        if processed is not None:
            self.processed_records = processed
        if successful is not None:
            self.successful_records = successful
        if failed is not None:
            self.failed_records = failed
        if status is not None:
            self.status = status
            if status == 'processing' and not self.started_at:
                self.started_at = timezone.now()
            elif status in ['completed', 'failed', 'cancelled']:
                self.completed_at = timezone.now()
        
        # Calculate progress percentage
        if self.total_records > 0:
            self.progress_percentage = (self.processed_records / self.total_records) * 100
        
        self.save(update_fields=[
            'processed_records', 'successful_records', 'failed_records', 
            'progress_percentage', 'status', 'started_at', 'completed_at', 'updated_at'
        ])
    
    def add_error(self, error_message, line_number=None, record_data=None):
        """Add an error to the error log"""
        error_entry = {
            'message': str(error_message),
            'timestamp': timezone.now().isoformat(),
        }
        
        if line_number is not None:
            error_entry['line_number'] = line_number
        if record_data is not None:
            error_entry['record_data'] = record_data
        
        if not self.error_log:
            self.error_log = []
        self.error_log.append(error_entry)
        self.save(update_fields=['error_log'])
    
    def get_error_summary(self):
        """Get a summary of errors"""
        if not self.error_log:
            return "No errors"
        
        error_types = {}
        for error in self.error_log:
            msg = error.get('message', 'Unknown error')
            error_types[msg] = error_types.get(msg, 0) + 1
        
        return error_types


class UploadJob(models.Model):
    """Model to track bulk upload progress and results"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # File information
    original_filename = models.CharField(max_length=255)
    file_size = models.BigIntegerField(help_text="File size in bytes")
    file_type = models.CharField(max_length=10, choices=[
        ('csv', 'CSV File'),
        ('json', 'JSON File'),
    ])
    
    # Upload configuration
    clear_existing = models.BooleanField(default=False, help_text="Clear existing standards before import")
    generate_embeddings = models.BooleanField(default=False, help_text="Generate embeddings after import")
    batch_size = models.IntegerField(default=100, help_text="Number of records to process per batch")
    
    # Job status
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('validating', 'Validating File'),
        ('processing', 'Processing'),
        ('generating_embeddings', 'Generating Embeddings'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    status = models.CharField(max_length=25, choices=STATUS_CHOICES, default='pending')
    
    # Progress tracking
    total_records = models.IntegerField(default=0)
    processed_records = models.IntegerField(default=0)
    successful_records = models.IntegerField(default=0)
    failed_records = models.IntegerField(default=0)
    progress_percentage = models.FloatField(default=0.0)
    
    # Results and errors
    error_log = models.JSONField(default=list, blank=True, help_text="List of errors encountered during processing")
    processing_summary = models.JSONField(default=dict, blank=True, help_text="Summary of processing results")
    
    # Timing information
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    estimated_completion = models.DateTimeField(null=True, blank=True)
    
    # User information
    uploaded_by = models.CharField(max_length=150, help_text="Username of the user who uploaded the file")
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status', 'created_at']),
            models.Index(fields=['uploaded_by', 'created_at']),
        ]
        verbose_name = "Upload Job"
        verbose_name_plural = "Upload Jobs"
    
    def __str__(self):
        return f"{self.original_filename} - {self.get_status_display()} ({self.progress_percentage:.1f}%)"
    
    @property
    def is_active(self):
        """Check if the job is currently active (not completed/failed/cancelled)"""
        return self.status in ['pending', 'validating', 'processing', 'generating_embeddings']
    
    @property
    def duration(self):
        """Get the duration of the job"""
        if self.started_at:
            end_time = self.completed_at if self.completed_at else timezone.now()
            return end_time - self.started_at
        return None
    
    def update_progress(self, processed=None, successful=None, failed=None, status=None):
        """Update job progress"""
        if processed is not None:
            self.processed_records = processed
        if successful is not None:
            self.successful_records = successful
        if failed is not None:
            self.failed_records = failed
        if status is not None:
            self.status = status
            if status == 'processing' and not self.started_at:
                self.started_at = timezone.now()
            elif status in ['completed', 'failed', 'cancelled']:
                self.completed_at = timezone.now()
        
        # Calculate progress percentage
        if self.total_records > 0:
            self.progress_percentage = (self.processed_records / self.total_records) * 100
        
        self.save(update_fields=[
            'processed_records', 'successful_records', 'failed_records', 
            'progress_percentage', 'status', 'started_at', 'completed_at', 'updated_at'
        ])
    
    def add_error(self, error_message, line_number=None, record_data=None):
        """Add an error to the error log"""
        error_entry = {
            'message': str(error_message),
            'timestamp': timezone.now().isoformat(),
        }
        
        if line_number is not None:
            error_entry['line_number'] = line_number
        if record_data is not None:
            error_entry['record_data'] = record_data
        
        if not self.error_log:
            self.error_log = []
        self.error_log.append(error_entry)
        self.save(update_fields=['error_log'])
    
    def get_error_summary(self):
        """Get a summary of errors"""
        if not self.error_log:
            return "No errors"
        
        error_types = {}
        for error in self.error_log:
            msg = error.get('message', 'Unknown error')
            error_types[msg] = error_types.get(msg, 0) + 1
        
        return error_types


class CorrelationJob(models.Model):
    """Model to track correlation generation progress and results"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Job configuration
    similarity_threshold = models.FloatField(
        default=0.8, 
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Minimum similarity threshold for creating correlations (0.0-1.0)"
    )
    clear_existing = models.BooleanField(default=False, help_text="Clear existing correlations before generation")
    batch_size = models.IntegerField(default=50, help_text="Number of standards to process per batch")
    
    # Filtering options
    subject_area_filter = models.ForeignKey(
        SubjectArea, on_delete=models.CASCADE, null=True, blank=True,
        help_text="Generate correlations only for this subject area"
    )
    state_filter = models.ManyToManyField(
        State, blank=True,
        help_text="Generate correlations only for these states (empty = all states)"
    )
    
    # Job status
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('analyzing', 'Analyzing Standards'),
        ('generating', 'Generating Correlations'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    status = models.CharField(max_length=25, choices=STATUS_CHOICES, default='pending')
    
    # Progress tracking
    total_standards = models.IntegerField(default=0, help_text="Total standards to process")
    processed_standards = models.IntegerField(default=0, help_text="Standards processed so far")
    correlations_created = models.IntegerField(default=0, help_text="New correlations created")
    correlations_updated = models.IntegerField(default=0, help_text="Existing correlations updated")
    correlations_skipped = models.IntegerField(default=0, help_text="Correlations skipped (duplicates)")
    progress_percentage = models.FloatField(default=0.0)
    
    # Performance metrics
    avg_processing_time_per_standard = models.FloatField(
        null=True, blank=True, 
        help_text="Average processing time per standard in seconds"
    )
    estimated_completion = models.DateTimeField(null=True, blank=True)
    
    # Results and errors
    error_log = models.JSONField(default=list, blank=True, help_text="List of errors encountered during processing")
    processing_summary = models.JSONField(default=dict, blank=True, help_text="Summary of processing results")
    statistics = models.JSONField(default=dict, blank=True, help_text="Generation statistics and metrics")
    
    # Timing information
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # User information
    created_by = models.CharField(max_length=150, help_text="Username of the user who started the job")
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status', 'created_at']),
            models.Index(fields=['created_by', 'created_at']),
            models.Index(fields=['similarity_threshold']),
        ]
        verbose_name = "Correlation Job"
        verbose_name_plural = "Correlation Jobs"
    
    def __str__(self):
        scope = f"{self.subject_area_filter.name}" if self.subject_area_filter else "All Subjects"
        return f"Correlation Generation: {scope} (Threshold: {self.similarity_threshold}) - {self.get_status_display()} ({self.progress_percentage:.1f}%)"
    
    @property
    def is_active(self):
        """Check if the job is currently active (not completed/failed/cancelled)"""
        return self.status in ['pending', 'analyzing', 'generating']
    
    @property
    def duration(self):
        """Get the duration of the job"""
        if self.started_at:
            end_time = self.completed_at if self.completed_at else timezone.now()
            return end_time - self.started_at
        return None
    
    @property
    def total_correlations_processed(self):
        """Get total correlations processed (created + updated + skipped)"""
        return self.correlations_created + self.correlations_updated + self.correlations_skipped
    
    def update_progress(self, processed=None, created=None, updated=None, skipped=None, status=None):
        """Update job progress"""
        if processed is not None:
            self.processed_standards = processed
        if created is not None:
            self.correlations_created = created
        if updated is not None:
            self.correlations_updated = updated
        if skipped is not None:
            self.correlations_skipped = skipped
        if status is not None:
            self.status = status
            if status == 'generating' and not self.started_at:
                self.started_at = timezone.now()
            elif status in ['completed', 'failed', 'cancelled']:
                self.completed_at = timezone.now()
        
        # Calculate progress percentage
        if self.total_standards > 0:
            self.progress_percentage = (self.processed_standards / self.total_standards) * 100
        
        # Update average processing time
        if self.processed_standards > 0 and self.started_at:
            elapsed = timezone.now() - self.started_at
            self.avg_processing_time_per_standard = elapsed.total_seconds() / self.processed_standards
            
            # Estimate completion time
            if self.total_standards > self.processed_standards:
                remaining_standards = self.total_standards - self.processed_standards
                estimated_remaining_seconds = remaining_standards * self.avg_processing_time_per_standard
                self.estimated_completion = timezone.now() + timezone.timedelta(seconds=estimated_remaining_seconds)
        
        self.save(update_fields=[
            'processed_standards', 'correlations_created', 'correlations_updated', 'correlations_skipped',
            'progress_percentage', 'status', 'started_at', 'completed_at', 'avg_processing_time_per_standard',
            'estimated_completion', 'updated_at'
        ])
    
    def add_error(self, error_message, standard_code=None, context=None):
        """Add an error to the error log"""
        error_entry = {
            'message': str(error_message),
            'timestamp': timezone.now().isoformat(),
        }
        
        if standard_code is not None:
            error_entry['standard_code'] = standard_code
        if context is not None:
            error_entry['context'] = context
        
        if not self.error_log:
            self.error_log = []
        self.error_log.append(error_entry)
        self.save(update_fields=['error_log'])
    
    def get_error_summary(self):
        """Get a summary of errors"""
        if not self.error_log:
            return "No errors"
        
        error_types = {}
        for error in self.error_log:
            msg = error.get('message', 'Unknown error')
            error_types[msg] = error_types.get(msg, 0) + 1
        
        return error_types
    
    def get_performance_summary(self):
        """Get performance summary"""
        summary = {
            'total_correlations': self.total_correlations_processed,
            'creation_rate': 0,
            'duration': str(self.duration) if self.duration else None,
        }
        
        if self.duration and self.total_correlations_processed > 0:
            summary['creation_rate'] = self.total_correlations_processed / self.duration.total_seconds()
        
        return summary
