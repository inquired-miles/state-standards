"""
Forms for dashboard topic management
"""
from django import forms
from standards.models import Concept, SubjectArea, GradeLevel


class ConceptForm(forms.ModelForm):
    """Form for creating and editing concepts"""
    
    keywords = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'placeholder': 'Enter keywords separated by commas',
            'class': 'form-control'
        }),
        help_text='Separate multiple keywords with commas'
    )
    
    class Meta:
        model = Concept
        fields = ['name', 'description', 'subject_areas', 'grade_levels', 'keywords']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., Fraction Operations'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Describe this educational concept...'
            }),
            'subject_areas': forms.CheckboxSelectMultiple(attrs={
                'class': 'form-check-input'
            }),
            'grade_levels': forms.CheckboxSelectMultiple(attrs={
                'class': 'form-check-input'
            }),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Populate keywords field from model array
        if self.instance and self.instance.keywords:
            self.fields['keywords'].initial = ', '.join(self.instance.keywords)
    
    def clean_keywords(self):
        """Convert comma-separated keywords to list"""
        keywords_str = self.cleaned_data.get('keywords', '')
        if keywords_str:
            keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
            return keywords
        return []


class TopicAnalysisForm(forms.Form):
    """Form for configuring topic analysis runs"""
    
    ANALYSIS_TYPES = [
        ('discover', 'Discover New Topics'),
        ('update_coverage', 'Update Coverage Statistics'),
        ('cluster_refinement', 'Refine Existing Clusters'),
        ('full_analysis', 'Full Topic Analysis'),
    ]
    
    analysis_type = forms.ChoiceField(
        choices=ANALYSIS_TYPES,
        widget=forms.Select(attrs={'class': 'form-select'}),
        initial='discover'
    )
    
    subject_area = forms.ModelChoiceField(
        queryset=SubjectArea.objects.all(),
        required=False,
        empty_label="All Subject Areas",
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    grade_levels = forms.ModelMultipleChoiceField(
        queryset=GradeLevel.objects.all().order_by('grade_numeric'),
        required=False,
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'})
    )
    
    min_standards_per_topic = forms.IntegerField(
        initial=5,
        min_value=2,
        max_value=50,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '2',
            'max': '50'
        }),
        help_text='Minimum number of standards required to form a topic cluster'
    )
    
    similarity_threshold = forms.FloatField(
        initial=0.75,
        min_value=0.1,
        max_value=1.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.05',
            'min': '0.1',
            'max': '1.0'
        }),
        help_text='Similarity threshold for clustering (0.1-1.0)'
    )
    
    update_existing = forms.BooleanField(
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text='Update existing topics with new analysis results'
    )
    
    generate_embeddings = forms.BooleanField(
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text='Generate embeddings for discovered topics'
    )


class TopicSearchForm(forms.Form):
    """Form for searching and filtering topics"""
    
    CATEGORY_CHOICES = [
        ('', 'All Categories'),
        ('must-have', 'Must-Have (90%+ states)'),
        ('important', 'Important (60-89% states)'),
        ('regional', 'Regional (30-59% states)'),
        ('specialized', 'Specialized (<30% states)'),
    ]
    
    SORT_CHOICES = [
        ('prevalence', 'Prevalence (High to Low)'),
        ('alphabetical', 'Alphabetical'),
        ('category', 'Category'),
        ('created', 'Recently Created'),
        ('updated', 'Recently Updated'),
    ]
    
    search_query = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Search topics by name or description...'
        })
    )
    
    subject_area = forms.ModelChoiceField(
        queryset=SubjectArea.objects.all(),
        required=False,
        empty_label="All Subjects",
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    category = forms.ChoiceField(
        choices=CATEGORY_CHOICES,
        required=False,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    grade_level = forms.ModelChoiceField(
        queryset=GradeLevel.objects.all().order_by('grade_numeric'),
        required=False,
        empty_label="All Grades",
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    sort_by = forms.ChoiceField(
        choices=SORT_CHOICES,
        initial='prevalence',
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    min_coverage = forms.IntegerField(
        required=False,
        min_value=0,
        max_value=50,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Min states',
            'min': '0',
            'max': '50'
        }),
        help_text='Minimum number of states covering the topic'
    )