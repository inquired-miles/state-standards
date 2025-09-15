"""
Forms for the standards app including bulk upload functionality
"""
import os
import json
import csv
from django import forms
from django.core.exceptions import ValidationError
from django.core.files.uploadedfile import UploadedFile
from .models import UploadJob, CorrelationJob, State, SubjectArea


class BulkUploadForm(forms.Form):
    """Form for bulk uploading standards data"""
    
    file = forms.FileField(
        help_text="Upload a CSV or JSON file containing standards data. Maximum file size: 50MB",
        widget=forms.ClearableFileInput(attrs={
            'accept': '.csv,.json',
            'class': 'form-control-file',
            'id': 'id_upload_file'
        })
    )
    
    clear_existing = forms.BooleanField(
        required=False,
        initial=False,
        help_text="Clear all existing standards before importing new ones (USE WITH CAUTION)",
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input',
            'id': 'id_clear_existing'
        })
    )
    
    generate_embeddings = forms.BooleanField(
        required=False,
        initial=True,
        help_text="Generate OpenAI embeddings for imported standards (recommended)",
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input',
            'id': 'id_generate_embeddings'
        })
    )
    
    batch_size = forms.IntegerField(
        initial=100,
        min_value=10,
        max_value=1000,
        help_text="Number of records to process in each batch (10-1000)",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'id': 'id_batch_size',
            'min': 10,
            'max': 1000
        })
    )
    
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
    
    def clean_file(self):
        """Validate uploaded file"""
        file = self.cleaned_data.get('file')
        
        if not file:
            raise ValidationError("Please select a file to upload.")
        
        # Check file size (50MB limit)
        max_size = 50 * 1024 * 1024  # 50MB
        if file.size > max_size:
            raise ValidationError(f"File is too large. Maximum size is {max_size // (1024*1024)}MB.")
        
        # Check file extension
        name = file.name.lower()
        if not (name.endswith('.csv') or name.endswith('.json')):
            raise ValidationError("Only CSV and JSON files are allowed.")
        
        # Validate file content
        self._validate_file_content(file)
        
        return file
    
    def _validate_file_content(self, file):
        """Validate the structure and content of the uploaded file"""
        file.seek(0)  # Reset file pointer
        
        try:
            if file.name.lower().endswith('.csv'):
                self._validate_csv_content(file)
            elif file.name.lower().endswith('.json'):
                self._validate_json_content(file)
        except Exception as e:
            raise ValidationError(f"File validation error: {str(e)}")
        finally:
            file.seek(0)  # Reset file pointer for later use
    
    def _validate_csv_content(self, file):
        """Validate CSV file structure"""
        # Read first few lines to check structure
        content = file.read(10000).decode('utf-8', errors='ignore')  # Read first 10KB
        file.seek(0)
        
        lines = content.split('\n')
        if len(lines) < 2:
            raise ValidationError("CSV file must contain at least a header row and one data row.")
        
        # Check if it looks like CSV
        reader = csv.DictReader(lines)
        # Only require: state, code, description (title is optional)
        required_fields = ['state', 'code', 'description']
        
        try:
            header = next(reader)
            fieldnames = reader.fieldnames or []
            
            missing_fields = [field for field in required_fields if field not in fieldnames]
            if missing_fields:
                raise ValidationError(
                    f"CSV file is missing required columns: {', '.join(missing_fields)}. "
                    f"Required fields are: state (must exist), code (unique per state), description (standard description)."
                )
            
            # Try to read first data row
            try:
                first_row = next(reader)
                if not first_row.get('state') or not first_row.get('code'):
                    raise ValidationError("First data row is missing required values for 'state' or 'code'.")
            except StopIteration:
                raise ValidationError("CSV file contains only headers, no data rows found.")
                
        except Exception as e:
            raise ValidationError(f"Invalid CSV format: {str(e)}")
    
    def _validate_json_content(self, file):
        """Validate JSON file structure"""
        try:
            content = file.read().decode('utf-8')
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON format: {str(e)}")
        except UnicodeDecodeError:
            raise ValidationError("File encoding error. Please ensure the file is UTF-8 encoded.")
        
        # Check if data is in expected format
        if isinstance(data, dict):
            # Could be organized by state
            if not data:
                raise ValidationError("JSON file is empty.")
            
            # Check if values are lists of standards
            sample_value = next(iter(data.values()))
            if not isinstance(sample_value, (list, dict)):
                raise ValidationError("JSON file format is not supported. Expected state-organized data or list of standards.")
                
        elif isinstance(data, list):
            if not data:
                raise ValidationError("JSON file contains an empty list.")
            
            # Check first record for required fields
            first_record = data[0]
            if not isinstance(first_record, dict):
                raise ValidationError("JSON file should contain a list of objects.")
            
            # Only require: state, code, description (title is optional)
            required_fields = ['state', 'code', 'description']
            missing_fields = [field for field in required_fields if field not in first_record]
            if missing_fields:
                raise ValidationError(
                    f"JSON records are missing required fields: {', '.join(missing_fields)}. "
                    f"Required fields are: state (must exist), code (unique per state), description (standard description)."
                )
        else:
            raise ValidationError("JSON file should contain either a list of standards or an object organized by state.")
    
    def clean(self):
        """Additional form validation"""
        cleaned_data = super().clean()
        
        # Warning for clear_existing
        if cleaned_data.get('clear_existing'):
            # This will be handled in the view with a confirmation step
            pass
        
        return cleaned_data
    
    def save(self):
        """Create an UploadJob instance"""
        if not self.is_valid():
            raise ValueError("Form is not valid")
        
        file = self.cleaned_data['file']
        
        # Determine file type
        file_type = 'csv' if file.name.lower().endswith('.csv') else 'json'
        
        # Create UploadJob
        upload_job = UploadJob.objects.create(
            original_filename=file.name,
            file_size=file.size,
            file_type=file_type,
            clear_existing=self.cleaned_data['clear_existing'],
            generate_embeddings=self.cleaned_data['generate_embeddings'],
            batch_size=self.cleaned_data['batch_size'],
            uploaded_by=self.user.username if self.user else 'unknown',
            status='pending'
        )
        
        return upload_job, file


class PreviewUploadForm(forms.Form):
    """Form for previewing upload before processing"""
    
    confirm_upload = forms.BooleanField(
        required=True,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input'
        }),
        label="I confirm that I want to proceed with this upload"
    )
    
    upload_job_id = forms.UUIDField(
        widget=forms.HiddenInput()
    )


class GenerateTemplateForm(forms.Form):
    """Form for generating sample templates"""
    
    TEMPLATE_FORMATS = [
        ('csv', 'CSV Format'),
        ('json', 'JSON Format'),
    ]
    
    format = forms.ChoiceField(
        choices=TEMPLATE_FORMATS,
        initial='csv',
        widget=forms.RadioSelect(attrs={
            'class': 'form-check-input'
        }),
        help_text="Select the format for the sample template"
    )
    
    include_sample_data = forms.BooleanField(
        required=False,
        initial=True,
        help_text="Include sample data rows in the template",
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input'
        })
    )
    
    states_count = forms.IntegerField(
        initial=3,
        min_value=1,
        max_value=10,
        help_text="Number of sample states to include (1-10)",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': 1,
            'max': 10
        })
    )


class BulkActionForm(forms.Form):
    """Form for bulk actions on standards"""
    
    ACTION_CHOICES = [
        ('generate_embeddings', 'Generate Embeddings'),
        ('recalculate_correlations', 'Recalculate Correlations'),
        ('export_csv', 'Export to CSV'),
        ('export_json', 'Export to JSON'),
        ('validate_data', 'Validate Data'),
    ]
    
    action = forms.ChoiceField(
        choices=ACTION_CHOICES,
        widget=forms.Select(attrs={
            'class': 'form-control'
        })
    )
    
    selected_standards = forms.CharField(
        widget=forms.HiddenInput(),
        help_text="Comma-separated list of standard IDs"
    )
    
    def clean_selected_standards(self):
        """Validate selected standards"""
        standards_str = self.cleaned_data.get('selected_standards', '')
        
        if not standards_str:
            raise ValidationError("No standards selected.")
        
        try:
            # Convert to list of UUIDs
            standard_ids = [id.strip() for id in standards_str.split(',') if id.strip()]
            if not standard_ids:
                raise ValidationError("No valid standard IDs provided.")
            
            return standard_ids
        except Exception as e:
            raise ValidationError(f"Invalid standard ID format: {str(e)}")


class GenerateCorrelationsForm(forms.Form):
    """Form for generating correlations between standards"""
    
    similarity_threshold = forms.FloatField(
        initial=0.8,
        min_value=0.0,
        max_value=1.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'id': 'id_similarity_threshold',
            'min': '0.0',
            'max': '1.0',
            'step': '0.01',
            'data-slider': 'true'
        }),
        help_text="Minimum similarity threshold for creating correlations (0.0 = very loose, 1.0 = exact match)"
    )
    
    clear_existing = forms.BooleanField(
        required=False,
        initial=False,
        help_text="Clear all existing correlations before generating new ones (USE WITH CAUTION)",
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input',
            'id': 'id_clear_existing'
        })
    )
    
    subject_area_filter = forms.ModelChoiceField(
        queryset=SubjectArea.objects.all(),
        required=False,
        empty_label="All Subjects",
        help_text="Generate correlations only for standards in this subject area",
        widget=forms.Select(attrs={
            'class': 'form-control',
            'id': 'id_subject_area_filter'
        })
    )
    
    state_filter = forms.ModelMultipleChoiceField(
        queryset=State.objects.all(),
        required=False,
        help_text="Generate correlations only for standards from these states (leave empty for all states)",
        widget=forms.SelectMultiple(attrs={
            'class': 'form-control',
            'id': 'id_state_filter',
            'data-multiselect': 'true'
        })
    )
    
    batch_size = forms.IntegerField(
        initial=50,
        min_value=10,
        max_value=500,
        help_text="Number of standards to process in each batch (10-500)",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'id': 'id_batch_size',
            'min': 10,
            'max': 500
        })
    )
    
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
    
    def clean_similarity_threshold(self):
        """Validate similarity threshold"""
        threshold = self.cleaned_data.get('similarity_threshold')
        
        if threshold is None:
            raise ValidationError("Similarity threshold is required.")
        
        if threshold < 0.0 or threshold > 1.0:
            raise ValidationError("Similarity threshold must be between 0.0 and 1.0.")
        
        # Warn for very low thresholds
        if threshold < 0.5:
            # This will be handled in the view with a warning
            pass
        
        return threshold
    
    def clean(self):
        """Additional form validation"""
        cleaned_data = super().clean()
        
        # Get preview information without executing
        state_filter = cleaned_data.get('state_filter', [])
        subject_filter = cleaned_data.get('subject_area_filter')
        
        # Calculate estimated scope
        from .models import Standard
        
        queryset = Standard.objects.filter(embedding__isnull=False)
        
        if subject_filter:
            queryset = queryset.filter(subject_area=subject_filter)
        
        if state_filter:
            queryset = queryset.filter(state__in=state_filter)
        
        standards_count = queryset.count()
        
        if standards_count == 0:
            raise ValidationError("No standards found matching the specified filters.")
        
        # Estimate number of correlations that might be generated
        # This is a rough estimate: N * avg_similar_per_standard
        estimated_correlations = standards_count * 5  # Rough estimate
        
        cleaned_data['_preview_info'] = {
            'standards_count': standards_count,
            'estimated_correlations': estimated_correlations,
        }
        
        return cleaned_data
    
    def save(self):
        """Create a CorrelationJob instance"""
        if not self.is_valid():
            raise ValueError("Form is not valid")
        
        # Create CorrelationJob
        correlation_job = CorrelationJob.objects.create(
            similarity_threshold=self.cleaned_data['similarity_threshold'],
            clear_existing=self.cleaned_data['clear_existing'],
            batch_size=self.cleaned_data['batch_size'],
            subject_area_filter=self.cleaned_data.get('subject_area_filter'),
            created_by=self.user.username if self.user else 'unknown',
            status='pending'
        )
        
        # Set state filter if provided
        state_filter = self.cleaned_data.get('state_filter')
        if state_filter:
            correlation_job.state_filter.set(state_filter)
        
        return correlation_job


class CorrelationPreviewForm(forms.Form):
    """Form for confirming correlation generation after preview"""
    
    confirm_generation = forms.BooleanField(
        required=True,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input'
        }),
        label="I confirm that I want to proceed with correlation generation"
    )
    
    correlation_job_id = forms.UUIDField(
        widget=forms.HiddenInput()
    )


class ThresholdAnalysisForm(forms.Form):
    """Form for analyzing correlation thresholds"""
    
    min_threshold = forms.FloatField(
        initial=0.5,
        min_value=0.0,
        max_value=1.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.05'
        }),
        help_text="Minimum threshold to analyze"
    )
    
    max_threshold = forms.FloatField(
        initial=1.0,
        min_value=0.0,
        max_value=1.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.05'
        }),
        help_text="Maximum threshold to analyze"
    )
    
    step_size = forms.FloatField(
        initial=0.05,
        min_value=0.01,
        max_value=0.1,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.01'
        }),
        help_text="Step size for threshold analysis"
    )
    
    sample_size = forms.IntegerField(
        initial=100,
        min_value=10,
        max_value=1000,
        widget=forms.NumberInput(attrs={
            'class': 'form-control'
        }),
        help_text="Number of standards to sample for analysis"
    )
    
    def clean(self):
        """Validate threshold range"""
        cleaned_data = super().clean()
        
        min_threshold = cleaned_data.get('min_threshold')
        max_threshold = cleaned_data.get('max_threshold')
        
        if min_threshold and max_threshold and min_threshold >= max_threshold:
            raise ValidationError("Minimum threshold must be less than maximum threshold.")
        
        return cleaned_data