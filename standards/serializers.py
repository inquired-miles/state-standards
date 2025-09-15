"""
Django REST Framework serializers for the EdTech Standards Alignment System
"""
from rest_framework import serializers
from .models import (
    State, SubjectArea, GradeLevel, Standard, StandardCorrelation,
    Concept, TopicCluster, ClusterMembership, CoverageAnalysis,
    ContentAlignment, ContentStandardMatch, StrategicPlan, CacheEntry
)


class StateSerializer(serializers.ModelSerializer):
    """Serializer for State model"""
    standards_count = serializers.SerializerMethodField()
    
    class Meta:
        model = State
        fields = ['code', 'name', 'standards_count', 'created_at', 'updated_at']
    
    def get_standards_count(self, obj):
        """Get count of standards for this state"""
        return obj.standards.count()


class SubjectAreaSerializer(serializers.ModelSerializer):
    """Serializer for SubjectArea model"""
    standards_count = serializers.SerializerMethodField()
    
    class Meta:
        model = SubjectArea
        fields = ['id', 'name', 'description', 'standards_count', 'created_at', 'updated_at']
    
    def get_standards_count(self, obj):
        """Get count of standards for this subject area"""
        return obj.standards.count()


class GradeLevelSerializer(serializers.ModelSerializer):
    """Serializer for GradeLevel model"""
    standards_count = serializers.SerializerMethodField()
    
    class Meta:
        model = GradeLevel
        fields = ['id', 'grade', 'grade_numeric', 'standards_count', 'created_at', 'updated_at']
    
    def get_standards_count(self, obj):
        """Get count of standards for this grade level"""
        return obj.standards.count()


class StandardSerializer(serializers.ModelSerializer):
    """Serializer for Standard model"""
    state_code = serializers.CharField(source='state.code', read_only=True)
    state_name = serializers.CharField(source='state.name', read_only=True)
    subject_name = serializers.CharField(source='subject_area.name', read_only=True)
    grade_names = serializers.SerializerMethodField()
    similarity_scores = serializers.SerializerMethodField()
    display_name = serializers.ReadOnlyField()
    
    class Meta:
        model = Standard
        fields = [
            'id', 'code', 'title', 'display_name', 'description', 'domain', 'cluster',
            'state_code', 'state_name', 'subject_name', 'grade_names',
            'keywords', 'skills', 'similarity_scores', 'created_at', 'updated_at'
        ]
    
    def get_grade_names(self, obj):
        """Get list of grade level names"""
        return [grade.grade for grade in obj.grade_levels.all()]
    
    def get_similarity_scores(self, obj):
        """Get top similarity scores for this standard"""
        correlations = obj.correlations_as_standard_1.all()[:5]
        return [
            {
                'related_standard_code': corr.standard_2.code,
                'related_state': corr.standard_2.state.code,
                'similarity_score': corr.similarity_score,
                'correlation_type': corr.correlation_type
            }
            for corr in correlations
        ]


class StandardDetailSerializer(StandardSerializer):
    """Detailed serializer for Standard model including embedding"""
    state = StateSerializer(read_only=True)
    subject_area = SubjectAreaSerializer(read_only=True)
    grade_levels = GradeLevelSerializer(many=True, read_only=True)
    
    class Meta(StandardSerializer.Meta):
        fields = StandardSerializer.Meta.fields + [
            'state', 'subject_area', 'grade_levels', 'embedding'
        ]


class StandardCorrelationSerializer(serializers.ModelSerializer):
    """Serializer for StandardCorrelation model"""
    standard_1_details = serializers.SerializerMethodField()
    standard_2_details = serializers.SerializerMethodField()
    
    class Meta:
        model = StandardCorrelation
        fields = [
            'id', 'standard_1_details', 'standard_2_details',
            'similarity_score', 'correlation_type', 'notes', 'verified',
            'created_at', 'updated_at'
        ]
    
    def get_standard_1_details(self, obj):
        """Get details for standard 1"""
        return {
            'id': obj.standard_1.id,
            'code': obj.standard_1.code,
            'title': obj.standard_1.title,
            'state_code': obj.standard_1.state.code,
            'subject': obj.standard_1.subject_area.name
        }
    
    def get_standard_2_details(self, obj):
        """Get details for standard 2"""
        return {
            'id': obj.standard_2.id,
            'code': obj.standard_2.code,
            'title': obj.standard_2.title,
            'state_code': obj.standard_2.state.code,
            'subject': obj.standard_2.subject_area.name
        }


class ConceptSerializer(serializers.ModelSerializer):
    """Serializer for Concept model"""
    subject_names = serializers.SerializerMethodField()
    grade_names = serializers.SerializerMethodField()
    
    class Meta:
        model = Concept
        fields = [
            'id', 'name', 'description', 'keywords', 'complexity_score',
            'states_covered', 'coverage_percentage', 'subject_names', 'grade_names',
            'created_at', 'updated_at'
        ]
    
    def get_subject_names(self, obj):
        """Get list of subject area names"""
        return [subject.name for subject in obj.subject_areas.all()]
    
    def get_grade_names(self, obj):
        """Get list of grade level names"""
        return [grade.grade for grade in obj.grade_levels.all()]


class ClusterMembershipSerializer(serializers.ModelSerializer):
    """Serializer for ClusterMembership model"""
    standard_code = serializers.CharField(source='standard.code', read_only=True)
    standard_title = serializers.CharField(source='standard.title', read_only=True)
    state_code = serializers.CharField(source='standard.state.code', read_only=True)
    
    class Meta:
        model = ClusterMembership
        fields = [
            'standard_code', 'standard_title', 'state_code',
            'membership_strength', 'created_at'
        ]


class TopicClusterSerializer(serializers.ModelSerializer):
    """Serializer for TopicCluster model"""
    subject_name = serializers.CharField(source='subject_area.name', read_only=True)
    grade_names = serializers.SerializerMethodField()
    top_standards = serializers.SerializerMethodField()
    
    class Meta:
        model = TopicCluster
        fields = [
            'id', 'name', 'description', 'subject_name', 'grade_names',
            'silhouette_score', 'cohesion_score', 'standards_count', 'states_represented',
            'common_terms', 'regional_pattern', 'top_standards',
            'created_at', 'updated_at'
        ]
    
    def get_grade_names(self, obj):
        """Get list of grade level names"""
        return [grade.grade for grade in obj.grade_levels.all()]
    
    def get_top_standards(self, obj):
        """Get top standards in this cluster by membership strength"""
        memberships = obj.clustermembership_set.order_by('-membership_strength')[:5]
        return ClusterMembershipSerializer(memberships, many=True).data


class CoverageAnalysisSerializer(serializers.ModelSerializer):
    """Serializer for CoverageAnalysis model"""
    state_name = serializers.CharField(source='state.name', read_only=True)
    subject_name = serializers.CharField(source='subject_area.name', read_only=True)
    grade_name = serializers.CharField(source='grade_level.grade', read_only=True)
    
    class Meta:
        model = CoverageAnalysis
        fields = [
            'id', 'analysis_type', 'state_name', 'subject_name', 'grade_name',
            'total_standards', 'covered_concepts', 'coverage_percentage',
            'bell_curve_data', 'gap_analysis', 'benchmark_comparison',
            'created_at', 'updated_at'
        ]


class ContentStandardMatchSerializer(serializers.ModelSerializer):
    """Serializer for ContentStandardMatch model"""
    standard_details = serializers.SerializerMethodField()
    
    class Meta:
        model = ContentStandardMatch
        fields = [
            'match_type', 'confidence_score', 'similarity_score',
            'matched_text_snippet', 'explanation', 'standard_details', 'created_at'
        ]
    
    def get_standard_details(self, obj):
        """Get details for the matched standard"""
        return {
            'id': obj.standard.id,
            'code': obj.standard.code,
            'title': obj.standard.title,
            'state_code': obj.standard.state.code,
            'subject': obj.standard.subject_area.name
        }


class ContentAlignmentSerializer(serializers.ModelSerializer):
    """Serializer for ContentAlignment model"""
    matched_standards = ContentStandardMatchSerializer(
        source='contentstandardmatch_set', many=True, read_only=True
    )
    full_alignment_states = serializers.StringRelatedField(
        source='states_with_full_alignment', many=True, read_only=True
    )
    partial_alignment_states = serializers.StringRelatedField(
        source='states_with_partial_alignment', many=True, read_only=True
    )
    
    class Meta:
        model = ContentAlignment
        fields = [
            'id', 'content_title', 'original_filename', 'file_type',
            'content_hash', 'total_standards_analyzed', 'overall_alignment_score',
            'exact_matches', 'semantic_matches', 'conceptual_matches',
            'full_alignment_states', 'partial_alignment_states',
            'alignment_report', 'improvement_suggestions', 'gap_analysis',
            'matched_standards', 'created_at', 'updated_at'
        ]


class ContentAlignmentCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating ContentAlignment instances"""
    content_file = serializers.FileField(write_only=True, required=False)
    
    class Meta:
        model = ContentAlignment
        fields = [
            'content_title', 'content_text', 'original_filename', 
            'file_type', 'content_file'
        ]
    
    def create(self, validated_data):
        """Create ContentAlignment instance with file processing"""
        content_file = validated_data.pop('content_file', None)
        
        if content_file:
            # Process file content based on file type
            validated_data['original_filename'] = content_file.name
            validated_data['file_type'] = self._get_file_type(content_file.name)
            validated_data['content_text'] = self._extract_text_from_file(content_file)
        
        # Generate content hash
        import hashlib
        validated_data['content_hash'] = hashlib.sha256(
            validated_data['content_text'].encode()
        ).hexdigest()
        
        return super().create(validated_data)
    
    def _get_file_type(self, filename):
        """Determine file type from filename"""
        if filename.lower().endswith('.pdf'):
            return 'pdf'
        elif filename.lower().endswith('.docx'):
            return 'docx'
        elif filename.lower().endswith('.txt'):
            return 'txt'
        else:
            return 'txt'
    
    def _extract_text_from_file(self, file):
        """Extract text content from uploaded file"""
        # This would be implemented with actual file processing logic
        # For now, return the file content as text
        try:
            return file.read().decode('utf-8')
        except UnicodeDecodeError:
            return "Error: Could not decode file content"


class StrategicPlanSerializer(serializers.ModelSerializer):
    """Serializer for StrategicPlan model"""
    target_state_names = serializers.StringRelatedField(
        source='target_states', many=True, read_only=True
    )
    target_subject_names = serializers.StringRelatedField(
        source='target_subjects', many=True, read_only=True
    )
    target_grade_names = serializers.StringRelatedField(
        source='target_grades', many=True, read_only=True
    )
    priority_concept_names = serializers.StringRelatedField(
        source='high_priority_concepts', many=True, read_only=True
    )
    
    class Meta:
        model = StrategicPlan
        fields = [
            'id', 'name', 'description', 'status', 'created_by',
            'target_coverage_percentage', 'target_state_names', 'target_subject_names',
            'target_grade_names', 'mvc_data', 'priority_matrix', 'roi_analysis',
            'timeline_data', 'estimated_completion_months', 'risk_factors',
            'implementation_difficulty', 'estimated_development_cost',
            'projected_market_reach', 'priority_concept_names',
            'created_at', 'updated_at'
        ]


class CacheEntrySerializer(serializers.ModelSerializer):
    """Serializer for CacheEntry model"""
    is_expired = serializers.ReadOnlyField()
    
    class Meta:
        model = CacheEntry
        fields = [
            'cache_key', 'cache_type', 'parameters_hash',
            'computation_time', 'expires_at', 'is_expired', 'created_at'
        ]


# Specialized serializers for API responses

class BellCurveDataSerializer(serializers.Serializer):
    """Serializer for bell curve visualization data"""
    concept_name = serializers.CharField()
    state_coverage = serializers.IntegerField()
    percentage = serializers.FloatField()
    frequency = serializers.IntegerField()
    cumulative_percentage = serializers.FloatField()


class SimilaritySearchResultSerializer(serializers.Serializer):
    """Serializer for similarity search results"""
    standard = StandardSerializer()
    similarity_score = serializers.FloatField()
    match_explanation = serializers.CharField()


class CoverageCalculatorSerializer(serializers.Serializer):
    """Serializer for coverage calculator requests"""
    concepts = serializers.ListField(child=serializers.CharField())
    target_percentage = serializers.FloatField(default=80.0)
    include_states = serializers.ListField(
        child=serializers.CharField(), required=False, default=list
    )
    exclude_states = serializers.ListField(
        child=serializers.CharField(), required=False, default=list
    )


class CoverageCalculatorResponseSerializer(serializers.Serializer):
    """Serializer for coverage calculator responses"""
    total_states_covered = serializers.IntegerField()
    coverage_percentage = serializers.FloatField()
    missing_states = serializers.ListField(child=serializers.CharField())
    coverage_by_concept = serializers.DictField()
    recommendations = serializers.ListField(child=serializers.CharField())