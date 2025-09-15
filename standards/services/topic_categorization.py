"""
LLM-based topic categorization service for creating hierarchical topic-based proxy standards.
"""
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    _HAS_OPENAI_CLIENT = True
except Exception:
    OpenAI = None
    _HAS_OPENAI_CLIENT = False

try:
    import openai
except Exception:
    openai = None

from django.db import transaction
from .base import BaseService
from .token_utils import TokenCounter
from ..models import Standard, TopicBasedProxy, GradeLevel


@dataclass
class TopicHierarchy:
    """Represents the topic hierarchy structure from LLM."""
    topics: List[Dict[str, Any]]
    
    def get_all_paths(self) -> List[Tuple[str, str, str]]:
        """Get all (topic, sub_topic, sub_sub_topic) paths."""
        paths = []
        for topic in self.topics:
            topic_name = topic.get('name', '')
            for sub_topic in topic.get('sub_topics', []):
                sub_topic_name = sub_topic.get('name', '')
                for sub_sub_topic in sub_topic.get('sub_sub_topics', []):
                    paths.append((topic_name, sub_topic_name, sub_sub_topic))
        return paths


@dataclass
class StandardCategorization:
    """Represents categorization of a single standard."""
    standard_id: str
    standard_obj: Optional[Standard]
    topic: str
    sub_topic: str
    sub_sub_topic: str
    is_outlier: bool = False
    
    # Enhanced fields for modern categorization (optional for backward compatibility)
    confidence_score: Optional[float] = None
    reasoning: Optional[str] = None
    key_concepts: Optional[List[str]] = None
    outlier_reason: Optional[str] = None
    outlier_complexity: Optional[str] = None
    suggested_category: Optional[str] = None


class TopicCategorizationService(BaseService):
    """Service for categorizing standards into hierarchical topics using LLM."""
    
    # Model configuration constants
    DEFAULT_MODEL = "gpt-4.1"
    DEFAULT_CHUNK_SIZE = 25  # Standards per LLM call
    SAFETY_MARGIN = 0.2  # 20% safety buffer for token calculations
    # Cap for dynamic chunk sizing to avoid oversized batches
    MAX_DYNAMIC_CHUNK_SIZE = 60
    
    # Enhanced JSON schema for standards categorization using Responses API
    CATEGORIZATION_SCHEMA = {
        "type": "object",
        "properties": {
            "pre_analysis": {
                "type": "object",
                "properties": {
                    "task_overview": {
                        "type": "string",
                        "description": "Brief overview of the categorization task and educational context."
                    },
                    "categorization_strategy": {
                        "type": "string", 
                        "description": "Strategy and approach for mapping standards to the hierarchy."
                    },
                    "identified_patterns": {
                        "type": "string",
                        "description": "Common educational patterns and themes observed in the standards."
                    },
                    "potential_challenges": {
                        "type": "string",
                        "description": "Standards that may be difficult to categorize and why."
                    }
                },
                "required": ["task_overview", "categorization_strategy", "identified_patterns", "potential_challenges"],
                "additionalProperties": False
            },
            "categorizations": {
                "type": "array",
                "description": "Successfully categorized educational standards.",
                "items": {
                    "type": "object",
                    "properties": {
                        "standard_id": {
                            "type": "string",
                            "description": "The unique identifier of the educational standard."
                        },
                        "topic": {
                            "type": "string",
                            "description": "The main topic category from the hierarchy."
                        },
                        "sub_topic": {
                            "type": "string",
                            "description": "The sub-topic category from the hierarchy."
                        }, 
                        "sub_sub_topic": {
                            "type": "string",
                            "description": "The specific sub-sub-topic category from the hierarchy."
                        },
                        "confidence_score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence level in this categorization (0.0 = uncertain, 1.0 = very confident)."
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief educational reasoning for why this standard fits this category."
                        },
                        "key_concepts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Key educational concepts identified in this standard.",
                            "minItems": 1,
                            "maxItems": 5
                        }
                    },
                    "required": ["standard_id", "topic", "sub_topic", "sub_sub_topic", "confidence_score", "reasoning", "key_concepts"],
                    "additionalProperties": False
                }
            },
            "outliers": {
                "type": "array",
                "description": "Standards that don't fit well into the provided hierarchy.",
                "items": {
                    "type": "object",
                    "properties": {
                        "standard_id": {
                            "type": "string",
                            "description": "The unique identifier of the outlier standard."
                        },
                        "reason": {
                            "type": "string",
                            "description": "Detailed explanation of why this standard doesn't fit the hierarchy."
                        },
                        "suggested_category": {
                            "type": "string",
                            "description": "Suggested new category or hierarchy extension if applicable."
                        },
                        "complexity_level": {
                            "type": "string",
                            "enum": ["too_specific", "too_broad", "cross_cutting", "unique_domain", "unclear"],
                            "description": "Classification of why this standard is an outlier."
                        },
                        "confidence_score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence that this truly doesn't fit (higher = more certain it's an outlier)."
                        }
                    },
                    "required": ["standard_id", "reason", "suggested_category", "complexity_level", "confidence_score"],
                    "additionalProperties": False
                }
            },
            "summary": {
                "type": "object", 
                "properties": {
                    "total_categorized": {
                        "type": "integer",
                        "description": "Number of standards successfully categorized."
                    },
                    "total_outliers": {
                        "type": "integer", 
                        "description": "Number of standards marked as outliers."
                    },
                    "avg_confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Average confidence score across all categorizations."
                    },
                    "dominant_topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Most frequently used topic categories in this batch.",
                        "maxItems": 5
                    }
                },
                "required": ["total_categorized", "total_outliers", "avg_confidence", "dominant_topics"],
                "additionalProperties": False
            }
        },
        "required": ["pre_analysis", "categorizations", "outliers", "summary"],
        "additionalProperties": False
    }
    
    def __init__(self):
        super().__init__()
        self.client = None
        self.chunk_size = self.DEFAULT_CHUNK_SIZE
        self.max_retries = 3
        self.token_counter = TokenCounter(self.DEFAULT_MODEL)
        
        # Initialize OpenAI client
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                if _HAS_OPENAI_CLIENT and OpenAI is not None:
                    self.client = OpenAI(api_key=api_key)
                elif openai is not None:
                    openai.api_key = api_key
                    self.client = openai
                logger.info("TopicCategorizationService OpenAI client initialized: %s", bool(self.client))
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")
            self.client = None
    
    def load_standards(self, grade_levels: List[int] = None, subject_area_id: int = None) -> List[Standard]:
        """Load standards with optional filtering, ensuring no data loss."""
        standards_query = Standard.objects.select_related('state', 'subject_area').prefetch_related('grade_levels')
        
        # Apply filters carefully to avoid data loss
        original_count = standards_query.count()
        
        # Filter by subject area if specified (apply first to get baseline)
        if subject_area_id:
            standards_query = standards_query.filter(subject_area_id=subject_area_id)
            after_subject_filter = standards_query.count()
            logger.info(f"📊 After subject area filter: {after_subject_filter:,} standards (from {original_count:,})")
        
        # Filter by grade levels if specified (be careful with distinct())
        if grade_levels:
            # Use proper filtering that doesn't eliminate multi-grade standards
            before_grade_filter = standards_query.count()
            standards_query = standards_query.filter(grade_levels__grade_numeric__in=grade_levels)
            
            # Count before distinct() to see how many matches we have
            matches_before_distinct = standards_query.count()
            
            # Only use distinct() if we actually have duplicates
            if matches_before_distinct > before_grade_filter:
                standards_query = standards_query.distinct()
                after_distinct = standards_query.count()
                logger.info(f"📊 Grade filter: {before_grade_filter:,} → {matches_before_distinct:,} matches → {after_distinct:,} after distinct()")
            else:
                logger.info(f"📊 Grade filter: {before_grade_filter:,} → {matches_before_distinct:,} standards (no duplicates)")
        
        standards = list(standards_query)
        
        if not standards:
            # Provide detailed error information
            error_msg = "No standards found for the specified criteria: "
            if subject_area_id:
                error_msg += f"subject_area_id={subject_area_id} "
            if grade_levels:
                error_msg += f"grade_levels={grade_levels} "
            
            # Check if it's a filter issue or no data
            total_standards = Standard.objects.count()
            error_msg += f"(Total standards in database: {total_standards:,})"
            
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate loaded standards
        self._validate_loaded_standards(standards, grade_levels, subject_area_id)
        
        logger.info(f"✅ Successfully loaded {len(standards):,} standards for topic categorization")
        return standards
    
    def _validate_loaded_standards(self, standards: List[Standard], 
                                 grade_levels: List[int] = None, 
                                 subject_area_id: int = None) -> None:
        """
        Validate that loaded standards match the expected criteria.
        
        Args:
            standards: Loaded standards to validate
            grade_levels: Expected grade levels filter
            subject_area_id: Expected subject area filter
        """
        if not standards:
            return
        
        logger.info(f"🔍 Validating {len(standards):,} loaded standards...")
        
        # Check subject area consistency
        if subject_area_id:
            subject_areas = {std.subject_area_id for std in standards if std.subject_area}
            if len(subject_areas) > 1:
                logger.warning(f"⚠️ Multiple subject areas found: {subject_areas} (expected: {subject_area_id})")
            elif subject_area_id not in subject_areas:
                logger.error(f"❌ No standards found with subject_area_id={subject_area_id}, found: {subject_areas}")
        
        # Check grade level distribution
        if grade_levels:
            found_grades = set()
            for std in standards:
                std_grades = {gl.grade_numeric for gl in std.grade_levels.all() if hasattr(gl, 'grade_numeric')}
                found_grades.update(std_grades)
            
            expected_grades = set(grade_levels)
            missing_grades = expected_grades - found_grades
            extra_grades = found_grades - expected_grades
            
            if missing_grades:
                logger.warning(f"⚠️ No standards found for grade levels: {sorted(missing_grades)}")
            if extra_grades:
                logger.info(f"ℹ️ Standards found for additional grade levels: {sorted(extra_grades)}")
        
        # Check for potential data quality issues
        standards_without_grades = [std for std in standards if not std.grade_levels.exists()]
        if standards_without_grades:
            logger.warning(f"⚠️ {len(standards_without_grades):,} standards have no grade level assignments")
        
        standards_without_subject = [std for std in standards if not std.subject_area]
        if standards_without_subject:
            logger.warning(f"⚠️ {len(standards_without_subject):,} standards have no subject area assignment")
        
        # Log distribution summary
        subject_distribution = {}
        for std in standards:
            if std.subject_area:
                subject = std.subject_area.name
                subject_distribution[subject] = subject_distribution.get(subject, 0) + 1
        
        logger.info(f"📊 Subject distribution: {dict(sorted(subject_distribution.items()))}")
        
        grade_distribution = {}
        for std in standards:
            for gl in std.grade_levels.all():
                if hasattr(gl, 'grade_numeric'):
                    grade = f"Grade {gl.grade_numeric}"
                    grade_distribution[grade] = grade_distribution.get(grade, 0) + 1
        
        logger.info(f"📊 Grade distribution: {dict(sorted(grade_distribution.items()))}")
        
        logger.info(f"✅ Standards validation completed")
    
    def generate_topic_hierarchy(self, standards: List[Standard], subject_area_name: str = None) -> TopicHierarchy:
        """Generate topic hierarchy based on educational standards using LLM."""
        if not self.client:
            raise ValueError("OpenAI client not available")
        
        # Convert Standard objects to dictionaries for token calculation
        standards_data = []
        for standard in standards:
            std_dict = {
                'code': standard.code,
                'title': standard.title,
                'description': standard.description or '',
            }
            standards_data.append(std_dict)
        
        # Calculate optimal number of standards to include using aggressive token calculation
        subject_hint = f" in {subject_area_name}" if subject_area_name else ""
        
        # Prepare prompt templates for token calculation
        system_prompt = (
            f"You are an education curriculum expert. Based on the provided educational standards{subject_hint}, "
            "create a hierarchical topic structure with 3 levels. The hierarchy should be a list of topics, sub-topics, and sub-sub-topics. :\n\n"
            "1. **Topics** (5-8 main areas) - Broad educational domains\n"
            "2. **Sub-topics** (3-6 per topic) - More specific subject areas within each topic\n"
            "3. **Sub-sub-topics** (4-8 per sub-topic) - Specific learning objectives/skills\n\n"
            f"Create a comprehensive hierarchy that could categorize educational standards{subject_hint}. "
            "Focus on creating logical groupings that educators would recognize.\n\n"
            "RULES:\n"
            "1. You will always create a minimum of 5 core topics\n"
            "2. You will always create a minimum of 3 sub-topics per topic\n"
            "3. You will always create a minimum of 4 sub-sub-topics per sub-topic\n"
        )
        
        user_prompt_template = "Standards:\n"  # The actual standards will be added dynamically
        
        # Calculate optimal number of standards to include with aggressive utilization
        optimal_standards_count = self.token_counter.calculate_optimal_standards_for_hierarchy(
            standards_data,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template
        )
        
        # For GPT-4.1, aggressively try to use ALL standards if they fit
        if self.token_counter.model_name in ['gpt-4.1', 'gpt-4.1-mini']:
            if optimal_standards_count >= len(standards):
                sample_standards = standards  # Use all standards
                logger.info(f"🚀 Using ALL {len(standards):,} standards for hierarchy generation with {self.token_counter.model_name}")
            else:
                # Use the calculated optimal count
                sample_standards = standards[:optimal_standards_count]
                logger.info(f"📊 Using {len(sample_standards):,} standards out of {len(standards):,} for hierarchy generation")
        else:
            # For other models, use calculated optimal
            sample_standards = standards[:optimal_standards_count]
            logger.info(f"📊 Using {len(sample_standards):,} standards out of {len(standards):,} for hierarchy generation (optimal_count: {optimal_standards_count:,})")
        
        # Build context from standards
        standards_context = []
        for standard in sample_standards:
            context = f"- {standard.code}: {standard.title}"
            if standard.description:
                context += f" - {standard.description[:150]}..."
            standards_context.append(context)
        
        subject_hint = f" in {subject_area_name}" if subject_area_name else ""
        
        # Prefer the modern Responses API with JSON schema for structured output
        if hasattr(self.client, 'responses'):
            try:
                schema = {
                    "type": "object",
                    "properties": {
                        "pre-analysis": {
                            "type": "object",
                            "properties": {
                                "task": {
                                    "type": "string",
                                    "description": "A brief description of the task at hand."
                                },
                                "rules": {
                                    "type": "string",
                                    "description": "A brief description of the rules that will be used to categorize the standards."
                                },
                                "topic_hierarchy_brainstorming": {
                                    "type": "string",
                                    "description": "Create a list of up to 20 topic hierarchy ideas for the task at hand."
                                }
                            },
                            "required": ["task", "rules", "topic_hierarchy_brainstorming"],
                            "additionalProperties": False
                        },
                        "topics": {
                            "type": "array",
                            "description": "Top-level topics representing broad educational domains.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "Name of the topic area."},
                                    "description": {"type": "string", "description": "A detailed description of the proxy standard for this topic, written like a standard."},
                                    "sub_topics": {
                                        "type": "array",
                                        "description": "List of sub-topics in this topic.",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string", "description": "Sub-topic name."},
                                                "description": {"type": "string", "description": "A detailed description of the proxy standard for this sub-topic, written like a standard."},
                                                "sub_sub_topics": {
                                                    "type": "array",
                                                    "description": "Learning objectives or specific skills in this sub-topic, written like a standard.",
                                                    "items": {"type": "string", "description": "Name of the sub-sub-topic."},
                                                    "minItems": 4,
                                                    "maxItems": 8
                                                }
                                            },
                                            "required": ["name", "description", "sub_sub_topics"],
                                            "additionalProperties": False
                                        },
                                        "minItems": 3,
                                        "maxItems": 6
                                    }
                                },
                                "required": ["name", "description", "sub_topics"],
                                "additionalProperties": False
                            },
                            "minItems": 5,
                            "maxItems": 8
                        }
                    },
                    "required": ["pre-analysis", "topics"],
                    "additionalProperties": False
                }

                system_text = system_prompt
                user_text = "Standards:\n" + "\n".join(standards_context)

                resp = self.client.responses.create(
                    model="gpt-4.1",
                    input=[
                        {"role": "system", "content": [{"type": "input_text", "text": system_text}]},
                        {"role": "user", "content": [{"type": "input_text", "text": user_text}]}
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "educational_topic_hierarchy",
                            "strict": True,
                            "schema": schema
                        }
                    },
                    temperature=0.3,
                    max_output_tokens=15614,
                    top_p=1,
                    store=True
                )

                #print the prompt and response
                print(f"Prompt: {system_text}\n\n{user_text}")
                print(f"Response: {resp.output_text}")

                content = getattr(resp, 'output_text', None)
                if not content:
                    try:
                        content = resp.output[0].content[0].text  # type: ignore[attr-defined]
                    except Exception:
                        content = None
                if not content:
                    raise ValueError("Empty response from Responses API")

                hierarchy_data = json.loads(content)
                if 'topics' not in hierarchy_data:
                    raise ValueError("Invalid hierarchy structure: missing 'topics'")

                logger.info(f"Generated hierarchy with {len(hierarchy_data['topics'])} topics")
                return TopicHierarchy(topics=hierarchy_data['topics'])

            except Exception as e:
                logger.warning(f"Responses API failed for topic hierarchy generation, falling back: {e}")
        
        # Fallback to existing prompt + chat completions flow
        prompt = f"""You are an education curriculum expert. Based on the following sample of educational standards{subject_hint}, create a hierarchical topic structure with 3 levels:\n\n
1. **Topics** (5-8 main areas) - Broad educational domains
2. **Sub-topics** (3-6 per topic) - More specific subject areas within each topic
3. **Sub-sub-topics** (4-8 per sub-topic) - Specific learning objectives/skills

Sample Standards:
{chr(10).join(standards_context)}

Create a comprehensive hierarchy that could categorize educational standards{subject_hint}. Focus on creating logical groupings that educators would recognize.

Return your response as JSON in exactly this format:
{{
  "topics": [
    {{
      "name": "Topic Name",
      "description": "Brief description of this topic area",
      "sub_topics": [
        {{
          "name": "Sub-topic Name", 
          "description": "Brief description",
          "sub_sub_topics": [
            "Sub-sub-topic 1",
            "Sub-sub-topic 2",
            "Sub-sub-topic 3"
          ]
        }}
      ]
    }}
  ]
}}"""

        try:
            response = self._call_openai_api(prompt, "topic_hierarchy_generation")
            hierarchy_data = json.loads(response)
            
            # Validate structure
            if 'topics' not in hierarchy_data:
                raise ValueError("Invalid hierarchy structure: missing 'topics'")
            
            logger.info(f"Generated hierarchy with {len(hierarchy_data['topics'])} topics")
            return TopicHierarchy(topics=hierarchy_data['topics'])
            
        except Exception as e:
            logger.error(f"Failed to generate topic hierarchy: {e}")
            raise
    
    def categorize_standards_chunk(self, 
                                 standards_chunk: List[Standard], 
                                 hierarchy: TopicHierarchy) -> List[StandardCategorization]:
        """Categorize a chunk of standards using the topic hierarchy with modern AI techniques."""
        if not self.client:
            raise ValueError("OpenAI client not available")
        
        logger.info(f"🔍 Starting enhanced categorization of {len(standards_chunk)} standards using {self.DEFAULT_MODEL}")
        
        # Prepare standards for LLM with enhanced formatting
        standards_data = self._prepare_standards_for_categorization(standards_chunk)
        
        # Prepare hierarchy paths for the prompt
        hierarchy_paths = self._format_hierarchy_paths(hierarchy)
        
        # Create enhanced system and user prompts
        system_prompt = self._create_categorization_system_prompt()
        user_prompt = self._create_categorization_user_prompt(standards_data, hierarchy_paths)
        
        # Try modern Responses API first, then fallback to basic completion
        try:
            categorization_data = self._call_enhanced_categorization_api(system_prompt, user_prompt)
            results = self._process_enhanced_categorization_response(categorization_data, standards_chunk)
            
            logger.info(f"Enhanced categorization data: {categorization_data}")
            logger.info(f"Enhanced categorization results: {results}")
            logger.info(f"✅ Enhanced categorization completed: {len(results)} standards processed")
            return results
            
        except Exception as e:
            logger.warning(f"Enhanced categorization failed, falling back to basic method: {e}")
            return self._fallback_basic_categorization(standards_chunk, hierarchy)
    
    def _prepare_standards_for_categorization(self, standards_chunk: List[Standard]) -> List[Dict[str, str]]:
        """Prepare standards data for enhanced LLM processing."""
        standards_data = []
        for standard in standards_chunk:
            # Smart truncation that preserves key educational content
            description = standard.description or ""
            if len(description) > 300:
                # Preserve first 200 chars and last 100 chars to keep context
                description = description[:200] + "..." + description[-100:]
            
            std_text = f"{standard.code}: {standard.title}"
            if description:
                std_text += f" - {description}"
                
            standards_data.append({
                "id": standard.code,
                "text": std_text,
                "grade_context": self._extract_grade_context(standard),
                "subject_context": standard.subject_area.name if standard.subject_area else "Unknown"
            })
        
        return standards_data
    
    def _extract_grade_context(self, standard: Standard) -> str:
        """Extract grade level context for better categorization."""
        if standard.grade_levels.exists():
            grades = [gl.grade for gl in standard.grade_levels.all()]
            return f"Grades: {', '.join(grades)}"
        return "Grade level not specified"
    
    def _format_hierarchy_paths(self, hierarchy: TopicHierarchy) -> List[str]:
        """Format hierarchy paths for enhanced prompt."""
        hierarchy_paths = []
        for topic in hierarchy.topics:
            topic_name = topic.get('name', '')
            topic_desc = topic.get('description', '')
            
            for sub_topic in topic.get('sub_topics', []):
                sub_topic_name = sub_topic.get('name', '')
                sub_topic_desc = sub_topic.get('description', '')
                
                for sub_sub_topic in sub_topic.get('sub_sub_topics', []):
                    path = f"  {topic_name} > {sub_topic_name} > {sub_sub_topic}"
                    if topic_desc or sub_topic_desc:
                        path += f" (Context: {topic_desc or sub_topic_desc})"
                    hierarchy_paths.append(path)
        
        return hierarchy_paths
    
    def _create_categorization_system_prompt(self) -> str:
        """Create enhanced system prompt for categorization."""
        return """You are an expert educational standards analyst with deep knowledge of K-12 curriculum design and educational taxonomy systems. Your expertise spans:

- Educational standards alignment and classification across all subject areas
- Cognitive complexity analysis using Bloom's Taxonomy and Depth of Knowledge (DOK) levels  
- Cross-state standards correlation and educational frameworks
- Learning progression identification and curriculum scope/sequence development
- Pattern recognition in educational content and pedagogical structures

CORE COMPETENCIES:
1. **Standards Analysis**: Identify key educational concepts, learning objectives, and cognitive demands in each standard
2. **Hierarchical Thinking**: Understand nested educational structures from broad domains to specific measurable skills
3. **Pattern Recognition**: Detect thematic connections and conceptual relationships across diverse educational standards
4. **Quality Assurance**: Maintain high accuracy in categorization while minimizing inappropriate classifications

CATEGORIZATION PRINCIPLES:
- Prioritize pedagogical logic and educational coherence over superficial keyword matching
- Consider grade-level appropriateness, developmental sequences, and learning progressions
- Recognize interdisciplinary connections and cross-cutting educational concepts
- Maintain consistency across similar standards from different states and educational systems
- Flag ambiguous or unclear cases rather than forcing inappropriate categorizations

Your task is to perform systematic, educationally-grounded categorization with detailed pre-analysis and high-confidence scoring."""
    
    def _create_categorization_user_prompt(self, standards_data: List[Dict[str, str]], hierarchy_paths: List[str]) -> str:
        """Create enhanced user prompt with pre-analysis requirements."""
        return f"""# EDUCATIONAL STANDARDS CATEGORIZATION TASK

## CONTEXT
You are categorizing {len(standards_data)} educational standards into a comprehensive topic hierarchy. Each standard represents specific learning expectations that students should achieve.

## PROVIDED TOPIC HIERARCHY
The following 3-level hierarchy represents the organizational structure for categorization:

**AVAILABLE CATEGORIZATION PATHS:**
{chr(10).join(hierarchy_paths)}

## STANDARDS TO CATEGORIZE
{chr(10).join(f"{i+1}. {std['text']} [{std['grade_context']}, Subject: {std['subject_context']}]" for i, std in enumerate(standards_data))}

## REQUIRED PRE-ANALYSIS
Before categorizing, you must analyze:

1. **Task Overview**: Summarize this categorization challenge and its educational context
2. **Categorization Strategy**: Explain your systematic approach for mapping standards to hierarchy
3. **Identified Patterns**: Describe common educational themes and patterns you observe in these standards
4. **Potential Challenges**: Identify standards that may be difficult to categorize and explain why

## CATEGORIZATION REQUIREMENTS

For each standard:
1. **Identify Core Educational Concepts**: What are the 1-5 key learning concepts?
2. **Assess Cognitive Complexity**: What type of thinking or skills are required?
3. **Find Best Hierarchical Fit**: Which sub-sub-topic most closely aligns with the standard's learning objective?
4. **Confidence Assessment**: Rate your certainty (0.0-1.0) where:
   - 0.8-1.0: Clear conceptual alignment with strong educational rationale
   - 0.6-0.8: Good fit with minor conceptual gaps
   - 0.4-0.6: Reasonable fit but uncertain placement
   - 0.0-0.4: Poor fit, likely outlier candidate

## OUTLIER IDENTIFICATION
Mark standards as outliers if they:
- State specific history or information that is extremely specific and not relevant to the task or topic hierarchy. Example, California gold rush, might be relevant to moving West. However, first governor of California, might be a bit too specific.
- Address educational concepts not adequately covered in the provided hierarchy
- Span multiple hierarchy branches with equal relevance (cross-cutting standards)
- Are significantly more specific or broader than the hierarchy's granularity level
- Contain unclear, ambiguous, or incomplete educational language
- Represent unique domains or interdisciplinary content not well-represented in hierarchy

## QUALITY EXPECTATIONS
- Target >85% successful categorization rate
- Maintain average confidence score >0.7 across all categorizations
- Provide clear educational reasoning for each categorization decision
- Identify potential hierarchy improvements through systematic outlier analysis

Return your response following the provided JSON schema with complete pre-analysis and detailed categorizations."""
    
    def _call_enhanced_categorization_api(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call the enhanced Responses API with comprehensive error handling."""
        # Prefer the modern Responses API with JSON schema for structured output
        if hasattr(self.client, 'responses'):
            try:
                logger.info("🚀 Using Responses API for enhanced categorization")
                
                resp = self.client.responses.create(
                    model=self.DEFAULT_MODEL,
                    input=[
                        {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                        {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "educational_standards_categorization",
                            "strict": True,
                            "schema": self.CATEGORIZATION_SCHEMA
                        }
                    },
                    temperature=0.1,  # Lower temperature for consistency
                    max_output_tokens=16000,  # Sufficient for large batches
                    top_p=1,
                    store=True
                )
                
                content = getattr(resp, 'output_text', None)
                if not content:
                    try:
                        content = resp.output[0].content[0].text  # type: ignore[attr-defined]
                    except Exception:
                        content = None
                if not content:
                    raise ValueError("Empty response from Responses API")
                
                logger.info("✅ Responses API call successful")
                return json.loads(content)
                
            except Exception as e:
                logger.warning(f"Responses API failed: {e}")
                raise
        else:
            raise ValueError("Responses API not available on this client")
    
    def _process_enhanced_categorization_response(self, categorization_data: Dict[str, Any], 
                                                standards_chunk: List[Standard]) -> List[StandardCategorization]:
        """Process the enhanced categorization response into StandardCategorization objects."""
        results = []
        standard_lookup = {std.code: std for std in standards_chunk}
        
        # Validate and process pre-analysis
        pre_analysis_quality = self._validate_and_process_pre_analysis(
            categorization_data.get('pre_analysis', {}), standards_chunk
        )
        
        # Adaptive processing based on pre-analysis quality
        processing_confidence = self._calculate_processing_confidence(pre_analysis_quality, categorization_data)
        
        # Process enhanced categorizations with quality checks
        valid_categorizations = 0
        confidence_scores = []
        
        for cat in categorization_data.get('categorizations', []):
            std_id = cat.get('standard_id')
            if std_id in standard_lookup:
                # Validate categorization quality
                cat_quality = self._validate_categorization_quality(cat, standards_chunk)
                
                if cat_quality['is_valid']:
                    confidence_score = cat.get('confidence_score', 0.0)
                    confidence_scores.append(confidence_score)
                    
                    results.append(StandardCategorization(
                        standard_id=std_id,
                        standard_obj=standard_lookup[std_id],
                        topic=cat.get('topic', ''),
                        sub_topic=cat.get('sub_topic', ''),
                        sub_sub_topic=cat.get('sub_sub_topic', ''),
                        is_outlier=False,
                        # Enhanced fields
                        confidence_score=confidence_score,
                        reasoning=cat.get('reasoning'),
                        key_concepts=cat.get('key_concepts', [])
                    ))
                    valid_categorizations += 1
                else:
                    logger.warning(f"⚠️ Low-quality categorization for {std_id}: {cat_quality['issues']}")
                    # Demote to outlier if quality is too low
                    results.append(StandardCategorization(
                        standard_id=std_id,
                        standard_obj=standard_lookup[std_id],
                        topic="Outliers",
                        sub_topic="Quality Issues",
                        sub_sub_topic=f"Low quality categorization: {cat_quality['primary_issue']}",
                        is_outlier=True,
                        confidence_score=0.2,  # Low confidence for quality issues
                        outlier_reason=f"Quality check failed: {cat_quality['primary_issue']}",
                        outlier_complexity="unclear"
                    ))
        
        # Process enhanced outliers with validation
        valid_outliers = 0
        for outlier in categorization_data.get('outliers', []):
            std_id = outlier.get('standard_id')
            if std_id in standard_lookup:
                # Validate outlier reasoning
                outlier_quality = self._validate_outlier_quality(outlier, standards_chunk)
                
                if outlier_quality['is_valid']:
                    results.append(StandardCategorization(
                        standard_id=std_id,
                        standard_obj=standard_lookup[std_id],
                        topic="Outliers",
                        sub_topic="Uncategorized",
                        sub_sub_topic=outlier.get('reason', 'Does not fit main taxonomy'),
                        is_outlier=True,
                        # Enhanced outlier fields
                        confidence_score=outlier.get('confidence_score'),
                        outlier_reason=outlier.get('reason'),
                        outlier_complexity=outlier.get('complexity_level'),
                        suggested_category=outlier.get('suggested_category')
                    ))
                    valid_outliers += 1
                else:
                    logger.warning(f"⚠️ Poor outlier reasoning for {std_id}, using generic classification")
                    results.append(StandardCategorization(
                        standard_id=std_id,
                        standard_obj=standard_lookup[std_id],
                        topic="Outliers",
                        sub_topic="Uncategorized",
                        sub_sub_topic="Generic outlier - insufficient reasoning",
                        is_outlier=True,
                        confidence_score=0.3,
                        outlier_reason="Generic outlier classification",
                        outlier_complexity="unclear"
                    ))
        
        # Enhanced summary with quality metrics
        summary = categorization_data.get('summary', {})
        total_categorized = summary.get('total_categorized', 0)
        total_outliers = summary.get('total_outliers', 0)
        reported_avg_confidence = summary.get('avg_confidence', 0.0)
        
        # Calculate actual confidence from processed results
        actual_avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Log comprehensive quality assessment
        logger.info(f"📈 Enhanced Processing Summary:")
        logger.info(f"📈   Pre-analysis quality: {pre_analysis_quality['score']:.2f}/1.0 ({pre_analysis_quality['quality_level']})")
        logger.info(f"📈   Processing confidence: {processing_confidence:.2f}/1.0")
        logger.info(f"📈   Valid categorizations: {valid_categorizations}/{total_categorized} ({100*valid_categorizations/max(total_categorized,1):.1f}%)")
        logger.info(f"📈   Valid outliers: {valid_outliers}/{total_outliers}")
        logger.info(f"📈   Reported confidence: {reported_avg_confidence:.2f}, Actual confidence: {actual_avg_confidence:.2f}")
        
        # Store quality metrics for future improvements
        self._store_quality_metrics({
            'pre_analysis_quality': pre_analysis_quality,
            'processing_confidence': processing_confidence,
            'categorization_success_rate': valid_categorizations / max(total_categorized, 1),
            'outlier_reasoning_quality': valid_outliers / max(total_outliers, 1),
            'confidence_accuracy': abs(reported_avg_confidence - actual_avg_confidence)
        })
        
        return results
    
    def _validate_and_process_pre_analysis(self, 
                                         pre_analysis: Dict[str, Any], 
                                         standards_chunk: Optional[List[Standard]] = None) -> Dict[str, Any]:
        """Validate the quality of pre-analysis and extract insights.
        
        Args:
            pre_analysis: Pre-analysis section from LLM response
            standards_chunk: Standards being processed
            
        Returns:
            Dictionary with quality assessment and insights
        """
        quality_assessment = {
            'score': 0.0,
            'quality_level': 'poor',
            'insights': [],
            'issues': [],
            'has_task_overview': False,
            'has_strategy': False,
            'has_patterns': False,
            'has_challenges': False
        }
        
        if not pre_analysis:
            quality_assessment['issues'].append('Missing pre-analysis section')
            logger.warning("⚠️ Pre-analysis validation: Missing pre-analysis section")
            return quality_assessment
        
        # Check for required components
        task_overview = pre_analysis.get('task_overview', '').strip()
        strategy = pre_analysis.get('categorization_strategy', '').strip()
        patterns = pre_analysis.get('identified_patterns', '').strip()
        challenges = pre_analysis.get('potential_challenges', '').strip()
        
        # Validate task overview
        if task_overview and len(task_overview) > 20:
            quality_assessment['has_task_overview'] = True
            quality_assessment['score'] += 0.2
            if 'standard' in task_overview.lower() and 'categor' in task_overview.lower():
                quality_assessment['score'] += 0.05
        else:
            quality_assessment['issues'].append('Insufficient task overview')
        
        # Validate strategy
        if strategy and len(strategy) > 30:
            quality_assessment['has_strategy'] = True
            quality_assessment['score'] += 0.25
            # Look for educational keywords
            edu_keywords = ['educational', 'learning', 'cognitive', 'bloom', 'complexity', 'grade', 'domain']
            if any(keyword in strategy.lower() for keyword in edu_keywords):
                quality_assessment['score'] += 0.1
                quality_assessment['insights'].append('Strategy shows educational awareness')
        else:
            quality_assessment['issues'].append('Insufficient categorization strategy')
        
        # Validate pattern identification
        if patterns and len(patterns) > 25:
            quality_assessment['has_patterns'] = True
            quality_assessment['score'] += 0.25
            # Check for specific pattern descriptions
            if any(word in patterns.lower() for word in ['pattern', 'theme', 'common', 'similar']):
                quality_assessment['score'] += 0.05
                quality_assessment['insights'].append('Good pattern recognition')
        else:
            quality_assessment['issues'].append('Insufficient pattern identification')
        
        # Validate challenge identification
        if challenges and len(challenges) > 20:
            quality_assessment['has_challenges'] = True
            quality_assessment['score'] += 0.15
            # Look for specific challenge types
            challenge_types = ['complex', 'ambiguous', 'cross-cutting', 'specific', 'broad']
            if any(challenge_type in challenges.lower() for challenge_type in challenge_types):
                quality_assessment['score'] += 0.05
                quality_assessment['insights'].append('Thoughtful challenge identification')
        else:
            quality_assessment['issues'].append('Insufficient challenge identification')
        
        # Determine quality level
        if quality_assessment['score'] >= 0.8:
            quality_assessment['quality_level'] = 'excellent'
        elif quality_assessment['score'] >= 0.6:
            quality_assessment['quality_level'] = 'good'
        elif quality_assessment['score'] >= 0.4:
            quality_assessment['quality_level'] = 'acceptable'
        else:
            quality_assessment['quality_level'] = 'poor'
        
        logger.info(f"🔍 Pre-analysis validation: {quality_assessment['quality_level']} (score: {quality_assessment['score']:.2f})")
        if quality_assessment['insights']:
            logger.info(f"🔍 Pre-analysis insights: {'; '.join(quality_assessment['insights'])}")
        if quality_assessment['issues']:
            logger.warning(f"⚠️ Pre-analysis issues: {'; '.join(quality_assessment['issues'])}")
        
        return quality_assessment
    
    def _calculate_processing_confidence(self, 
                                       pre_analysis_quality: Dict[str, Any], 
                                       categorization_data: Dict[str, Any]) -> float:
        """Calculate overall processing confidence based on multiple factors.
        
        Args:
            pre_analysis_quality: Quality assessment of pre-analysis
            categorization_data: Full categorization response
            
        Returns:
            Processing confidence score (0.0-1.0)
        """
        confidence = 0.0
        
        # Base confidence from pre-analysis quality (40% weight)
        confidence += pre_analysis_quality['score'] * 0.4
        
        # Confidence from summary statistics (30% weight)
        summary = categorization_data.get('summary', {})
        reported_confidence = summary.get('avg_confidence', 0.0)
        if reported_confidence > 0.7:
            confidence += 0.3
        elif reported_confidence > 0.5:
            confidence += 0.2
        elif reported_confidence > 0.3:
            confidence += 0.1
        
        # Confidence from response completeness (20% weight)
        total_expected = len(categorization_data.get('categorizations', [])) + len(categorization_data.get('outliers', []))
        if total_expected > 0:
            confidence += 0.2
        
        # Confidence from consistency checks (10% weight)
        summary_totals = summary.get('total_categorized', 0) + summary.get('total_outliers', 0)
        actual_totals = len(categorization_data.get('categorizations', [])) + len(categorization_data.get('outliers', []))
        
        if summary_totals == actual_totals:
            confidence += 0.1
        elif abs(summary_totals - actual_totals) <= 1:  # Allow small discrepancies
            confidence += 0.05
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def _validate_categorization_quality(self, 
                                       categorization: Dict[str, Any], 
                                       standards_chunk: Optional[List[Standard]] = None) -> Dict[str, Any]:
        """Validate the quality of a single categorization.
        
        Args:
            categorization: Single categorization to validate
            standards_chunk: Context standards for validation
            
        Returns:
            Quality assessment dictionary
        """
        quality = {
            'is_valid': True,
            'score': 1.0,
            'issues': [],
            'primary_issue': None
        }
        
        # Check required fields
        required_fields = ['standard_id', 'topic', 'sub_topic', 'sub_sub_topic', 'confidence_score', 'reasoning']
        for field in required_fields:
            if not categorization.get(field):
                quality['issues'].append(f'Missing {field}')
                quality['score'] -= 0.2
        
        # Validate confidence score
        confidence = categorization.get('confidence_score')
        if confidence is not None:
            if not (0.0 <= confidence <= 1.0):
                quality['issues'].append('Invalid confidence score range')
                quality['score'] -= 0.3
            elif confidence < 0.3:
                quality['issues'].append('Very low confidence score')
                quality['score'] -= 0.1
        
        # Validate reasoning quality
        reasoning = categorization.get('reasoning', '').strip()
        if len(reasoning) < 10:
            quality['issues'].append('Insufficient reasoning')
            quality['score'] -= 0.2
        elif len(reasoning) < 30:
            quality['issues'].append('Brief reasoning')
            quality['score'] -= 0.1
        
        # Validate key concepts
        key_concepts = categorization.get('key_concepts', [])
        if not key_concepts:
            quality['issues'].append('Missing key concepts')
            quality['score'] -= 0.15
        elif len(key_concepts) > 5:
            quality['issues'].append('Too many key concepts')
            quality['score'] -= 0.05
        
        # Determine validity
        if quality['score'] < 0.5:
            quality['is_valid'] = False
            quality['primary_issue'] = quality['issues'][0] if quality['issues'] else 'Unknown issue'
        
        return quality
    
    def _validate_outlier_quality(self, 
                                outlier: Dict[str, Any], 
                                standards_chunk: Optional[List[Standard]] = None) -> Dict[str, Any]:
        """Validate the quality of outlier reasoning.
        
        Args:
            outlier: Outlier entry to validate
            standards_chunk: Context standards for validation
            
        Returns:
            Quality assessment dictionary
        """
        quality = {
            'is_valid': True,
            'score': 1.0,
            'issues': []
        }
        
        # Check required fields
        if not outlier.get('standard_id'):
            quality['issues'].append('Missing standard_id')
            quality['score'] -= 0.3
        
        # Validate reason
        reason = outlier.get('reason', '').strip()
        if len(reason) < 15:
            quality['issues'].append('Insufficient outlier reasoning')
            quality['score'] -= 0.4
        elif 'does not fit' in reason.lower() and len(reason) < 30:
            quality['issues'].append('Generic outlier reasoning')
            quality['score'] -= 0.2
        
        # Validate complexity level
        complexity = outlier.get('complexity_level')
        valid_complexities = ['too_specific', 'too_broad', 'cross_cutting', 'unique_domain', 'unclear']
        if complexity not in valid_complexities:
            quality['issues'].append('Invalid complexity level')
            quality['score'] -= 0.2
        
        # Validate confidence
        confidence = outlier.get('confidence_score')
        if confidence is not None:
            if not (0.0 <= confidence <= 1.0):
                quality['issues'].append('Invalid confidence range')
                quality['score'] -= 0.2
        
        # Determine validity
        if quality['score'] < 0.6:  # Higher threshold for outliers
            quality['is_valid'] = False
        
        return quality
    
    def _store_quality_metrics(self, metrics: Dict[str, Any]) -> None:
        """Store quality metrics for future analysis and improvement.
        
        Args:
            metrics: Quality metrics to store
        """
        # For now, just log the metrics. In the future, this could store to database
        # or a metrics collection service for analysis and model improvement
        
        logger.info("📉 Quality Metrics Summary:")
        for metric_name, value in metrics.items():
            if isinstance(value, dict):
                logger.info(f"📉   {metric_name}: {value.get('score', 'N/A')} ({value.get('quality_level', 'N/A')})")
            elif isinstance(value, float):
                logger.info(f"📉   {metric_name}: {value:.3f}")
            else:
                logger.info(f"📉   {metric_name}: {value}")
        
        # Calculate overall quality score
        overall_score = (
            metrics.get('pre_analysis_quality', {}).get('score', 0.0) * 0.3 +
            metrics.get('processing_confidence', 0.0) * 0.3 +
            metrics.get('categorization_success_rate', 0.0) * 0.2 +
            metrics.get('outlier_reasoning_quality', 0.0) * 0.1 +
            max(0, 1.0 - metrics.get('confidence_accuracy', 1.0)) * 0.1  # Invert accuracy for scoring
        )
        
        logger.info(f"🏆 Overall Processing Quality Score: {overall_score:.3f}/1.0")
        
        # TODO: In future iterations, this could:
        # 1. Store metrics in database for historical analysis
        # 2. Trigger adaptive prompt improvements
        # 3. Adjust model selection based on performance
        # 4. Generate quality reports for system monitoring
    
    def _fallback_basic_categorization(self, standards_chunk: List[Standard], 
                                     hierarchy: TopicHierarchy) -> List[StandardCategorization]:
        """Fallback to basic categorization method if enhanced method fails."""
        logger.info("🔄 Using fallback basic categorization method")
        
        # Prepare standards for basic LLM processing
        standards_data = []
        for standard in standards_chunk:
            std_text = f"{standard.code}: {standard.title}"
            if standard.description:
                std_text += f" - {standard.description[:200]}..."
            standards_data.append({
                "id": standard.code,
                "text": std_text
            })
        
        # Prepare hierarchy paths for basic prompt
        hierarchy_paths = []
        for topic in hierarchy.topics:
            topic_name = topic.get('name', '')
            for sub_topic in topic.get('sub_topics', []):
                sub_topic_name = sub_topic.get('name', '')
                for sub_sub_topic in sub_topic.get('sub_sub_topics', []):
                    hierarchy_paths.append(f"  {topic_name} > {sub_topic_name} > {sub_sub_topic}")
        
        # Create basic prompt
        prompt = f"""You are an education curriculum expert. Categorize each standard into the provided topic hierarchy.

**Topic Hierarchy:**
{chr(10).join(hierarchy_paths)}

**Standards to Categorize:**
{chr(10).join(f"{i+1}. {std['text']}" for i, std in enumerate(standards_data))}

Return JSON with categorizations and outliers:
{{
  "categorizations": [
    {{
      "standard_id": "CA.3.1.1",
      "topic": "Topic Name",
      "sub_topic": "Sub-Topic Name", 
      "sub_sub_topic": "Sub-Sub-Topic Name"
    }}
  ],
  "outliers": [
    {{
      "standard_id": "TX.3.4D",
      "reason": "Brief explanation"
    }}
  ]
}}"""
        
        try:
            response = self._call_openai_api(prompt, "standard_categorization_fallback")
            categorization_data = json.loads(response)
            
            # Process basic categorizations
            results = []
            standard_lookup = {std.code: std for std in standards_chunk}
            
            # Process regular categorizations
            for cat in categorization_data.get('categorizations', []):
                std_id = cat.get('standard_id')
                if std_id in standard_lookup:
                    results.append(StandardCategorization(
                        standard_id=std_id,
                        standard_obj=standard_lookup[std_id],
                        topic=cat.get('topic', ''),
                        sub_topic=cat.get('sub_topic', ''),
                        sub_sub_topic=cat.get('sub_sub_topic', ''),
                        is_outlier=False
                    ))
            
            # Process outliers
            for outlier in categorization_data.get('outliers', []):
                std_id = outlier.get('standard_id')
                if std_id in standard_lookup:
                    results.append(StandardCategorization(
                        standard_id=std_id,
                        standard_obj=standard_lookup[std_id],
                        topic="Outliers",
                        sub_topic="Uncategorized",
                        sub_sub_topic=outlier.get('reason', 'Does not fit main taxonomy'),
                        is_outlier=True,
                        outlier_reason=outlier.get('reason')
                    ))
            
            logger.info(f"✅ Fallback categorization completed: {len(results)} standards processed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Fallback categorization also failed: {e}")
            raise
    
    def create_topic_proxies(self, categorizations: List[StandardCategorization], 
                           filter_grade_levels: List[int] = None, 
                           filter_subject_area_id: int = None,
                           filter_criteria: dict = None) -> List[TopicBasedProxy]:
        """Create TopicBasedProxy objects from categorizations with comprehensive validation.
        
        Args:
            categorizations: List of standard categorizations
            filter_grade_levels: Original grade levels used as filter criteria
            filter_subject_area_id: Original subject area ID used as filter
            filter_criteria: Complete original filtering criteria
        """
        if not categorizations:
            logger.warning("⚠️ No categorizations provided for proxy creation")
            return []
        
        logger.info(f"🏗️ Creating topic proxies from {len(categorizations):,} categorizations")
        
        # Validate categorizations before processing
        valid_categorizations = self._validate_categorizations_for_proxy_creation(categorizations)
        
        # Group categorizations by (topic, sub_topic, sub_sub_topic)
        groups = {}
        categorization_tracking = {
            'processed': 0,
            'skipped_invalid': 0,
            'standards_added': 0,
            'duplicate_standards': 0
        }
        
        for cat in valid_categorizations:
            key = (cat.topic, cat.sub_topic, cat.sub_sub_topic, cat.is_outlier)
            if key not in groups:
                groups[key] = []
            groups[key].append(cat)
            categorization_tracking['processed'] += 1
        
        categorization_tracking['skipped_invalid'] = len(categorizations) - len(valid_categorizations)
        
        logger.info(f"📊 Categorization grouping: {len(groups)} unique topic groups from {len(valid_categorizations)} valid categorizations")
        
        created_proxies = []
        
        # Import here to avoid circular imports
        from ..models import SubjectArea
        
        # Get subject area object if ID provided
        filter_subject_area = None
        if filter_subject_area_id:
            try:
                filter_subject_area = SubjectArea.objects.get(id=filter_subject_area_id)
                logger.info(f"📚 Using subject area filter: {filter_subject_area.name}")
            except SubjectArea.DoesNotExist:
                logger.warning(f"⚠️ Subject area {filter_subject_area_id} not found")
        
        with transaction.atomic():
            for (topic, sub_topic, sub_sub_topic, is_outlier), group_cats in groups.items():
                try:
                    # Generate proxy ID
                    proxy_id = self._generate_proxy_id(topic, sub_topic, sub_sub_topic, is_outlier)
                    
                    # Prepare defaults with filter criteria
                    proxy_defaults = {
                        'topic': topic,
                        'sub_topic': sub_topic,
                        'sub_sub_topic': sub_sub_topic,
                        'outlier_category': is_outlier,
                        'filter_grade_levels': filter_grade_levels or [],
                        'filter_subject_area': filter_subject_area,
                        'filter_criteria': filter_criteria or {},
                    }
                    
                    # Create or get existing proxy
                    proxy, created = TopicBasedProxy.objects.get_or_create(
                        proxy_id=proxy_id,
                        defaults=proxy_defaults
                    )
                    
                    # Process member standards with validation
                    standards_to_add = []
                    existing_standard_ids = set(proxy.member_standards.values_list('id', flat=True))
                    
                    for cat in group_cats:
                        if cat.standard_obj:
                            if cat.standard_obj.id not in existing_standard_ids:
                                standards_to_add.append(cat.standard_obj)
                                categorization_tracking['standards_added'] += 1
                            else:
                                categorization_tracking['duplicate_standards'] += 1
                    
                    # Add new member standards if any
                    if standards_to_add:
                        proxy.member_standards.add(*standards_to_add)
                        logger.debug(f"Added {len(standards_to_add)} new standards to proxy {proxy_id}")
                    
                    # Update proxy metadata
                    self._update_proxy_metadata(proxy)
                    
                    created_proxies.append(proxy)
                    
                    status = "Created" if created else "Updated"
                    total_standards = proxy.member_standards.count()
                    logger.info(f"✅ {status} proxy {proxy_id}: {total_standards} total standards ({len(standards_to_add)} new)")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to create proxy for topic '{topic}' > '{sub_topic}' > '{sub_sub_topic}': {e}")
                    continue
        
        # Final validation and reporting
        self._validate_proxy_creation_results(created_proxies, categorizations, categorization_tracking)
        
        logger.info(f"🏆 Created {len(created_proxies)} topic proxies with {categorization_tracking['standards_added']} standard associations")
        return created_proxies

    def _get_or_create_topic_proxy(self,
                                   topic: str,
                                   sub_topic: str,
                                   sub_sub_topic: str,
                                   is_outlier: bool,
                                   filter_grade_levels: Optional[List[int]],
                                   filter_subject_area_id: Optional[int],
                                   filter_criteria: Optional[dict]) -> 'TopicBasedProxy':
        """Fetch an existing topic proxy by hierarchy or create a new one.

        Uses the hierarchy fields to look up an existing proxy to avoid duplicates
        across chunks. If not found, creates a new proxy with a generated ID.
        """
        existing = TopicBasedProxy.objects.filter(
            topic=topic,
            sub_topic=sub_topic,
            sub_sub_topic=sub_sub_topic,
            outlier_category=is_outlier
        ).first()
        if existing:
            return existing

        filter_subject_area = None
        if filter_subject_area_id:
            try:
                from ..models import SubjectArea  # avoid circular import at module load time
                filter_subject_area = SubjectArea.objects.get(id=filter_subject_area_id)
            except Exception:
                filter_subject_area = None

        proxy_id = self._generate_proxy_id(topic, sub_topic, sub_sub_topic, is_outlier)
        return TopicBasedProxy.objects.create(
            proxy_id=proxy_id,
            topic=topic,
            sub_topic=sub_topic,
            sub_sub_topic=sub_sub_topic,
            outlier_category=is_outlier,
            filter_grade_levels=filter_grade_levels or [],
            filter_subject_area=filter_subject_area,
            filter_criteria=filter_criteria or {},
        )

    def _upsert_topic_proxies_for_chunk(self,
                                        chunk_categorizations: List[StandardCategorization],
                                        filter_grade_levels: Optional[List[int]],
                                        filter_subject_area_id: Optional[int],
                                        filter_criteria: Optional[dict]) -> List['TopicBasedProxy']:
        """Upsert `TopicBasedProxy` rows for a single chunk of categorizations."""
        if not chunk_categorizations:
            return []

        groups: Dict[Tuple[str, str, str, bool], List[StandardCategorization]] = {}
        for cat in chunk_categorizations:
            key = (cat.topic, cat.sub_topic, cat.sub_sub_topic, cat.is_outlier)
            groups.setdefault(key, []).append(cat)

        touched: List[TopicBasedProxy] = []
        with transaction.atomic():
            for (topic, sub_topic, sub_sub_topic, is_outlier), group_cats in groups.items():
                try:
                    proxy = self._get_or_create_topic_proxy(
                        topic,
                        sub_topic,
                        sub_sub_topic,
                        is_outlier,
                        filter_grade_levels,
                        filter_subject_area_id,
                        filter_criteria or {},
                    )

                    existing_ids = set(proxy.member_standards.values_list('id', flat=True))
                    standards_to_add = [
                        cat.standard_obj for cat in group_cats
                        if cat.standard_obj and cat.standard_obj.id not in existing_ids
                    ]

                    if standards_to_add:
                        proxy.member_standards.add(*standards_to_add)

                    self._update_proxy_metadata(proxy)
                    touched.append(proxy)
                except Exception as e:
                    logger.error(f"❌ Failed to upsert proxy for '{topic}' > '{sub_topic}' > '{sub_sub_topic}': {e}")

        return touched
    
    def _validate_categorizations_for_proxy_creation(self, categorizations: List[StandardCategorization]) -> List[StandardCategorization]:
        """
        Validate categorizations before creating proxies.
        
        Args:
            categorizations: Raw categorizations to validate
            
        Returns:
            List of valid categorizations
        """
        valid_categorizations = []
        validation_issues = {
            'missing_standard_obj': 0,
            'empty_topic': 0,
            'empty_sub_topic': 0,
            'empty_sub_sub_topic': 0,
            'duplicate_standard_ids': set()
        }
        
        seen_standard_ids = set()
        
        for cat in categorizations:
            # Check for missing standard object
            if not cat.standard_obj:
                validation_issues['missing_standard_obj'] += 1
                continue
            
            # Check for empty topic fields
            if not cat.topic or cat.topic.strip() == "":
                validation_issues['empty_topic'] += 1
                continue
                
            if not cat.sub_topic or cat.sub_topic.strip() == "":
                validation_issues['empty_sub_topic'] += 1
                continue
                
            if not cat.sub_sub_topic or cat.sub_sub_topic.strip() == "":
                validation_issues['empty_sub_sub_topic'] += 1
                continue
            
            # Check for duplicate standard IDs
            if cat.standard_id in seen_standard_ids:
                validation_issues['duplicate_standard_ids'].add(cat.standard_id)
                continue
            
            seen_standard_ids.add(cat.standard_id)
            valid_categorizations.append(cat)
        
        # Log validation results
        if any(validation_issues.values()):
            logger.warning("⚠️ Categorization validation issues found:")
            if validation_issues['missing_standard_obj']:
                logger.warning(f"  📝 Missing standard objects: {validation_issues['missing_standard_obj']}")
            if validation_issues['empty_topic']:
                logger.warning(f"  📝 Empty topics: {validation_issues['empty_topic']}")
            if validation_issues['empty_sub_topic']:
                logger.warning(f"  📝 Empty sub-topics: {validation_issues['empty_sub_topic']}")
            if validation_issues['empty_sub_sub_topic']:
                logger.warning(f"  📝 Empty sub-sub-topics: {validation_issues['empty_sub_sub_topic']}")
            if validation_issues['duplicate_standard_ids']:
                logger.warning(f"  📝 Duplicate standard IDs: {len(validation_issues['duplicate_standard_ids'])}")
        
        logger.info(f"✅ Validation: {len(valid_categorizations):,} valid categorizations from {len(categorizations):,} total")
        return valid_categorizations
    
    def _update_proxy_metadata(self, proxy: 'TopicBasedProxy') -> None:
        """
        Update proxy metadata based on member standards.
        
        Args:
            proxy: TopicBasedProxy to update
        """
        try:
            # Update grade level information
            all_grade_levels = set()
            for standard in proxy.member_standards.all():
                for grade_level in standard.grade_levels.all():
                    all_grade_levels.add(grade_level)
            
            if all_grade_levels:
                proxy.grade_levels.set(all_grade_levels)
                
                # Update min/max grade
                grade_numerics = [gl.grade_numeric for gl in all_grade_levels if hasattr(gl, 'grade_numeric')]
                if grade_numerics:
                    proxy.min_grade = min(grade_numerics)
                    proxy.max_grade = max(grade_numerics)
            
            proxy.save()
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to update metadata for proxy {proxy.proxy_id}: {e}")
    
    def _validate_proxy_creation_results(self, created_proxies: List['TopicBasedProxy'], 
                                       original_categorizations: List[StandardCategorization],
                                       tracking: Dict[str, Any]) -> None:
        """
        Validate the results of proxy creation.
        
        Args:
            created_proxies: List of created/updated proxies
            original_categorizations: Original categorizations
            tracking: Processing tracking data
        """
        logger.info("📋 Proxy Creation Validation Report:")
        logger.info(f"  📥 Input categorizations: {len(original_categorizations):,}")
        logger.info(f"  🏗️ Proxies created/updated: {len(created_proxies):,}")
        logger.info(f"  ✅ Categorizations processed: {tracking['processed']:,}")
        logger.info(f"  ❌ Categorizations skipped (invalid): {tracking['skipped_invalid']:,}")
        logger.info(f"  📊 Standards added to proxies: {tracking['standards_added']:,}")
        logger.info(f"  🔄 Duplicate standards skipped: {tracking['duplicate_standards']:,}")
        
        # Validate that all standards are accounted for
        input_standard_ids = {cat.standard_id for cat in original_categorizations if cat.standard_obj}
        proxy_standard_ids = set()
        
        for proxy in created_proxies:
            proxy_standard_ids.update(proxy.member_standards.values_list('code', flat=True))
        
        missing_standards = input_standard_ids - proxy_standard_ids
        extra_standards = proxy_standard_ids - input_standard_ids
        
        if missing_standards:
            logger.error(f"❌ Standards missing from proxies: {len(missing_standards)}")
            logger.error(f"   First few missing: {sorted(list(missing_standards))[:5]}")
        
        if extra_standards:
            logger.warning(f"⚠️ Extra standards in proxies: {len(extra_standards)}")
        
        # Calculate coverage statistics
        coverage_rate = len(proxy_standard_ids) / len(input_standard_ids) * 100 if input_standard_ids else 0
        logger.info(f"📈 Standard coverage rate: {coverage_rate:.1f}%")
        
        if coverage_rate < 95:
            logger.error(f"❌ Low coverage rate: {coverage_rate:.1f}% - investigate data loss issues")
        else:
            logger.info(f"✅ Good coverage rate: {coverage_rate:.1f}%")
    
    def _generate_proxy_id(self, topic: str, sub_topic: str, sub_sub_topic: str, is_outlier: bool) -> str:
        """Generate a unique proxy ID from the topic hierarchy."""
        if is_outlier:
            prefix = "TP-OUT"
            # Use hash of sub_sub_topic for outliers
            topic_hash = hashlib.md5(sub_sub_topic.encode()).hexdigest()[:4].upper()
            base_id = f"{prefix}-{topic_hash}"
        else:
            prefix = "TP"
            # Create abbreviated versions of topic parts
            topic_abbrev = self._abbreviate_topic(topic)
            subtopic_abbrev = self._abbreviate_topic(sub_topic)
            base_id = f"{prefix}-{topic_abbrev}-{subtopic_abbrev}"
        
        # Find next available ID with this base
        counter = 1
        while True:
            proxy_id = f"{base_id}-{counter:03d}"
            if not TopicBasedProxy.objects.filter(proxy_id=proxy_id).exists():
                return proxy_id
            counter += 1
            if counter > 999:  # Safety break
                proxy_id = f"{base_id}-{counter}"
                break
        
        return proxy_id
    
    def _abbreviate_topic(self, topic: str) -> str:
        """Create abbreviated version of topic for ID generation."""
        # Remove common words and abbreviate
        common_words = {'and', 'or', 'the', 'of', 'in', 'for', 'with', 'to', 'a', 'an'}
        words = [w for w in topic.split() if w.lower() not in common_words]
        
        if len(words) == 1:
            return words[0][:8].title()
        elif len(words) == 2:
            return f"{words[0][:4]}{words[1][:4]}".title()
        else:
            # Use first letters of first 3 words
            return ''.join(w[0].upper() for w in words[:3])
    
    def _call_openai_api(self, prompt: str, operation_type: str) -> str:
        """Call OpenAI API with error handling and retries."""
        for attempt in range(self.max_retries):
            try:
                # Prefer Chat Completions API
                if hasattr(self.client, 'chat') and hasattr(self.client.chat, 'completions'):
                    response = self.client.chat.completions.create(
                        model=self.DEFAULT_MODEL,  # Use the same model as everywhere else
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": "You are an education curriculum expert. Always respond with valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=16000  # Increased for GPT-4.1's larger context
                    )
                    return response.choices[0].message.content
                
                # Legacy fallback
                elif hasattr(self.client, 'ChatCompletion'):
                    response = self.client.ChatCompletion.create(
                        model=self.DEFAULT_MODEL,  # Use the same model as everywhere else
                        messages=[
                            {"role": "system", "content": "You are an education curriculum expert. Always respond with valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=16000  # Increased for GPT-4.1's larger context
                    )
                    return response["choices"][0]["message"]["content"]
                
                else:
                    raise ValueError("No compatible OpenAI API interface found")
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {operation_type}: {e}")
                if attempt == self.max_retries - 1:
                    raise
        
        raise Exception(f"Failed to complete {operation_type} after {self.max_retries} attempts")
    
    def calculate_optimal_chunk_size(self, 
                                   standards: List[Standard],
                                   hierarchy: TopicHierarchy = None,
                                   mode: str = "enhanced") -> int:
        """Calculate optimal chunk size for processing standards with aggressive utilization.
        
        Args:
            standards: List of standards to process
            hierarchy: Optional topic hierarchy for context calculation
            mode: Processing mode ("enhanced" or "basic")
            
        Returns:
            Optimal chunk size (number of standards per chunk)
        """
        if not standards:
            return 50  # Reasonable minimum for empty case
        
        # Convert Standard objects to dictionaries for token calculation
        standards_data = []
        for standard in standards:
            std_dict = {
                'code': standard.code,
                'title': standard.title,
                'description': standard.description or '',
                'domain': getattr(standard, 'domain', ''),
                'cluster': getattr(standard, 'cluster', ''),
                'subject': standard.subject_area.name if standard.subject_area else 'Unknown'
            }
            standards_data.append(std_dict)
        
        # Prepare hierarchy text for token calculation
        hierarchy_text = None
        if hierarchy:
            hierarchy_paths = []
            for topic in hierarchy.topics:
                topic_name = topic.get('name', '')
                for sub_topic in topic.get('sub_topics', []):
                    sub_topic_name = sub_topic.get('name', '')
                    for sub_sub_topic in sub_topic.get('sub_sub_topics', []):
                        hierarchy_paths.append(f"  {topic_name} > {sub_topic_name} > {sub_sub_topic}")
            hierarchy_text = "\n".join(hierarchy_paths)
        
        # Calculate optimal chunk size with aggressive utilization
        logger.info(f"🧠 Calculating optimal chunk size using model: {self.token_counter.model_name} ({mode} mode)")
        optimal_size = self.token_counter.calculate_optimal_chunk_size(
            standards_data,
            topic_hierarchy=hierarchy_text,
            mode=mode
        )
        
        logger.info(f"🎯 Calculated optimal chunk size: {optimal_size:,} standards per chunk")
        logger.info(f"📊 Expected chunks for {len(standards):,} standards: {(len(standards) + optimal_size - 1) // optimal_size}")
        return optimal_size
    
    def _analyze_educational_context(self, standards: List[Standard]) -> Dict[str, Any]:
        """Analyze educational context of standards for intelligent chunking.
        
        Args:
            standards: List of standards to analyze
            
        Returns:
            Dictionary containing educational context analysis
        """
        context = {
            'grade_distribution': {},
            'subject_distribution': {},
            'domain_distribution': {},
            'complexity_distribution': {},
            'total_standards': len(standards)
        }
        
        # Analyze grade level distribution
        for standard in standards:
            grade_levels = standard.grade_levels.all()
            for grade in grade_levels:
                grade_key = f"Grade {grade.grade_numeric}" if hasattr(grade, 'grade_numeric') else grade.name
                context['grade_distribution'][grade_key] = context['grade_distribution'].get(grade_key, 0) + 1
        
        # Analyze subject area distribution
        for standard in standards:
            if standard.subject_area:
                subject = standard.subject_area.name
                context['subject_distribution'][subject] = context['subject_distribution'].get(subject, 0) + 1
        
        # Analyze domain distribution
        for standard in standards:
            if hasattr(standard, 'domain') and standard.domain:
                domain = standard.domain
                context['domain_distribution'][domain] = context['domain_distribution'].get(domain, 0) + 1
        
        # Analyze complexity distribution (based on description length and keywords)
        for standard in standards:
            complexity = self._estimate_standard_complexity(standard)
            context['complexity_distribution'][complexity] = context['complexity_distribution'].get(complexity, 0) + 1
        
        logger.info(f"📊 Educational context analysis: {context['total_standards']} standards")
        logger.info(f"📊 Grade distribution: {context['grade_distribution']}")
        logger.info(f"📊 Subject distribution: {context['subject_distribution']}")
        
        return context
    
    def _estimate_standard_complexity(self, standard: Standard) -> str:
        """Estimate the cognitive complexity of a standard.
        
        Args:
            standard: Standard to analyze
            
        Returns:
            Complexity level string
        """
        description = (standard.description or '').lower()
        title = (standard.title or '').lower()
        text = f"{title} {description}"
        
        # Bloom's taxonomy keywords for complexity estimation
        complexity_keywords = {
            'high': ['analyze', 'evaluate', 'create', 'synthesize', 'critique', 'design', 'construct', 
                    'develop', 'formulate', 'investigate', 'justify', 'assess', 'compare', 'contrast'],
            'medium': ['apply', 'demonstrate', 'solve', 'use', 'show', 'classify', 'organize', 
                      'relate', 'calculate', 'modify', 'prepare', 'produce', 'examine'],
            'low': ['remember', 'recall', 'identify', 'describe', 'define', 'list', 'name', 
                   'state', 'recognize', 'understand', 'explain', 'interpret', 'summarize']
        }
        
        # Count complexity indicators
        high_count = sum(1 for keyword in complexity_keywords['high'] if keyword in text)
        medium_count = sum(1 for keyword in complexity_keywords['medium'] if keyword in text)
        # low_count = sum(1 for keyword in complexity_keywords['low'] if keyword in text)  # Available for future use
        
        # Determine complexity level
        if high_count > 0 or len(description) > 200:
            return 'high'
        elif medium_count > 0 or len(description) > 100:
            return 'medium'
        else:
            return 'low'
    
    def create_educational_chunks(self, 
                                standards: List[Standard], 
                                base_chunk_size: int,
                                hierarchy: Optional[TopicHierarchy] = None) -> List[List[Standard]]:
        """Create educationally-aware chunks of standards.
        
        Args:
            standards: List of standards to chunk
            base_chunk_size: Base chunk size from token calculation
            hierarchy: Optional topic hierarchy for context
            
        Returns:
            List of standard chunks optimized for educational coherence
        """
        if not standards:
            return []
        
        logger.info(f"🎓 Creating educational chunks for {len(standards)} standards (base_size: {base_chunk_size})")
        
        # Analyze educational context
        context = self._analyze_educational_context(standards)
        
        # For small datasets, use simple chunking
        if len(standards) <= base_chunk_size:
            logger.info(f"📝 Small dataset, using single chunk of {len(standards)} standards")
            return [standards]
        
        # Sort standards for optimal chunking
        sorted_standards = self._sort_standards_for_chunking(standards)
        
        # Create chunks with educational awareness
        chunks = []
        current_chunk = []
        current_chunk_size = 0
        
        # Adjust chunk size based on educational factors
        adjusted_chunk_size = self._adjust_chunk_size_for_education(
            base_chunk_size, context, len(standards)
        )
        
        for i, standard in enumerate(sorted_standards):
            # Check if adding this standard would exceed chunk size
            if current_chunk_size >= adjusted_chunk_size and current_chunk:
                # Look ahead to see if we can keep related standards together
                should_break_chunk = self._should_break_chunk(
                    current_chunk, standard, sorted_standards[i:], adjusted_chunk_size
                )
                
                if should_break_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_chunk_size = 0
            
            current_chunk.append(standard)
            current_chunk_size += 1
        
        # Add final chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        # Log chunking results
        chunk_sizes = [len(chunk) for chunk in chunks]
        logger.info(f"📚 Created {len(chunks)} educational chunks: sizes {chunk_sizes}")
        logger.info(f"📚 Average chunk size: {sum(chunk_sizes) / len(chunk_sizes):.1f} standards")
        
        return chunks
    
    def _sort_standards_for_chunking(self, standards: List[Standard]) -> List[Standard]:
        """Sort standards to optimize educational coherence in chunks.
        
        Args:
            standards: List of standards to sort
            
        Returns:
            Sorted list of standards
        """
        def sort_key(standard):
            # Primary sort: Subject area
            subject = standard.subject_area.name if standard.subject_area else 'ZZ_Unknown'
            
            # Secondary sort: Grade level (average if multiple)
            grades = standard.grade_levels.all()
            if grades:
                avg_grade = sum(getattr(g, 'grade_numeric', 0) for g in grades) / len(grades)
            else:
                avg_grade = 999  # Put unknown grades at end
            
            # Tertiary sort: Domain
            domain = getattr(standard, 'domain', '') or 'ZZ_Unknown'
            
            # Quaternary sort: Complexity
            complexity_order = {'low': 1, 'medium': 2, 'high': 3}
            complexity = complexity_order.get(self._estimate_standard_complexity(standard), 4)
            
            return (subject, avg_grade, domain, complexity, standard.code)
        
        sorted_standards = sorted(standards, key=sort_key)
        logger.info(f"🔄 Sorted {len(standards)} standards for educational coherence")
        
        return sorted_standards
    
    def _adjust_chunk_size_for_education(self, 
                                       base_chunk_size: int, 
                                       context: Dict[str, Any], 
                                       total_standards: int) -> int:
        """Adjust chunk size based on educational factors.
        
        Args:
            base_chunk_size: Base chunk size from token calculation
            context: Educational context analysis
            total_standards: Total number of standards
            
        Returns:
            Adjusted chunk size
        """
        adjusted_size = base_chunk_size
        
        # Adjust for subject diversity
        subject_count = len(context['subject_distribution'])
        if subject_count > 1:
            # Prefer slightly smaller chunks for mixed subjects
            adjusted_size = max(int(adjusted_size * 0.9), 5)
            logger.info(f"🔧 Adjusted chunk size for {subject_count} subjects: {adjusted_size}")
        
        # Adjust for grade diversity
        grade_count = len(context['grade_distribution'])
        if grade_count > 3:
            # Prefer smaller chunks for wide grade ranges
            adjusted_size = max(int(adjusted_size * 0.85), 5)
            logger.info(f"🔧 Adjusted chunk size for {grade_count} grade levels: {adjusted_size}")
        
        # Adjust for complexity distribution
        complexity_levels = len(context['complexity_distribution'])
        if complexity_levels > 1:
            # Mixed complexity - slight adjustment
            adjusted_size = max(int(adjusted_size * 0.95), 5)
        
        # Ensure we don't create too many tiny chunks
        min_chunk_size = max(5, total_standards // 50)  # At most 50 chunks
        adjusted_size = max(adjusted_size, min_chunk_size)
        
        # Ensure we don't exceed reasonable maximums for educational processing
        max_educational_chunk = 200  # Educational maximum
        adjusted_size = min(adjusted_size, max_educational_chunk)
        
        return adjusted_size
    
    def _should_break_chunk(self, 
                          current_chunk: List[Standard], 
                          next_standard: Standard,
                          remaining_standards: Optional[List[Standard]],
                          target_size: int) -> bool:
        """Decide whether to break a chunk based on educational coherence.
        
        Args:
            current_chunk: Current chunk being built
            next_standard: Next standard to potentially add
            remaining_standards: Remaining standards to process
            target_size: Target chunk size
            
        Returns:
            True if chunk should be broken, False if standard should be added
        """
        if not current_chunk:
            return False
        
        # If chunk is significantly oversized, break it
        if len(current_chunk) > target_size * 1.2:
            return True
        
        # If chunk is undersized, keep adding
        if len(current_chunk) < target_size * 0.7:
            return False
        
        # Check educational coherence factors
        last_standard = current_chunk[-1]
        
        # Subject area coherence
        if (last_standard.subject_area and next_standard.subject_area and 
            last_standard.subject_area != next_standard.subject_area):
            logger.debug(f"🔄 Breaking chunk due to subject change: {last_standard.subject_area} -> {next_standard.subject_area}")
            return True
        
        # Grade level coherence (allow 1-2 grade difference)
        last_grades = set(g.grade_numeric for g in last_standard.grade_levels.all() if hasattr(g, 'grade_numeric'))
        next_grades = set(g.grade_numeric for g in next_standard.grade_levels.all() if hasattr(g, 'grade_numeric'))
        
        if last_grades and next_grades:
            min_last, max_last = min(last_grades), max(last_grades)
            min_next, max_next = min(next_grades), max(next_grades)
            
            # Break if grade ranges don't overlap and are more than 2 levels apart
            if (max_last < min_next and min_next - max_last > 2) or \
               (max_next < min_last and min_last - max_next > 2):
                logger.debug(f"🔄 Breaking chunk due to grade gap: {last_grades} -> {next_grades}")
                return True
        
        # Domain coherence
        if (hasattr(last_standard, 'domain') and hasattr(next_standard, 'domain') and
            last_standard.domain and next_standard.domain and
            last_standard.domain != next_standard.domain):
            logger.debug(f"🔄 Breaking chunk due to domain change: {last_standard.domain} -> {next_standard.domain}")
            return True
        
        # Default: don't break if we've made it this far
        return False
    
    def run_full_categorization(self, 
                              grade_levels: List[int] = None, 
                              subject_area_id: int = None,
                              progress_callback=None,
                              use_dynamic_chunk_size: bool = True,
                              override_chunk_size: Optional[int] = None,
                              incremental_proxy_updates: bool = True) -> Tuple[TopicHierarchy, List[TopicBasedProxy]]:
        """Run the complete topic categorization process.
        
        Args:
            grade_levels: Optional list of grade levels to filter by
            subject_area_id: Optional subject area ID to filter by
            progress_callback: Optional callback for progress updates
            use_dynamic_chunk_size: Whether to use dynamic chunk sizing (default True)
            override_chunk_size: Optional manual override for chunk size
            
        Returns:
            Tuple of (TopicHierarchy, List[TopicBasedProxy])
        """
        logger.info(f"Starting topic categorization with model: {self.DEFAULT_MODEL} (token_counter: {self.token_counter.model_name})")
        logger.info(f"Parameters: grade_levels={grade_levels}, subject_area_id={subject_area_id}, use_dynamic_chunk_size={use_dynamic_chunk_size}, override_chunk_size={override_chunk_size}")
        if progress_callback:
            progress_callback(5, "Loading standards...")
        
        # Load standards
        standards = self.load_standards(grade_levels, subject_area_id)
        
        if progress_callback:
            progress_callback(15, f"Generating topic hierarchy from {len(standards)} standards...")
        
        # Generate topic hierarchy
        subject_area_name = None
        if subject_area_id:
            from ..models import SubjectArea
            try:
                subject_area_name = SubjectArea.objects.get(id=subject_area_id).name
            except SubjectArea.DoesNotExist:
                pass
        
        hierarchy = self.generate_topic_hierarchy(standards, subject_area_name)
        
        if progress_callback:
            progress_callback(25, "Calculating optimal chunk size...")
        
        # Determine chunk size with aggressive optimization
        if override_chunk_size:
            chunk_size = override_chunk_size
            logger.info(f"🔧 Using override chunk size: {chunk_size:,} standards per chunk")
        elif use_dynamic_chunk_size:
            calculated = self.calculate_optimal_chunk_size(standards, hierarchy, mode="enhanced")
            chunk_size = min(calculated, self.MAX_DYNAMIC_CHUNK_SIZE)
            if calculated != chunk_size:
                logger.info(f"📏 Clamped dynamic chunk size from {calculated:,} to {chunk_size:,} (MAX_DYNAMIC_CHUNK_SIZE)")
            logger.info(f"🤖 Using dynamic chunk size: {chunk_size:,} standards per chunk (model: {self.token_counter.model_name})")
        else:
            # Use a much larger default for modern models
            if self.token_counter.model_name in ['gpt-4.1', 'gpt-4.1-mini']:
                chunk_size = min(1000, len(standards))  # Up to 1000 for GPT-4.1
            elif self.token_counter.model_name in ['gpt-5', 'gpt-5-mini']:
                chunk_size = min(500, len(standards))   # Up to 500 for GPT-5
            else:
                chunk_size = min(200, len(standards))   # Up to 200 for other models
            # Also apply cap for safety
            if chunk_size > self.MAX_DYNAMIC_CHUNK_SIZE:
                logger.info(f"📏 Clamped default chunk size from {chunk_size:,} to {self.MAX_DYNAMIC_CHUNK_SIZE:,} (MAX_DYNAMIC_CHUNK_SIZE)")
                chunk_size = self.MAX_DYNAMIC_CHUNK_SIZE
            logger.info(f"📐 Using enhanced default chunk size: {chunk_size:,} standards per chunk")
        
        # Log context window utilization for debugging
        if hasattr(self.token_counter, 'MODEL_LIMITS') and self.token_counter.model_name in self.token_counter.MODEL_LIMITS:
            context_window = self.token_counter.MODEL_LIMITS[self.token_counter.model_name]['context_window']
            logger.info(f"🧠 Model {self.token_counter.model_name} context window: {context_window:,} tokens")
        
        logger.info(f"📊 Preparing to process {len(standards)} standards with target chunk size: {chunk_size}")
        logger.info(f"🎓 Educational chunking will optimize for subject coherence, grade alignment, and complexity grouping")
        
        if progress_callback:
            progress_callback(30, f"Categorizing standards in chunks of {chunk_size}...")
        
        # Create educationally-aware chunks
        if progress_callback:
            progress_callback(28, "Creating educational chunks...")
        
        educational_chunks = self.create_educational_chunks(
            standards, chunk_size, hierarchy
        )
        
        if progress_callback:
            progress_callback(30, f"Processing {len(educational_chunks)} educational chunks...")
        
        # Process standards with robust error handling and validation
        all_categorizations = []
        # Prepare filter criteria once (used for both incremental and batch modes)
        filter_criteria: Dict[str, Any] = {}
        if grade_levels:
            if len(grade_levels) == 1:
                filter_criteria['grade_selection'] = {'type': 'specific', 'grades': grade_levels}
            else:
                # Check if it's a range
                sorted_grades = sorted(grade_levels)
                if sorted_grades == list(range(sorted_grades[0], sorted_grades[-1] + 1)):
                    filter_criteria['grade_selection'] = {'type': 'range', 'min_grade': sorted_grades[0], 'max_grade': sorted_grades[-1]}
                else:
                    filter_criteria['grade_selection'] = {'type': 'specific', 'grades': grade_levels}
        else:
            filter_criteria['grade_selection'] = {'type': 'all'}
        
        if subject_area_id:
            filter_criteria['subject_area_id'] = subject_area_id
        
        touched_proxies: Dict[str, TopicBasedProxy] = {}
        total_chunks = len(educational_chunks)
        processing_state = {
            'processed_standards': set(),
            'failed_standards': set(),
            'total_input_standards': len(standards),
            'chunks_processed': 0,
            'chunks_failed': 0
        }
        
        for chunk_num, chunk in enumerate(educational_chunks, 1):
            if progress_callback:
                progress = 30 + (45 * chunk_num / total_chunks)
                progress_callback(progress, f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk):,} standards)...")
            
            chunk_standard_ids = {std.code for std in chunk}
            logger.info(f"🎨 Processing chunk {chunk_num}/{total_chunks}: {len(chunk):,} standards")
            
            try:
                chunk_categorizations = self._process_chunk_with_retry(chunk, hierarchy, chunk_num, max_retries=3)
                
                # Validate chunk results
                categorized_ids = {cat.standard_id for cat in chunk_categorizations}
                missing_ids = chunk_standard_ids - categorized_ids
                
                if missing_ids:
                    logger.warning(f"⚠️ Chunk {chunk_num}: {len(missing_ids)} standards not categorized, attempting individual processing")
                    # Process missing standards individually
                    for std in chunk:
                        if std.code in missing_ids:
                            try:
                                individual_cat = self._categorize_single_standard(std, hierarchy)
                                if individual_cat:
                                    chunk_categorizations.append(individual_cat)
                                    categorized_ids.add(std.code)
                            except Exception as e:
                                logger.error(f"❌ Failed to categorize individual standard {std.code}: {e}")
                                processing_state['failed_standards'].add(std.code)
                
                all_categorizations.extend(chunk_categorizations)
                processing_state['processed_standards'].update(categorized_ids)
                processing_state['chunks_processed'] += 1
                
                logger.info(f"✅ Chunk {chunk_num} completed: {len(chunk_categorizations):,} categorizations")
                logger.info(f"📊 Progress: {len(processing_state['processed_standards']):,}/{processing_state['total_input_standards']:,} standards processed")
                
            except Exception as e:
                logger.error(f"❌ Failed to process chunk {chunk_num} after retries: {e}")
                processing_state['chunks_failed'] += 1
                processing_state['failed_standards'].update(chunk_standard_ids)
                
                # Attempt individual processing for entire failed chunk
                logger.info(f"🔄 Attempting individual processing for failed chunk {chunk_num}")
                for std in chunk:
                    try:
                        individual_cat = self._categorize_single_standard(std, hierarchy)
                        if individual_cat:
                            all_categorizations.append(individual_cat)
                            processing_state['processed_standards'].add(std.code)
                            processing_state['failed_standards'].discard(std.code)
                    except Exception as individual_e:
                        logger.error(f"❌ Individual processing failed for {std.code}: {individual_e}")

            # Incrementally upsert proxies for this chunk if enabled
            if incremental_proxy_updates:
                chunk_cats = [
                    cat for cat in all_categorizations
                    if cat.standard_obj and cat.standard_id in chunk_standard_ids
                ]
                updated = self._upsert_topic_proxies_for_chunk(
                    chunk_cats,
                    filter_grade_levels=grade_levels,
                    filter_subject_area_id=subject_area_id,
                    filter_criteria=filter_criteria,
                )
                for p in updated:
                    touched_proxies[p.proxy_id] = p
        
        # Final validation
        self._validate_processing_completeness(standards, all_categorizations, processing_state)
        
        if progress_callback:
            progress_callback(80, "Creating topic-based proxy standards...")
        
        # Create proxy standards with filter criteria (batch) if not using incremental updates
        if incremental_proxy_updates:
            proxies = list(touched_proxies.values())
        else:
            proxies = self.create_topic_proxies(
                all_categorizations, 
                filter_grade_levels=grade_levels,
                filter_subject_area_id=subject_area_id,
                filter_criteria=filter_criteria
            )
        
        if progress_callback:
            progress_callback(100, f"Created {len(proxies)} topic-based proxy standards")
        
        # Comprehensive final reporting
        self._generate_final_processing_report(standards, all_categorizations, proxies, processing_state)
        
        logger.info(f"🎉 Topic categorization complete: {len(proxies):,} proxies created from {len(standards):,} standards")
        return hierarchy, proxies
    
    def _generate_final_processing_report(self, input_standards: List[Standard], 
                                        categorizations: List[StandardCategorization],
                                        proxies: List['TopicBasedProxy'],
                                        processing_state: Dict[str, Any]) -> None:
        """
        Generate comprehensive final processing report.
        
        Args:
            input_standards: Original input standards
            categorizations: Generated categorizations  
            proxies: Created topic proxies
            processing_state: Processing state tracking
        """
        logger.info("="*80)
        logger.info("🏁 FINAL PROCESSING REPORT")
        logger.info("="*80)
        
        # Input summary
        logger.info(f"📥 INPUT SUMMARY:")
        logger.info(f"   Total standards loaded: {len(input_standards):,}")
        
        # Processing summary
        logger.info(f"🔧 PROCESSING SUMMARY:")
        logger.info(f"   Standards processed: {len(processing_state['processed_standards']):,}")
        logger.info(f"   Standards failed: {len(processing_state['failed_standards']):,}")
        logger.info(f"   Chunks processed successfully: {processing_state['chunks_processed']:,}")
        logger.info(f"   Chunks failed: {processing_state['chunks_failed']:,}")
        
        # Model utilization summary
        logger.info(f"🧠 MODEL UTILIZATION:")
        logger.info(f"   Model used: {self.token_counter.model_name}")
        if hasattr(self.token_counter, 'MODEL_LIMITS'):
            limits = self.token_counter.MODEL_LIMITS.get(self.token_counter.model_name, {})
            context_window = limits.get('context_window', 'Unknown')
            logger.info(f"   Context window: {context_window:,} tokens" if isinstance(context_window, int) else f"   Context window: {context_window}")
        
        # Categorization summary
        categorized_standards = {cat.standard_id for cat in categorizations}
        outlier_categorizations = [cat for cat in categorizations if cat.is_outlier]
        regular_categorizations = [cat for cat in categorizations if not cat.is_outlier]
        
        logger.info(f"📊 CATEGORIZATION SUMMARY:")
        logger.info(f"   Total categorizations: {len(categorizations):,}")
        logger.info(f"   Regular categorizations: {len(regular_categorizations):,}")
        logger.info(f"   Outlier categorizations: {len(outlier_categorizations):,}")
        logger.info(f"   Categorization rate: {len(categorized_standards)/len(input_standards)*100:.1f}%")
        
        # Topic summary
        unique_topics = set()
        unique_sub_topics = set()
        unique_sub_sub_topics = set()
        
        for cat in regular_categorizations:
            unique_topics.add(cat.topic)
            unique_sub_topics.add((cat.topic, cat.sub_topic))
            unique_sub_sub_topics.add((cat.topic, cat.sub_topic, cat.sub_sub_topic))
        
        logger.info(f"🏷️ TOPIC HIERARCHY SUMMARY:")
        logger.info(f"   Unique topics: {len(unique_topics):,}")
        logger.info(f"   Unique sub-topics: {len(unique_sub_topics):,}")
        logger.info(f"   Unique sub-sub-topics: {len(unique_sub_sub_topics):,}")
        
        # Proxy summary
        logger.info(f"🏗️ PROXY CREATION SUMMARY:")
        logger.info(f"   Topic proxies created: {len(proxies):,}")
        
        total_proxy_standards = 0
        for proxy in proxies:
            total_proxy_standards += proxy.member_standards.count()
        
        logger.info(f"   Total standard-proxy associations: {total_proxy_standards:,}")
        
        # Quality metrics
        failed_rate = len(processing_state['failed_standards']) / len(input_standards) * 100
        
        logger.info(f"📈 QUALITY METRICS:")
        logger.info(f"   Processing success rate: {100-failed_rate:.1f}%")
        logger.info(f"   Outlier rate: {len(outlier_categorizations)/len(categorizations)*100:.1f}%")
        
        # Performance assessment
        if failed_rate > 5:
            logger.error(f"❌ HIGH FAILURE RATE: {failed_rate:.1f}% - investigate system issues")
        elif failed_rate > 1:
            logger.warning(f"⚠️ MODERATE FAILURE RATE: {failed_rate:.1f}% - monitor system health")
        else:
            logger.info(f"✅ LOW FAILURE RATE: {failed_rate:.1f}% - system performing well")
        
        # Recommendations
        logger.info(f"💡 RECOMMENDATIONS:")
        
        if len(processing_state['failed_standards']) > 0:
            logger.info(f"   🔧 Review failed standards for patterns: {sorted(list(processing_state['failed_standards']))[:3]}{'...' if len(processing_state['failed_standards']) > 3 else ''}")
        
        if len(outlier_categorizations) / len(categorizations) > 0.2:
            logger.info(f"   🎯 High outlier rate ({len(outlier_categorizations)/len(categorizations)*100:.1f}%) - consider refining topic hierarchy")
        
        if processing_state['chunks_failed'] > 0:
            logger.info(f"   🔄 {processing_state['chunks_failed']} chunks failed - consider smaller chunk sizes or API rate limiting")
        
        # Context utilization assessment for GPT-4.1
        if self.token_counter.model_name in ['gpt-4.1', 'gpt-4.1-mini']:
            total_chunks = processing_state['chunks_processed'] + processing_state['chunks_failed']
            if total_chunks > 5:
                logger.info(f"   🚀 Consider larger chunk sizes for GPT-4.1 - currently using {total_chunks} chunks")
        
        logger.info("="*80)
    
    # Backward Compatibility Methods
    
    def simple_categorize_standards(self, 
                                  standards: List[Standard], 
                                  subject_area_name: str = None) -> Tuple[TopicHierarchy, List[TopicBasedProxy]]:
        """Simple interface for backward compatibility - categorizes standards without advanced options.
        
        This method provides a simplified interface for existing code that doesn't need
        the advanced chunking and analysis features.
        
        Args:
            standards: List of standards to categorize
            subject_area_name: Optional subject area name for context
            
        Returns:
            Tuple of (TopicHierarchy, List[TopicBasedProxy])
        """
        logger.info(f"🔄 Simple categorization mode for {len(standards)} standards")
        
        if not self.client:
            raise ValueError("OpenAI client not available")
        
        # Generate hierarchy
        hierarchy = self.generate_topic_hierarchy(standards, subject_area_name)
        
        # Use basic chunking (not educational chunks)
        chunk_size = min(self.DEFAULT_CHUNK_SIZE, len(standards))
        
        # Process in simple chunks
        all_categorizations = []
        for i in range(0, len(standards), chunk_size):
            chunk = standards[i:i + chunk_size]
            try:
                chunk_categorizations = self.categorize_standards_chunk(chunk, hierarchy)
                all_categorizations.extend(chunk_categorizations)
            except Exception as e:
                logger.error(f"Failed to process simple chunk: {e}")
                continue
        
        # Create proxies
        proxies = self.create_topic_proxies(all_categorizations)
        
        logger.info(f"✅ Simple categorization complete: {len(proxies)} proxies created")
        return hierarchy, proxies
    
    def get_basic_categorizations(self, 
                                standards: List[Standard], 
                                hierarchy: TopicHierarchy) -> List[StandardCategorization]:
        """Get basic categorizations without enhanced features for backward compatibility.
        
        Args:
            standards: Standards to categorize
            hierarchy: Topic hierarchy to use
            
        Returns:
            List of basic StandardCategorization objects (without enhanced fields)
        """
        logger.info(f"📋 Basic categorization mode for {len(standards)} standards")
        
        # Process in small chunks to avoid token limits
        chunk_size = min(self.DEFAULT_CHUNK_SIZE, len(standards))
        all_categorizations = []
        
        for i in range(0, len(standards), chunk_size):
            chunk = standards[i:i + chunk_size]
            try:
                # Force fallback to basic method
                chunk_categorizations = self._fallback_basic_categorization(chunk, hierarchy)
                all_categorizations.extend(chunk_categorizations)
            except Exception as e:
                logger.error(f"Failed to process basic chunk: {e}")
                continue
        
        return all_categorizations
    
    def is_enhanced_mode_available(self) -> bool:
        """Check if enhanced mode features are available.
        
        Returns:
            True if enhanced features (Responses API, pre-analysis) are available
        """
        return (
            self.client is not None and 
            hasattr(self.client, 'responses') and
            self.token_counter is not None
        )
    
    def get_compatibility_info(self) -> Dict[str, Any]:
        """Get information about feature compatibility and availability.
        
        Returns:
            Dictionary with compatibility information
        """
        info = {
            'version': '2.0-enhanced',
            'backward_compatible': True,
            'enhanced_features_available': self.is_enhanced_mode_available(),
            'features': {
                'basic_categorization': True,
                'topic_hierarchy_generation': bool(self.client),
                'enhanced_chunking': self.is_enhanced_mode_available(),
                'pre_analysis': self.is_enhanced_mode_available(),
                'quality_validation': self.is_enhanced_mode_available(),
                'responses_api': bool(self.client and hasattr(self.client, 'responses')),
                'educational_context_awareness': True,
                'confidence_scoring': self.is_enhanced_mode_available()
            },
            'models': {
                'default_model': self.DEFAULT_MODEL,
                'supports_large_context': self.DEFAULT_MODEL in ['gpt-4.1', 'gpt-4.1-mini'],
                'token_counter_model': getattr(self.token_counter, 'model_name', 'unknown') if self.token_counter else None
            },
            'chunking': {
                'default_chunk_size': self.DEFAULT_CHUNK_SIZE,
                'supports_dynamic_sizing': bool(self.token_counter),
                'supports_educational_chunking': True
            }
        }
        
        logger.info(f"🔍 Compatibility Info: Enhanced features {'available' if info['enhanced_features_available'] else 'not available'}")
        return info
    
    def migrate_from_basic_usage(self, 
                               standards: List[Standard],
                               enable_enhanced_features: bool = True,
                               subject_area_name: str = None) -> Dict[str, Any]:
        """Helper method to migrate from basic usage to enhanced features.
        
        This method helps existing code transition to enhanced features gradually.
        
        Args:
            standards: Standards to process
            enable_enhanced_features: Whether to enable enhanced features
            subject_area_name: Optional subject area for context
            
        Returns:
            Dictionary with migration results and recommendations
        """
        logger.info(f"🚀 Migration helper: Processing {len(standards)} standards (enhanced: {enable_enhanced_features})")
        
        migration_result = {
            'success': False,
            'hierarchy': None,
            'proxies': [],
            'categorizations': [],
            'enhanced_features_used': [],
            'recommendations': [],
            'performance_metrics': {}
        }
        
        try:
            start_time = time.time()
            
            if enable_enhanced_features and self.is_enhanced_mode_available():
                # Use full enhanced pipeline
                hierarchy, proxies = self.run_full_categorization(
                    grade_levels=None,
                    subject_area_id=None,
                    use_dynamic_chunk_size=True
                )
                migration_result['enhanced_features_used'] = [
                    'educational_chunking', 'pre_analysis', 'quality_validation', 'responses_api'
                ]
                migration_result['recommendations'].append(
                    "Enhanced features successfully enabled. Consider using grade_levels and subject_area_id filters for better results."
                )
            else:
                # Use simple compatibility mode
                hierarchy, proxies = self.simple_categorize_standards(standards, subject_area_name)
                migration_result['recommendations'].append(
                    "Used basic compatibility mode. Consider upgrading OpenAI client for enhanced features."
                )
                
                if not self.client:
                    migration_result['recommendations'].append(
                        "OpenAI client not available. Set OPENAI_API_KEY environment variable."
                    )
            
            # Collect all categorizations for analysis
            all_categorizations = []
            for proxy in proxies:
                for standard in proxy.member_standards.all():
                    all_categorizations.append(StandardCategorization(
                        standard_id=standard.code,
                        standard_obj=standard,
                        topic=proxy.topic,
                        sub_topic=proxy.sub_topic,
                        sub_sub_topic=proxy.sub_sub_topic,
                        is_outlier=proxy.outlier_category
                    ))
            
            migration_result.update({
                'success': True,
                'hierarchy': hierarchy,
                'proxies': proxies,
                'categorizations': all_categorizations,
                'performance_metrics': {
                    'processing_time_seconds': time.time() - start_time,
                    'standards_processed': len(standards),
                    'proxies_created': len(proxies),
                    'categorizations_created': len(all_categorizations)
                }
            })
            
            # Add performance recommendations
            processing_time = migration_result['performance_metrics']['processing_time_seconds']
            if processing_time > 60:
                migration_result['recommendations'].append(
                    f"Processing took {processing_time:.1f}s. Consider using smaller chunks or filtering by grade/subject."
                )
            
            logger.info(f"✅ Migration completed successfully in {processing_time:.1f}s")
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            migration_result['error'] = str(e)
            migration_result['recommendations'].append(
                f"Migration failed: {e}. Try using simple_categorize_standards() method instead."
            )
        
        return migration_result
    
    def validate_backward_compatibility(self) -> Dict[str, Any]:
        """Validate that backward compatibility is maintained.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'compatible': True,
            'issues': [],
            'warnings': [],
            'method_availability': {}
        }
        
        # Check essential methods are available
        essential_methods = [
            'load_standards',
            'generate_topic_hierarchy', 
            'categorize_standards_chunk',
            'create_topic_proxies',
            'run_full_categorization'
        ]
        
        for method_name in essential_methods:
            if hasattr(self, method_name):
                validation['method_availability'][method_name] = 'available'
            else:
                validation['method_availability'][method_name] = 'missing'
                validation['issues'].append(f"Essential method {method_name} not available")
                validation['compatible'] = False
        
        # Check optional enhanced methods
        enhanced_methods = [
            'simple_categorize_standards',
            'get_basic_categorizations',
            'create_educational_chunks'
        ]
        
        for method_name in enhanced_methods:
            if hasattr(self, method_name):
                validation['method_availability'][method_name] = 'available'
            else:
                validation['method_availability'][method_name] = 'missing'
                validation['warnings'].append(f"Enhanced method {method_name} not available")
        
        # Check client availability
        if not self.client:
            validation['warnings'].append("OpenAI client not available - limited functionality")
        
        # Check token counter
        if not self.token_counter:
            validation['warnings'].append("Token counter not available - will use default chunk sizes")
        
        logger.info(f"🔍 Backward compatibility validation: {'✅ Compatible' if validation['compatible'] else '❌ Issues found'}")
        if validation['issues']:
            logger.warning(f"⚠️ Compatibility issues: {validation['issues']}")
        if validation['warnings']:
            logger.info(f"ℹ️ Compatibility warnings: {validation['warnings']}")
        
        return validation
    
    def _process_chunk_with_retry(self, chunk: List[Standard], hierarchy: TopicHierarchy, 
                                chunk_num: int, max_retries: int = 3) -> List[StandardCategorization]:
        """
        Process a chunk with retry logic and exponential backoff.
        
        Args:
            chunk: Standards to process
            hierarchy: Topic hierarchy
            chunk_num: Chunk number for logging
            max_retries: Maximum retry attempts
            
        Returns:
            List of categorizations
        """
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    # Exponential backoff: 1s, 2s, 4s, etc.
                    wait_time = 2 ** attempt
                    logger.info(f"⏳ Retrying chunk {chunk_num} after {wait_time}s delay (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                
                categorizations = self.categorize_standards_chunk(chunk, hierarchy)
                
                # Validate the results have reasonable coverage
                if len(categorizations) < len(chunk) * 0.5:  # At least 50% should be categorized
                    logger.warning(f"⚠️ Low categorization rate for chunk {chunk_num}: {len(categorizations)}/{len(chunk)}")
                
                return categorizations
                
            except Exception as e:
                logger.warning(f"⚠️ Attempt {attempt + 1} failed for chunk {chunk_num}: {e}")
                if attempt == max_retries - 1:
                    raise  # Re-raise on final attempt
        
        return []  # Should never reach here
    
    def _categorize_single_standard(self, standard: Standard, hierarchy: TopicHierarchy) -> Optional[StandardCategorization]:
        """
        Categorize a single standard using a simplified approach.
        
        Args:
            standard: Standard to categorize
            hierarchy: Topic hierarchy
            
        Returns:
            Categorization result or None if failed
        """
        try:
            # Use the fallback basic categorization for single standards
            result = self._fallback_basic_categorization([standard], hierarchy)
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Failed to categorize single standard {standard.code}: {e}")
            
            # Create a generic outlier categorization as last resort
            return StandardCategorization(
                standard_id=standard.code,
                standard_obj=standard,
                topic="Outliers",
                sub_topic="Processing Failed",
                sub_sub_topic=f"Failed to categorize: {str(e)[:100]}",
                is_outlier=True,
                confidence_score=0.1,
                outlier_reason=f"Processing error: {str(e)}",
                outlier_complexity="unclear"
            )
    
    def _validate_processing_completeness(self, input_standards: List[Standard], 
                                        categorizations: List[StandardCategorization],
                                        processing_state: Dict[str, Any]) -> None:
        """
        Validate that all input standards were processed and provide comprehensive reporting.
        
        Args:
            input_standards: Original input standards
            categorizations: Generated categorizations
            processing_state: Processing state tracking
        """
        input_ids = {std.code for std in input_standards}
        categorized_ids = {cat.standard_id for cat in categorizations}
        
        missing_ids = input_ids - categorized_ids
        extra_ids = categorized_ids - input_ids
        
        # Log comprehensive processing summary
        logger.info("📋 Processing Completeness Report:")
        logger.info(f"  📥 Input standards: {len(input_standards):,}")
        logger.info(f"  📤 Generated categorizations: {len(categorizations):,}")
        logger.info(f"  ✅ Successfully processed: {len(categorized_ids):,}")
        logger.info(f"  ❌ Failed to process: {len(missing_ids):,}")
        logger.info(f"  🔍 Extra categorizations: {len(extra_ids):,}")
        logger.info(f"  📊 Success rate: {len(categorized_ids)/len(input_standards)*100:.1f}%")
        logger.info(f"  🧩 Chunks processed: {processing_state['chunks_processed']}")
        logger.info(f"  💥 Chunks failed: {processing_state['chunks_failed']}")
        
        if missing_ids:
            logger.error(f"❌ Missing standards: {sorted(list(missing_ids))[:10]}{'...' if len(missing_ids) > 10 else ''}")
            
            # Create placeholder categorizations for missing standards
            logger.info(f"🔧 Creating placeholder categorizations for {len(missing_ids)} missing standards")
            for std in input_standards:
                if std.code in missing_ids:
                    placeholder = StandardCategorization(
                        standard_id=std.code,
                        standard_obj=std,
                        topic="Outliers",
                        sub_topic="Processing Incomplete",
                        sub_sub_topic="Standard not processed due to system errors",
                        is_outlier=True,
                        confidence_score=0.0,
                        outlier_reason="Failed to process during categorization",
                        outlier_complexity="unclear"
                    )
                    categorizations.append(placeholder)
        
        if extra_ids:
            logger.warning(f"⚠️ Extra categorizations found: {sorted(list(extra_ids))[:5]}{'...' if len(extra_ids) > 5 else ''}")
        
        # Final validation
        final_completeness = len(categorized_ids) / len(input_standards) * 100
        if final_completeness < 95:
            logger.error(f"❌ Low processing completeness: {final_completeness:.1f}% - investigate system issues")
        else:
            logger.info(f"✅ Processing completeness acceptable: {final_completeness:.1f}%")