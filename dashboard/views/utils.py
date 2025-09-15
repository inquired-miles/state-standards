"""
Utility functions for dashboard calculations and data processing
"""
from typing import Dict, List, Any, Set
from django.db.models import Q, Avg, Count
from django.core.cache import cache
import logging

from standards.models import (
    State, Standard, StandardCorrelation, Concept, TopicCluster
)

logger = logging.getLogger(__name__)


def calculate_state_groupings() -> Dict[str, Any]:
    """
    Calculate state similarity groupings based on correlations
    Returns dictionary with groupings and unique states
    """
    try:
        # Get all states with standards
        states = list(State.objects.filter(standards__isnull=False).distinct())
        state_codes = [state.code for state in states]
        
        if not states:
            return {'groupings': [], 'unique_states': []}
        
        # Calculate similarity matrix between states based on correlations
        similarity_matrix = {}
        
        for state1 in states:
            similarity_matrix[state1.code] = {}
            for state2 in states:
                if state1.code == state2.code:
                    similarity_matrix[state1.code][state2.code] = 1.0
                else:
                    # Calculate similarity based on shared correlations
                    similarity = _calculate_state_similarity(state1, state2)
                    similarity_matrix[state1.code][state2.code] = similarity
        
        # Find clusters of similar states
        groupings = []
        processed_states: Set[str] = set()
        
        for state_code in state_codes:
            if state_code in processed_states:
                continue
                
            # Find similar states (similarity > 0.8)
            similar_states = [state_code]
            for other_state, similarity in similarity_matrix[state_code].items():
                if (similarity > 0.8 and other_state != state_code 
                    and other_state not in processed_states):
                    similar_states.append(other_state)
            
            if len(similar_states) > 1:
                # Calculate average similarity for the group
                total_similarity = 0
                count = 0
                for s1 in similar_states:
                    for s2 in similar_states:
                        if s1 != s2:
                            total_similarity += similarity_matrix[s1][s2]
                            count += 1
                
                avg_similarity = (total_similarity / count) * 100 if count > 0 else 0
                
                groupings.append({
                    'name': f"Group {len(groupings) + 1}",
                    'states': similar_states,
                    'similarity_percentage': round(avg_similarity, 1),
                    'state_count': len(similar_states),
                    'region': _determine_region(similar_states)
                })
                
                processed_states.update(similar_states)
        
        # Find unique states (not in any group)
        unique_states = [state for state in state_codes if state not in processed_states]
        
        return {
            'groupings': groupings,
            'unique_states': unique_states
        }
    
    except Exception as e:
        logger.error(f"Error calculating state groupings: {e}")
        return {'groupings': [], 'unique_states': []}


def _calculate_state_similarity(state1: State, state2: State) -> float:
    """Calculate similarity between two states based on correlations"""
    try:
        # Find correlations between the states
        correlations = StandardCorrelation.objects.filter(
            Q(standard_1__state=state1, standard_2__state=state2) |
            Q(standard_1__state=state2, standard_2__state=state1)
        )
        
        if not correlations.exists():
            return 0.0
        
        # Calculate average similarity score
        avg_similarity = correlations.aggregate(
            avg_score=Avg('similarity_score')
        )['avg_score'] or 0.0
        
        return min(1.0, max(0.0, avg_similarity))  # Clamp between 0 and 1
    
    except Exception as e:
        logger.error(f"Error calculating similarity between {state1.code} and {state2.code}: {e}")
        return 0.0


def _determine_region(state_codes: List[str]) -> str:
    """Determine geographic region for a group of states"""
    # US Census Bureau regions
    regions = {
        'Northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
        'Southeast': ['DE', 'MD', 'VA', 'WV', 'KY', 'TN', 'NC', 'SC', 'GA', 'FL', 'AL', 'MS', 'AR', 'LA'],
        'Midwest': ['OH', 'MI', 'IN', 'IL', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
        'Southwest': ['TX', 'OK', 'NM', 'AZ'],
        'West': ['CO', 'WY', 'MT', 'ID', 'UT', 'NV', 'CA', 'OR', 'WA'],
        'Pacific': ['AK', 'HI']
    }
    
    # Count states per region
    region_counts = {}
    for region, states in regions.items():
        count = sum(1 for state in state_codes if state in states)
        if count > 0:
            region_counts[region] = count
    
    if not region_counts:
        return 'Mixed'
    
    # Return region with most states
    return max(region_counts.items(), key=lambda x: x[1])[0]


def calculate_topic_prevalence() -> Dict[str, Any]:
    """Calculate topic prevalence across states"""
    try:
        topics_data = []
        
        # Method 1: Use existing Concept model if available
        if Concept.objects.exists():
            concepts = Concept.objects.select_related().prefetch_related('subject_areas')
            for concept in concepts:
                # Calculate prevalence safely
                state_count = getattr(concept, 'states_covered', 0)
                prevalence = (state_count / 50) * 100 if state_count else 0
                
                # Get subject areas safely
                subject_areas = []
                if hasattr(concept, 'subject_areas'):
                    try:
                        subject_areas = list(concept.subject_areas.values_list('name', flat=True))
                    except:
                        subject_areas = []
                
                topics_data.append({
                    'name': concept.name,
                    'description': concept.description or '',
                    'state_count': state_count,
                    'prevalence_percentage': round(prevalence, 1),
                    'category': _categorize_topic_importance(prevalence),
                    'subject_areas': subject_areas
                })
        
        # Method 2: Extract from TopicCluster model
        clusters = TopicCluster.objects.select_related('subject_area')
        for cluster in clusters:
            state_count = getattr(cluster, 'states_represented', 0)
            prevalence = (state_count / 50) * 100 if state_count else 0
            
            topics_data.append({
                'name': cluster.name,
                'description': cluster.description or '',
                'state_count': state_count,
                'prevalence_percentage': round(prevalence, 1),
                'category': _categorize_topic_importance(prevalence),
                'subject_areas': [cluster.subject_area.name] if cluster.subject_area else []
            })
        
        # Remove duplicates and sort by prevalence
        seen_names = set()
        unique_topics = []
        for topic in topics_data:
            if topic['name'] not in seen_names:
                unique_topics.append(topic)
                seen_names.add(topic['name'])
        
        unique_topics.sort(key=lambda x: x['prevalence_percentage'], reverse=True)
        
        return {'topics': unique_topics}
    
    except Exception as e:
        logger.error(f"Error calculating topic prevalence: {e}")
        return {'topics': []}


def _categorize_topic_importance(prevalence: float) -> str:
    """Categorize topic importance based on prevalence percentage"""
    if prevalence >= 90:
        return 'must-have'
    elif prevalence >= 60:
        return 'important'
    elif prevalence >= 30:
        return 'regional'
    else:
        return 'specialized'


def _categorize_theme_importance(percentage: float) -> str:
    """Categorize theme importance based on adoption percentage"""
    if percentage >= 80:
        return 'universal'
    elif percentage >= 60:
        return 'widespread'
    elif percentage >= 40:
        return 'common'
    elif percentage >= 20:
        return 'regional'
    else:
        return 'specialized'


def extract_concepts_from_content(content: str) -> List[str]:
    """
    Extract educational concepts from content using keyword matching
    This is a simplified approach - could be enhanced with NLP
    """
    educational_keywords = [
        'fraction', 'algebra', 'geometry', 'statistics', 'measurement',
        'reading', 'writing', 'literature', 'grammar', 'vocabulary',
        'science', 'physics', 'chemistry', 'biology', 'earth science',
        'history', 'geography', 'civics', 'government', 'economics',
        'maps', 'timeline', 'democracy', 'constitution', 'community'
    ]
    
    content_lower = content.lower()
    found_concepts = []
    
    for keyword in educational_keywords:
        if keyword in content_lower:
            found_concepts.append(keyword)
    
    return found_concepts[:10]  # Limit to top 10 concepts