"""
Secure API views for embeddings visualization with proper validation and caching
"""
from typing import Dict, List, Optional, Any
from django.http import JsonResponse
from django.core.exceptions import ValidationError
from django.db.models import Q, Avg
from django.core.cache import cache
import logging
import numpy as np
import umap
import hdbscan
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

from standards.models import (
    Standard, State, TopicCluster, StandardCorrelation, SubjectArea
)
from standards.services.search import SearchService
from .base import BaseAPIView, api_endpoint
from .utils import _categorize_theme_importance

logger = logging.getLogger(__name__)


class EmbeddingsAPIView(BaseAPIView):
    """Base class for embeddings-related API views"""
    
    def validate_clustering_params(self, cluster_size: Any, epsilon: Any) -> tuple:
        """Validate clustering parameters"""
        cluster_size = self.validate_integer(cluster_size, 2, 100, "cluster_size")
        epsilon = self.validate_float(epsilon, 0.01, 2.0, "epsilon")
        return cluster_size, epsilon
    
    def get_cached_or_calculate(self, cache_key: str, calculation_func, ttl: int = 3600):
        """Get data from cache or calculate and cache it"""
        data = cache.get(cache_key)
        if data is None:
            logger.info(f"Cache MISS for key: {cache_key} - executing calculation function")
            try:
                data = calculation_func()
                cache.set(cache_key, data, ttl)
                logger.info(f"Cache SET for key: {cache_key} - data calculated and cached")
            except Exception as e:
                logger.error(f"Error calculating data for cache key {cache_key}: {e}")
                data = {}
        else:
            logger.info(f"Cache HIT for key: {cache_key} - returning cached data")
            # Check if cached data looks like it has problematic cluster counts
            if isinstance(data, dict) and 'clusters' in data:
                cluster_count = len(data.get('clusters', []))
                total_standards = data.get('total_standards', 0)
                if total_standards > 100 and cluster_count <= 2:
                    logger.warning(f"Cache contains stale clustering data: {cluster_count} clusters for {total_standards} standards - clearing cache")
                    cache.delete(cache_key)
                    logger.info(f"Cache CLEARED for key: {cache_key} - recalculating...")
                    return self.get_cached_or_calculate(cache_key, calculation_func, ttl)
        return data


@api_endpoint(['GET'])
def embeddings_visualization_data_api(request):
    """API endpoint for scatter plot and cluster data with comprehensive validation"""
    view = EmbeddingsAPIView()
    
    try:
        # Validate parameters
        grade_level = request.GET.get('grade_level')
        if grade_level and grade_level.strip():
            grade_level = view.validate_integer(grade_level, 0, 12, "grade_level")
        else:
            grade_level = None
        
        subject_area_id = request.GET.get('subject_area')
        if subject_area_id:
            subject_area_id = view.validate_integer(subject_area_id, 1, None, "subject_area")
            # Verify subject area exists
            try:
                SubjectArea.objects.get(id=subject_area_id)
            except SubjectArea.DoesNotExist:
                return view.error_response(f"Subject area {subject_area_id} not found", status=404)
        
        cluster_size, epsilon = view.validate_clustering_params(
            request.GET.get('cluster_size', 5),
            request.GET.get('epsilon', 0.5)
        )
        
        # Get visualization mode (2d or 3d)
        viz_mode = request.GET.get('viz_mode', '2d')
        if viz_mode not in ['2d', '3d']:
            viz_mode = '2d'
        
        # Create cache key for this specific request
        cache_key = f"embeddings_viz_{grade_level}_{subject_area_id}_{cluster_size}_{epsilon}_{viz_mode}"
        logger.info(f"Cache key: {cache_key} (cluster_size={cluster_size}, epsilon={epsilon})")
        
        def calculate_visualization_data():
            # Filter standards with proper query optimization
            standards_query = Standard.objects.filter(
                embedding__isnull=False
            ).select_related('state', 'subject_area')
            
            if grade_level is not None:
                standards_query = standards_query.filter(
                    grade_levels__grade_numeric=grade_level
                ).distinct()
            
            if subject_area_id:
                standards_query = standards_query.filter(subject_area_id=subject_area_id)
            
            # Limit for performance and memory management
            standards = list(standards_query[:2000])  # Increased limit but still reasonable
            
            if not standards:
                # Provide detailed information about why no data was found
                criteria_info = []
                if grade_level is not None:
                    criteria_info.append(f"grade level {grade_level}")
                if subject_area_id:
                    try:
                        from standards.models import SubjectArea
                        subject_area = SubjectArea.objects.get(id=subject_area_id)
                        criteria_info.append(f"subject area '{subject_area.name}'")
                    except:
                        criteria_info.append(f"subject area ID {subject_area_id}")
                
                criteria_text = " and ".join(criteria_info) if criteria_info else "your filter criteria"
                
                # Check if standards exist without embeddings for better debugging
                standards_without_embeddings_query = Standard.objects.all()
                if grade_level is not None:
                    standards_without_embeddings_query = standards_without_embeddings_query.filter(
                        grade_levels__grade_numeric=grade_level
                    ).distinct()
                if subject_area_id:
                    standards_without_embeddings_query = standards_without_embeddings_query.filter(
                        subject_area_id=subject_area_id
                    )
                
                total_matching_standards = standards_without_embeddings_query.count()
                standards_with_embeddings_count = standards_without_embeddings_query.filter(
                    embedding__isnull=False
                ).count()
                
                # Provide specific guidance based on the situation
                if total_matching_standards == 0:
                    message = f'No standards found for {criteria_text}. Try different filter criteria.'
                elif standards_with_embeddings_count == 0:
                    message = f'Found {total_matching_standards} standards for {criteria_text}, but none have embeddings. Run the generate_embeddings management command to create embeddings.'
                else:
                    message = f'No standards with embeddings found for {criteria_text}'
                
                return {
                    'scatter_data': [],
                    'clusters': [],
                    'state_colors': {},
                    'total_standards': 0,
                    'message': message,
                    'debug_info': {
                        'total_standards_in_db': Standard.objects.count(),
                        'standards_with_embeddings': Standard.objects.filter(embedding__isnull=False).count(),
                        'standards_matching_criteria': total_matching_standards,
                        'standards_matching_criteria_with_embeddings': standards_with_embeddings_count,
                        'filters_applied': {
                            'grade_level': grade_level,
                            'subject_area_id': subject_area_id
                        }
                    }
                }
            
            # Prepare color palette
            color_palette = [
                '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
                '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43',
                '#48CAE4', '#023E8A', '#FFB3BA', '#B5EAD7', '#C7CEEA'
            ]
            
            # Build state color mapping
            state_colors = {}
            scatter_data = []
            
            for standard in standards:
                state_code = standard.state.code if standard.state else 'Unknown'
                
                if state_code not in state_colors:
                    color_index = len(state_colors) % len(color_palette)
                    state_colors[state_code] = color_palette[color_index]
                
            # Prepare embeddings for UMAP and clustering
            embeddings_matrix = []
            valid_standards = []
            
            for standard in standards:
                try:
                    embedding = standard.embedding
                    if embedding is not None and len(embedding) >= 2:
                        try:
                            # Convert to numpy array if needed
                            if isinstance(embedding, list):
                                embedding_array = np.array(embedding, dtype=np.float32)
                            else:
                                embedding_array = np.array(embedding, dtype=np.float32)
                            
                            embeddings_matrix.append(embedding_array)
                            valid_standards.append(standard)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Invalid embedding for standard {standard.id}: {e}")
                            continue
                except Exception as e:
                    logger.error(f"Error processing standard {standard.id}: {e}")
                    continue
            
            if not embeddings_matrix:
                return {
                    'scatter_data': [],
                    'clusters': [],
                    'state_colors': state_colors,
                    'total_standards': 0,
                    'message': 'No valid embeddings found for visualization'
                }
            
            # Convert to numpy matrix
            embeddings_matrix = np.array(embeddings_matrix)
            logger.info(f"Processing {len(embeddings_matrix)} embeddings with shape {embeddings_matrix.shape}")
            
            # Perform UMAP dimensionality reduction (simplified)
            try:
                # Set n_components based on visualization mode
                n_components = 3 if viz_mode == '3d' else 2
                
                # Phase 1: Adaptive UMAP parameters for better clustering
                n_samples = len(embeddings_matrix)
                
                # Adaptive n_neighbors: smaller datasets need fewer neighbors for local structure
                # Formula: min(max(5, sqrt(n_samples)), 50) capped by dataset size
                import math
                adaptive_n_neighbors = min(max(5, int(math.sqrt(n_samples))), 50, n_samples - 1)
                
                # Adaptive min_dist: smaller datasets benefit from tighter clustering
                if n_samples < 100:
                    adaptive_min_dist = 0.01  # Tight clusters for small datasets
                elif n_samples < 500:
                    adaptive_min_dist = 0.05  # Balanced for medium datasets
                else:
                    adaptive_min_dist = 0.1   # Standard for large datasets
                
                logger.info(f"UMAP adaptive parameters: n_samples={n_samples}, n_neighbors={adaptive_n_neighbors}, min_dist={adaptive_min_dist}")
                
                umap_reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=adaptive_n_neighbors,
                    min_dist=adaptive_min_dist,
                    metric='cosine',
                    random_state=42
                )
                umap_embeddings = umap_reducer.fit_transform(embeddings_matrix)
                logger.info(f"UMAP completed successfully")
                
                # Basic check for embedding quality
                embedding_variance = np.var(umap_embeddings, axis=0).mean()
                logger.info(f"UMAP embedding variance: {embedding_variance:.4f}")
                
                if embedding_variance < 0.01:
                    logger.warning("Low UMAP embedding variance - may affect clustering quality")
                
            except Exception as e:
                logger.error(f"UMAP failed: {e}")
                # Fallback based on visualization mode
                if viz_mode == '3d':
                    if embeddings_matrix.shape[1] >= 3:
                        umap_embeddings = embeddings_matrix[:, :3]
                    else:
                        # Pad with zeros if needed
                        umap_embeddings = np.pad(embeddings_matrix[:, :2], ((0, 0), (0, 1)), mode='constant')
                else:
                    umap_embeddings = embeddings_matrix[:, :2]
            
            # Perform HDBSCAN clustering with Phase 1 improvements
            try:
                # Use cluster_size directly as min_cluster_size (intuitive behavior)
                # Smaller cluster_size = smaller min_cluster_size = MORE clusters
                # Larger cluster_size = larger min_cluster_size = FEWER clusters
                min_cluster_size = max(2, cluster_size)  # Direct control, minimum of 2
                
                # Phase 1: Add min_samples parameter for better noise handling
                # min_samples controls how conservative the clustering is
                # Lower min_samples = more aggressive clustering (finds more clusters)
                min_samples = max(1, min_cluster_size // 2)
                
                logger.info(f"HDBSCAN parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}, epsilon={epsilon}")
                
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric='euclidean',
                    cluster_selection_epsilon=epsilon
                )
                cluster_labels = clusterer.fit_predict(umap_embeddings)
                
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                n_noise = (cluster_labels == -1).sum()
                n_total = len(embeddings_matrix)  # Add this back
                
                logger.info(f"HDBSCAN clustering: {n_clusters} clusters, {n_noise} noise points")
                
                # Log cluster size distribution for debugging
                from collections import Counter
                cluster_sizes = Counter(cluster_labels)
                non_noise_sizes = [size for label, size in cluster_sizes.items() if label != -1]
                if non_noise_sizes:
                    logger.info(f"Cluster sizes: min={min(non_noise_sizes)}, max={max(non_noise_sizes)}, avg={sum(non_noise_sizes)/len(non_noise_sizes):.1f}")
                    
                    # Check for clustering quality issues
                    max_cluster_size = max(non_noise_sizes)
                    if max_cluster_size > n_total * 0.7:
                        logger.warning(f"Dominant cluster detected: {max_cluster_size}/{n_total} ({max_cluster_size/n_total:.1%})")
                else:
                    logger.warning("No valid clusters found - all points are noise")
                
                # Enhanced retry logic with min_samples adjustment
                if n_clusters == 0 and len(embeddings_matrix) > 10:
                    logger.warning(f"Got zero clusters, retrying with more permissive settings")
                    
                    # Phase 1: Enhanced retry with both min_cluster_size and min_samples adjustment
                    retry_min_size = max(2, min_cluster_size // 2)
                    retry_min_samples = max(1, retry_min_size // 2)  # Adjust min_samples too
                    retry_epsilon = min(1.0, epsilon * 1.5)  # More permissive
                    
                    logger.info(f"Retry with min_cluster_size={retry_min_size}, min_samples={retry_min_samples}, epsilon={retry_epsilon}")
                    
                    retry_clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=retry_min_size,
                        min_samples=retry_min_samples,
                        metric='euclidean',
                        cluster_selection_epsilon=retry_epsilon
                    )
                    retry_labels = retry_clusterer.fit_predict(umap_embeddings)
                    retry_n_clusters = len(set(retry_labels)) - (1 if -1 in retry_labels else 0)
                    
                    logger.info(f"Retry result: {retry_n_clusters} clusters")
                    
                    # Use retry result if we got any clusters
                    if retry_n_clusters > 0:
                        cluster_labels = retry_labels
                        logger.info(f"Using retry clustering with {retry_n_clusters} clusters")
                    else:
                        logger.warning("Retry also failed, keeping original results")
                
            except Exception as e:
                logger.error(f"HDBSCAN failed: {e}")
                # Try KMeans as fallback clustering method
                logger.info("HDBSCAN failed completely, trying KMeans fallback")
                cluster_labels = _fallback_kmeans_clustering(umap_embeddings, len(embeddings_matrix), cluster_size, logger)
            
            # First, identify weak clusters (less than 3 members)
            from collections import Counter
            cluster_sizes_check = Counter(cluster_labels)
            weak_cluster_ids = {label for label, count in cluster_sizes_check.items() 
                               if label != -1 and count < 3}
            
            # Build scatter plot data with UMAP coordinates
            for i, standard in enumerate(valid_standards):
                state_code = standard.state.code if standard.state else 'Unknown'
                
                if state_code not in state_colors:
                    color_index = len(state_colors) % len(color_palette)
                    state_colors[state_code] = color_palette[color_index]
                
                # Use description as title if title is empty
                display_title = standard.title or standard.description or f"Standard {str(standard.id)[:8]}"
                if len(display_title) > 80:
                    display_title = display_title[:80] + '...'
                
                # Mark weak clusters as -1 (unclustered) for visual consistency
                cluster_label = cluster_labels[i]
                if cluster_label in weak_cluster_ids:
                    cluster_label = -1  # Treat weak clusters as unclustered visually
                
                # Build data point with coordinates based on visualization mode
                data_point = {
                    'x': float(umap_embeddings[i, 0]),
                    'y': float(umap_embeddings[i, 1]),
                    'state': state_code,
                    'title': display_title,
                    'color': state_colors[state_code],
                    'id': str(standard.id),
                    'subject': standard.subject_area.name if standard.subject_area else 'Unknown',
                    'cluster': int(cluster_label) if cluster_label != -1 else -1
                }
                
                # Add z-coordinate for 3D visualization
                if viz_mode == '3d' and umap_embeddings.shape[1] >= 3:
                    data_point['z'] = float(umap_embeddings[i, 2])
                
                scatter_data.append(data_point)
            
            # Generate real clusters from HDBSCAN results
            clusters = _generate_real_clusters(valid_standards, umap_embeddings, cluster_labels, state_colors, viz_mode)
            
            # Calculate meaningful clustering statistics
            from collections import Counter
            cluster_sizes = Counter(cluster_labels)
            
            # Log cluster distribution for debugging
            logger.info(f"Cluster distribution: {dict(cluster_sizes)}")
            
            # Define meaningful clusters (3+ members for visual significance)
            MIN_MEANINGFUL_CLUSTER_SIZE = 3
            
            # Count standards in meaningful clusters
            meaningful_clustered = 0
            weak_clustered = 0
            for label, count in cluster_sizes.items():
                if label != -1:  # Not noise
                    if count >= MIN_MEANINGFUL_CLUSTER_SIZE:
                        meaningful_clustered += count
                    else:
                        weak_clustered += count
            
            # Noise points (-1) plus weak clusters are considered unclustered
            noise_count = cluster_sizes.get(-1, 0)
            effectively_unclustered = noise_count + weak_clustered
            
            logger.info(f"Clustering analysis: {meaningful_clustered} in strong clusters, "
                       f"{weak_clustered} in weak clusters, {noise_count} noise points")
            
            return {
                'scatter_data': scatter_data,
                'clusters': clusters,
                'state_colors': state_colors,
                'total_standards': len(scatter_data),
                'clustered_standards': meaningful_clustered,  # Only count meaningful clusters
                'unclustered_standards': effectively_unclustered,  # Include weak clusters as unclustered
                'clustering_stats': {
                    'total_standards': len(scatter_data),
                    'clustered': meaningful_clustered,  # Standards in clusters with 3+ members
                    'unclustered': effectively_unclustered,  # Noise + weak clusters
                    'clustering_rate': round((meaningful_clustered / len(scatter_data)) * 100, 1) if scatter_data else 0,
                    'num_clusters': len([c for c in clusters if len(c.get('standards', [])) >= MIN_MEANINGFUL_CLUSTER_SIZE]),
                    'weak_clusters': weak_clustered,  # Additional stat for transparency
                    'noise_points': noise_count  # Pure noise points
                }
            }
        
        # Cache enabled with auto-clearing of stale data
        visualization_data = view.get_cached_or_calculate(cache_key, calculate_visualization_data)
        
        # Add parameters to response
        visualization_data['parameters'] = {
            'grade_level': grade_level,
            'subject_area': subject_area_id,
            'cluster_size': cluster_size,
            'epsilon': epsilon,
            'viz_mode': viz_mode
        }
        
        return view.success_response(visualization_data)
        
    except ValidationError as e:
        return view.validation_error_response(str(e))
    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['GET'])
def embeddings_similarity_matrix_api(request):
    """API endpoint for state-to-state similarity matrix with caching"""
    view = EmbeddingsAPIView()
    
    try:
        # Validate parameters
        grade_level = request.GET.get('grade_level')
        if grade_level and grade_level.strip():
            grade_level = view.validate_integer(grade_level, 0, 12, "grade_level")
        else:
            grade_level = None
        
        subject_area_id = request.GET.get('subject_area')
        if subject_area_id:
            subject_area_id = view.validate_integer(subject_area_id, 1, None, "subject_area")
        
        # Create cache key
        cache_key = f"similarity_matrix_{grade_level}_{subject_area_id}"
        
        def calculate_similarity_matrix():
            # Get states that have standards matching criteria
            states_query = State.objects.filter(
                standards__embedding__isnull=False
            ).distinct()
            
            if grade_level is not None:
                states_query = states_query.filter(
                    standards__grade_levels__grade_numeric=grade_level
                )
            
            if subject_area_id:
                states_query = states_query.filter(
                    standards__subject_area_id=subject_area_id
                )
            
            states = list(states_query.order_by('code'))
            
            if not states:
                return {
                    'states': [],
                    'similarity_matrix': [],
                    'error': 'No states found with standards matching the criteria'
                }
            
            # Calculate similarity matrix with performance optimization
            similarity_matrix = []
            state_codes = [state.code for state in states]
            
            # Use batch queries for better performance
            correlations_cache = {}
            all_correlations = StandardCorrelation.objects.filter(
                Q(standard_1__state__in=states) | Q(standard_2__state__in=states)
            ).select_related('standard_1__state', 'standard_2__state')
            
            # Build correlation lookup
            for corr in all_correlations:
                state1_code = corr.standard_1.state.code
                state2_code = corr.standard_2.state.code
                key = tuple(sorted([state1_code, state2_code]))
                
                if key not in correlations_cache:
                    correlations_cache[key] = []
                correlations_cache[key].append(corr.similarity_score)
            
            # Build matrix
            for state1 in states:
                row = []
                for state2 in states:
                    if state1.code == state2.code:
                        similarity = 1.0
                    else:
                        key = tuple(sorted([state1.code, state2.code]))
                        scores = correlations_cache.get(key, [])
                        
                        if scores:
                            similarity = sum(scores) / len(scores)
                        else:
                            similarity = 0.0
                    
                    row.append(round(similarity, 3))
                similarity_matrix.append(row)
            
            return {
                'states': state_codes,
                'similarity_matrix': similarity_matrix,
                'total_states': len(state_codes)
            }
        
        matrix_data = view.get_cached_or_calculate(cache_key, calculate_similarity_matrix)
        
        # Add parameters to response
        matrix_data['parameters'] = {
            'grade_level': grade_level,
            'subject_area': subject_area_id
        }
        
        return view.success_response(matrix_data)
        
    except ValidationError as e:
        return view.validation_error_response(str(e))
    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['GET'])
def embeddings_network_graph_api(request):
    """API endpoint for network graph showing concept relationships and state implementations"""
    view = EmbeddingsAPIView()
    
    try:
        # Validate parameters
        grade_level = request.GET.get('grade_level')
        if grade_level and grade_level.strip():
            grade_level = view.validate_integer(grade_level, 0, 12, "grade_level")
        else:
            grade_level = None
        
        subject_area_id = request.GET.get('subject_area')
        if subject_area_id:
            subject_area_id = view.validate_integer(subject_area_id, 1, None, "subject_area")
        
        # Get clustering parameters to sync with scatter plot
        cluster_size, epsilon = view.validate_clustering_params(
            request.GET.get('cluster_size', 5),
            request.GET.get('epsilon', 0.5)
        )
        
        # Get display parameters
        max_standards = int(request.GET.get('max_standards', 500))
        standards_per_cluster = int(request.GET.get('standards_per_cluster', 25))
        secondary_threshold = float(request.GET.get('secondary_threshold', 0.5))
        
        # Create cache key including clustering parameters
        cache_key = f"network_graph_{grade_level}_{subject_area_id}_{cluster_size}_{epsilon}"
        
        def calculate_network_graph():
            # Get standards with embeddings for analysis
            standards_query = Standard.objects.filter(
                embedding__isnull=False
            ).select_related('state', 'subject_area')
            
            if grade_level is not None:
                standards_query = standards_query.filter(
                    grade_levels__grade_numeric=grade_level
                ).distinct()
            
            if subject_area_id:
                standards_query = standards_query.filter(subject_area_id=subject_area_id)
            
            # Use the max_standards parameter from outer scope
            standards = list(standards_query[:max_standards])
            
            if len(standards) < 10:
                return {
                    'nodes': [],
                    'edges': [],
                    'error': 'Not enough standards for network analysis'
                }
            
            # Prepare embeddings for clustering
            embeddings_matrix = []
            valid_standards = []
            
            for standard in standards:
                try:
                    embedding = standard.embedding
                    if embedding is not None and len(embedding) >= 2:
                        if isinstance(embedding, list):
                            embedding_array = np.array(embedding, dtype=np.float32)
                        else:
                            embedding_array = np.array(embedding, dtype=np.float32)
                        
                        embeddings_matrix.append(embedding_array)
                        valid_standards.append(standard)
                except Exception:
                    continue
            
            if len(embeddings_matrix) < 10:
                return {
                    'nodes': [],
                    'edges': [],
                    'error': 'Not enough valid embeddings for clustering'
                }
            
            embeddings_matrix = np.array(embeddings_matrix)
            
            # Perform clustering to identify concepts
            try:
                import umap
                import hdbscan
                from sklearn.metrics.pairwise import cosine_similarity
                
                # UMAP for dimensionality reduction
                n_samples = len(embeddings_matrix)
                import math
                adaptive_n_neighbors = min(max(5, int(math.sqrt(n_samples))), 30, n_samples - 1)
                
                umap_reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=adaptive_n_neighbors,
                    min_dist=0.1,
                    metric='cosine',
                    random_state=42
                )
                umap_embeddings = umap_reducer.fit_transform(embeddings_matrix)
                
                # HDBSCAN clustering to identify concepts - use same params as scatter plot
                # Use cluster_size directly as min_cluster_size for consistency
                min_cluster_size = max(2, cluster_size)  # Same as scatter plot
                min_samples = max(1, min_cluster_size // 2)  # Same as scatter plot
                
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric='euclidean',
                    cluster_selection_epsilon=epsilon  # Use epsilon from request
                )
                cluster_labels = clusterer.fit_predict(umap_embeddings)
                
                # Build enhanced network graph with core vs state-specific analysis
                nodes = []
                edges = []
                
                # Get all states in the database for proper coverage calculation
                # We want to know coverage across ALL states, not just ones in current query
                from standards.models import State
                total_states_in_db = State.objects.filter(
                    standards__embedding__isnull=False
                ).distinct().count()
                
                # Use a reasonable minimum (10 states) to avoid inflated coverage percentages
                total_states = max(total_states_in_db, 10)
                
                # Track states actually in current dataset
                all_states = set(std.state.code for std in valid_standards if std.state)
                
                # Get unique clusters (excluding noise -1)
                unique_clusters = [int(c) for c in set(cluster_labels) if c != -1]
                
                # Analyze each cluster for state coverage
                cluster_analysis = {}
                
                for cluster_id in unique_clusters:
                    cluster_mask = cluster_labels == cluster_id
                    cluster_standards = [valid_standards[i] for i in range(len(valid_standards)) if cluster_mask[i]]
                    
                    if len(cluster_standards) < 2:
                        continue
                    
                    # Calculate state coverage for this cluster
                    states_in_cluster = set(std.state.code for std in cluster_standards if std.state)
                    coverage_percentage = len(states_in_cluster) / max(total_states, 1)
                    
                    # Categorize concept by actual state count (not percentage)
                    state_count = len(states_in_cluster)
                    
                    if state_count == 1:
                        concept_type = 'state_specific'
                        node_color = '#E67E22'  # Orange for state-specific (only 1 state)
                        importance = 1
                    elif state_count >= 5:  # Present in 5+ states
                        concept_type = 'common'
                        node_color = '#27AE60'  # Green for common concepts
                        importance = 3
                    else:  # Present in 2-4 states
                        concept_type = 'semi_common'
                        node_color = '#3498DB'  # Blue for semi-common concepts
                        importance = 2
                    
                    # Generate concept name and extract keywords
                    concept_name = _extract_cluster_theme(cluster_standards, cluster_id)
                    key_topics = _extract_key_topics(cluster_standards)[:8]  # Get top 8 keywords
                    
                    cluster_analysis[cluster_id] = {
                        'standards': cluster_standards,
                        'states': states_in_cluster,
                        'coverage': coverage_percentage,
                        'type': concept_type,
                        'name': concept_name,
                        'color': node_color,
                        'importance': importance,
                        'keywords': key_topics
                    }
                    
                    # Count how many standards will be shown vs total
                    total_standards_in_cluster = len(cluster_standards)
                    shown_standards = min(total_standards_in_cluster, standards_per_cluster)
                    
                    # Create concept node with enhanced metadata
                    nodes.append({
                        'id': f'concept_{cluster_id}',
                        'type': 'concept',
                        'concept_type': concept_type,
                        'label': concept_name,
                        'color': node_color,
                        'size': 15 + (coverage_percentage * 50),  # Size based on coverage
                        'cluster_size': int(len(cluster_standards)),
                        'total_standards': total_standards_in_cluster,
                        'shown_standards': shown_standards,
                        'state_coverage': int(len(states_in_cluster)),
                        'coverage_percentage': round(coverage_percentage * 100, 1),
                        'implementing_states': list(states_in_cluster),
                        'importance': int(importance),
                        'keywords': key_topics  # Add keywords for tooltip display
                    })
                
                # Add standard nodes and create multiple connections based on similarity
                standard_to_clusters = {}  # Track which clusters each standard belongs to
                standard_nodes_added = set()  # Track which standards we've already added as nodes
                
                # First pass: Add standard nodes with their primary cluster
                # Using standards_per_cluster from outer scope
                for cluster_id, analysis in cluster_analysis.items():
                    for standard in analysis['standards'][:standards_per_cluster]:
                        state_code = standard.state.code if standard.state else 'Unknown'
                        standard_id = f'std_{standard.id}'
                        
                        # Track cluster relationships
                        if standard_id not in standard_to_clusters:
                            standard_to_clusters[standard_id] = []
                        standard_to_clusters[standard_id].append({
                            'cluster_id': cluster_id,
                            'type': analysis['type'],
                            'importance': analysis['importance']
                        })
                        
                        # Add node only once (color based on individual standard's state coverage)
                        if standard_id not in standard_nodes_added:
                            # Check how many states this specific standard appears in
                            # For now, individual standards only appear in one state, so they're all state-specific
                            # Standards get clustered, but each individual standard is from one state
                            std_color = '#F0B27A'  # Light orange - all individual standards are state-specific
                            std_concept_type = 'state_specific'  # Individual standards are always state-specific
                            
                            nodes.append({
                                'id': standard_id,
                                'type': 'standard',
                                'label': (standard.title or standard.description or 'Standard')[:40] + '...',
                                'state': state_code,
                                'color': std_color,
                                'size': 8 + (analysis['importance'] * 2),
                                'concept_type': std_concept_type,  # Use standard-specific type
                                'cluster_id': cluster_id,
                                'cluster_type': analysis['type'],  # Keep cluster type for reference
                                'full_text': standard.description or standard.title or ''
                            })
                            standard_nodes_added.add(standard_id)
                
                # Second pass: Create primary edges from concept nodes to their cluster standards
                # This ensures each concept node is connected to all its standards
                for cluster_id, analysis in cluster_analysis.items():
                    concept_node_id = f'concept_{cluster_id}'
                    
                    # Connect each standard in this cluster to the concept node
                    for standard in analysis['standards'][:standards_per_cluster]:
                        standard_id = f'std_{standard.id}'
                        if standard_id in standard_nodes_added:
                            edges.append({
                                'source': concept_node_id,
                                'target': standard_id,
                                'type': 'implements',
                                'weight': 1.0,
                                'style': 'solid',
                                'color': '#95A5A6',
                                'width': 2
                            })
                
                # Third pass: Add secondary edges based on cross-cluster similarity (optional)
                # Calculate similarities between each standard and other cluster centroids
                cluster_centroids = {}
                for cluster_id, analysis in cluster_analysis.items():
                    # Calculate centroid embedding for this cluster
                    cluster_embeddings = []
                    for standard in analysis['standards']:
                        if hasattr(standard, 'embedding') and standard.embedding is not None:
                            try:
                                if isinstance(standard.embedding, list):
                                    cluster_embeddings.append(np.array(standard.embedding, dtype=np.float32))
                                else:
                                    cluster_embeddings.append(np.array(standard.embedding, dtype=np.float32))
                            except (ValueError, TypeError):
                                continue
                    
                    if cluster_embeddings:
                        cluster_centroids[cluster_id] = np.mean(cluster_embeddings, axis=0)
                
                # Create secondary edges for standards with high similarity to other clusters
                for cluster_id, analysis in cluster_analysis.items():
                    for standard in analysis['standards'][:standards_per_cluster]:
                        standard_id = f'std_{standard.id}'
                        if standard_id not in standard_nodes_added:
                            continue
                        
                        if hasattr(standard, 'embedding') and standard.embedding is not None:
                            try:
                                std_embedding = np.array(standard.embedding, dtype=np.float32)
                            except (ValueError, TypeError):
                                continue
                            
                            # Calculate similarity to other cluster centroids (not its primary cluster)
                            for other_cluster_id, centroid in cluster_centroids.items():
                                if other_cluster_id == cluster_id:  # Skip primary cluster
                                    continue
                                
                                similarity = float(cosine_similarity([std_embedding], [centroid])[0][0])
                                
                                # Connect to secondary clusters if similarity is high enough
                                # Using secondary_threshold from outer scope
                                if similarity > secondary_threshold:
                                    edges.append({
                                        'source': f'concept_{other_cluster_id}',
                                        'target': standard_id,
                                        'type': 'secondary_relation',
                                        'weight': similarity,
                                        'style': 'dashed',
                                        'color': '#BDC3C7',
                                        'width': 1
                                    })
                
                # Update standard_to_cluster for compatibility with existing code
                standard_to_cluster = {}
                for std_id, clusters in standard_to_clusters.items():
                    if clusters:
                        standard_to_cluster[std_id] = clusters[0]['cluster_id']  # Use primary cluster
                
                # Calculate similarities between all standards for cross-connections
                similarity_matrix = cosine_similarity(embeddings_matrix)
                
                # Create edges for similar standards across different clusters
                for i, std1 in enumerate(valid_standards):
                    std1_id = f'std_{std1.id}'
                    if std1_id not in standard_to_cluster:
                        continue
                    
                    for j, std2 in enumerate(valid_standards[i+1:], i+1):
                        std2_id = f'std_{std2.id}'
                        if std2_id not in standard_to_cluster:
                            continue
                        
                        # Skip if same state and same cluster
                        if (std1.state == std2.state and 
                            standard_to_cluster[std1_id] == standard_to_cluster[std2_id]):
                            continue
                        
                        similarity = float(similarity_matrix[i, j])
                        
                        # Create edges with different styles based on similarity strength
                        if similarity > 0.85:
                            edge_style = 'solid'
                            edge_width = 3
                            edge_color = '#E74C3C'  # Red for strong similarity
                            edge_type = 'strong_similarity'
                        elif similarity > 0.75:
                            edge_style = 'solid'
                            edge_width = 2
                            edge_color = '#F39C12'  # Orange for moderate
                            edge_type = 'moderate_similarity'
                        elif similarity > 0.65 and std1.state != std2.state:
                            edge_style = 'dashed'
                            edge_width = 1
                            edge_color = '#95A5A6'  # Gray for weak
                            edge_type = 'weak_similarity'
                        else:
                            continue  # Skip very weak similarities
                        
                        edges.append({
                            'source': std1_id,
                            'target': std2_id,
                            'type': edge_type,
                            'weight': similarity,
                            'style': edge_style,
                            'width': edge_width,
                            'color': edge_color,
                            'cross_state': std1.state != std2.state
                        })
                
                # Calculate summary statistics
                core_concepts = [n for n in nodes if n.get('concept_type') == 'core']
                regional_concepts = [n for n in nodes if n.get('concept_type') == 'regional']
                state_specific_concepts = [n for n in nodes if n.get('concept_type') == 'state_specific']
                
                # Convert cluster_analysis to ensure all values are JSON serializable
                serializable_cluster_analysis = {}
                for cid, analysis in cluster_analysis.items():
                    serializable_cluster_analysis[str(cid)] = {
                        'states': list(analysis['states']),
                        'coverage': float(analysis['coverage']),
                        'type': analysis['type'],
                        'name': analysis['name'],
                        'color': analysis['color'],
                        'importance': int(analysis['importance']),
                        'total_standards': len(analysis['standards']),
                        'keywords': analysis.get('keywords', [])  # Include keywords
                    }
                
                return {
                    'nodes': nodes,
                    'edges': edges,
                    'total_nodes': int(len(nodes)),
                    'total_edges': int(len(edges)),
                    'concept_count': int(len(unique_clusters)),
                    'standard_count': int(len([n for n in nodes if n['type'] == 'standard'])),
                    'core_concept_count': int(len(core_concepts)),
                    'regional_concept_count': int(len(regional_concepts)),
                    'state_specific_count': int(len(state_specific_concepts)),
                    'states_represented': int(len(all_states)),
                    'cluster_analysis': serializable_cluster_analysis
                }
                
            except Exception as e:
                logger.error(f"Network graph calculation failed: {e}")
                return {
                    'nodes': [],
                    'edges': [],
                    'error': f'Network analysis failed: {str(e)}'
                }
        
        network_data = view.get_cached_or_calculate(cache_key, calculate_network_graph)
        
        # Add parameters to response
        network_data['parameters'] = {
            'grade_level': grade_level,
            'subject_area': subject_area_id
        }
        
        return view.success_response(network_data)
        
    except ValidationError as e:
        return view.validation_error_response(str(e))
    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['POST'])
def clear_embeddings_cache_api(request):
    """API endpoint to clear problematic cache entries"""
    view = EmbeddingsAPIView()
    
    try:
        data = view.parse_json_body(request) if request.body else {}
        force_clear_all = data.get('force_clear_all', False)
        
        if force_clear_all:
            # Clear all embeddings-related cache keys
            cache_patterns = ['embeddings_viz_', 'similarity_matrix_', 'theme_coverage_']
            cleared_count = 0
            
            # Note: Django's cache doesn't support pattern-based deletion directly
            # So we'll clear specific keys that might be problematic
            test_keys = []
            
            # Generate common problematic cache keys
            for grade in [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                for subject in [None, 1, 2, 3, 4]:
                    for cluster_size in [5, 8, 10, 15, 20]:
                        for epsilon in [0.3, 0.5, 0.7]:
                            for viz_mode in ['2d', '3d']:
                                cache_key = f"embeddings_viz_{grade}_{subject}_{cluster_size}_{epsilon}_{viz_mode}"
                                test_keys.append(cache_key)
            
            for key in test_keys:
                if cache.get(key) is not None:
                    cache.delete(key)
                    cleared_count += 1
            
            logger.info(f"Force cleared {cleared_count} cache entries")
            return view.success_response({
                'cleared_count': cleared_count,
                'message': f'Cleared {cleared_count} embeddings cache entries'
            })
        else:
            # Smart clearing - only clear entries with problematic cluster counts
            cleared_keys = []
            
            # Check common cache keys for problematic data
            test_keys = []
            for grade in [None, 0, 1, 2, 3, 4, 5]:
                for subject in [None, 1, 2]:
                    for cluster_size in [5, 8, 10]:
                        for epsilon in [0.3, 0.5]:
                            cache_key = f"embeddings_viz_{grade}_{subject}_{cluster_size}_{epsilon}_2d"
                            test_keys.append(cache_key)
            
            for cache_key in test_keys:
                data = cache.get(cache_key)
                if data and isinstance(data, dict):
                    cluster_count = len(data.get('clusters', []))
                    total_standards = data.get('total_standards', 0)
                    if total_standards > 50 and cluster_count <= 2:
                        cache.delete(cache_key)
                        cleared_keys.append(cache_key)
                        logger.info(f"Cleared problematic cache key: {cache_key} ({cluster_count} clusters, {total_standards} standards)")
            
            return view.success_response({
                'cleared_keys': cleared_keys,
                'message': f'Cleared {len(cleared_keys)} problematic cache entries'
            })
        
    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['POST'])
def embeddings_semantic_search_api(request):
    """API endpoint for semantic concept search with validation and rate limiting"""
    view = EmbeddingsAPIView()
    
    try:
        data = view.parse_json_body(request)
        
        # Validate search parameters
        query = view.validate_string(
            data.get('query', ''),
            min_length=2,
            max_length=500,
            field_name="query"
        )
        
        grade_level = data.get('grade_level')
        if grade_level is not None and grade_level != "":
            grade_level = view.validate_integer(grade_level, 0, 12, "grade_level")
        else:
            grade_level = None
        
        subject_area_id = data.get('subject_area')
        if subject_area_id and subject_area_id != "":
            subject_area_id = view.validate_integer(subject_area_id, 1, None, "subject_area")
        else:
            subject_area_id = None
        
        limit = view.validate_integer(data.get('limit', 20), 1, 500, "limit")
        
        # Use the existing search service
        search_service = SearchService()
        
        # Prepare filters
        filters = {
            'limit': limit,
            'threshold': 0.7  # Reasonable similarity threshold
        }
        
        if grade_level is not None:
            filters['grade_id'] = grade_level
        
        if subject_area_id:
            filters['subject_id'] = subject_area_id
        
        # Perform semantic search with error handling
        try:
            results = search_service.semantic_search(query, **filters)
        except Exception as e:
            logger.error(f"Semantic search failed for query '{query}': {e}")
            return view.error_response(
                "Search service temporarily unavailable",
                status=503,
                error_code='SEARCH_UNAVAILABLE'
            )
        
        # Group results by state for emphasis analysis
        state_emphasis = {}
        
        for result in results:
            standard = result['standard']
            state_code = standard.state.code if standard.state else 'Unknown'
            if state_code not in state_emphasis:
                state_emphasis[state_code] = {
                    'count': 0,
                    'avg_similarity': 0,
                    'standards': []
                }

            state_emphasis[state_code]['count'] += 1
            state_emphasis[state_code]['standards'].append({
                'id': standard.id,
                'title': standard.title,
                'similarity': result.get('similarity_score', 0)
            })
        
        # Calculate average similarities safely
        for state_data in state_emphasis.values():
            if state_data['standards']:
                total_similarity = sum(s['similarity'] for s in state_data['standards'])
                state_data['avg_similarity'] = round(
                    total_similarity / len(state_data['standards']), 3
                )
        
        # Sort states by emphasis (count and average similarity)
        sorted_states = sorted(
            state_emphasis.items(),
            key=lambda x: (x[1]['count'], x[1]['avg_similarity']),
            reverse=True
        )
        
        # Format results for frontend
        formatted_results = []
        for result in results:
            standard = result['standard']
            formatted_results.append({
                'id': standard.id,
                'code': standard.code,
                'title': standard.title,
                'description': standard.description,
                'state': standard.state.code if standard.state else 'Unknown',
                'similarity_score': result.get('similarity_score', 0),
                'alignment_category': result.get('alignment_category', 'minimal'),
                'alignment_label': result.get('alignment_label', 'Minimal Match'),
                'match_explanation': result.get('match_explanation', '')
            })

        response_data = {
            'query': query,
            'total_results': len(results),
            'results': formatted_results,
            'state_emphasis': dict(sorted_states),
            'top_emphasizing_states': [state for state, _ in sorted_states[:10]],
            'search_metadata': {
                'filters_applied': {
                    'grade_level': grade_level,
                    'subject_area': subject_area_id,
                    'limit': limit
                },
                'states_found': len(state_emphasis)
            }
        }
        
        return view.success_response(response_data)
        
    except ValidationError as e:
        return view.validation_error_response(str(e))
    except Exception as e:
        return view.handle_exception(request, e)


def _generate_real_clusters(standards: List[Standard], umap_embeddings: np.ndarray, 
                          cluster_labels: np.ndarray, state_colors: Dict[str, str], viz_mode: str = '2d') -> List[Dict]:
    """
    Generate real cluster analysis from HDBSCAN results
    Only include meaningful clusters (3+ members) for visual display
    """
    MIN_CLUSTER_SIZE_FOR_DISPLAY = 3  # Only show clusters with 3+ members
    clusters = []
    unique_clusters = set(cluster_labels)
    unique_clusters.discard(-1)  # Remove noise cluster
    
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_standards = [standards[i] for i in range(len(standards)) if cluster_mask[i]]
        
        # Skip clusters that are too small to be meaningful
        if len(cluster_standards) < MIN_CLUSTER_SIZE_FOR_DISPLAY:
            logger.debug(f"Skipping weak cluster {cluster_id} with only {len(cluster_standards)} members")
            continue
            
        cluster_embeddings = umap_embeddings[cluster_mask]
        
        if len(cluster_standards) == 0:
            continue
            
        # Calculate cluster center based on visualization mode
        center_x = float(np.mean(cluster_embeddings[:, 0]))
        center_y = float(np.mean(cluster_embeddings[:, 1]))
        
        cluster_center = {'x': center_x, 'y': center_y}
        
        # Calculate cluster radius based on dimensionality
        if viz_mode == '3d' and cluster_embeddings.shape[1] >= 3:
            # 3D case
            center_z = float(np.mean(cluster_embeddings[:, 2]))
            cluster_center['z'] = center_z
            
            # Calculate 3D radius (distance from center to furthest point)
            distances_3d = np.sqrt(
                (cluster_embeddings[:, 0] - center_x) ** 2 + 
                (cluster_embeddings[:, 1] - center_y) ** 2 + 
                (cluster_embeddings[:, 2] - center_z) ** 2
            )
            cluster_radius = float(np.max(distances_3d)) * 1.2  # Add 20% padding
            cluster_radius_3d = cluster_radius
        else:
            # 2D case
            distances = np.sqrt(
                (cluster_embeddings[:, 0] - center_x) ** 2 + 
                (cluster_embeddings[:, 1] - center_y) ** 2
            )
            cluster_radius = float(np.max(distances)) * 1.2  # Add 20% padding
            cluster_radius_3d = None
        
        # Analyze cluster content
        cluster_states = [s.state.code for s in cluster_standards if s.state]
        state_counts = Counter(cluster_states)
        dominant_states = [state for state, count in state_counts.most_common(5)]
        
        # Extract topic themes using TF-IDF and domain knowledge
        cluster_name = _extract_cluster_theme(cluster_standards, cluster_id)
        
        # Extract key topics instead of philosophy
        key_topics = _extract_key_topics(cluster_standards)
        
        # Prepare detailed standards information for frontend
        standards_details = []
        for standard in cluster_standards:
            # Get grade levels for this standard
            grade_levels = list(standard.grade_levels.values_list('grade_numeric', flat=True))
            
            # Use description as title if title is empty for display
            display_title = standard.title or standard.description or f"Standard {str(standard.id)[:8]}"
            display_description = standard.description or "No description available"
            
            if len(display_description) > 200:
                display_description = display_description[:200] + '...'
                
            standards_details.append({
                'id': str(standard.id),
                'title': display_title,
                'description': display_description,
                'state': standard.state.code if standard.state else 'Unknown',
                'state_name': standard.state.name if standard.state else 'Unknown',
                'subject_area': standard.subject_area.name if standard.subject_area else 'Unknown',
                'grade_levels': sorted(grade_levels) if grade_levels else [],
                'domain': getattr(standard, 'domain', ''),
                'cluster_id': getattr(standard, 'cluster', '')
            })
        
        cluster_data = {
            'id': int(cluster_id),
            'name': cluster_name,
            'center': cluster_center,
            'radius': cluster_radius,
            'states': dominant_states,
            'key_topics': key_topics,
            'size': len(cluster_standards),
            'standards_count': len(cluster_standards),
            'state_distribution': dict(state_counts),
            'standards': standards_details  # Add the detailed standards list
        }
        
        # Add 3D-specific data if available
        if cluster_radius_3d is not None:
            cluster_data['radius_3d'] = cluster_radius_3d
            cluster_data['viz_mode'] = '3d'
        
        clusters.append(cluster_data)
    
    # Sort clusters by size (largest first)
    clusters.sort(key=lambda x: x['standards_count'], reverse=True)
    
    return clusters


def _extract_cluster_theme(standards: List[Standard], cluster_id: int) -> str:
    """Extract meaningful and specific topic themes using semantic analysis"""
    
    if not standards:
        return f'Educational Cluster {cluster_id + 1}'
    
    logger.info(f"Extracting theme for cluster {cluster_id} with {len(standards)} standards")
    
    # Analyze state composition with higher threshold for state-specificity
    state_counts = Counter([s.state.code for s in standards if s.state])
    dominant_state = state_counts.most_common(1)[0][0] if state_counts else None
    state_focus_threshold = 0.75  # Raised from 60% to 75% to reduce false positives
    
    is_state_specific = False
    state_percentage = 0
    if dominant_state and len(standards) > 0:
        state_percentage = state_counts[dominant_state] / len(standards)
        is_state_specific = state_percentage >= state_focus_threshold
        logger.info(f"Cluster {cluster_id}: {state_percentage:.1%} from {dominant_state}, state_specific={is_state_specific}")
    
    # Combine all text from standards for analysis
    combined_text = []
    for standard in standards:
        full_text = f"{standard.title} {standard.description or ''}"
        combined_text.append(full_text)
    
    if not combined_text:
        return f'Educational Cluster {cluster_id + 1}'
    
    # Step 1: Categorize educational content domain
    educational_domain = _categorize_educational_content(combined_text, standards)
    logger.info(f"Cluster {cluster_id}: Educational domain identified as '{educational_domain}'")
    
    # Step 2: Extract semantic themes using improved TF-IDF
    semantic_theme = _extract_semantic_theme(combined_text, standards, cluster_id)
    logger.info(f"Cluster {cluster_id}: Semantic theme identified as '{semantic_theme}'")
    
    # Step 3: Check for genuine state-specific content (not just state dominance)
    genuine_state_content = _has_genuine_state_content(combined_text, dominant_state) if is_state_specific else False
    
    # Step 4: Generate final theme based on analysis
    final_theme = _generate_final_theme(
        educational_domain, semantic_theme, dominant_state, 
        genuine_state_content, state_percentage, cluster_id
    )
    
    logger.info(f"Cluster {cluster_id}: Final theme = '{final_theme}'")
    return final_theme


def _categorize_educational_content(combined_text: List[str], standards: List[Standard]) -> str:
    """Categorize the educational domain of the cluster content"""
    
    all_text = ' '.join(combined_text).lower()
    
    # Define domain indicators with weighted importance
    domain_indicators = {
        'Mathematics': {
            'keywords': ['math', 'number', 'algebra', 'geometry', 'fraction', 'decimal', 'equation', 
                        'measurement', 'data', 'graph', 'calculate', 'solve', 'formula', 'operation',
                        'addition', 'subtraction', 'multiplication', 'division', 'probability', 'statistics'],
            'weight': 1.0
        },
        'Science': {
            'keywords': ['science', 'experiment', 'hypothesis', 'observation', 'energy', 'matter',
                        'biology', 'chemistry', 'physics', 'ecosystem', 'organism', 'cell', 'force',
                        'motion', 'earth', 'space', 'planet', 'solar', 'weather', 'climate'],
            'weight': 1.0
        },
        'Social Studies - Geography': {
            'keywords': ['map', 'geography', 'geographic', 'location', 'region', 'continent', 'country',
                        'latitude', 'longitude', 'cardinal', 'direction', 'compass', 'grid', 'scale',
                        'physical features', 'landforms', 'bodies water', 'mountains', 'rivers'],
            'weight': 1.2  # Higher weight for geographic specificity
        },
        'Social Studies - History': {
            'keywords': ['history', 'historical', 'past', 'timeline', 'period', 'era', 'century',
                        'colonial', 'revolution', 'civil war', 'world war', 'depression', 'ancient',
                        'explorer', 'settlement', 'migration', 'immigration', 'culture', 'tradition'],
            'weight': 1.0
        },
        'Social Studies - Government': {
            'keywords': ['government', 'constitution', 'democracy', 'republic', 'citizen', 'rights',
                        'responsibility', 'law', 'rule', 'leader', 'president', 'congress', 'court',
                        'election', 'vote', 'branches', 'legislative', 'executive', 'judicial'],
            'weight': 1.0
        },
        'Social Studies - Economics': {
            'keywords': ['economy', 'economic', 'business', 'trade', 'goods', 'services', 'money',
                        'resource', 'production', 'consumption', 'supply', 'demand', 'market',
                        'entrepreneur', 'job', 'career', 'industry', 'agriculture', 'manufacturing'],
            'weight': 1.0
        },
        'Language Arts': {
            'keywords': ['reading', 'writing', 'literature', 'story', 'narrative', 'poem', 'author',
                        'character', 'plot', 'setting', 'theme', 'vocabulary', 'grammar', 'sentence',
                        'paragraph', 'essay', 'communication', 'language', 'comprehension', 'fluency'],
            'weight': 1.0
        }
    }
    
    # Score each domain
    domain_scores = {}
    for domain, data in domain_indicators.items():
        score = 0
        keywords = data['keywords']
        weight = data['weight']
        
        for keyword in keywords:
            if keyword in all_text:
                # Count occurrences with weight
                occurrences = all_text.count(keyword)
                score += occurrences * weight
        
        domain_scores[domain] = score
    
    # Find the highest scoring domain
    if domain_scores:
        best_domain = max(domain_scores, key=domain_scores.get)
        if domain_scores[best_domain] > 0:
            return best_domain
    
    # Fallback: use subject area from standards
    subject_areas = [s.subject_area.name for s in standards if s.subject_area]
    if subject_areas:
        most_common_subject = Counter(subject_areas).most_common(1)[0][0]
        return most_common_subject
    
    return 'General Education'


def _extract_semantic_theme(combined_text: List[str], standards: List[Standard], cluster_id: int) -> str:
    """Extract semantic theme using improved TF-IDF analysis"""
    
    try:
        # Create TF-IDF vectorizer with educational focus
        vectorizer = TfidfVectorizer(
            max_features=50,
            stop_words='english',
            ngram_range=(1, 3),  # Reduced from (1,4) for better performance
            min_df=1,  # At least 1 document (more lenient)
            max_df=0.8,  # Exclude very common terms
            lowercase=True
        )
        
        tfidf_matrix = vectorizer.fit_transform(combined_text)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get average TF-IDF scores
        mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # Get top terms sorted by importance
        top_indices = np.argsort(mean_scores)[::-1]
        top_terms = [feature_names[i] for i in top_indices[:10]]
        
        # Filter and clean terms
        meaningful_terms = []
        educational_stopwords = {
            'student', 'students', 'learn', 'learning', 'understand', 'understanding',
            'skill', 'skills', 'knowledge', 'ability', 'abilities', 'grade', 'level',
            'standard', 'standards', 'objective', 'objectives', 'including', 'various',
            'describe', 'explain', 'identify', 'analyze', 'compare', 'evaluate'
        }
        
        for term in top_terms:
            term_clean = term.lower().strip()
            if (len(term_clean) > 2 and 
                term_clean not in educational_stopwords and
                any(char.isalpha() for char in term_clean) and
                not term_clean.startswith(('the ', 'and ', 'for ', 'with '))):
                meaningful_terms.append(term.title())
        
        # Create semantic theme from meaningful terms
        if meaningful_terms:
            # Take top 2-3 most meaningful terms
            primary_terms = meaningful_terms[:3]
            return ' & '.join(primary_terms)
        
    except Exception as e:
        logger.warning(f"Semantic theme extraction failed for cluster {cluster_id}: {e}")
    
    return None


def _has_genuine_state_content(combined_text: List[str], state_code: str) -> bool:
    """Check if content genuinely mentions state-specific information"""
    
    if not state_code:
        return False
    
    all_text = ' '.join(combined_text).lower()
    
    # State name mappings
    state_names = {
        'AL': 'alabama', 'AK': 'alaska', 'AZ': 'arizona', 'AR': 'arkansas', 'CA': 'california',
        'CO': 'colorado', 'CT': 'connecticut', 'DE': 'delaware', 'FL': 'florida', 'GA': 'georgia',
        'HI': 'hawaii', 'ID': 'idaho', 'IL': 'illinois', 'IN': 'indiana', 'IA': 'iowa',
        'KS': 'kansas', 'KY': 'kentucky', 'LA': 'louisiana', 'ME': 'maine', 'MD': 'maryland',
        'MA': 'massachusetts', 'MI': 'michigan', 'MN': 'minnesota', 'MS': 'mississippi', 'MO': 'missouri',
        'MT': 'montana', 'NE': 'nebraska', 'NV': 'nevada', 'NH': 'new hampshire', 'NJ': 'new jersey',
        'NM': 'new mexico', 'NY': 'new york', 'NC': 'north carolina', 'ND': 'north dakota', 'OH': 'ohio',
        'OK': 'oklahoma', 'OR': 'oregon', 'PA': 'pennsylvania', 'RI': 'rhode island', 'SC': 'south carolina',
        'SD': 'south dakota', 'TN': 'tennessee', 'TX': 'texas', 'UT': 'utah', 'VT': 'vermont',
        'VA': 'virginia', 'WA': 'washington', 'WV': 'west virginia', 'WI': 'wisconsin', 'WY': 'wyoming'
    }
    
    state_name = state_names.get(state_code, '').lower()
    
    # Check for explicit state mentions
    if state_name and state_name in all_text:
        return True
    
    # Check for state-specific geographic features or history
    state_specific_indicators = {
        'TX': ['texas', 'lone star', 'alamo', 'austin', 'houston', 'dallas'],
        'CA': ['california', 'golden state', 'sacramento', 'los angeles', 'san francisco'],
        'FL': ['florida', 'sunshine state', 'tallahassee', 'miami', 'orlando', 'everglades'],
        'NY': ['new york', 'empire state', 'albany', 'manhattan', 'brooklyn'],
        'OK': ['oklahoma', 'sooner state', 'oklahoma city', 'tulsa', 'dust bowl'],
    }
    
    indicators = state_specific_indicators.get(state_code, [])
    for indicator in indicators:
        if indicator in all_text:
            return True
    
    return False


def _generate_final_theme(educational_domain: str, semantic_theme: str, dominant_state: str,
                         genuine_state_content: bool, state_percentage: float, cluster_id: int) -> str:
    """Generate the final theme name based on all analysis"""
    
    # Priority 1: Use semantic theme if it's meaningful and domain-specific
    if semantic_theme and len(semantic_theme) > 3:
        # Check if semantic theme already contains domain context
        domain_keywords = {
            'Mathematics': ['math', 'number', 'algebra', 'geometry', 'fraction'],
            'Science': ['science', 'experiment', 'energy', 'biology', 'chemistry'],
            'Social Studies - Geography': ['geography', 'map', 'location', 'region'],
            'Social Studies - History': ['history', 'historical', 'period', 'era'],
            'Social Studies - Government': ['government', 'constitution', 'democracy'],
            'Social Studies - Economics': ['economic', 'business', 'trade', 'goods']
        }
        
        semantic_lower = semantic_theme.lower()
        theme_has_domain_context = any(
            any(keyword in semantic_lower for keyword in keywords)
            for keywords in domain_keywords.values()
        )
        
        if theme_has_domain_context:
            # Semantic theme already includes domain context
            base_theme = semantic_theme
        else:
            # Add domain context to semantic theme
            if educational_domain.startswith('Social Studies'):
                domain_suffix = educational_domain.split(' - ')[1] if ' - ' in educational_domain else 'Studies'
                base_theme = f"{semantic_theme} ({domain_suffix})"
            else:
                base_theme = f"{semantic_theme} ({educational_domain})"
    else:
        # Priority 2: Use educational domain
        base_theme = educational_domain
    
    # Add state context only if genuinely state-specific
    if genuine_state_content and dominant_state and state_percentage > 0.8:
        return f"{dominant_state} {base_theme}"
    elif genuine_state_content and dominant_state and state_percentage > 0.75:
        return f"{base_theme} ({dominant_state})"
    else:
        return base_theme


def _enhance_topic_with_context(top_terms: List[str], combined_text: List[str], 
                               standards: List[Standard], dominant_state: str, 
                               is_state_specific: bool) -> str:
    """Enhanced topic extraction with geographic and contextual awareness"""
    
    # Combine all text for pattern matching
    all_text = ' '.join(combined_text).lower()
    
    # State-specific patterns for more precise naming
    state_specific_patterns = {
        'oklahoma': f'{dominant_state} State Studies',
        'texas': f'{dominant_state} State Studies', 
        'california': f'{dominant_state} State Studies',
        'florida': f'{dominant_state} State Studies',
        'new york': f'{dominant_state} State Studies'
    }
    
    # Enhanced domain-specific topic mappings with more specificity
    # Geography patterns should be checked FIRST as they're very distinctive
    geography_patterns = {
        # Map and geographic tools - these are VERY specific indicators
        'map features': 'Geography & Map Skills',
        'geographic tools': 'Geography & Map Skills', 
        'cardinal directions': 'Geography & Map Skills',
        'latitude longitude': 'Geography & Map Skills',
        'equator prime meridian': 'Geography & Map Skills',
        'map scale': 'Geography & Map Skills',
        'compass rose': 'Geography & Map Skills',
        'grid system': 'Geography & Map Skills',
        'thematic maps': 'Geography & Map Skills',
        'spatial thinking': 'Geography & Map Skills',
        'locate map': 'Geography & Map Skills',
        'geographic terminology': 'Geography & Map Skills',
        'maps globes': 'Geography & Map Skills',
        
        # Physical geography
        'physical geography': 'Physical Geography',
        'landforms': 'Physical Geography',
        'mountains valleys': 'Physical Geography',
        'oceans continents': 'Physical Geography',
        'bodies water': 'Physical Geography',
        'natural features': 'Physical Geography',
        'geographic regions': 'Regional Geography',
        'region unique': 'Regional Geography',
        'local region': 'Regional Geography',
        'human geography': 'Human Geography',
        'geographic awareness': 'Geographic Studies'
    }
    
    specific_topics = {
        # State-specific content
        'oklahoma': 'Oklahoma History & Government',
        'texas': 'Texas History & Government', 
        'california': 'California Studies',
        'florida': 'Florida Studies',
        'colonial america': 'Colonial American History',
        'westward expansion': 'Westward Expansion & Migration',
        'dust bowl': 'Dust Bowl & Great Depression',
        'oil gas': 'Energy & Natural Resources',
        'cattle industry': 'Agricultural & Ranching History',
        'native american tribes': 'Indigenous Peoples & Cultures',
        'buffalo soldiers': 'Military & Civil Rights History',
        
        # Economic Systems (more specific)
        'goods services': 'Economic Fundamentals',
        'entrepreneurs business': 'Entrepreneurship & Business',
        'supply demand': 'Market Economics',
        'resources production': 'Resource Economics',
        'trade exports': 'Trade & Commerce',
        'labor capital': 'Economic Factors',
        
        # Government and Civics
        'government branches': 'Government Structure',
        'constitution rights': 'Constitutional Principles',
        'citizenship democracy': 'Civic Participation',
        'voting elections': 'Democratic Processes',
        'laws regulations': 'Legal Systems',
        
        # Historical Periods (more specific)
        'civil war reconstruction': 'Civil War & Reconstruction',
        'world war': 'World War Era',
        'great depression': 'Depression Era Studies',
        'cold war': 'Cold War Period',
        'progressive era': 'Progressive Reform Era',
        
        # Migration and Demographics (but NOT geographic content)
        'immigration migration': 'Immigration & Migration Patterns',
        'population movement': 'Demographic Changes',
        'ethnic communities': 'Community & Identity',
        
        # Science and Math (more specific)
        'fraction decimal': 'Fraction & Decimal Operations',
        'algebra equation': 'Algebraic Problem Solving',
        'geometry measurement': 'Geometric Measurement',
        'data analysis': 'Data Analysis & Statistics',
        'scientific method': 'Scientific Inquiry Process',
        'ecosystem environment': 'Environmental Science'
    }
    
    # FIRST: Check for geography patterns - these are most distinctive and should override other patterns
    for pattern, topic in geography_patterns.items():
        if pattern in all_text:
            if is_state_specific and dominant_state:
                return f"{dominant_state} {topic}"
            return topic
    
    # SECOND: Check for other specific patterns
    for pattern, topic in specific_topics.items():
        if pattern in all_text:
            if is_state_specific and dominant_state:
                # Add state context for state-specific clusters
                if not topic.startswith(dominant_state):
                    return f"{dominant_state} {topic}"
                else:
                    return topic
            return topic
    
    # Check for state-specific content
    if is_state_specific and dominant_state:
        state_name_lower = None
        state_names = {
            'OK': 'oklahoma',
            'TX': 'texas', 
            'CA': 'california',
            'FL': 'florida',
            'NY': 'new york'
        }
        
        if dominant_state in state_names:
            state_name_lower = state_names[dominant_state]
            if state_name_lower in all_text:
                return f"{dominant_state} Regional Studies"
    
    # Look for distinctive terms that appear frequently
    distinctive_terms = []
    for term in top_terms[:5]:  # Focus on top 5 most distinctive terms
        term_lower = term.lower()
        
        # Filter out generic educational terms
        generic_terms = {
            'student', 'learn', 'understand', 'skill', 'knowledge', 'ability',
            'describe', 'explain', 'identify', 'analyze', 'compare', 'evaluate',
            'grade', 'level', 'standard', 'objective', 'including', 'various'
        }
        
        if (len(term) > 3 and 
            not any(generic in term_lower for generic in generic_terms) and
            any(char.isalpha() for char in term) and
            term_lower not in ['and', 'the', 'for', 'with', 'from']):
            distinctive_terms.append(term.title())
    
    # Create more specific names based on distinctive terms
    if distinctive_terms:
        primary_term = distinctive_terms[0]
        
        # Add context based on content analysis - prioritize geographic indicators
        if any(word in all_text for word in ['map', 'geographic', 'geography', 'latitude', 'longitude', 'cardinal', 'direction', 'compass', 'region', 'location', 'spatial', 'coordinate']):
            context = 'Geographic'
        elif any(word in all_text for word in ['economy', 'economic', 'business', 'trade', 'goods', 'services']):
            context = 'Economic'
        elif any(word in all_text for word in ['government', 'civic', 'political', 'democracy', 'constitution']):
            context = 'Civic'
        elif any(word in all_text for word in ['history', 'historical', 'period', 'era', 'past', 'colonial']):
            context = 'Historical'
        elif any(word in all_text for word in ['culture', 'social', 'community', 'people', 'diversity']):
            context = 'Cultural'
        else:
            context = 'Educational'
        
        # Combine with state context if applicable
        if is_state_specific and dominant_state:
            return f"{dominant_state} {context} Studies: {primary_term}"
        else:
            return f"{context} Studies: {primary_term}"
    
    return None


def _advanced_fallback_extraction(standards: List[Standard], cluster_id: int, 
                                dominant_state: str, is_state_specific: bool) -> str:
    """Advanced fallback method with context awareness"""
    
    if not standards:
        return f'Educational Cluster {cluster_id + 1}'
    
    # Combine all text
    all_text = ' '.join([f"{s.title} {s.description or ''}" for s in standards]).lower()
    
    # Enhanced pattern matching with specificity
    patterns = [
        # State-specific patterns first
        ('oklahoma', f'{dominant_state} State Studies' if dominant_state == 'OK' else 'Oklahoma Studies'),
        ('texas', f'{dominant_state} State Studies' if dominant_state == 'TX' else 'Texas Studies'),
        
        # More specific historical patterns
        ('civil war', 'Civil War Era'),
        ('reconstruction', 'Reconstruction Period'),
        ('world war', 'World War Studies'),
        ('great depression', 'Depression Era'),
        ('dust bowl', 'Dust Bowl History'),
        ('colonial', 'Colonial America'),
        ('revolution', 'Revolutionary Period'),
        
        # Specific economic patterns
        ('entrepreneur', 'Entrepreneurship & Business'),
        ('goods services', 'Economic Fundamentals'),
        ('trade', 'Trade & Commerce'),
        ('business', 'Business Studies'),
        ('economy economic', 'Economic Systems'),
        
        # Government and civic patterns
        ('government', 'Government & Civics'),
        ('constitution', 'Constitutional Studies'),
        ('democracy', 'Democratic Principles'),
        ('citizenship', 'Civic Education'),
        
        # Geographic patterns
        ('geography geographic', 'Geographic Studies'),
        ('environment', 'Environmental Studies'),
        ('region', 'Regional Studies'),
        ('migration', 'Migration & Settlement'),
        
        # Cultural patterns
        ('native american', 'Indigenous Studies'),
        ('culture', 'Cultural Studies'),
        ('immigration', 'Immigration Studies'),
        ('diversity', 'Cultural Diversity'),
        
        # STEM patterns
        ('math mathematical', 'Mathematical Concepts'),
        ('science scientific', 'Scientific Concepts'),
        ('fraction', 'Fraction Operations'),
        ('algebra', 'Algebraic Thinking'),
        ('geometry', 'Geometric Concepts')
    ]
    
    # Check patterns in order of specificity
    for pattern, topic in patterns:
        if pattern in all_text:
            if is_state_specific and dominant_state and not topic.startswith(dominant_state):
                return f"{dominant_state} {topic}"
            return topic
    
    # Final fallback - use grade level and subject area context
    grade_levels = set()
    subject_areas = set()
    
    for standard in standards:
        if hasattr(standard, 'grade_levels'):
            for grade in standard.grade_levels.all():
                grade_levels.add(grade.grade_numeric)
        if standard.subject_area:
            subject_areas.add(standard.subject_area.name)
    
    # Create descriptive name based on context
    context_parts = []
    
    if len(subject_areas) == 1:
        context_parts.append(list(subject_areas)[0])
    
    if grade_levels:
        min_grade = min(grade_levels)
        max_grade = max(grade_levels)
        if min_grade == max_grade:
            context_parts.append(f"Grade {min_grade}")
        else:
            context_parts.append(f"Grades {min_grade}-{max_grade}")
    
    if is_state_specific and dominant_state:
        context_parts.insert(0, dominant_state)
    
    if context_parts:
        return f"{' '.join(context_parts)} Studies"
    
    return f'Educational Cluster {cluster_id + 1}'


def _fallback_topic_extraction(combined_text: List[str], cluster_id: int) -> str:
    """Legacy fallback method - kept for compatibility"""
    all_text = ' '.join(combined_text).lower()
    
    # Basic pattern matching for common educational terms
    if 'history' in all_text or 'historical' in all_text:
        return 'Historical Studies'
    elif 'math' in all_text or 'number' in all_text:
        return 'Mathematical Concepts'
    elif 'science' in all_text or 'scientific' in all_text:
        return 'Scientific Concepts'
    elif 'read' in all_text or 'literature' in all_text:
        return 'Reading & Literature'
    elif 'write' in all_text or 'writing' in all_text:
        return 'Writing Skills'
    elif 'geography' in all_text or 'geographic' in all_text:
        return 'Geographic Studies'
    elif 'government' in all_text or 'civic' in all_text:
        return 'Civics & Government'
    else:
        return f'Educational Cluster {cluster_id + 1}'


def _extract_key_topics(standards: List[Standard]) -> List[str]:
    """Extract 8-10 categorized key educational topics from cluster standards"""
    
    # Combine all text from standards
    all_text = ' '.join([
        f"{s.title} {s.description or ''}" 
        for s in standards
    ]).lower()
    
    # Use enhanced TF-IDF to find most important terms
    try:
        documents = [f"{s.title} {s.description or ''}" for s in standards]
        
        # Enhanced vectorizer for better topic extraction
        vectorizer = TfidfVectorizer(
            max_features=40,  # Increased from 20
            stop_words='english',
            ngram_range=(1, 3),  # Expanded to include 3-grams for better phrases
            min_df=1,
            max_df=0.85,  # Slightly more restrictive
            lowercase=True
        )
        
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get average TF-IDF scores
        mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # Get top terms sorted by importance
        top_indices = np.argsort(mean_scores)[::-1]
        
        # Enhanced filtering with categorization
        categorized_topics = _categorize_topics(feature_names, top_indices, mean_scores, all_text)
        
        # Convert to display format with categories
        display_topics = []
        for category, topics in categorized_topics.items():
            if topics:  # Only include categories that have topics
                for topic in topics[:3]:  # Limit per category
                    display_topics.append(f"{topic}")
        
        # If we have too few topics, add some from raw TF-IDF
        if len(display_topics) < 6:
            raw_topics = _extract_raw_tfidf_topics(feature_names, top_indices, mean_scores)
            for topic in raw_topics:
                if topic not in display_topics and len(display_topics) < 10:
                    display_topics.append(topic)
        
        # Return 8-10 unique topics
        unique_topics = []
        for topic in display_topics:
            if topic not in unique_topics:
                unique_topics.append(topic)
        
        return unique_topics[:10]  # Return up to 10 topics
        
    except Exception as e:
        logger.warning(f"Enhanced key topics extraction failed: {e}")
        
        # Fallback to improved simple keyword extraction
        return _extract_enhanced_keywords(all_text)


def _categorize_topics(feature_names, top_indices, mean_scores, all_text) -> dict:
    """Categorize extracted topics into educational domains"""
    
    # Define topic categories with keywords
    topic_categories = {
        'Core Concepts': {
            'math': ['number', 'fraction', 'decimal', 'equation', 'algebra', 'geometry', 'measurement'],
            'science': ['experiment', 'observation', 'hypothesis', 'energy', 'matter', 'organism', 'ecosystem'],
            'language': ['reading', 'writing', 'vocabulary', 'comprehension', 'grammar', 'literature'],
            'social_studies': ['history', 'geography', 'government', 'economics', 'culture', 'society']
        },
        'Skills & Methods': {
            'analysis': ['analyze', 'compare', 'evaluate', 'examine', 'investigate', 'research'],
            'problem_solving': ['solve', 'calculate', 'determine', 'find', 'compute', 'figure'],
            'communication': ['explain', 'describe', 'discuss', 'present', 'communicate', 'express'],
            'critical_thinking': ['interpret', 'infer', 'conclude', 'reason', 'judge', 'assess']
        },
        'Geographic & Spatial': {
            'location': ['map', 'location', 'region', 'area', 'place', 'position', 'coordinate'],
            'features': ['landform', 'mountain', 'river', 'ocean', 'continent', 'country', 'city'],
            'tools': ['compass', 'grid', 'scale', 'direction', 'cardinal', 'latitude', 'longitude']
        },
        'Historical & Cultural': {
            'time': ['period', 'era', 'century', 'timeline', 'past', 'historical', 'ancient'],
            'events': ['war', 'revolution', 'exploration', 'settlement', 'migration', 'discovery'],
            'people': ['leader', 'explorer', 'inventor', 'citizen', 'community', 'society']
        },
        'Government & Civics': {
            'structure': ['government', 'constitution', 'law', 'rule', 'branch', 'system'],
            'participation': ['citizen', 'vote', 'election', 'democracy', 'republic', 'rights'],
            'leaders': ['president', 'governor', 'mayor', 'representative', 'official', 'leader']
        }
    }
    
    categorized = {category: [] for category in topic_categories.keys()}
    categorized['Other'] = []
    
    # Enhanced educational stopwords
    educational_stopwords = {
        'student', 'students', 'learn', 'learning', 'understand', 'understanding',
        'skill', 'skills', 'knowledge', 'ability', 'abilities', 'grade', 'level',
        'standard', 'standards', 'objective', 'objectives', 'including', 'various',
        'describe', 'explain', 'identify', 'analyze', 'compare', 'evaluate',
        'will', 'can', 'use', 'using', 'make', 'work', 'time', 'way', 'new',
        'given', 'provided', 'appropriate', 'different', 'such', 'example'
    }
    
    # Process top terms
    for i in top_indices[:25]:  # Look at more terms for categorization
        term = feature_names[i].lower().strip()
        score = mean_scores[i]
        
        # Skip if too short, generic, or stopword
        if (len(term) <= 2 or 
            term in educational_stopwords or
            not any(char.isalpha() for char in term) or
            score < 0.01):  # Minimum relevance threshold
            continue
        
        # Clean and format term
        display_term = ' '.join(word.capitalize() for word in term.split())
        
        # Categorize the term
        categorized_term = False
        for category, subcategories in topic_categories.items():
            for subcat, keywords in subcategories.items():
                if any(keyword in term for keyword in keywords):
                    if display_term not in categorized[category]:
                        categorized[category].append(display_term)
                        categorized_term = True
                        break
            if categorized_term:
                break
        
        # If not categorized, check if it's a meaningful term for 'Other'
        if not categorized_term and len(display_term) > 3:
            # Additional quality check for 'Other' category
            if (not any(generic in term for generic in ['the', 'and', 'for', 'with', 'from']) and
                any(char.isalpha() for char in term)):
                categorized['Other'].append(display_term)
    
    # Remove empty categories and limit items per category
    filtered_categorized = {}
    for category, terms in categorized.items():
        if terms:
            # Sort by length/specificity and take best terms
            terms_sorted = sorted(terms, key=lambda x: (len(x), x.count(' ')), reverse=True)
            filtered_categorized[category] = terms_sorted[:4]  # Max 4 per category
    
    return filtered_categorized


def _extract_raw_tfidf_topics(feature_names, top_indices, mean_scores) -> List[str]:
    """Extract raw topics from TF-IDF when categorization yields too few results"""
    
    educational_stopwords = {
        'student', 'students', 'learn', 'learning', 'understand', 'understanding',
        'skill', 'skills', 'knowledge', 'ability', 'abilities', 'grade', 'level',
        'standard', 'standards', 'objective', 'objectives', 'including', 'various',
        'describe', 'explain', 'identify', 'analyze', 'compare', 'evaluate'
    }
    
    raw_topics = []
    for i in top_indices[:15]:  # Look at top 15 terms
        term = feature_names[i].lower().strip()
        score = mean_scores[i]
        
        if (len(term) > 2 and 
            term not in educational_stopwords and
            any(char.isalpha() for char in term) and
            score > 0.015 and  # Higher threshold for raw topics
            not term.startswith(('the ', 'and ', 'for ', 'with '))):
            
            display_term = ' '.join(word.capitalize() for word in term.split())
            raw_topics.append(display_term)
    
    return raw_topics


def _extract_enhanced_keywords(text: str) -> List[str]:
    """Enhanced fallback keyword extraction with better categorization"""
    
    # Enhanced educational keywords by category
    priority_keywords = {
        'Mathematics': ['math', 'algebra', 'geometry', 'fraction', 'decimal', 'equation', 'measurement', 'data', 'graph'],
        'Science': ['science', 'experiment', 'hypothesis', 'energy', 'matter', 'biology', 'chemistry', 'physics'],
        'Geography': ['geography', 'map', 'location', 'region', 'continent', 'latitude', 'longitude', 'cardinal'],
        'History': ['history', 'historical', 'timeline', 'period', 'era', 'colonial', 'revolution', 'civil war'],
        'Government': ['government', 'constitution', 'democracy', 'citizen', 'rights', 'law', 'election', 'vote'],
        'Economics': ['economics', 'business', 'trade', 'goods', 'services', 'money', 'resource', 'production'],
        'Language Arts': ['reading', 'writing', 'literature', 'vocabulary', 'grammar', 'comprehension', 'communication']
    }
    
    found_keywords = []
    for category, keywords in priority_keywords.items():
        category_found = []
        for keyword in keywords:
            if keyword in text:
                category_found.append(keyword.title())
        
        # Add up to 2 keywords per category
        found_keywords.extend(category_found[:2])
        
        if len(found_keywords) >= 10:  # Stop when we have enough
            break
    
    return found_keywords[:10] if found_keywords else ['General Education']


def _extract_simple_keywords(text: str) -> List[str]:
    """Simple fallback keyword extraction"""
    
    # Common educational keywords to prioritize
    priority_keywords = [
        'history', 'historical', 'geography', 'government', 'constitution',
        'civil war', 'revolution', 'colonial', 'economics', 'culture',
        'math', 'algebra', 'geometry', 'fraction', 'equation', 'measurement',
        'science', 'biology', 'chemistry', 'physics', 'experiment',
        'reading', 'writing', 'literature', 'language', 'communication'
    ]
    
    found_keywords = []
    for keyword in priority_keywords:
        if keyword in text and keyword.title() not in found_keywords:
            found_keywords.append(keyword.title())
            
    return found_keywords[:5] if found_keywords else ['General Education']


def _fallback_kmeans_clustering(embeddings: np.ndarray, n_total: int, desired_clusters: int, logger) -> np.ndarray:
    """
    Simple KMeans fallback when HDBSCAN fails completely.
    """
    try:
        from sklearn.cluster import KMeans
        
        # Use a reasonable number of clusters
        n_clusters = max(3, min(desired_clusters, n_total // 5))
        
        logger.info(f"KMeans fallback with {n_clusters} clusters")
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        logger.info(f"KMeans fallback successful: {len(set(cluster_labels))} clusters")
        return cluster_labels
        
    except ImportError:
        logger.error("sklearn not available for KMeans fallback")
        return np.array([-1] * len(embeddings))
    except Exception as e:
        logger.error(f"KMeans fallback failed: {e}")
        return np.array([-1] * len(embeddings))


@api_endpoint(['GET'])
def embeddings_cluster_matrix_api(request):
    """API endpoint for topic coverage heat map (topics × states matrix)"""
    view = EmbeddingsAPIView()
    
    try:
        # Validate parameters (same as visualization endpoint for consistency)
        grade_level = request.GET.get('grade_level')
        if grade_level and grade_level.strip():
            grade_level = view.validate_integer(grade_level, 0, 12, "grade_level")
        else:
            grade_level = None
        
        subject_area_id = request.GET.get('subject_area')
        if subject_area_id:
            subject_area_id = view.validate_integer(subject_area_id, 1, None, "subject_area")
        
        cluster_size, epsilon = view.validate_clustering_params(
            request.GET.get('cluster_size', 5),
            request.GET.get('epsilon', 0.5)
        )
        
        # Create cache key
        cache_key = f"topic_coverage_matrix_{grade_level}_{subject_area_id}_{cluster_size}_{epsilon}"
        
        def calculate_cluster_matrix():
            # Get the same clustering data as the scatter plot
            # This ensures consistency between visualizations
            standards_query = Standard.objects.filter(
                embedding__isnull=False
            ).select_related('state', 'subject_area')
            
            if grade_level is not None:
                standards_query = standards_query.filter(
                    grade_levels__grade_numeric=grade_level
                ).distinct()
            
            if subject_area_id:
                standards_query = standards_query.filter(subject_area_id=subject_area_id)
            
            standards = list(standards_query[:2000])
            
            if len(standards) < 10:
                return {
                    'topic_names': [],
                    'state_codes': [],
                    'coverage_matrix': [],
                    'error': 'Not enough standards for topic coverage analysis'
                }
            
            # Prepare embeddings (same logic as visualization endpoint)
            embeddings_matrix = []
            valid_standards = []
            
            for standard in standards:
                try:
                    embedding = standard.embedding
                    if embedding is not None and len(embedding) >= 2:
                        if isinstance(embedding, list):
                            embedding_array = np.array(embedding, dtype=np.float32)
                        else:
                            embedding_array = np.array(embedding, dtype=np.float32)
                        
                        embeddings_matrix.append(embedding_array)
                        valid_standards.append(standard)
                except Exception as e:
                    continue
            
            if not embeddings_matrix:
                return {
                    'topic_names': [],
                    'state_codes': [],
                    'coverage_matrix': [],
                    'error': 'No valid embeddings found'
                }
            
            embeddings_matrix = np.array(embeddings_matrix)
            logger.info(f"Calculating topic coverage matrix for {len(embeddings_matrix)} embeddings")
            
            # Perform same UMAP + HDBSCAN as visualization
            try:
                # Phase 1: Adaptive UMAP parameters
                n_samples = len(embeddings_matrix)
                import math
                adaptive_n_neighbors = min(max(5, int(math.sqrt(n_samples))), 50, n_samples - 1)
                
                if n_samples < 100:
                    adaptive_min_dist = 0.01
                elif n_samples < 500:
                    adaptive_min_dist = 0.05
                else:
                    adaptive_min_dist = 0.1
                
                umap_reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=adaptive_n_neighbors,
                    min_dist=adaptive_min_dist,
                    metric='cosine',
                    random_state=42
                )
                umap_embeddings = umap_reducer.fit_transform(embeddings_matrix)
                
                # Phase 1: HDBSCAN with min_samples
                min_cluster_size = max(2, cluster_size)
                min_samples = max(1, min_cluster_size // 2)
                
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric='euclidean',
                    cluster_selection_epsilon=epsilon
                )
                cluster_labels = clusterer.fit_predict(umap_embeddings)
                
                # Get unique cluster IDs (excluding noise -1)
                unique_clusters = [c for c in set(cluster_labels) if c != -1]
                unique_clusters.sort()
                
                if len(unique_clusters) < 2:
                    return {
                        'topic_names': [],
                        'state_codes': [],
                        'coverage_matrix': [],
                        'error': f'Only {len(unique_clusters)} topics found - need at least 2 for coverage analysis'
                    }
                
                logger.info(f"Found {len(unique_clusters)} topic clusters for coverage matrix")
                
                # Group standards by cluster and state
                cluster_info = {}
                states_in_data = set()
                
                for cluster_id in unique_clusters:
                    cluster_mask = cluster_labels == cluster_id
                    cluster_standards = [valid_standards[i] for i in range(len(valid_standards)) if cluster_mask[i]]
                    
                    # Generate cluster name using existing theme extraction
                    cluster_name = _extract_cluster_theme(cluster_standards, cluster_id)
                    cluster_info[cluster_id] = {
                        'name': cluster_name,
                        'standards_by_state': {},
                        'total_standards': len(cluster_standards)
                    }
                    
                    # Group standards by state
                    for standard in cluster_standards:
                        state_code = standard.state.code
                        states_in_data.add(state_code)
                        
                        if state_code not in cluster_info[cluster_id]['standards_by_state']:
                            cluster_info[cluster_id]['standards_by_state'][state_code] = []
                        cluster_info[cluster_id]['standards_by_state'][state_code].append(standard)
                
                # Sort states for consistent ordering
                state_codes = sorted(list(states_in_data))
                
                # Calculate coverage matrix: Topics (rows) × States (columns)
                coverage_matrix = []
                topic_names = []
                topic_metadata = []
                
                for cluster_id in unique_clusters:
                    info = cluster_info[cluster_id]
                    topic_names.append(info['name'])
                    
                    # Calculate coverage for each state
                    coverage_row = []
                    max_coverage = max([len(info['standards_by_state'].get(state, [])) for state in state_codes] + [1])
                    
                    for state_code in state_codes:
                        state_standards = info['standards_by_state'].get(state_code, [])
                        # Normalize coverage as percentage of maximum coverage for this topic
                        coverage = len(state_standards) / max_coverage if max_coverage > 0 else 0
                        coverage_row.append(round(coverage, 3))
                    
                    coverage_matrix.append(coverage_row)
                    
                    # Create metadata for tooltips
                    sample_standards = []
                    for state in state_codes[:3]:  # Sample from first few states
                        state_standards = info['standards_by_state'].get(state, [])
                        if state_standards:
                            sample_standards.extend([
                                f"{state}: {(s.title or s.description or f'Standard {str(s.id)[:8]}')[:50]}..."
                                for s in state_standards[:2]
                            ])
                    
                    topic_metadata.append({
                        'cluster_id': int(cluster_id),
                        'name': info['name'],
                        'total_standards': int(info['total_standards']),
                        'states_covered': int(len([s for s in state_codes if info['standards_by_state'].get(s)])),
                        'sample_standards': sample_standards[:5]  # Limit samples
                    })
                
                return {
                    'topic_names': topic_names,
                    'state_codes': state_codes,
                    'coverage_matrix': coverage_matrix,
                    'topic_metadata': topic_metadata,
                    'total_topics': int(len(unique_clusters)),
                    'total_states': int(len(state_codes)),
                    'total_standards': int(len(valid_standards))
                }
                
            except Exception as e:
                logger.error(f"Topic coverage matrix calculation failed: {e}")
                return {
                    'topic_names': [],
                    'state_codes': [],
                    'coverage_matrix': [],
                    'error': f'Topic clustering failed: {str(e)}'
                }
        
        matrix_data = view.get_cached_or_calculate(cache_key, calculate_cluster_matrix)
        
        # Add parameters to response
        matrix_data['parameters'] = {
            'grade_level': grade_level,
            'subject_area': subject_area_id,
            'cluster_size': cluster_size,
            'epsilon': epsilon
        }
        
        return view.success_response(matrix_data)
        
    except ValidationError as e:
        return view.validation_error_response(str(e))
    except Exception as e:
        return view.handle_exception(request, e)


@api_endpoint(['GET'])  
def embeddings_enhanced_similarity_matrix_api(request):
    """Enhanced similarity matrix with multiple view types and better data"""
    view = EmbeddingsAPIView()
    
    try:
        # Validate parameters
        grade_level = request.GET.get('grade_level')
        if grade_level and grade_level.strip():
            grade_level = view.validate_integer(grade_level, 0, 12, "grade_level")
        else:
            grade_level = None
        
        subject_area_id = request.GET.get('subject_area')
        if subject_area_id:
            subject_area_id = view.validate_integer(subject_area_id, 1, None, "subject_area")
        
        # New parameter for matrix type
        matrix_type = request.GET.get('matrix_type', 'state')  # 'state', 'topic', 'density'
        
        # Create cache key
        cache_key = f"enhanced_matrix_{matrix_type}_{grade_level}_{subject_area_id}"
        
        def calculate_enhanced_matrix():
            if matrix_type == 'topic':
                return _calculate_topic_coverage_matrix(grade_level, subject_area_id)
            elif matrix_type == 'density':
                return _calculate_density_matrix(grade_level, subject_area_id)
            else:  # Default to enhanced state matrix
                return _calculate_enhanced_state_matrix(grade_level, subject_area_id)
        
        matrix_data = view.get_cached_or_calculate(cache_key, calculate_enhanced_matrix)
        
        # Add parameters to response
        matrix_data['parameters'] = {
            'grade_level': grade_level,
            'subject_area': subject_area_id,
            'matrix_type': matrix_type
        }
        
        return view.success_response(matrix_data)
        
    except ValidationError as e:
        return view.validation_error_response(str(e))
    except Exception as e:
        return view.handle_exception(request, e)


def _calculate_topic_coverage_matrix(grade_level: int = None, subject_area_id: int = None) -> Dict:
    """Calculate topic/theme coverage across states matrix"""
    
    # Get topic clusters
    themes_query = TopicCluster.objects.all()
    if subject_area_id:
        themes_query = themes_query.filter(subject_area_id=subject_area_id)
    
    themes = list(themes_query.order_by('-states_represented')[:20])  # Limit for performance
    
    # Get states with standards
    states = list(State.objects.filter(
        standards__embedding__isnull=False
    ).distinct().order_by('code'))
    
    if not themes or not states:
        return {
            'row_labels': [],
            'col_labels': [],
            'matrix': [],
            'error': 'Insufficient data for topic coverage matrix'
        }
    
    # Build topic coverage matrix: themes × states
    matrix = []
    for theme in themes:
        row = []
        for state in states:
            # Calculate coverage percentage for this theme in this state
            # This is a simplified calculation - in practice you'd want more sophisticated logic
            coverage = min(100, theme.states_represented * 100 / 50) if hasattr(theme, 'states_represented') else 0
            row.append(round(coverage, 1))
        matrix.append(row)
    
    return {
        'row_labels': [theme.name for theme in themes],
        'col_labels': [state.code for state in states],
        'matrix': matrix,
        'matrix_type': 'topic_coverage',
        'total_themes': int(len(themes)),
        'total_states': int(len(states))
    }


def _calculate_density_matrix(grade_level: int = None, subject_area_id: int = None) -> Dict:
    """Calculate standards density across grades and subjects"""
    
    # Get grade levels and subjects for matrix dimensions
    if grade_level is not None:
        grade_levels = [grade_level]
    else:
        grade_levels = list(range(0, 13))  # K-12
    
    if subject_area_id:
        subjects = list(SubjectArea.objects.filter(id=subject_area_id))
    else:
        subjects = list(SubjectArea.objects.all())
    
    # Build density matrix: grades × subjects
    matrix = []
    for grade in grade_levels:
        row = []
        for subject in subjects:
            # Count standards for this grade/subject combination
            count = Standard.objects.filter(
                grade_levels__grade_numeric=grade,
                subject_area=subject,
                embedding__isnull=False
            ).count()
            
            row.append(count)
        matrix.append(row)
    
    return {
        'row_labels': [f"Grade {g}" if g > 0 else "Kindergarten" for g in grade_levels],
        'col_labels': [subject.name for subject in subjects],
        'matrix': matrix,
        'matrix_type': 'density',
        'total_grades': int(len(grade_levels)),
        'total_subjects': int(len(subjects))
    }


def _calculate_enhanced_state_matrix(grade_level: int = None, subject_area_id: int = None) -> Dict:
    """Calculate enhanced state similarity matrix with better data"""
    
    # Use embedding-based similarities instead of just correlations
    states_query = State.objects.filter(
        standards__embedding__isnull=False
    ).distinct()
    
    if grade_level is not None:
        states_query = states_query.filter(
            standards__grade_levels__grade_numeric=grade_level
        )
    
    if subject_area_id:
        states_query = states_query.filter(
            standards__subject_area_id=subject_area_id
        )
    
    states = list(states_query.order_by('code'))
    
    if len(states) < 2:
        return {
            'row_labels': [],
            'col_labels': [],
            'matrix': [],
            'error': 'Need at least 2 states for similarity matrix'
        }
    
    # Calculate state centroids from embeddings
    state_centroids = {}
    
    for state in states:
        standards_query = Standard.objects.filter(
            state=state,
            embedding__isnull=False
        )
        
        if grade_level is not None:
            standards_query = standards_query.filter(
                grade_levels__grade_numeric=grade_level
            ).distinct()
        
        if subject_area_id:
            standards_query = standards_query.filter(subject_area_id=subject_area_id)
        
        standards = list(standards_query[:500])  # Limit for performance
        
        if standards:
            # Calculate average embedding (centroid)
            embeddings = []
            for standard in standards:
                try:
                    if isinstance(standard.embedding, list):
                        embeddings.append(np.array(standard.embedding, dtype=np.float32))
                    else:
                        embeddings.append(np.array(standard.embedding, dtype=np.float32))
                except:
                    continue
            
            if embeddings:
                centroid = np.mean(embeddings, axis=0)
                state_centroids[state.code] = centroid
    
    # Calculate similarity matrix using centroids
    state_codes = [state.code for state in states if state.code in state_centroids]
    n_states = len(state_codes)
    
    if n_states < 2:
        return {
            'row_labels': [],
            'col_labels': [],
            'matrix': [],
            'error': 'Insufficient embedding data for similarity calculation'
        }
    
    similarity_matrix = []
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    for i, state1 in enumerate(state_codes):
        row = []
        for j, state2 in enumerate(state_codes):
            if i == j:
                similarity = 1.0
            else:
                # Calculate cosine similarity between state centroids
                centroid1 = state_centroids[state1].reshape(1, -1)
                centroid2 = state_centroids[state2].reshape(1, -1)
                similarity = float(cosine_similarity(centroid1, centroid2)[0][0])
            
            row.append(round(similarity, 3))
        similarity_matrix.append(row)
    
    return {
        'row_labels': state_codes,
        'col_labels': state_codes,
        'matrix': similarity_matrix,
        'matrix_type': 'enhanced_state',
        'total_states': int(len(state_codes))
    }