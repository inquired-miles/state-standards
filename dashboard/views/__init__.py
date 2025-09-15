"""
Dashboard views package - modularized for better maintainability
"""
from .dashboard_views import (
    dashboard_home,
    coverage_dashboard,
    groupings_dashboard,
    topics_dashboard,
    proxies_dashboard,
    embeddings_dashboard,
    custom_clusters_dashboard,
    proxy_runs_dashboard
)

from .proxy_api_views import (
    proxy_run_coverage_api,
    proxy_run_proxies_api,
    generate_proxies_api,
    proxy_job_status_api,
    generate_standard_proxies_api,
    standard_proxy_job_status_api,
    proxy_detail_api
)

from .topic_api_views import (
    validate_topic_criteria,
    generate_topic_proxies_api,
    topic_proxy_job_status_api,
    create_topic_api,
    run_topic_analysis_api,
    topic_analysis_status_api,
    analyze_coverage_api
)

from .embeddings_api_views import (
    embeddings_visualization_data_api,
    embeddings_similarity_matrix_api,
    embeddings_cluster_matrix_api,
    embeddings_enhanced_similarity_matrix_api,
    embeddings_network_graph_api,
    embeddings_semantic_search_api,
    clear_embeddings_cache_api
)

from .custom_cluster_api_views import (
    custom_clusters_api,
    custom_cluster_detail_api,
    cluster_reports_api,
    cluster_report_detail_api,
)

from .atomization_api_views import (
    atomize_standards_api,
    atomize_job_status_api,
    generate_atom_embeddings_api,
    atom_embeddings_job_status_api,
    run_proxy_pipeline_api,
    proxy_pipeline_status_api
)

__all__ = [
    # Dashboard views
    'dashboard_home',
    'coverage_dashboard', 
    'groupings_dashboard',
    'topics_dashboard',
    'proxies_dashboard',
    'embeddings_dashboard',
    'custom_clusters_dashboard',
    'proxy_runs_dashboard',
    
    # Proxy API views
    'proxy_run_coverage_api',
    'proxy_run_proxies_api',
    'generate_proxies_api',
    'proxy_job_status_api',
    'generate_standard_proxies_api',
    'standard_proxy_job_status_api',
    'proxy_detail_api',
    
    # Topic API views
    'validate_topic_criteria',
    'generate_topic_proxies_api',
    'topic_proxy_job_status_api',
    'create_topic_api',
    'run_topic_analysis_api',
    'topic_analysis_status_api',
    'analyze_coverage_api',

    # Embeddings API views
    'embeddings_visualization_data_api',
    'embeddings_similarity_matrix_api',
    'embeddings_cluster_matrix_api',
    'embeddings_enhanced_similarity_matrix_api',
    'embeddings_network_graph_api',
    'embeddings_semantic_search_api',
    'clear_embeddings_cache_api',

    # Custom cluster APIs
    'custom_clusters_api',
    'custom_cluster_detail_api',
    'cluster_reports_api',
    'cluster_report_detail_api',

    # Atomization API views
    'atomize_standards_api',
    'atomize_job_status_api',
    'generate_atom_embeddings_api',
    'atom_embeddings_job_status_api',
    'run_proxy_pipeline_api',
    'proxy_pipeline_status_api',
]
