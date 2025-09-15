"""
URL configuration for dashboard app with modular view imports
"""
from django.urls import path
from .views import (
    # Dashboard views
    dashboard_home, coverage_dashboard, groupings_dashboard, topics_dashboard,
    proxies_dashboard, proxy_runs_dashboard, embeddings_dashboard,
    
    # Proxy API views
    proxy_run_coverage_api, proxy_run_proxies_api, generate_proxies_api,
    proxy_job_status_api, generate_standard_proxies_api, standard_proxy_job_status_api,
    proxy_detail_api,
    
    # Topic API views  
    validate_topic_criteria, generate_topic_proxies_api, topic_proxy_job_status_api,
    create_topic_api, run_topic_analysis_api, topic_analysis_status_api, analyze_coverage_api,
    
    # Embeddings API views
    embeddings_visualization_data_api, embeddings_similarity_matrix_api,
    embeddings_network_graph_api, embeddings_semantic_search_api,
    clear_embeddings_cache_api, embeddings_cluster_matrix_api,
    embeddings_enhanced_similarity_matrix_api,
    
    # Atomization API views
    atomize_standards_api, atomize_job_status_api, generate_atom_embeddings_api,
    atom_embeddings_job_status_api, run_proxy_pipeline_api, proxy_pipeline_status_api,
)

app_name = 'dashboard'

urlpatterns = [
    # Main dashboard
    path('', dashboard_home, name='home'),
    
    # Dashboard views
    path('coverage/', coverage_dashboard, name='coverage'),
    path('groupings/', groupings_dashboard, name='groupings'), 
    path('topics/', topics_dashboard, name='topics'),
    path('proxies/', proxies_dashboard, name='proxies'),
    path('proxy-runs/', proxy_runs_dashboard, name='proxy_runs'),
    path('embeddings/', embeddings_dashboard, name='embeddings'),
    
    # API endpoints
    path('api/analyze-coverage/', analyze_coverage_api, name='analyze_coverage_api'),
    path('api/proxy-run-coverage/', proxy_run_coverage_api, name='proxy_run_coverage_api'),
    path('api/proxy-run-proxies/', proxy_run_proxies_api, name='proxy_run_proxies_api'),
    
    # Atom-level proxy endpoints
    path('api/generate-proxies/', generate_proxies_api, name='generate_proxies_api'),
    path('api/proxy-job-status/<str:job_id>/', proxy_job_status_api, name='proxy_job_status_api'),
    
    # Standard-level proxy endpoints
    path('api/generate-standard-proxies/', generate_standard_proxies_api, name='generate_standard_proxies_api'),
    path('api/standard-proxy-job-status/<str:job_id>/', standard_proxy_job_status_api, name='standard_proxy_job_status_api'),
    
    # Topic-based proxy endpoints
    path('api/validate-topic-criteria/', validate_topic_criteria, name='validate_topic_criteria'),
    path('api/generate-topic-proxies/', generate_topic_proxies_api, name='generate_topic_proxies_api'),
    path('api/topic-proxy-job-status/<str:job_id>/', topic_proxy_job_status_api, name='topic_proxy_job_status_api'),
    
    # Shared proxy endpoints
    path('api/proxy-detail/<uuid:proxy_id>/', proxy_detail_api, name='proxy_detail_api'),
    
    # Atomization & embeddings APIs
    path('api/atomize-standards/', atomize_standards_api, name='atomize_standards_api'),
    path('api/atomize-job-status/<str:job_id>/', atomize_job_status_api, name='atomize_job_status_api'),
    path('api/generate-atom-embeddings/', generate_atom_embeddings_api, name='generate_atom_embeddings_api'),
    path('api/atom-embeddings-job-status/<str:job_id>/', atom_embeddings_job_status_api, name='atom_embeddings_job_status_api'),
    
    # Full pipeline (atomize + embed + proxies)
    path('api/proxy-pipeline/', run_proxy_pipeline_api, name='run_proxy_pipeline_api'),
    path('api/proxy-pipeline-status/<str:job_id>/', proxy_pipeline_status_api, name='proxy_pipeline_status_api'),
    
    # Topic management API endpoints
    path('api/topics/create/', create_topic_api, name='create_topic_api'),
    path('api/topics/analyze/', run_topic_analysis_api, name='run_topic_analysis_api'),
    path('api/topics/job-status/<str:job_id>/', topic_analysis_status_api, name='topic_analysis_status_api'),
    
    # Embeddings visualization API endpoints
    path('api/embeddings/visualization-data/', embeddings_visualization_data_api, name='embeddings_visualization_data'),
    path('api/embeddings/similarity-matrix/', embeddings_similarity_matrix_api, name='embeddings_similarity_matrix'),
    path('api/embeddings/cluster-matrix/', embeddings_cluster_matrix_api, name='embeddings_cluster_matrix'),
    path('api/embeddings/enhanced-matrix/', embeddings_enhanced_similarity_matrix_api, name='embeddings_enhanced_matrix'),
    path('api/embeddings/network-graph/', embeddings_network_graph_api, name='embeddings_network_graph'),
    path('api/embeddings/semantic-search/', embeddings_semantic_search_api, name='embeddings_semantic_search'),
    path('api/embeddings/clear-cache/', clear_embeddings_cache_api, name='clear_embeddings_cache'),
]