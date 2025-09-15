"""
Dashboard-specific settings for security and performance configurations
"""

# Dashboard Security Settings
DASHBOARD_SETTINGS = {
    # Input validation limits
    'MAX_QUERY_LENGTH': 500,
    'MAX_CONTENT_LENGTH': 10000,
    'MAX_BATCH_SIZE': 200,
    'MAX_PROCESSING_LIMIT': 1000,
    
    # Cache settings (in seconds)
    'CACHE_TTL_SHORT': 300,    # 5 minutes
    'CACHE_TTL_MEDIUM': 1800,  # 30 minutes  
    'CACHE_TTL_LONG': 3600,    # 1 hour
    
    # Pagination settings
    'DEFAULT_PAGE_SIZE': 25,
    'MAX_PAGE_SIZE': 100,
    
    # Background job settings
    'JOB_TIMEOUT': 3600,  # 1 hour
    'PROGRESS_UPDATE_INTERVAL': 10,  # seconds
    
    # Clustering default parameters
    'DEFAULT_MIN_CLUSTER_SIZE': 8,
    'DEFAULT_EPSILON': 0.15,
    'DEFAULT_SIMILARITY_THRESHOLD': 0.7,
    
    # Rate limiting (requests per minute)
    'API_RATE_LIMIT': 60,
    'SEARCH_RATE_LIMIT': 30,
    
    # Resource limits
    'MAX_EMBEDDINGS_BATCH': 25,
    'MAX_CONCURRENT_JOBS': 5,
    'MEMORY_LIMIT_MB': 512,
}

# Embedding configuration
EMBEDDING_SETTINGS = {
    'MODEL': 'text-embedding-3-small',
    'DIMENSIONS': 1536,
    'BATCH_SIZE': 50,  # Conservative for memory management
    'TIMEOUT_SECONDS': 30,
    'RETRY_COUNT': 3,
    'RETRY_DELAY': 1,  # seconds
}

# Security headers and CSRF settings
SECURITY_SETTINGS = {
    'CSRF_COOKIE_SECURE': True,
    'CSRF_COOKIE_HTTPONLY': True,
    'CSRF_COOKIE_SAMESITE': 'Strict',
    'SESSION_COOKIE_SECURE': True,
    'SESSION_COOKIE_HTTPONLY': True,
    'SESSION_COOKIE_SAMESITE': 'Strict',
    'SECURE_BROWSER_XSS_FILTER': True,
    'SECURE_CONTENT_TYPE_NOSNIFF': True,
    'SECURE_REFERRER_POLICY': 'strict-origin-when-cross-origin',
}

# Database optimization settings
DATABASE_SETTINGS = {
    'CONNECTION_POOL_SIZE': 10,
    'MAX_OVERFLOW': 20,
    'POOL_TIMEOUT': 30,
    'POOL_RECYCLE': 3600,
    'QUERY_TIMEOUT': 30,
}

def get_dashboard_setting(key: str, default=None):
    """Get a dashboard setting with fallback to default"""
    return DASHBOARD_SETTINGS.get(key, default)

def get_embedding_setting(key: str, default=None):
    """Get an embedding setting with fallback to default"""
    return EMBEDDING_SETTINGS.get(key, default)