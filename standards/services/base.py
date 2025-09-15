"""
Base service class with common functionality
"""
import time
import hashlib
import logging
from typing import Any, Dict, Optional
from django.conf import settings
from django.core.cache import cache
from django.utils import timezone


logger = logging.getLogger(__name__)


class BaseService:
    """Base service class with common utilities"""
    
    def __init__(self):
        self.cache_timeout = getattr(settings, 'EDTECH_SETTINGS', {}).get('CACHE_TIMEOUT_HOURS', 24) * 3600
    
    def generate_cache_key(self, prefix: str, **kwargs) -> str:
        """Generate a cache key from parameters"""
        key_data = f"{prefix}:" + ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if available"""
        return cache.get(cache_key)
    
    def set_cached_result(self, cache_key: str, data: Any, timeout: Optional[int] = None) -> None:
        """Set cached result"""
        timeout = timeout or self.cache_timeout
        cache.set(cache_key, data, timeout)
    
    def measure_execution_time(self, func, *args, **kwargs) -> tuple[Any, float]:
        """Measure execution time of a function"""
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    
    def log_performance(self, operation: str, execution_time: float, **metadata):
        """Log performance metrics"""
        logger.info(
            f"Performance: {operation} took {execution_time:.3f}s",
            extra={'operation': operation, 'execution_time': execution_time, **metadata}
        )
    
    def get_edtech_setting(self, key: str, default: Any = None) -> Any:
        """Get EdTech-specific setting"""
        return getattr(settings, 'EDTECH_SETTINGS', {}).get(key, default)
    
    def validate_parameters(self, required_params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Validate required parameters"""
        validated = {}
        for param, param_type in required_params.items():
            if param not in kwargs:
                raise ValueError(f"Required parameter '{param}' is missing")
            
            value = kwargs[param]
            if not isinstance(value, param_type):
                raise TypeError(f"Parameter '{param}' must be of type {param_type.__name__}")
            
            validated[param] = value
        
        return validated
    
    def handle_service_error(self, operation: str, error: Exception, **context):
        """Handle service errors consistently"""
        logger.error(
            f"Service error in {operation}: {str(error)}",
            extra={'operation': operation, 'error_type': type(error).__name__, **context},
            exc_info=True
        )
        raise error