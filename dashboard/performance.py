"""
Performance optimization utilities for dashboard operations
"""
import time
import psutil
from typing import Any, Dict, List, Optional, Callable
from django.core.cache import cache
from django.db import connection
from django.db.models import QuerySet
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    @staticmethod
    @contextmanager
    def measure_time(operation_name: str, log_slow_queries: bool = True):
        """Context manager to measure operation time"""
        start_time = time.time()
        start_queries = len(connection.queries)
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            # Count database queries
            query_count = len(connection.queries) - start_queries
            
            # Log performance metrics
            metrics = {
                'operation': operation_name,
                'duration_seconds': round(duration, 3),
                'query_count': query_count,
                'memory_usage_mb': round(psutil.Process().memory_info().rss / 1024 / 1024, 2)
            }
            
            if duration > 1.0 or query_count > 10:  # Log slow operations
                logger.warning(f"SLOW_OPERATION: {operation_name}", extra=metrics)
            else:
                logger.info(f"PERFORMANCE: {operation_name}", extra=metrics)
            
            # Log slow queries if enabled
            if log_slow_queries and query_count > 0:
                recent_queries = connection.queries[-query_count:]
                slow_queries = [q for q in recent_queries if float(q.get('time', 0)) > 0.1]
                
                for query in slow_queries:
                    logger.warning(f"SLOW_QUERY: {query['sql'][:200]}...", extra={
                        'duration': query['time'],
                        'sql': query['sql']
                    })


class CacheManager:
    """Advanced caching utilities with performance optimization"""
    
    DEFAULT_TTL = 1800  # 30 minutes
    SHORT_TTL = 300     # 5 minutes
    LONG_TTL = 3600     # 1 hour
    
    @staticmethod
    def get_or_set_with_lock(key: str, factory_func: Callable, ttl: int = DEFAULT_TTL,
                            lock_timeout: int = 30) -> Any:
        """
        Get cached value or set it using factory function with distributed locking
        to prevent cache stampede
        """
        # Try to get cached value
        value = cache.get(key)
        if value is not None:
            return value
        
        # Try to acquire lock
        lock_key = f"lock:{key}"
        if cache.add(lock_key, "locked", lock_timeout):
            try:
                # We got the lock, compute the value
                with PerformanceMonitor.measure_time(f"cache_factory:{key}"):
                    value = factory_func()
                
                # Cache the computed value
                cache.set(key, value, ttl)
                return value
            finally:
                # Release lock
                cache.delete(lock_key)
        else:
            # Someone else is computing, wait and try again
            time.sleep(0.1)
            value = cache.get(key)
            if value is not None:
                return value
            
            # If still not available, compute without lock (fallback)
            logger.warning(f"Cache lock timeout for key: {key}, computing without lock")
            return factory_func()
    
    @staticmethod
    def invalidate_pattern(pattern: str):
        """Invalidate cache keys matching a pattern"""
        # Note: This requires a cache backend that supports pattern deletion
        # For Redis: cache.delete_pattern(pattern)
        # For other backends, this is a placeholder
        logger.info(f"Cache invalidation requested for pattern: {pattern}")
    
    @staticmethod
    def get_cache_stats() -> Dict[str, Any]:
        """Get cache performance statistics"""
        try:
            # This depends on cache backend
            return {
                'backend': cache.__class__.__name__,
                'hit_rate': 'N/A',  # Would need cache backend stats
                'size': 'N/A'
            }
        except Exception:
            return {'error': 'Cache stats not available'}


class QueryOptimizer:
    """Database query optimization utilities"""
    
    @staticmethod
    def optimize_queryset(queryset: QuerySet, select_related: List[str] = None,
                         prefetch_related: List[str] = None) -> QuerySet:
        """Apply common query optimizations"""
        if select_related:
            queryset = queryset.select_related(*select_related)
        
        if prefetch_related:
            queryset = queryset.prefetch_related(*prefetch_related)
        
        return queryset
    
    @staticmethod
    def batch_process(queryset: QuerySet, batch_size: int = 1000,
                     process_func: Callable = None) -> int:
        """Process large querysets in batches to manage memory"""
        processed = 0
        
        # Use iterator() to avoid loading all objects into memory
        for obj in queryset.iterator(chunk_size=batch_size):
            if process_func:
                process_func(obj)
            processed += 1
            
            # Log progress for large operations
            if processed % batch_size == 0:
                logger.info(f"Batch processed: {processed} objects")
        
        return processed
    
    @staticmethod
    def explain_query(queryset: QuerySet) -> str:
        """Get query execution plan for debugging"""
        try:
            return str(queryset.explain())
        except Exception as e:
            return f"Could not explain query: {e}"


class MemoryManager:
    """Memory usage monitoring and management"""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': round(memory_info.rss / 1024 / 1024, 2),
            'vms_mb': round(memory_info.vms / 1024 / 1024, 2),
            'percent': round(process.memory_percent(), 2)
        }
    
    @staticmethod
    def check_memory_limit(limit_mb: int = 512) -> bool:
        """Check if memory usage is within limit"""
        usage = MemoryManager.get_memory_usage()
        return usage['rss_mb'] <= limit_mb
    
    @staticmethod
    def log_memory_usage(operation: str):
        """Log current memory usage for an operation"""
        usage = MemoryManager.get_memory_usage()
        logger.info(f"MEMORY_USAGE: {operation}", extra=usage)


class ConnectionPoolManager:
    """Database connection pool monitoring"""
    
    @staticmethod
    def get_connection_stats() -> Dict[str, Any]:
        """Get database connection statistics"""
        try:
            from django.db import connections
            
            stats = {}
            for alias, conn in connections.all():
                if hasattr(conn, 'pool'):
                    stats[alias] = {
                        'pool_size': getattr(conn.pool, 'size', 'N/A'),
                        'checked_out': getattr(conn.pool, 'checkedout', 'N/A'),
                        'overflow': getattr(conn.pool, 'overflow', 'N/A'),
                    }
                else:
                    stats[alias] = {'status': 'no_pool'}
            
            return stats
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def close_old_connections():
        """Close old database connections to free resources"""
        from django.db import connections
        
        for conn in connections.all():
            conn.close_if_unusable_or_obsolete()


def performance_profile(func):
    """Decorator to profile function performance"""
    def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}"
        
        with PerformanceMonitor.measure_time(func_name):
            result = func(*args, **kwargs)
        
        return result
    
    return wrapper


def memory_limit(limit_mb: int = 512):
    """Decorator to enforce memory limits"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not MemoryManager.check_memory_limit(limit_mb):
                current_usage = MemoryManager.get_memory_usage()
                raise MemoryError(
                    f"Memory limit exceeded: {current_usage['rss_mb']}MB > {limit_mb}MB"
                )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def cache_result(key_template: str, ttl: int = CacheManager.DEFAULT_TTL):
    """Decorator to cache function results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key from template and arguments
            cache_key = key_template.format(*args, **kwargs)
            
            def factory():
                return func(*args, **kwargs)
            
            return CacheManager.get_or_set_with_lock(cache_key, factory, ttl)
        
        return wrapper
    return decorator