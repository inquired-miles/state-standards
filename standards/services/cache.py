"""
Caching service for the EdTech Standards Alignment System
"""
import json
import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
from django.core.cache import cache
from django.utils import timezone
from django.db import models
from .base import BaseService
from ..models import CacheEntry


class CacheService(BaseService):
    """Enhanced caching service with database backup"""
    
    def __init__(self):
        super().__init__()
        self.default_timeout = self.cache_timeout
    
    def get_or_compute(
        self,
        cache_key: str,
        compute_func,
        cache_type: str = 'general',
        timeout: Optional[int] = None,
        use_db_cache: bool = True,
        *args,
        **kwargs
    ) -> Any:
        """
        Get cached result or compute and cache it
        
        Args:
            cache_key: Unique cache key
            compute_func: Function to compute the result
            cache_type: Type of cache for categorization
            timeout: Cache timeout in seconds
            use_db_cache: Whether to use database cache as backup
            *args, **kwargs: Arguments for compute_func
            
        Returns:
            Cached or computed result
        """
        timeout = timeout or self.default_timeout
        
        # Try Redis cache first
        result = cache.get(cache_key)
        if result is not None:
            return result
        
        # Try database cache if enabled
        if use_db_cache:
            db_result = self._get_from_db_cache(cache_key)
            if db_result is not None:
                # Refresh Redis cache
                cache.set(cache_key, db_result, timeout)
                return db_result
        
        # Compute result
        start_time = timezone.now()
        result = compute_func(*args, **kwargs)
        computation_time = (timezone.now() - start_time).total_seconds()
        
        # Cache result
        self._set_cache(cache_key, result, cache_type, timeout, computation_time, use_db_cache)
        
        return result
    
    def _set_cache(
        self,
        cache_key: str,
        data: Any,
        cache_type: str,
        timeout: int,
        computation_time: float,
        use_db_cache: bool
    ):
        """Set cache in both Redis and database"""
        
        # Set Redis cache
        cache.set(cache_key, data, timeout)
        
        # Set database cache if enabled
        if use_db_cache:
            self._set_db_cache(cache_key, data, cache_type, timeout, computation_time)
    
    def _get_from_db_cache(self, cache_key: str) -> Optional[Any]:
        """Get result from database cache"""
        try:
            cache_entry = CacheEntry.objects.get(cache_key=cache_key)
            
            if cache_entry.is_expired:
                # Clean up expired entry
                cache_entry.delete()
                return None
            
            return cache_entry.cache_value
            
        except CacheEntry.DoesNotExist:
            return None
    
    def _set_db_cache(
        self,
        cache_key: str,
        data: Any,
        cache_type: str,
        timeout: int,
        computation_time: float
    ):
        """Set result in database cache"""
        try:
            # Generate parameters hash for the cache key
            parameters_hash = hashlib.md5(cache_key.encode()).hexdigest()
            
            # Calculate expiry time
            expires_at = timezone.now() + timedelta(seconds=timeout)
            
            # Create or update cache entry
            CacheEntry.objects.update_or_create(
                cache_key=cache_key,
                defaults={
                    'cache_value': data,
                    'cache_type': cache_type,
                    'parameters_hash': parameters_hash,
                    'computation_time': computation_time,
                    'expires_at': expires_at
                }
            )
            
        except Exception as e:
            self.handle_service_error("_set_db_cache", e, cache_key=cache_key)
    
    def invalidate_cache(self, cache_key: str):
        """Invalidate cache entry"""
        # Remove from Redis
        cache.delete(cache_key)
        
        # Remove from database
        try:
            CacheEntry.objects.filter(cache_key=cache_key).delete()
        except Exception as e:
            self.handle_service_error("invalidate_cache", e, cache_key=cache_key)
    
    def invalidate_cache_pattern(self, pattern: str):
        """Invalidate cache entries matching a pattern"""
        # For Redis, this would require additional setup
        # For now, just handle database cache
        try:
            CacheEntry.objects.filter(cache_key__icontains=pattern).delete()
        except Exception as e:
            self.handle_service_error("invalidate_cache_pattern", e, pattern=pattern)
    
    def clean_expired_cache(self):
        """Clean up expired cache entries"""
        try:
            expired_count = CacheEntry.objects.filter(
                expires_at__lt=timezone.now()
            ).count()
            
            CacheEntry.objects.filter(expires_at__lt=timezone.now()).delete()
            
            return expired_count
            
        except Exception as e:
            self.handle_service_error("clean_expired_cache", e)
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            stats = {
                'total_entries': CacheEntry.objects.count(),
                'expired_entries': CacheEntry.objects.filter(
                    expires_at__lt=timezone.now()
                ).count(),
                'cache_types': dict(
                    CacheEntry.objects.values('cache_type').annotate(
                        count=models.Count('id')
                    ).values_list('cache_type', 'count')
                ),
                'average_computation_time': CacheEntry.objects.aggregate(
                    avg_time=models.Avg('computation_time')
                )['avg_time'] or 0
            }
            
            return stats
            
        except Exception as e:
            self.handle_service_error("get_cache_stats", e)
            return {}