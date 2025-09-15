"""
Security utilities for dashboard operations
"""
import re
import html
import hashlib
import time
from typing import Dict, Any, Optional, List
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.utils.html import strip_tags
import logging

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Centralized security validation for dashboard inputs"""
    
    # Regex patterns for validation
    UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    STATE_CODE_PATTERN = re.compile(r'^[A-Z]{2}$')
    ALPHANUMERIC_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\s]+$')
    
    @staticmethod
    def validate_uuid(value: str, field_name: str = "UUID") -> str:
        """Validate UUID format"""
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string")
        
        value = value.strip()
        if not SecurityValidator.UUID_PATTERN.match(value):
            raise ValidationError(f"{field_name} must be a valid UUID")
        
        return value
    
    @staticmethod
    def validate_state_code(value: str, field_name: str = "state_code") -> str:
        """Validate US state code format"""
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string")
        
        value = value.strip().upper()
        if not SecurityValidator.STATE_CODE_PATTERN.match(value):
            raise ValidationError(f"{field_name} must be a valid 2-letter state code")
        
        return value
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000, 
                       allow_html: bool = False) -> str:
        """Sanitize string input to prevent XSS and injection attacks"""
        if not isinstance(value, str):
            return str(value)
        
        # Strip or escape HTML
        if not allow_html:
            value = strip_tags(value)
        else:
            value = html.escape(value)
        
        # Limit length
        if len(value) > max_length:
            value = value[:max_length]
        
        return value.strip()
    
    @staticmethod
    def validate_json_structure(data: Dict[str, Any], 
                               required_fields: List[str] = None,
                               allowed_fields: List[str] = None) -> Dict[str, Any]:
        """Validate JSON structure and fields"""
        if not isinstance(data, dict):
            raise ValidationError("Data must be a JSON object")
        
        # Check required fields
        if required_fields:
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise ValidationError(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Filter allowed fields
        if allowed_fields:
            filtered_data = {k: v for k, v in data.items() if k in allowed_fields}
            return filtered_data
        
        return data
    
    @staticmethod
    def validate_sql_safe_string(value: str, field_name: str = "value") -> str:
        """Validate string is safe for SQL operations (no SQL injection patterns)"""
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string")
        
        # Check for SQL injection patterns
        dangerous_patterns = [
            r"'.*?'",  # Single quotes
            r"--",     # SQL comments
            r";",      # Statement terminators
            r"union\s+select",  # UNION SELECT
            r"drop\s+table",    # DROP TABLE
            r"delete\s+from",   # DELETE FROM
            r"insert\s+into",   # INSERT INTO
            r"update\s+.*?set", # UPDATE SET
        ]
        
        value_lower = value.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, value_lower):
                raise ValidationError(f"{field_name} contains potentially dangerous characters")
        
        return value


class RateLimiter:
    """Rate limiting for API endpoints"""
    
    @staticmethod
    def check_rate_limit(request, endpoint: str, limit: int = 60, 
                        window: int = 60) -> bool:
        """
        Check if request is within rate limit
        
        Args:
            request: Django request object
            endpoint: API endpoint identifier
            limit: Number of requests allowed per window
            window: Time window in seconds
            
        Returns:
            True if within limit, False if rate limited
        """
        # Get client identifier (IP + user if available)
        client_ip = get_client_ip(request)
        user_id = request.user.id if request.user.is_authenticated else 'anonymous'
        cache_key = f"rate_limit:{endpoint}:{client_ip}:{user_id}"
        
        # Get current request count
        current_requests = cache.get(cache_key, 0)
        
        if current_requests >= limit:
            logger.warning(f"Rate limit exceeded for {client_ip} on {endpoint}")
            return False
        
        # Increment counter
        cache.set(cache_key, current_requests + 1, window)
        return True
    
    @staticmethod
    def get_rate_limit_info(request, endpoint: str, limit: int = 60,
                           window: int = 60) -> Dict[str, Any]:
        """Get rate limit information for client"""
        client_ip = get_client_ip(request)
        user_id = request.user.id if request.user.is_authenticated else 'anonymous'
        cache_key = f"rate_limit:{endpoint}:{client_ip}:{user_id}"
        
        current_requests = cache.get(cache_key, 0)
        remaining = max(0, limit - current_requests)
        
        return {
            'limit': limit,
            'used': current_requests,
            'remaining': remaining,
            'reset_in': cache.ttl(cache_key) or window
        }


class CSRFValidator:
    """CSRF protection utilities"""
    
    @staticmethod
    def validate_csrf_token(request) -> bool:
        """Validate CSRF token for API requests"""
        from django.middleware.csrf import get_token
        from django.views.decorators.csrf import csrf_exempt
        
        # Check if view is CSRF exempt
        view_func = getattr(request, 'resolver_match', None)
        if view_func and hasattr(view_func.func, 'csrf_exempt'):
            return True
        
        # Validate CSRF token
        token = request.META.get('HTTP_X_CSRFTOKEN') or request.POST.get('csrfmiddlewaretoken')
        expected_token = get_token(request)
        
        return token == expected_token


def get_client_ip(request) -> str:
    """Get client IP address from request"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        return x_forwarded_for.split(',')[0].strip()
    
    x_real_ip = request.META.get('HTTP_X_REAL_IP')
    if x_real_ip:
        return x_real_ip.strip()
    
    return request.META.get('REMOTE_ADDR', 'unknown')


def hash_sensitive_data(data: str, salt: str = '') -> str:
    """Hash sensitive data for logging/caching without exposing values"""
    hasher = hashlib.sha256()
    hasher.update((data + salt).encode('utf-8'))
    return hasher.hexdigest()[:16]  # First 16 chars for brevity


def audit_log(request, action: str, details: Dict[str, Any] = None):
    """Log security-relevant actions for audit trail"""
    client_ip = get_client_ip(request)
    user_id = request.user.id if request.user.is_authenticated else None
    
    log_data = {
        'timestamp': time.time(),
        'action': action,
        'client_ip': client_ip,
        'user_id': user_id,
        'user_agent': request.META.get('HTTP_USER_AGENT', 'unknown'),
        'details': details or {}
    }
    
    logger.info(f"AUDIT: {action}", extra=log_data)