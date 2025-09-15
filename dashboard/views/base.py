"""
Base view classes and utilities for dashboard views
"""
from typing import Dict, Any, Optional, List, Union
from django.http import JsonResponse, Http404
from django.core.exceptions import ValidationError, PermissionDenied
from django.views.decorators.csrf import csrf_protect
from django.utils.decorators import method_decorator
from django.contrib.admin.views.decorators import staff_member_required
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.db.models import QuerySet
import logging
import json

logger = logging.getLogger(__name__)


class ValidationMixin:
    """Mixin for input validation"""
    
    @staticmethod
    def validate_integer(value: Any, min_val: Optional[int] = None, max_val: Optional[int] = None, 
                        field_name: str = "value") -> int:
        """Validate integer input with bounds checking"""
        try:
            int_val = int(value)
            if min_val is not None and int_val < min_val:
                raise ValidationError(f"{field_name} must be >= {min_val}")
            if max_val is not None and int_val > max_val:
                raise ValidationError(f"{field_name} must be <= {max_val}")
            return int_val
        except (ValueError, TypeError):
            raise ValidationError(f"{field_name} must be a valid integer")
    
    @staticmethod
    def validate_float(value: Any, min_val: Optional[float] = None, max_val: Optional[float] = None,
                      field_name: str = "value") -> float:
        """Validate float input with bounds checking"""
        try:
            float_val = float(value)
            if min_val is not None and float_val < min_val:
                raise ValidationError(f"{field_name} must be >= {min_val}")
            if max_val is not None and float_val > max_val:
                raise ValidationError(f"{field_name} must be <= {max_val}")
            return float_val
        except (ValueError, TypeError):
            raise ValidationError(f"{field_name} must be a valid number")
    
    @staticmethod
    def validate_string(value: Any, max_length: Optional[int] = None, 
                       min_length: Optional[int] = None, field_name: str = "value") -> str:
        """Validate string input with length checking"""
        if value is None:
            raise ValidationError(f"{field_name} is required")
        
        str_val = str(value).strip()
        
        if min_length is not None and len(str_val) < min_length:
            raise ValidationError(f"{field_name} must be at least {min_length} characters")
        if max_length is not None and len(str_val) > max_length:
            raise ValidationError(f"{field_name} must be no more than {max_length} characters")
        
        return str_val
    
    @staticmethod
    def validate_list_of_integers(value: Union[str, List], separator: str = ",", 
                                 field_name: str = "value") -> List[int]:
        """Validate comma-separated integers or list of integers"""
        if isinstance(value, list):
            try:
                return [int(v) for v in value]
            except (ValueError, TypeError):
                raise ValidationError(f"{field_name} must be a list of integers")
        
        if isinstance(value, str):
            try:
                if not value.strip():
                    return []
                return [int(v.strip()) for v in value.split(separator) if v.strip()]
            except ValueError:
                raise ValidationError(f"{field_name} must be comma-separated integers")
        
        raise ValidationError(f"{field_name} must be a string or list")


class PaginationMixin:
    """Mixin for pagination utilities"""
    
    DEFAULT_PAGE_SIZE = 25
    MAX_PAGE_SIZE = 100
    
    def paginate_queryset(self, queryset: QuerySet, request, page_size: Optional[int] = None) -> Dict[str, Any]:
        """Paginate a queryset and return pagination data"""
        if page_size is None:
            page_size = self.DEFAULT_PAGE_SIZE
        
        # Validate page size
        try:
            page_size = min(int(page_size), self.MAX_PAGE_SIZE)
        except (ValueError, TypeError):
            page_size = self.DEFAULT_PAGE_SIZE
        
        paginator = Paginator(queryset, page_size)
        
        page = request.GET.get('page', 1)
        try:
            page_obj = paginator.page(page)
        except PageNotAnInteger:
            page_obj = paginator.page(1)
        except EmptyPage:
            page_obj = paginator.page(paginator.num_pages)
        
        return {
            'results': list(page_obj),
            'pagination': {
                'current_page': page_obj.number,
                'total_pages': paginator.num_pages,
                'total_count': paginator.count,
                'page_size': page_size,
                'has_next': page_obj.has_next(),
                'has_previous': page_obj.has_previous(),
                'next_page': page_obj.next_page_number() if page_obj.has_next() else None,
                'previous_page': page_obj.previous_page_number() if page_obj.has_previous() else None,
            }
        }


class APIResponseMixin:
    """Mixin for standardized API responses"""
    
    @staticmethod
    def success_response(data: Any = None, message: str = None, status: int = 200) -> JsonResponse:
        """Return a standardized success response"""
        response_data = {'success': True}
        if data is not None:
            response_data['data'] = data
        if message:
            response_data['message'] = message
        return JsonResponse(response_data, status=status)
    
    @staticmethod
    def error_response(message: str, status: int = 400, error_code: str = None, 
                      details: Dict = None) -> JsonResponse:
        """Return a standardized error response"""
        response_data = {
            'success': False,
            'error': message
        }
        if error_code:
            response_data['error_code'] = error_code
        if details:
            response_data['details'] = details
        return JsonResponse(response_data, status=status)
    
    @staticmethod
    def validation_error_response(errors: Union[Dict, List, str]) -> JsonResponse:
        """Return a validation error response"""
        if isinstance(errors, str):
            message = errors
            details = None
        elif isinstance(errors, dict):
            message = "Validation failed"
            details = errors
        else:
            message = "Validation failed"
            details = {'errors': errors}
        
        return APIResponseMixin.error_response(
            message=message,
            status=400,
            error_code='VALIDATION_ERROR',
            details=details
        )


class BaseAPIView(ValidationMixin, PaginationMixin, APIResponseMixin):
    """Base class for API views with common functionality"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__module__)
    
    def handle_exception(self, request, e: Exception) -> JsonResponse:
        """Handle exceptions with appropriate logging and responses"""
        self.logger.error(f"Error in {self.__class__.__name__}: {str(e)}", exc_info=True)
        
        if isinstance(e, ValidationError):
            return self.validation_error_response(str(e))
        elif isinstance(e, PermissionDenied):
            return self.error_response("Permission denied", status=403)
        elif isinstance(e, Http404):
            return self.error_response("Resource not found", status=404)
        else:
            # Don't expose internal errors in production
            message = "An internal error occurred"
            return self.error_response(message, status=500, error_code='INTERNAL_ERROR')
    
    def parse_json_body(self, request) -> Dict[str, Any]:
        """Safely parse JSON request body"""
        try:
            if not request.body:
                return {}
            return json.loads(request.body)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON: {str(e)}")


def secure_staff_required(view_func):
    """Decorator that combines staff requirement with CSRF protection"""
    @method_decorator(csrf_protect, name='dispatch')
    @staff_member_required
    def wrapper(*args, **kwargs):
        return view_func(*args, **kwargs)
    return wrapper


def api_endpoint(methods: List[str] = None, csrf_exempt: bool = True):
    """Decorator for API endpoints with security and validation"""
    if methods is None:
        methods = ['GET']
    
    def decorator(view_func):
        from django.views.decorators.http import require_http_methods
        from django.views.decorators.csrf import csrf_exempt as django_csrf_exempt
        
        decorated_view = staff_member_required(view_func)
        decorated_view = require_http_methods(methods)(decorated_view)
        
        if csrf_exempt:
            decorated_view = django_csrf_exempt(decorated_view)
        
        return decorated_view
    
    return decorator