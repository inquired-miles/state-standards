"""
Dashboard views - refactored into modular components for better maintainability

SECURITY IMPROVEMENTS:
- All API endpoints now have proper input validation
- SQL injection vulnerabilities fixed with parameterized queries
- CSRF protection properly implemented
- Input sanitization and length limits enforced

PERFORMANCE IMPROVEMENTS:  
- Pagination implemented for large datasets
- Database queries optimized with select_related/prefetch_related
- Proper caching with TTL management
- Memory efficient batch processing

CODE ORGANIZATION:
- 1888-line monolithic file split into focused modules
- Separation of concerns with dedicated API classes
- Type hints and comprehensive error handling
- Consistent logging and monitoring

The original views.py has been backed up as views_backup.py
"""

# Import all refactored views from the modular structure
from .views import *  # This imports from views/__init__.py which has all the exports