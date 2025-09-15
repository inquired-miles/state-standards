"""
Enhanced service layer for the EdTech Standards Alignment System
"""
from .base import BaseService
from .embedding import EmbeddingService
from .coverage import CoverageAnalysisService
from .discovery import TopicDiscoveryService
from .alignment import ContentAlignmentService
from .planning import StrategicPlanningService
from .cache import CacheService
from .search import SearchService
from .bell_curve import BellCurveAnalysisService
from .storyline import StorylineDiscoveryService
from .correlation import StandardCorrelationService

__all__ = [
    'BaseService',
    'EmbeddingService',
    'CoverageAnalysisService',
    'TopicDiscoveryService',
    'ContentAlignmentService',
    'StrategicPlanningService',
    'CacheService',
    'SearchService',
    'BellCurveAnalysisService',
    'StorylineDiscoveryService',
    'StandardCorrelationService',
]