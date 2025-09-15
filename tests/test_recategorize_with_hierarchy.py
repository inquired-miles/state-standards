#!/usr/bin/env python
"""
Test script for re-categorizing standards with an existing topic hierarchy.
This allows testing the categorization phase independently from hierarchy generation.

Usage:
    python test_recategorize_with_hierarchy.py --run-id "topic-xyz" [options]
    
Options:
    --run-id: Existing ProxyRun ID to extract hierarchy from
    --save-hierarchy: Save hierarchy to JSON file
    --load-hierarchy: Load hierarchy from JSON file
    --test-mode: "enhanced" or "basic" (default: enhanced)
    --chunk-size: Number of standards per chunk (default: dynamic)
    --grades: Space-separated grade levels (e.g., 3 4 5)
    --subject: Subject area name
    --create-new-run: Create a new ProxyRun for results
    --dry-run: Test without saving to database
"""

import os
import sys
import django
import json
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'state_standards_project.settings')
django.setup()

from django.core.cache import cache
from django.db import transaction
from django.utils import timezone

from standards.models import (
    Standard, TopicBasedProxy, ProxyRun, State, SubjectArea, GradeLevel
)
from standards.services.topic_categorization import (
    TopicCategorizationService, TopicHierarchy, StandardCategorization
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HierarchyManager:
    """Manages extraction, saving, and loading of topic hierarchies."""
    
    @staticmethod
    def extract_from_run(run_id: str) -> Optional[TopicHierarchy]:
        """Extract topic hierarchy from an existing ProxyRun."""
        try:
            # First, try to get from cache
            cache_key = f'topic_categorization_job_{run_id.replace("topic-", "")}'
            cached_data = cache.get(cache_key)
            
            if cached_data and 'results' in cached_data:
                hierarchy_data = cached_data.get('results', {}).get('hierarchy', {})
                if hierarchy_data and 'topics' in hierarchy_data:
                    logger.info(f"✅ Found hierarchy in cache for run {run_id}")
                    return TopicHierarchy(topics=hierarchy_data['topics'])
            
            # If not in cache, reconstruct from TopicBasedProxy objects
            logger.info(f"Cache miss, reconstructing hierarchy from proxies for run {run_id}")
            return HierarchyManager.reconstruct_from_proxies(run_id)
            
        except Exception as e:
            logger.error(f"Failed to extract hierarchy from run {run_id}: {e}")
            return None
    
    @staticmethod
    def reconstruct_from_proxies(run_id: str) -> Optional[TopicHierarchy]:
        """Reconstruct hierarchy from TopicBasedProxy objects associated with a run."""
        try:
            # Get the ProxyRun
            proxy_run = ProxyRun.objects.get(run_id=run_id)
            
            # Get all TopicBasedProxy objects created in the same time window
            # (within 1 hour of the run completion)
            time_window_start = proxy_run.started_at
            time_window_end = proxy_run.completed_at or (proxy_run.started_at + timezone.timedelta(hours=1))
            
            proxies = TopicBasedProxy.objects.filter(
                created_at__gte=time_window_start,
                created_at__lte=time_window_end
            ).exclude(outlier_category=True)
            
            # Build hierarchy structure from proxies
            hierarchy_dict = {}
            
            for proxy in proxies:
                topic = proxy.topic
                sub_topic = proxy.sub_topic
                sub_sub_topic = proxy.sub_sub_topic
                
                if topic not in hierarchy_dict:
                    hierarchy_dict[topic] = {
                        'name': topic,
                        'description': f'Topic category: {topic}',
                        'sub_topics': {}
                    }
                
                if sub_topic not in hierarchy_dict[topic]['sub_topics']:
                    hierarchy_dict[topic]['sub_topics'][sub_topic] = {
                        'name': sub_topic,
                        'description': f'Sub-topic: {sub_topic}',
                        'sub_sub_topics': set()
                    }
                
                hierarchy_dict[topic]['sub_topics'][sub_topic]['sub_sub_topics'].add(sub_sub_topic)
            
            # Convert to proper format
            topics = []
            for topic_data in hierarchy_dict.values():
                sub_topics = []
                for sub_topic_data in topic_data['sub_topics'].values():
                    sub_topics.append({
                        'name': sub_topic_data['name'],
                        'description': sub_topic_data['description'],
                        'sub_sub_topics': list(sub_topic_data['sub_sub_topics'])
                    })
                
                topics.append({
                    'name': topic_data['name'],
                    'description': topic_data['description'],
                    'sub_topics': sub_topics
                })
            
            logger.info(f"✅ Reconstructed hierarchy with {len(topics)} topics from {len(proxies)} proxies")
            return TopicHierarchy(topics=topics)
            
        except ProxyRun.DoesNotExist:
            logger.error(f"ProxyRun with ID {run_id} not found")
            return None
        except Exception as e:
            logger.error(f"Failed to reconstruct hierarchy: {e}")
            return None
    
    @staticmethod
    def save_to_file(hierarchy: TopicHierarchy, filename: str):
        """Save hierarchy to JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump({'topics': hierarchy.topics}, f, indent=2)
            logger.info(f"✅ Saved hierarchy to {filename}")
        except Exception as e:
            logger.error(f"Failed to save hierarchy: {e}")
    
    @staticmethod
    def load_from_file(filename: str) -> Optional[TopicHierarchy]:
        """Load hierarchy from JSON file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            if 'topics' not in data:
                logger.error("Invalid hierarchy file: missing 'topics' key")
                return None
            
            logger.info(f"✅ Loaded hierarchy from {filename}")
            return TopicHierarchy(topics=data['topics'])
            
        except FileNotFoundError:
            logger.error(f"Hierarchy file not found: {filename}")
            return None
        except Exception as e:
            logger.error(f"Failed to load hierarchy: {e}")
            return None


class CategorizationTester:
    """Manages testing of categorization with different configurations."""
    
    def __init__(self, service: TopicCategorizationService):
        self.service = service
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'total_standards': 0,
            'successful_categorizations': 0,
            'outliers': 0,
            'chunks_processed': 0,
            'avg_confidence': 0.0,
            'errors': []
        }
    
    def run_test(self, 
                 hierarchy: TopicHierarchy,
                 standards: List[Standard],
                 test_mode: str = 'enhanced',
                 chunk_size: Optional[int] = None,
                 dry_run: bool = False) -> Tuple[List[StandardCategorization], Dict[str, Any]]:
        """Run categorization test with specified configuration."""
        
        logger.info(f"\n🧪 Starting categorization test")
        logger.info(f"   Mode: {test_mode}")
        logger.info(f"   Standards: {len(standards)}")
        logger.info(f"   Chunk size: {chunk_size or 'dynamic'}")
        logger.info(f"   Dry run: {dry_run}")
        
        self.metrics['start_time'] = time.time()
        self.metrics['total_standards'] = len(standards)
        
        all_categorizations = []
        
        try:
            if test_mode == 'enhanced':
                # Use enhanced chunking and categorization
                if chunk_size:
                    chunks = self._create_fixed_chunks(standards, chunk_size)
                else:
                    # Use educational chunking
                    base_chunk_size = self.service.calculate_optimal_chunk_size(standards, hierarchy)
                    chunks = self.service.create_educational_chunks(standards, base_chunk_size, hierarchy)
                    logger.info(f"📚 Created {len(chunks)} educational chunks")
            else:
                # Use basic chunking
                chunk_size = chunk_size or self.service.DEFAULT_CHUNK_SIZE
                chunks = self._create_fixed_chunks(standards, chunk_size)
            
            # Process each chunk
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"\n📦 Processing chunk {i}/{len(chunks)} ({len(chunk)} standards)")
                
                try:
                    if test_mode == 'enhanced':
                        categorizations = self.service.categorize_standards_chunk(chunk, hierarchy)
                    else:
                        # Use basic fallback method
                        categorizations = self.service._fallback_basic_categorization(chunk, hierarchy)
                    
                    all_categorizations.extend(categorizations)
                    self.metrics['chunks_processed'] += 1
                    
                    # Collect metrics
                    successful = [c for c in categorizations if not c.is_outlier]
                    outliers = [c for c in categorizations if c.is_outlier]
                    
                    self.metrics['successful_categorizations'] += len(successful)
                    self.metrics['outliers'] += len(outliers)
                    
                    # Calculate average confidence if available
                    if successful and hasattr(successful[0], 'confidence_score'):
                        confidences = [c.confidence_score for c in successful if c.confidence_score is not None]
                        if confidences:
                            chunk_avg_confidence = sum(confidences) / len(confidences)
                            logger.info(f"   ✅ Categorized: {len(successful)}, Outliers: {len(outliers)}, Avg confidence: {chunk_avg_confidence:.2f}")
                    else:
                        logger.info(f"   ✅ Categorized: {len(successful)}, Outliers: {len(outliers)}")
                    
                except Exception as e:
                    logger.error(f"   ❌ Chunk {i} failed: {e}")
                    self.metrics['errors'].append(f"Chunk {i}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Test failed: {e}")
            self.metrics['errors'].append(f"Test failed: {str(e)}")
        
        finally:
            self.metrics['end_time'] = time.time()
            self.metrics['duration'] = self.metrics['end_time'] - self.metrics['start_time']
            
            # Calculate final average confidence
            if all_categorizations:
                confidences = [c.confidence_score for c in all_categorizations 
                             if hasattr(c, 'confidence_score') and c.confidence_score is not None]
                if confidences:
                    self.metrics['avg_confidence'] = sum(confidences) / len(confidences)
        
        return all_categorizations, self.metrics
    
    def _create_fixed_chunks(self, standards: List[Standard], chunk_size: int) -> List[List[Standard]]:
        """Create fixed-size chunks of standards."""
        chunks = []
        for i in range(0, len(standards), chunk_size):
            chunks.append(standards[i:i + chunk_size])
        return chunks
    
    def save_results(self, 
                    categorizations: List[StandardCategorization],
                    create_new_run: bool = False,
                    run_name: str = None) -> Optional[ProxyRun]:
        """Save categorization results to database."""
        
        if not categorizations:
            logger.warning("No categorizations to save")
            return None
        
        try:
            with transaction.atomic():
                # Create new ProxyRun if requested
                proxy_run = None
                if create_new_run:
                    run_id = f"test-{timezone.now().strftime('%Y%m%d-%H%M%S')}"
                    proxy_run = ProxyRun.objects.create(
                        run_id=run_id,
                        name=run_name or f"Test categorization {timezone.now().strftime('%Y-%m-%d %H:%M')}",
                        description="Test categorization with existing hierarchy",
                        run_type='topics',
                        status='completed',
                        started_at=timezone.now() - timezone.timedelta(seconds=self.metrics['duration']),
                        completed_at=timezone.now(),
                        duration_seconds=int(self.metrics['duration']),
                        total_input_standards=self.metrics['total_standards'],
                        total_proxies_created=len(categorizations),
                        outlier_proxies_count=self.metrics['outliers']
                    )
                    logger.info(f"✅ Created new ProxyRun: {run_id}")
                
                # Create TopicBasedProxy objects
                proxies = self.service.create_topic_proxies(categorizations)
                logger.info(f"✅ Created {len(proxies)} TopicBasedProxy objects")
                
                return proxy_run
                
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return None


def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(description='Test categorization with existing topic hierarchy')
    parser.add_argument('--run-id', required=True, help='Existing ProxyRun ID to extract hierarchy from')
    parser.add_argument('--save-hierarchy', help='Save hierarchy to JSON file')
    parser.add_argument('--load-hierarchy', help='Load hierarchy from JSON file (overrides --run-id)')
    parser.add_argument('--test-mode', choices=['enhanced', 'basic'], default='enhanced', 
                       help='Categorization mode to test')
    parser.add_argument('--chunk-size', type=int, help='Fixed chunk size (overrides dynamic sizing)')
    parser.add_argument('--grades', nargs='+', type=int, help='Grade levels to filter standards')
    parser.add_argument('--subject', help='Subject area name')
    parser.add_argument('--create-new-run', action='store_true', help='Create new ProxyRun for results')
    parser.add_argument('--run-name', help='Name for new ProxyRun')
    parser.add_argument('--dry-run', action='store_true', help='Test without saving to database')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize service
    service = TopicCategorizationService()
    
    # Check if service is properly configured
    if not service.client:
        logger.error("❌ OpenAI client not configured. Set OPENAI_API_KEY environment variable.")
        return 1
    
    # Load or extract hierarchy
    hierarchy = None
    
    if args.load_hierarchy:
        logger.info(f"Loading hierarchy from file: {args.load_hierarchy}")
        hierarchy = HierarchyManager.load_from_file(args.load_hierarchy)
    else:
        logger.info(f"Extracting hierarchy from run: {args.run_id}")
        hierarchy = HierarchyManager.extract_from_run(args.run_id)
        
        if hierarchy and args.save_hierarchy:
            HierarchyManager.save_to_file(hierarchy, args.save_hierarchy)
    
    if not hierarchy:
        logger.error("❌ Failed to load topic hierarchy")
        return 1
    
    logger.info(f"✅ Loaded hierarchy with {len(hierarchy.topics)} topics")
    
    # Load standards
    try:
        # Build filter parameters
        subject_area_id = None
        if args.subject:
            try:
                subject_area = SubjectArea.objects.get(name__icontains=args.subject)
                subject_area_id = subject_area.id
                logger.info(f"✅ Found subject area: {subject_area.name}")
            except SubjectArea.DoesNotExist:
                logger.warning(f"Subject area '{args.subject}' not found, proceeding without filter")
        
        standards = service.load_standards(
            grade_levels=args.grades,
            subject_area_id=subject_area_id
        )
        
        logger.info(f"✅ Loaded {len(standards)} standards")
        
    except Exception as e:
        logger.error(f"❌ Failed to load standards: {e}")
        return 1
    
    # Run categorization test
    tester = CategorizationTester(service)
    categorizations, metrics = tester.run_test(
        hierarchy=hierarchy,
        standards=standards,
        test_mode=args.test_mode,
        chunk_size=args.chunk_size,
        dry_run=args.dry_run
    )
    
    # Display results
    logger.info("\n" + "="*60)
    logger.info("📊 TEST RESULTS")
    logger.info("="*60)
    logger.info(f"Duration: {metrics['duration']:.2f} seconds")
    logger.info(f"Standards processed: {metrics['total_standards']}")
    logger.info(f"Successful categorizations: {metrics['successful_categorizations']}")
    logger.info(f"Outliers: {metrics['outliers']}")
    logger.info(f"Chunks processed: {metrics['chunks_processed']}")
    
    if metrics['avg_confidence'] > 0:
        logger.info(f"Average confidence: {metrics['avg_confidence']:.2f}")
    
    if metrics['errors']:
        logger.info(f"Errors encountered: {len(metrics['errors'])}")
        for error in metrics['errors'][:5]:  # Show first 5 errors
            logger.error(f"  - {error}")
    
    # Save results if not dry run
    if not args.dry_run and categorizations:
        logger.info("\n💾 Saving results...")
        proxy_run = tester.save_results(
            categorizations,
            create_new_run=args.create_new_run,
            run_name=args.run_name
        )
        
        if proxy_run:
            logger.info(f"✅ Results saved to ProxyRun: {proxy_run.run_id}")
    
    logger.info("\n✅ Test completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())