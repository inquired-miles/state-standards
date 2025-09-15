"""
ProxyRunAnalyzer service for comprehensive analysis of proxy generation runs.
Provides coverage analysis, topic prevalence, bell curve visualizations, and state comparisons.
"""
import logging
import time
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
import statistics

logger = logging.getLogger(__name__)

from django.db.models import Count, Q, Avg
from .base import BaseService
from ..models import (
    ProxyRun, ProxyRunReport, ProxyStandard, TopicBasedProxy, 
    Standard, State, SubjectArea, GradeLevel
)


class ProxyRunAnalyzer(BaseService):
    """Service for analyzing completed proxy runs and generating comprehensive reports."""
    
    def __init__(self):
        super().__init__()
    
    def analyze_run(self, proxy_run: ProxyRun, force_regenerate: bool = False) -> ProxyRunReport:
        """Generate comprehensive analysis report for a proxy run."""
        start_time = time.time()
        
        # Check if report already exists
        if hasattr(proxy_run, 'report') and not force_regenerate:
            logger.info(f"Report already exists for run {proxy_run.run_id}")
            return proxy_run.report
        
        logger.info(f"Analyzing proxy run: {proxy_run.name} ({proxy_run.run_type})")
        
        # Get associated proxies
        proxies = proxy_run.get_associated_proxies()
        if not proxies.exists():
            logger.warning(f"No proxies found for run {proxy_run.run_id}")
            return self._create_empty_report(proxy_run)
        
        # Perform different analyses based on run type
        if proxy_run.run_type == 'topics':
            analysis_data = self._analyze_topic_run(proxy_run, proxies)
        else:
            analysis_data = self._analyze_clustering_run(proxy_run, proxies)
        
        # Create or update report
        report, created = ProxyRunReport.objects.get_or_create(
            run=proxy_run,
            defaults=analysis_data
        )
        
        if not created:
            # Update existing report
            for key, value in analysis_data.items():
                setattr(report, key, value)
            report.save()
        
        # Record generation time
        generation_time = time.time() - start_time
        report.generation_time_seconds = generation_time
        report.save(update_fields=['generation_time_seconds'])
        
        logger.info(f"Analysis completed in {generation_time:.2f}s for run {proxy_run.run_id}")
        return report
    
    def _analyze_topic_run(self, proxy_run: ProxyRun, proxies) -> Dict[str, Any]:
        """Analyze a topic-based categorization run."""
        analysis_data = {}
        
        # State-by-state coverage analysis
        analysis_data['state_coverage'] = self._calculate_state_coverage_topics(proxies)
        
        # Topic prevalence across states
        analysis_data['topic_prevalence'] = self._calculate_topic_prevalence(proxies)
        
        # Coverage distribution (bell curve data)
        analysis_data['coverage_distribution'] = self._calculate_coverage_distribution_topics(proxies)
        
        # Outlier analysis
        analysis_data['outlier_analysis'] = self._analyze_outliers_topics(proxies)
        
        # Topic hierarchy statistics
        analysis_data['topic_hierarchy_stats'] = self._calculate_hierarchy_stats(proxies)
        
        # Cross-state commonality
        analysis_data['cross_state_commonality'] = self._calculate_cross_state_commonality(proxies)
        
        # Quality metrics (simplified for topic-based)
        analysis_data['silhouette_scores'] = {}
        analysis_data['coverage_gaps'] = self._identify_coverage_gaps_topics(proxies)
        
        return analysis_data
    
    def _analyze_clustering_run(self, proxy_run: ProxyRun, proxies) -> Dict[str, Any]:
        """Analyze a traditional clustering run (atoms or standards)."""
        analysis_data = {}
        
        # State-by-state coverage analysis
        analysis_data['state_coverage'] = self._calculate_state_coverage_clustering(proxies)
        
        # For clustering runs, we'll use proxy titles as "topics" for prevalence
        analysis_data['topic_prevalence'] = self._calculate_proxy_prevalence(proxies)
        
        # Coverage distribution
        analysis_data['coverage_distribution'] = self._calculate_coverage_distribution_clustering(proxies)
        
        # No outlier analysis for traditional clustering
        analysis_data['outlier_analysis'] = {}
        
        # No hierarchy stats for traditional clustering
        analysis_data['topic_hierarchy_stats'] = {}
        analysis_data['cross_state_commonality'] = {}
        
        # Clustering quality metrics
        analysis_data['silhouette_scores'] = self._calculate_clustering_quality(proxies)
        analysis_data['coverage_gaps'] = self._identify_coverage_gaps_clustering(proxies)
        
        return analysis_data
    
    def _calculate_state_coverage_topics(self, proxies) -> Dict[str, Any]:
        """Calculate state-by-state coverage for topic-based proxies."""
        state_coverage = {}
        
        # Get all states that have standards in these proxies
        states_with_standards = set()
        for proxy in proxies:
            for standard in proxy.member_standards.select_related('state').all():
                if standard.state:
                    states_with_standards.add(standard.state.code)
        
        for state_code in states_with_standards:
            state = State.objects.get(code=state_code)
            
            # Count proxies that have standards from this state
            proxies_with_state = 0
            total_standards_in_state = 0
            topics_covered = set()
            
            for proxy in proxies:
                state_standards = proxy.member_standards.filter(state=state)
                if state_standards.exists():
                    proxies_with_state += 1
                    total_standards_in_state += state_standards.count()
                    if not proxy.outlier_category:
                        topics_covered.add(proxy.topic)
            
            # Calculate total standards for this state in the filtered dataset
            total_state_standards = Standard.objects.filter(state=state).count()
            
            state_coverage[state_code] = {
                'state_name': state.name,
                'proxies_count': proxies_with_state,
                'standards_covered': total_standards_in_state,
                'total_state_standards': total_state_standards,
                'coverage_percentage': (total_standards_in_state / total_state_standards * 100) if total_state_standards > 0 else 0,
                'topics_covered': len(topics_covered),
                'topics_list': list(topics_covered)
            }
        
        return state_coverage
    
    def _calculate_topic_prevalence(self, proxies) -> Dict[str, Any]:
        """Calculate how many states each topic appears in (topic intelligence)."""
        topic_state_map = defaultdict(set)
        topic_standards_count = defaultdict(int)
        topic_proxy_count = defaultdict(int)
        
        # Collect data for each topic
        for proxy in proxies:
            if proxy.outlier_category:
                continue  # Skip outliers for topic prevalence
                
            topic = proxy.topic
            topic_proxy_count[topic] += 1
            
            # Count standards and states for this topic
            for standard in proxy.member_standards.select_related('state').all():
                if standard.state:
                    topic_state_map[topic].add(standard.state.code)
                    topic_standards_count[topic] += 1
        
        # Build prevalence data
        prevalence = {}
        total_states = len(set().union(*topic_state_map.values())) if topic_state_map else 0
        
        for topic, states in topic_state_map.items():
            state_count = len(states)
            prevalence_percentage = (state_count / total_states * 100) if total_states > 0 else 0
            
            # Categorize topic importance
            if prevalence_percentage >= 80:
                category = 'must_have'
            elif prevalence_percentage >= 60:
                category = 'important'
            else:
                category = 'regional'
            
            prevalence[topic] = {
                'state_count': state_count,
                'total_states': total_states,
                'prevalence_percentage': prevalence_percentage,
                'category': category,
                'states_list': list(states),
                'standards_count': topic_standards_count[topic],
                'proxy_count': topic_proxy_count[topic]
            }
        
        # Sort by prevalence
        prevalence = dict(sorted(prevalence.items(), key=lambda x: x[1]['state_count'], reverse=True))
        
        return prevalence
    
    def _calculate_coverage_distribution_topics(self, proxies) -> Dict[str, Any]:
        """Calculate coverage distribution for bell curve visualization."""
        coverage_by_topic = defaultdict(list)
        coverage_by_state = defaultdict(int)
        
        # Collect coverage data
        for proxy in proxies:
            if proxy.outlier_category:
                continue
                
            topic = proxy.topic
            state_counts = defaultdict(int)
            
            for standard in proxy.member_standards.select_related('state').all():
                if standard.state:
                    state_counts[standard.state.code] += 1
            
            # Record coverage for this topic in each state
            for state_code, count in state_counts.items():
                coverage_by_topic[topic].append(count)
                coverage_by_state[state_code] += count
        
        # Calculate distribution statistics
        all_coverages = list(coverage_by_state.values())
        if all_coverages:
            distribution = {
                'mean': statistics.mean(all_coverages),
                'median': statistics.median(all_coverages),
                'std_dev': statistics.stdev(all_coverages) if len(all_coverages) > 1 else 0,
                'min': min(all_coverages),
                'max': max(all_coverages),
                'data_points': all_coverages,
                'histogram_data': self._create_histogram_data(all_coverages)
            }
        else:
            distribution = {
                'mean': 0, 'median': 0, 'std_dev': 0, 'min': 0, 'max': 0,
                'data_points': [], 'histogram_data': []
            }
        
        return {
            'overall_distribution': distribution,
            'coverage_by_topic': dict(coverage_by_topic),
            'coverage_by_state': dict(coverage_by_state)
        }
    
    def _analyze_outliers_topics(self, proxies) -> Dict[str, Any]:
        """Analyze outlier patterns in topic-based categorization."""
        outlier_proxies = proxies.filter(outlier_category=True)
        regular_proxies = proxies.filter(outlier_category=False)
        
        outlier_analysis = {
            'total_outliers': outlier_proxies.count(),
            'total_regular': regular_proxies.count(),
            'outlier_percentage': (outlier_proxies.count() / proxies.count() * 100) if proxies.count() > 0 else 0,
            'outlier_patterns': [],
            'common_outlier_reasons': []
        }
        
        # Analyze outlier patterns
        outlier_reasons = []
        for outlier in outlier_proxies:
            if outlier.sub_sub_topic:
                outlier_reasons.append(outlier.sub_sub_topic)
        
        # Count common outlier reasons
        if outlier_reasons:
            reason_counts = Counter(outlier_reasons)
            outlier_analysis['common_outlier_reasons'] = [
                {'reason': reason, 'count': count} 
                for reason, count in reason_counts.most_common(10)
            ]
        
        return outlier_analysis
    
    def _calculate_hierarchy_stats(self, proxies) -> Dict[str, Any]:
        """Calculate statistics about topic hierarchy usage."""
        topic_counts = Counter()
        subtopic_counts = Counter()
        subsubtopic_counts = Counter()
        
        regular_proxies = proxies.filter(outlier_category=False)
        
        for proxy in regular_proxies:
            topic_counts[proxy.topic] += 1
            subtopic_counts[f"{proxy.topic} > {proxy.sub_topic}"] += 1
            subsubtopic_counts[f"{proxy.topic} > {proxy.sub_topic} > {proxy.sub_sub_topic}"] += 1
        
        return {
            'total_topics': len(topic_counts),
            'total_subtopics': len(subtopic_counts),
            'total_subsubtopics': len(subsubtopic_counts),
            'topic_distribution': dict(topic_counts.most_common()),
            'subtopic_distribution': dict(subtopic_counts.most_common(20)),
            'avg_proxies_per_topic': sum(topic_counts.values()) / len(topic_counts) if topic_counts else 0
        }
    
    def _calculate_cross_state_commonality(self, proxies) -> Dict[str, Any]:
        """Calculate topics that appear across multiple states."""
        topic_state_coverage = defaultdict(set)
        
        for proxy in proxies:
            if proxy.outlier_category:
                continue
                
            for standard in proxy.member_standards.select_related('state').all():
                if standard.state:
                    topic_state_coverage[proxy.sub_sub_topic].add(standard.state.code)
        
        # Categorize by commonality
        total_states = len(set().union(*topic_state_coverage.values())) if topic_state_coverage else 0
        
        commonality = {
            'universal_topics': {},  # 80%+ states
            'common_topics': {},     # 60-79% states
            'regional_topics': {},   # <60% states
            'total_states_analyzed': total_states
        }
        
        for topic, states in topic_state_coverage.items():
            state_count = len(states)
            percentage = (state_count / total_states * 100) if total_states > 0 else 0
            
            topic_data = {
                'state_count': state_count,
                'percentage': percentage,
                'states': list(states)
            }
            
            if percentage >= 80:
                commonality['universal_topics'][topic] = topic_data
            elif percentage >= 60:
                commonality['common_topics'][topic] = topic_data
            else:
                commonality['regional_topics'][topic] = topic_data
        
        return commonality
    
    def _calculate_state_coverage_clustering(self, proxies) -> Dict[str, Any]:
        """Calculate state coverage for traditional clustering runs."""
        state_coverage = {}
        
        # Get all states that have standards/atoms in these proxies
        states_with_content = set()
        for proxy in proxies:
            if hasattr(proxy, 'member_atoms') and proxy.member_atoms.exists():
                # Atom-based clustering
                for atom in proxy.member_atoms.select_related('standard__state').all():
                    if atom.standard and atom.standard.state:
                        states_with_content.add(atom.standard.state.code)
            elif hasattr(proxy, 'member_standards') and proxy.member_standards.exists():
                # Standard-based clustering
                for standard in proxy.member_standards.select_related('state').all():
                    if standard.state:
                        states_with_content.add(standard.state.code)
        
        for state_code in states_with_content:
            state = State.objects.get(code=state_code)
            
            # Count proxies that have content from this state
            proxies_with_state = 0
            total_content_in_state = 0
            
            for proxy in proxies:
                if hasattr(proxy, 'member_atoms') and proxy.member_atoms.exists():
                    # Atom-based clustering
                    state_atoms = proxy.member_atoms.filter(standard__state=state)
                    if state_atoms.exists():
                        proxies_with_state += 1
                        total_content_in_state += state_atoms.count()
                elif hasattr(proxy, 'member_standards') and proxy.member_standards.exists():
                    # Standard-based clustering
                    state_standards = proxy.member_standards.filter(state=state)
                    if state_standards.exists():
                        proxies_with_state += 1
                        total_content_in_state += state_standards.count()
            
            # Calculate total standards for this state
            total_state_standards = Standard.objects.filter(state=state).count()
            
            state_coverage[state_code] = {
                'state_name': state.name,
                'proxies_count': proxies_with_state,
                'standards_covered': total_content_in_state,
                'total_state_standards': total_state_standards,
                'coverage_percentage': (total_content_in_state / total_state_standards * 100) if total_state_standards > 0 else 0,
                'topics_covered': 0,  # Not applicable for clustering
                'topics_list': []
            }
        
        return state_coverage
    
    def _calculate_proxy_prevalence(self, proxies) -> Dict[str, Any]:
        """Calculate prevalence of proxy titles (for clustering runs)."""
        proxy_state_map = defaultdict(set)
        proxy_content_count = defaultdict(int)
        
        # Collect data for each proxy title
        for proxy in proxies:
            title = proxy.title or f"Proxy {proxy.id}"
            
            # Count content and states for this proxy
            if hasattr(proxy, 'member_atoms') and proxy.member_atoms.exists():
                # Atom-based clustering
                for atom in proxy.member_atoms.select_related('standard__state').all():
                    if atom.standard and atom.standard.state:
                        proxy_state_map[title].add(atom.standard.state.code)
                        proxy_content_count[title] += 1
            elif hasattr(proxy, 'member_standards') and proxy.member_standards.exists():
                # Standard-based clustering
                for standard in proxy.member_standards.select_related('state').all():
                    if standard.state:
                        proxy_state_map[title].add(standard.state.code)
                        proxy_content_count[title] += 1
        
        # Build prevalence data similar to topic prevalence
        prevalence = {}
        total_states = len(set().union(*proxy_state_map.values())) if proxy_state_map else 0
        
        for proxy_title, states in proxy_state_map.items():
            state_count = len(states)
            prevalence_percentage = (state_count / total_states * 100) if total_states > 0 else 0
            
            # Categorize proxy importance
            if prevalence_percentage >= 80:
                category = 'must_have'
            elif prevalence_percentage >= 60:
                category = 'important'
            else:
                category = 'regional'
            
            prevalence[proxy_title] = {
                'state_count': state_count,
                'total_states': total_states,
                'prevalence_percentage': prevalence_percentage,
                'category': category,
                'states_list': list(states),
                'standards_count': proxy_content_count[proxy_title],
                'proxy_count': 1  # Each proxy is unique
            }
        
        # Sort by prevalence
        prevalence = dict(sorted(prevalence.items(), key=lambda x: x[1]['state_count'], reverse=True))
        
        return prevalence
    
    def _calculate_coverage_distribution_clustering(self, proxies) -> Dict[str, Any]:
        """Calculate coverage distribution for clustering runs."""
        coverage_by_proxy = defaultdict(list)
        coverage_by_state = defaultdict(int)
        
        # Collect coverage data
        for proxy in proxies:
            proxy_title = proxy.title or f"Proxy {proxy.id}"
            state_counts = defaultdict(int)
            
            if hasattr(proxy, 'member_atoms') and proxy.member_atoms.exists():
                # Atom-based clustering
                for atom in proxy.member_atoms.select_related('standard__state').all():
                    if atom.standard and atom.standard.state:
                        state_counts[atom.standard.state.code] += 1
            elif hasattr(proxy, 'member_standards') and proxy.member_standards.exists():
                # Standard-based clustering
                for standard in proxy.member_standards.select_related('state').all():
                    if standard.state:
                        state_counts[standard.state.code] += 1
            
            # Record coverage for this proxy in each state
            for state_code, count in state_counts.items():
                coverage_by_proxy[proxy_title].append(count)
                coverage_by_state[state_code] += count
        
        # Calculate distribution statistics
        all_coverages = list(coverage_by_state.values())
        if all_coverages:
            distribution = {
                'mean': statistics.mean(all_coverages),
                'median': statistics.median(all_coverages),
                'std_dev': statistics.stdev(all_coverages) if len(all_coverages) > 1 else 0,
                'min': min(all_coverages),
                'max': max(all_coverages),
                'data_points': all_coverages,
                'histogram_data': self._create_histogram_data(all_coverages)
            }
        else:
            distribution = {
                'mean': 0, 'median': 0, 'std_dev': 0, 'min': 0, 'max': 0,
                'data_points': [], 'histogram_data': []
            }
        
        return {
            'overall_distribution': distribution,
            'coverage_by_proxy': dict(coverage_by_proxy),
            'coverage_by_state': dict(coverage_by_state)
        }
    
    def _calculate_clustering_quality(self, proxies) -> Dict[str, Any]:
        """Calculate clustering quality metrics."""
        quality_metrics = {
            'total_clusters': proxies.count(),
            'avg_cluster_size': 0,
            'cluster_size_distribution': {},
            'empty_clusters': 0,
            'large_clusters': 0,
            'single_item_clusters': 0
        }
        
        if not proxies.exists():
            return quality_metrics
        
        cluster_sizes = []
        size_distribution = defaultdict(int)
        
        for proxy in proxies:
            # Calculate cluster size based on content type
            cluster_size = 0
            if hasattr(proxy, 'member_atoms') and proxy.member_atoms.exists():
                cluster_size = proxy.member_atoms.count()
            elif hasattr(proxy, 'member_standards') and proxy.member_standards.exists():
                cluster_size = proxy.member_standards.count()
            
            cluster_sizes.append(cluster_size)
            size_distribution[cluster_size] += 1
            
            # Count cluster types
            if cluster_size == 0:
                quality_metrics['empty_clusters'] += 1
            elif cluster_size == 1:
                quality_metrics['single_item_clusters'] += 1
            elif cluster_size > 20:  # Arbitrary threshold for "large"
                quality_metrics['large_clusters'] += 1
        
        if cluster_sizes:
            quality_metrics['avg_cluster_size'] = statistics.mean(cluster_sizes)
            quality_metrics['median_cluster_size'] = statistics.median(cluster_sizes)
            quality_metrics['std_dev_cluster_size'] = statistics.stdev(cluster_sizes) if len(cluster_sizes) > 1 else 0
            quality_metrics['min_cluster_size'] = min(cluster_sizes)
            quality_metrics['max_cluster_size'] = max(cluster_sizes)
        
        quality_metrics['cluster_size_distribution'] = dict(size_distribution)
        
        return quality_metrics
    
    def _identify_coverage_gaps_topics(self, proxies) -> Dict[str, Any]:
        """Identify areas with low coverage in topic-based runs."""
        gaps = {
            'states_with_low_coverage': [],
            'topics_with_few_standards': [],
            'grade_level_gaps': []
        }
        
        # Identify states with low overall coverage
        state_coverage_percentages = []
        for proxy in proxies:
            for standard in proxy.member_standards.select_related('state').all():
                if standard.state:
                    # This is simplified - would need more sophisticated calculation
                    pass
        
        return gaps
    
    def _identify_coverage_gaps_clustering(self, proxies) -> Dict[str, Any]:
        """Identify coverage gaps for clustering runs."""
        gaps = {
            'states_with_low_coverage': [],
            'small_clusters': [],
            'grade_level_gaps': [],
            'subject_area_gaps': []
        }
        
        # Calculate state coverage
        state_coverage_percentages = {}
        for proxy in proxies:
            if hasattr(proxy, 'member_atoms') and proxy.member_atoms.exists():
                # Atom-based clustering
                for atom in proxy.member_atoms.select_related('standard__state').all():
                    if atom.standard and atom.standard.state:
                        state_code = atom.standard.state.code
                        if state_code not in state_coverage_percentages:
                            total_standards = Standard.objects.filter(state=atom.standard.state).count()
                            covered_standards = sum(1 for p in proxies 
                                                  if p.member_atoms.filter(standard__state=atom.standard.state).exists())
                            state_coverage_percentages[state_code] = {
                                'state_name': atom.standard.state.name,
                                'coverage_percentage': (covered_standards / total_standards * 100) if total_standards > 0 else 0
                            }
            elif hasattr(proxy, 'member_standards') and proxy.member_standards.exists():
                # Standard-based clustering
                for standard in proxy.member_standards.select_related('state').all():
                    if standard.state:
                        state_code = standard.state.code
                        if state_code not in state_coverage_percentages:
                            total_standards = Standard.objects.filter(state=standard.state).count()
                            covered_standards = sum(1 for p in proxies 
                                                  if p.member_standards.filter(state=standard.state).exists())
                            state_coverage_percentages[state_code] = {
                                'state_name': standard.state.name,
                                'coverage_percentage': (covered_standards / total_standards * 100) if total_standards > 0 else 0
                            }
        
        # Identify states with low coverage (< 30%)
        for state_code, data in state_coverage_percentages.items():
            if data['coverage_percentage'] < 30:
                gaps['states_with_low_coverage'].append({
                    'state_code': state_code,
                    'state_name': data['state_name'],
                    'coverage_percentage': data['coverage_percentage']
                })
        
        # Identify small clusters (potentially noise)
        for proxy in proxies:
            cluster_size = 0
            if hasattr(proxy, 'member_atoms') and proxy.member_atoms.exists():
                cluster_size = proxy.member_atoms.count()
            elif hasattr(proxy, 'member_standards') and proxy.member_standards.exists():
                cluster_size = proxy.member_standards.count()
            
            if cluster_size <= 2:  # Very small clusters might indicate noise
                gaps['small_clusters'].append({
                    'proxy_id': proxy.id,
                    'proxy_title': proxy.title or f"Proxy {proxy.id}",
                    'cluster_size': cluster_size
                })
        
        return gaps
    
    def _create_histogram_data(self, data_points: List[float], bins: int = 10) -> List[Dict[str, Any]]:
        """Create histogram data for visualization."""
        if not data_points:
            return []
        
        min_val, max_val = min(data_points), max(data_points)
        if min_val == max_val:
            return [{'bin_start': min_val, 'bin_end': max_val, 'count': len(data_points)}]
        
        bin_width = (max_val - min_val) / bins
        histogram = []
        
        for i in range(bins):
            bin_start = min_val + i * bin_width
            bin_end = bin_start + bin_width
            
            if i == bins - 1:  # Last bin includes max value
                count = len([x for x in data_points if bin_start <= x <= bin_end])
            else:
                count = len([x for x in data_points if bin_start <= x < bin_end])
            
            histogram.append({
                'bin_start': round(bin_start, 2),
                'bin_end': round(bin_end, 2),
                'count': count,
                'percentage': (count / len(data_points) * 100) if data_points else 0
            })
        
        return histogram
    
    def _create_empty_report(self, proxy_run: ProxyRun) -> ProxyRunReport:
        """Create an empty report for runs with no proxies."""
        return ProxyRunReport.objects.create(
            run=proxy_run,
            state_coverage={},
            topic_prevalence={},
            coverage_distribution={},
            outlier_analysis={},
            topic_hierarchy_stats={},
            cross_state_commonality={},
            silhouette_scores={},
            coverage_gaps={}
        )
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple proxy runs side by side."""
        runs = ProxyRun.objects.filter(run_id__in=run_ids, status='completed')
        
        comparison = {
            'runs': [],
            'coverage_comparison': {},
            'topic_overlap': {},
            'efficiency_metrics': {}
        }
        
        for run in runs:
            if hasattr(run, 'report'):
                run_data = {
                    'run_id': run.run_id,
                    'name': run.name,
                    'run_type': run.run_type,
                    'total_proxies': run.total_proxies_created,
                    'coverage_percentage': run.coverage_percentage,
                    'duration': run.duration_display,
                    'filter_summary': run.filter_summary,
                    'algorithm_summary': run.algorithm_summary
                }
                comparison['runs'].append(run_data)
        
        return comparison
    
    def get_topic_prevalence_chart_data(self, report: ProxyRunReport) -> Dict[str, Any]:
        """Get formatted data for topic prevalence chart visualization."""
        prevalence = report.topic_prevalence
        total_states = max((data.get('total_states', 0) for data in prevalence.values()), default=0)
        
        chart_data = {
            'must_have': [],
            'important': [],
            'regional': [],
            'total_states': total_states
        }
        
        for topic, data in prevalence.items():
            state_count = data.get('state_count', 0)
            category = data.get('category', 'regional')
            
            topic_entry = {
                'topic': topic,
                'state_count': state_count,
                'total_states': total_states,
                'percentage': data.get('prevalence_percentage', 0),
                'bar_fill_percentage': (state_count / total_states * 100) if total_states > 0 else 0,
                'standards_count': data.get('standards_count', 0)
            }
            
            chart_data[category].append(topic_entry)
        
        # Sort each category by state count descending
        for category in chart_data:
            if isinstance(chart_data[category], list):
                chart_data[category].sort(key=lambda x: x.get('state_count', 0), reverse=True)
        
        return chart_data