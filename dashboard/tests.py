from django.test import TestCase, RequestFactory
from django.contrib.auth.models import User
from django.utils import timezone
from django.core.cache import cache

from dashboard.views import proxy_run_proxies_api, proxy_run_coverage_api
from standards.models import (
    ProxyRun,
    Standard,
    State,
    SubjectArea,
    GradeLevel,
    TopicCluster,
    ClusterMembership,
    ClusterReport,
    ClusterReportEntry,
    ProxyStandard,
)
from standards.services.discovery import CustomClusterService
import json
from datetime import timedelta


class ProxyRunProxiesAPITestCase(TestCase):
    def setUp(self):
        """Set up test data"""
        self.factory = RequestFactory()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass'
        )
        self.user.is_staff = True
        self.user.save()
        
        # Create test data
        self.state = State.objects.create(name='Test State', code='TS')
        self.subject_area = SubjectArea.objects.create(name='Test Subject')
        self.grade_level = GradeLevel.objects.create(grade='3', grade_numeric=3)
        
        # Create test standard with description
        self.standard = Standard.objects.create(
            state=self.state,
            subject_area=self.subject_area,
            code='TS.3.TEST.1',
            title='Test Standard',
            description='This is a test standard description for verifying API responses.'
        )
        self.standard.grade_levels.add(self.grade_level)
        
        # Create a test proxy run
        self.proxy_run = ProxyRun.objects.create(
            run_id='test-run-123',
            name='Test Run',
            run_type='topics',
            status='completed'
        )

    def test_proxy_run_proxies_api_includes_description(self):
        """Test that the proxy_run_proxies_api includes description field in not_covered standards"""
        request = self.factory.get(f'/api/proxy-run-proxies/?run_id={self.proxy_run.run_id}')
        request.user = self.user
        
        response = proxy_run_proxies_api(request)
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        payload = data.get('data', data)
        
        # Check that not_covered array exists
        self.assertIn('not_covered', payload)
        not_covered = payload['not_covered']
        
        if not_covered:
            # Check that each not_covered standard has description field
            for standard in not_covered:
                self.assertIn('description', standard, 
                            "Description field should be present in not_covered standards")
                self.assertIn('code', standard)
                self.assertIn('title', standard)
                self.assertIn('state__code', standard)
                
                # If this is our test standard, verify the description content
                if standard.get('code') == 'TS.3.TEST.1':
                    self.assertEqual(
                        standard['description'],
                        'This is a test standard description for verifying API responses.'
                    )


class BaseReportScopeTestCase(TestCase):
    """Shared fixture setup for coverage report scoped proxy run tests."""

    def setUp(self):
        cache.clear()
        self.factory = RequestFactory()
        self.user = User.objects.create_user(
            username='staffer',
            email='staff@example.com',
            password='pass1234'
        )
        self.user.is_staff = True
        self.user.save()

        self.state_ts = State.objects.create(name='Test State', code='TS')
        self.state_rs = State.objects.create(name='Reference State', code='RS')
        self.subject_area = SubjectArea.objects.create(name='English Language Arts')
        self.grade_level = GradeLevel.objects.create(grade='3', grade_numeric=3)

        self.standard_a = Standard.objects.create(
            state=self.state_ts,
            subject_area=self.subject_area,
            code='TS.ELA.3.1',
            title='Main Idea',
            description='Determine the main idea of a text.'
        )
        self.standard_b = Standard.objects.create(
            state=self.state_ts,
            subject_area=self.subject_area,
            code='TS.ELA.3.2',
            title='Supporting Details',
            description='Identify supporting details in informational texts.'
        )
        self.standard_c = Standard.objects.create(
            state=self.state_rs,
            subject_area=self.subject_area,
            code='RS.ELA.3.3',
            title='Summarize',
            description='Summarize a text including main ideas.'
        )

        self.standard_extra = Standard.objects.create(
            state=self.state_ts,
            subject_area=self.subject_area,
            code='TS.ELA.3.99',
            title='Author Purpose',
            description='Determine author purpose for third grade texts.'
        )

        for standard in [self.standard_a, self.standard_b, self.standard_c, self.standard_extra]:
            standard.grade_levels.add(self.grade_level)

        self.cluster_one = TopicCluster.objects.create(
            name='ELA Main Idea',
            description='Focus on main idea skills',
            origin='custom',
            subject_area=self.subject_area,
            created_by=self.user,
            is_shared=False,
        )
        self.cluster_one.grade_levels.add(self.grade_level)
        self.cluster_one.standards_count = 2
        self.cluster_one.states_represented = 1
        self.cluster_one.save(update_fields=['standards_count', 'states_represented'])

        self.cluster_two = TopicCluster.objects.create(
            name='ELA Summaries',
            description='Summarizing standards',
            origin='custom',
            subject_area=self.subject_area,
            created_by=self.user,
            is_shared=False,
        )
        self.cluster_two.grade_levels.add(self.grade_level)
        self.cluster_two.standards_count = 1
        self.cluster_two.states_represented = 1
        self.cluster_two.save(update_fields=['standards_count', 'states_represented'])

        ClusterMembership.objects.create(
            standard=self.standard_a,
            cluster=self.cluster_one,
            membership_strength=0.95,
            selection_order=0
        )
        ClusterMembership.objects.create(
            standard=self.standard_b,
            cluster=self.cluster_one,
            membership_strength=0.9,
            selection_order=1
        )
        ClusterMembership.objects.create(
            standard=self.standard_c,
            cluster=self.cluster_two,
            membership_strength=0.92,
            selection_order=0
        )

        self.report = ClusterReport.objects.create(
            title='Benchmark Coverage Report',
            description='Compare proxy coverage against curated clusters',
            created_by=self.user,
            is_shared=False,
        )
        ClusterReportEntry.objects.create(
            report=self.report,
            cluster=self.cluster_one,
            selection_order=0,
            notes='Key third grade skills'
        )
        ClusterReportEntry.objects.create(
            report=self.report,
            cluster=self.cluster_two,
            selection_order=1,
            notes='Supplemental skills'
        )

        self.proxy_run = ProxyRun.objects.create(
            run_id='report-run-1',
            name='Report Driven Run',
            run_type='standards',
            status='completed'
        )
        self.proxy_run.started_at = timezone.now() - timedelta(minutes=10)
        self.proxy_run.save(update_fields=['started_at'])

        self.proxy_a = ProxyStandard.objects.create(
            proxy_id='proxy-A',
            title='Main Idea Proxy',
            source_type='standards',
            cluster_id=1
        )
        self.proxy_a.member_standards.add(self.standard_a)
        self.proxy_a.coverage_count = 1
        self.proxy_a.save(update_fields=['coverage_count'])

        self.proxy_c = ProxyStandard.objects.create(
            proxy_id='proxy-C',
            title='Summary Proxy',
            source_type='standards',
            cluster_id=2
        )
        self.proxy_c.member_standards.add(self.standard_c)
        self.proxy_c.coverage_count = 1
        self.proxy_c.save(update_fields=['coverage_count'])

        self.proxy_run.completed_at = timezone.now()
        self.proxy_run.save(update_fields=['completed_at'])


class CustomClusterServiceReportScopeTestCase(BaseReportScopeTestCase):
    def setUp(self):
        super().setUp()
        self.service = CustomClusterService()

    def test_build_report_scope_returns_expected_metadata(self):
        scope = self.service.build_report_scope(self.report, use_cache=False)

        self.assertEqual(scope['report_id'], str(self.report.id))
        self.assertEqual(scope['title'], self.report.title)
        self.assertEqual(scope['total_clusters'], 2)
        self.assertEqual(set(scope['standard_ids']), {str(self.standard_a.id), str(self.standard_b.id), str(self.standard_c.id)})

        clusters = scope['clusters']
        self.assertEqual(len(clusters), 2)

        cluster_one_scope = next(c for c in clusters if c['cluster_name'] == 'ELA Main Idea')
        self.assertTrue(all(isinstance(sid, str) for sid in cluster_one_scope['standard_ids']))
        self.assertEqual(cluster_one_scope['states_breakdown'], {'TS': 2})

        cluster_two_scope = next(c for c in clusters if c['cluster_name'] == 'ELA Summaries')
        self.assertEqual(cluster_two_scope['states_breakdown'], {'RS': 1})


class ProxyRunCoverageReportAPITestCase(BaseReportScopeTestCase):
    def test_proxy_run_coverage_scopes_to_report(self):
        request = self.factory.get(
            '/api/proxy-run-coverage/',
            {
                'run_id': self.proxy_run.run_id,
                'report_id': str(self.report.id)
            }
        )
        request.user = self.user

        response = proxy_run_coverage_api(request)
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.content)
        payload = data.get('data', data)

        self.assertEqual(payload['run_id'], self.proxy_run.run_id)
        self.assertIn('report', payload)
        report_meta = payload['report']
        self.assertEqual(report_meta['id'], str(self.report.id))
        self.assertEqual(report_meta['standard_count'], 3)

        clusters = report_meta['clusters']
        self.assertEqual(len(clusters), 2)

        cluster_one = next(c for c in clusters if c['cluster_name'] == 'ELA Main Idea')
        self.assertEqual(cluster_one['covered_count'], 1)
        self.assertEqual(cluster_one['not_covered_count'], 1)
        self.assertIn(str(self.standard_a.id), cluster_one['covered_standard_ids'])
        self.assertIn(str(self.standard_b.id), cluster_one['not_covered_standard_ids'])

        cluster_two = next(c for c in clusters if c['cluster_name'] == 'ELA Summaries')
        self.assertEqual(cluster_two['covered_count'], 1)
        self.assertEqual(cluster_two['not_covered_count'], 0)
        self.assertIn(str(self.standard_c.id), cluster_two['covered_standard_ids'])

        states = payload['states']
        self.assertEqual({state['state'] for state in states}, {'TS', 'RS'})
        ts_state = next(state for state in states if state['state'] == 'TS')
        self.assertEqual(ts_state['total_count'], 2)
        self.assertEqual(ts_state['covered_count'], 1)
        self.assertEqual(len(ts_state['not_covered']), 1)
        self.assertEqual(ts_state['not_covered'][0]['code'], 'TS.ELA.3.2')

    def test_proxy_run_proxies_scopes_to_report(self):
        request = self.factory.get(
            '/api/proxy-run-proxies/',
            {
                'run_id': self.proxy_run.run_id,
                'report_id': str(self.report.id)
            }
        )
        request.user = self.user

        response = proxy_run_proxies_api(request)
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.content)
        payload = data.get('data', data)

        self.assertEqual(payload['run_id'], self.proxy_run.run_id)
        self.assertIn('report', payload)
        report_meta = payload['report']
        cluster_one = next(c for c in report_meta['clusters'] if c['cluster_name'] == 'ELA Main Idea')
        self.assertEqual(cluster_one['covered_count'], 1)
        self.assertEqual(cluster_one['not_covered_count'], 1)

        not_covered_ids = {str(item['id']) for item in payload['not_covered']}
        self.assertIn(str(self.standard_b.id), not_covered_ids)
        self.assertNotIn(str(self.standard_c.id), not_covered_ids)
        self.assertNotIn(str(self.standard_extra.id), not_covered_ids)

        self.assertEqual(payload['standards_in_scope_count'], 3)

    def test_coverage_report_only_returns_states(self):
        request = self.factory.get(
            '/api/proxy-run-coverage/',
            {
                'report_id': str(self.report.id)
            }
        )
        request.user = self.user

        response = proxy_run_coverage_api(request)
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.content)
        payload = data.get('data', data)

        self.assertIsNone(payload.get('run_id'))
        self.assertIn('report', payload)
        report_meta = payload['report']
        self.assertEqual(report_meta['id'], str(self.report.id))

        states = payload['states']
        self.assertEqual({state['state'] for state in states}, {'TS', 'RS'})
        for state in states:
            self.assertEqual(state['covered_count'], state['total_count'])
            self.assertEqual(state['coverage_percentage'], 100.0 if state['covered_count'] else 0.0)
            self.assertEqual(state['not_covered'], [])

        clusters = report_meta['clusters']
        cluster_one = next(c for c in clusters if c['cluster_name'] == 'ELA Main Idea')
        self.assertEqual(cluster_one['covered_count'], 2)
        self.assertEqual(cluster_one['not_covered_count'], 0)

    def test_proxy_run_proxies_report_only_returns_clusters(self):
        request = self.factory.get(
            '/api/proxy-run-proxies/',
            {
                'report_id': str(self.report.id)
            }
        )
        request.user = self.user

        response = proxy_run_proxies_api(request)
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.content)
        payload = data.get('data', data)

        self.assertIsNone(payload.get('run_id'))
        self.assertIn('proxies', payload)
        proxy_map = {item['title']: item for item in payload['proxies']}
        self.assertIn('ELA Main Idea', proxy_map)
        self.assertIn('ELA Summaries', proxy_map)

        self.assertEqual(proxy_map['ELA Main Idea']['covered_count'], 2)
        self.assertEqual(proxy_map['ELA Summaries']['covered_count'], 1)
        not_covered_codes = {item['code'] for item in payload['not_covered']}
        self.assertIn('TS.ELA.3.99', not_covered_codes)
        self.assertEqual(payload['standards_in_scope_count'], 4)
