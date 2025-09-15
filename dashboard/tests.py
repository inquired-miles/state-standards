from django.test import TestCase, RequestFactory
from django.contrib.auth.models import User
from dashboard.views import proxy_run_proxies_api
from standards.models import ProxyRun, Standard, State, SubjectArea, GradeLevel
import json


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
        
        # Check that not_covered array exists
        self.assertIn('not_covered', data)
        not_covered = data['not_covered']
        
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
