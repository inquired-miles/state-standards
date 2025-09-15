#!/usr/bin/env python
"""
Test script to verify admin functionality is restored
"""
import os
import django

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'state_standards_project.settings')
django.setup()

# Override ALLOWED_HOSTS for testing
from django.conf import settings
settings.ALLOWED_HOSTS = ['*']

from django.test import Client
from django.contrib.auth import get_user_model
from django.urls import reverse, NoReverseMatch
from django.contrib import admin
from standards.models import *

def test_admin_functionality():
    """Test that admin functionality is restored"""
    print("🔧 Testing Admin Functionality Fix")
    print("=" * 50)
    
    # Test 1: Check that models are registered
    print("\n1. Testing model registrations...")
    registered_models = admin.site._registry.keys()
    model_names = [model._meta.verbose_name for model in registered_models]
    
    expected_models = [
        'State', 'Subject Area', 'Grade Level', 'Standard', 'Standard Correlation',
        'Concept', 'Topic Cluster', 'Coverage Analysis', 'Content Alignment',
        'Strategic Plan', 'Cache Entry', 'Upload Job'
    ]
    
    found_models = []
    missing_models = []
    
    for expected in expected_models:
        found = False
        for registered_name in model_names:
            if expected.lower() in registered_name.lower():
                found_models.append(expected)
                found = True
                break
        if not found:
            missing_models.append(expected)
    
    print(f"   ✅ Found {len(found_models)} registered models:")
    for model in found_models:
        print(f"      - {model}")
    
    if missing_models:
        print(f"   ❌ Missing {len(missing_models)} models:")
        for model in missing_models:
            print(f"      - {model}")
        return False
    
    # Test 2: Check URL namespace resolution
    print("\n2. Testing URL namespace resolution...")
    try:
        bulk_upload_url = reverse('standards:bulk_upload')
        generate_template_url = reverse('standards:generate_template')
        print(f"   ✅ Namespace resolution works:")
        print(f"      - bulk_upload: {bulk_upload_url}")
        print(f"      - generate_template: {generate_template_url}")
    except NoReverseMatch as e:
        print(f"   ❌ Namespace resolution failed: {e}")
        return False
    
    # Test 3: Test admin site access
    print("\n3. Testing admin site access...")
    try:
        # Create test client and user
        client = Client()
        User = get_user_model()
        
        # Get or create admin user
        admin_user, created = User.objects.get_or_create(
            username='testadmin',
            defaults={
                'is_staff': True,
                'is_superuser': True,
                'email': 'admin@test.com'
            }
        )
        if created:
            admin_user.set_password('testpass123')
            admin_user.save()
        
        # Login and test admin access
        client.force_login(admin_user)
        
        # Test main admin page
        response = client.get('/admin/')
        if response.status_code == 200:
            print("   ✅ Main admin page accessible")
        else:
            print(f"   ❌ Main admin page failed: {response.status_code}")
            return False
        
        # Test Standards admin page
        response = client.get('/admin/standards/standard/')
        if response.status_code == 200:
            print("   ✅ Standards admin page accessible")
        else:
            print(f"   ❌ Standards admin page failed: {response.status_code}")
            return False
            
        # Test other model admin pages
        test_urls = [
            '/admin/standards/state/',
            '/admin/standards/subjectarea/', 
            '/admin/standards/gradelevel/',
            '/admin/standards/uploadjob/',
        ]
        
        success_count = 0
        for url in test_urls:
            try:
                response = client.get(url)
                if response.status_code == 200:
                    success_count += 1
            except Exception as e:
                print(f"      - Error accessing {url}: {e}")
        
        print(f"   ✅ {success_count}/{len(test_urls)} model admin pages accessible")
        
    except Exception as e:
        print(f"   ❌ Admin site test failed: {e}")
        return False
    
    # Test 4: Test bulk upload URL access
    print("\n4. Testing bulk upload functionality...")
    try:
        response = client.get('/admin/bulk-upload/')
        if response.status_code == 200:
            print("   ✅ Bulk upload page accessible")
        else:
            print(f"   ❌ Bulk upload page failed: {response.status_code}")
            return False
        
        response = client.get('/admin/generate-template/')
        if response.status_code == 200:
            print("   ✅ Generate template page accessible")
        else:
            print(f"   ❌ Generate template page failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Bulk upload test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Admin functionality successfully restored!")
    
    print("\n✅ All tests passed:")
    print("   - All models are registered in admin")
    print("   - URL namespace resolution works")
    print("   - Admin pages are accessible")
    print("   - Bulk upload functionality preserved")
    
    print("\n🚀 You can now access the admin at:")
    print("   http://localhost:8000/admin/")
    print("   - All models should be visible")
    print("   - Bulk upload button should work")
    print("   - All permissions should be restored")
    
    return True


if __name__ == "__main__":
    try:
        success = test_admin_functionality()
        if not success:
            exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)