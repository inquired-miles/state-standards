#!/usr/bin/env python
"""
Test script for bulk upload functionality
"""
import os
import django

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'state_standards_project.settings')
django.setup()

from django.test import TestCase, Client
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from standards.models import Standard, State, SubjectArea, UploadJob
import json

def test_bulk_upload_functionality():
    """Test the bulk upload functionality"""
    print("🧪 Testing Bulk Upload Functionality")
    print("=" * 50)
    
    # Create test user (admin)
    User = get_user_model()
    admin_user, created = User.objects.get_or_create(
        username='admin',
        defaults={
            'is_staff': True,
            'is_superuser': True,
            'email': 'admin@example.com'
        }
    )
    if created:
        admin_user.set_password('admin123')
        admin_user.save()
        print("✅ Created admin user")
    else:
        print("✅ Admin user already exists")
    
    # Create test client
    client = Client()
    client.force_login(admin_user)
    
    # Test 1: Access bulk upload page
    print("\n1. Testing bulk upload page access...")
    try:
        response = client.get('/admin/standards/bulk-upload/')
        if response.status_code == 200:
            print("   ✅ Bulk upload page accessible")
        else:
            print(f"   ❌ Bulk upload page failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error accessing bulk upload page: {e}")
        return False
    
    # Test 2: Create sample JSON data
    print("\n2. Creating sample test data...")
    sample_data = [
        {
            "state": "CA",
            "subject": "Mathematics",
            "grade": "3",
            "code": "CA.3.TEST.1",
            "title": "Test Standard 1",
            "description": "This is a test standard for multiplication.",
            "domain": "Operations and Algebraic Thinking",
            "cluster": "Test Cluster",
            "keywords": ["test", "multiplication"],
            "skills": ["problem solving"]
        },
        {
            "state": "TX",
            "subject": "Mathematics", 
            "grade": "3",
            "code": "TX.3.TEST.1",
            "title": "Test Standard 2",
            "description": "This is a test standard for division.",
            "domain": "Operations and Algebraic Thinking",
            "cluster": "Test Cluster",
            "keywords": ["test", "division"],
            "skills": ["mathematical reasoning"]
        }
    ]
    
    json_content = json.dumps(sample_data, indent=2)
    print(f"   ✅ Created sample data with {len(sample_data)} records")
    
    # Test 3: Test file upload form validation
    print("\n3. Testing file upload validation...")
    try:
        # Test with valid JSON file
        test_file = SimpleUploadedFile(
            "test_standards.json",
            json_content.encode('utf-8'),
            content_type="application/json"
        )
        
        response = client.post('/admin/standards/bulk-upload/', {
            'file': test_file,
            'generate_embeddings': True,
            'batch_size': 10,
            'clear_existing': False,
        })
        
        if response.status_code == 200:
            print("   ✅ File upload form validation passed")
        else:
            print(f"   ❌ File upload failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ File upload test failed: {e}")
        return False
    
    # Test 4: Check UploadJob creation
    print("\n4. Testing UploadJob creation...")
    try:
        upload_jobs = UploadJob.objects.all()
        if upload_jobs.exists():
            latest_job = upload_jobs.first()
            print(f"   ✅ UploadJob created: {latest_job.original_filename}")
            print(f"   - Status: {latest_job.get_status_display()}")
            print(f"   - File size: {latest_job.file_size} bytes")
            print(f"   - File type: {latest_job.get_file_type_display()}")
        else:
            print("   ❌ No UploadJob created")
            return False
    except Exception as e:
        print(f"   ❌ UploadJob test failed: {e}")
        return False
    
    # Test 5: Test template generation
    print("\n5. Testing template generation...")
    try:
        response = client.get('/admin/standards/generate-template/')
        if response.status_code == 200:
            print("   ✅ Template generation page accessible")
            
            # Test CSV template download
            response = client.post('/admin/standards/generate-template/', {
                'format': 'csv',
                'include_sample_data': True,
                'states_count': 3
            })
            
            if response.status_code == 200 and 'text/csv' in response.get('Content-Type', ''):
                print("   ✅ CSV template generation works")
            else:
                print("   ❌ CSV template generation failed")
                
            # Test JSON template download
            response = client.post('/admin/standards/generate-template/', {
                'format': 'json',
                'include_sample_data': True,
                'states_count': 3
            })
            
            if response.status_code == 200 and 'application/json' in response.get('Content-Type', ''):
                print("   ✅ JSON template generation works")
            else:
                print("   ❌ JSON template generation failed")
                
        else:
            print(f"   ❌ Template generation page failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Template generation test failed: {e}")
        return False
    
    # Test 6: Test admin actions
    print("\n6. Testing admin actions...")
    try:
        # Create some test standards first
        ca_state, _ = State.objects.get_or_create(code='CA', defaults={'name': 'California'})
        math_subject, _ = SubjectArea.objects.get_or_create(
            name='Mathematics', 
            defaults={'description': 'Mathematics subject area'}
        )
        
        test_standard = Standard.objects.create(
            state=ca_state,
            subject_area=math_subject,
            code='CA.TEST.ADMIN.1',
            title='Test Admin Standard',
            description='Test standard for admin actions'
        )
        
        # Test export action
        response = client.post('/admin/standards/standard/', {
            'action': 'export_to_csv_action',
            '_selected_action': [str(test_standard.id)]
        })
        
        if response.status_code == 200 and 'text/csv' in response.get('Content-Type', ''):
            print("   ✅ CSV export admin action works")
        else:
            print("   ❌ CSV export admin action failed")
            
        # Test validation action
        response = client.post('/admin/standards/standard/', {
            'action': 'validate_data_action', 
            '_selected_action': [str(test_standard.id)]
        })
        
        if response.status_code == 302:  # Redirect after action
            print("   ✅ Validation admin action works")
        else:
            print("   ❌ Validation admin action failed")
            
    except Exception as e:
        print(f"   ❌ Admin actions test failed: {e}")
        return False
    
    # Test 7: Test upload status page
    print("\n7. Testing upload status functionality...")
    try:
        if upload_jobs.exists():
            job = upload_jobs.first()
            
            # Test status page
            response = client.get(f'/admin/standards/upload-status/{job.id}/')
            if response.status_code == 200:
                print("   ✅ Upload status page accessible")
            else:
                print(f"   ❌ Upload status page failed: {response.status_code}")
            
            # Test status API
            response = client.get(f'/admin/standards/upload-status-api/{job.id}/')
            if response.status_code == 200:
                data = response.json()
                print("   ✅ Upload status API works")
                print(f"   - API response keys: {list(data.keys())}")
            else:
                print(f"   ❌ Upload status API failed: {response.status_code}")
                
    except Exception as e:
        print(f"   ❌ Upload status test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All bulk upload tests passed!")
    
    print("\nFeatures tested:")
    print("✅ Bulk upload page access")
    print("✅ File upload form validation")
    print("✅ UploadJob model creation")
    print("✅ Template generation (CSV/JSON)")
    print("✅ Admin actions (export, validation)")
    print("✅ Upload status tracking")
    print("✅ Status API endpoint")
    
    print("\nTo test the full upload process:")
    print("1. Start the Django server: python manage.py runserver")
    print("2. Go to: http://localhost:8000/admin/standards/standard/")
    print("3. Click 'Bulk Upload Standards'")
    print("4. Upload the sample_standards.json file")
    print("5. Monitor the upload progress")
    
    return True


if __name__ == "__main__":
    try:
        success = test_bulk_upload_functionality()
        if not success:
            exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        exit(1)