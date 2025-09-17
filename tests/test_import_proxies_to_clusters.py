#!/usr/bin/env python
"""
Tests for importing proxies as editable custom clusters.
"""
import os
import django

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'state_standards_project.settings')
django.setup()

from django.test import Client
from django.contrib.auth import get_user_model
from django.utils import timezone
from standards.models import State, SubjectArea, GradeLevel, Standard, TopicBasedProxy, ProxyRun, TopicCluster
import json


def ensure_admin_user():
    User = get_user_model()
    admin_user, created = User.objects.get_or_create(
        username='admin',
        defaults={'is_staff': True, 'is_superuser': True, 'email': 'admin@example.com'}
    )
    if created:
        admin_user.set_password('admin123')
        admin_user.save()
    return admin_user


def create_minimal_standard(state, subject, grades, code_suffix):
    std = Standard.objects.create(
        state=state,
        subject_area=subject,
        code=f"{state.code}.{code_suffix}",
        title=f"Standard {code_suffix}",
        description=f"Desc {code_suffix}",
        domain="Test",
        cluster="Test",
    )
    std.grade_levels.set(list(GradeLevel.objects.filter(grade_numeric__in=grades)))
    return std


def test_import_topic_based_proxy_as_cluster():
    print("🧪 Testing import from topic-based proxy...")
    admin = ensure_admin_user()
    client = Client()
    client.force_login(admin)

    # Setup minimal taxonomy
    ca, _ = State.objects.get_or_create(code='CA', defaults={'name': 'California'})
    tx, _ = State.objects.get_or_create(code='TX', defaults={'name': 'Texas'})
    math, _ = SubjectArea.objects.get_or_create(name='Mathematics')
    for g in range(0, 3):
        GradeLevel.objects.get_or_create(grade=str(g if g > 0 else 'K'), defaults={'grade_numeric': g})

    # Create standards
    s1 = create_minimal_standard(ca, math, [0, 1], 'MATH.1.A')
    s2 = create_minimal_standard(tx, math, [1], 'MATH.1.B')

    # Topic-based proxy with member standards
    tproxy = TopicBasedProxy.objects.create(
        proxy_id='TP-TEST-001',
        topic='Numbers',
        sub_topic='Counting',
        sub_sub_topic='Basic',
        title='Counting Basics',
        description='Counting basics proxy'
    )
    tproxy.member_standards.add(s1, s2)
    tproxy.save()

    # Proxy run
    run = ProxyRun.objects.create(
        run_id='topics-test-run-1',
        name='Topic Run 1',
        run_type='topics',
        status='completed',
        filter_parameters={'subject_area_id': math.id, 'grade_selection': {'type': 'specific', 'grades': [0, 1]}},
        started_at=timezone.now(),
        completed_at=timezone.now(),
    )

    payload = {
        'imports': [
            { 'run_id': run.run_id, 'proxy_type': 'topics', 'proxy_id': tproxy.proxy_id }
        ]
    }
    url = '/dashboard/api/custom-clusters/import-from-proxies/'
    resp = client.post(url, data=json.dumps(payload), content_type='application/json')
    assert resp.status_code in (200, 201), f"Unexpected status: {resp.status_code} {resp.content}"
    data = resp.json().get('data', {})
    created = data.get('created', [])
    assert created, 'No cluster created'
    cluster = created[0]
    assert cluster.get('origin') == 'custom'
    assert cluster.get('copied_from_proxy') is True
    assert cluster.get('source', {}).get('run_id') == run.run_id
    assert cluster.get('source', {}).get('proxy_id') == tproxy.proxy_id
    members = cluster.get('members', [])
    assert len(members) == 2, 'Imported cluster should have 2 members'

    # Ensure cluster exists in DB with source metadata
    db_cluster = TopicCluster.objects.get(id=cluster['id'])
    assert db_cluster.source_run_id == run.id
    assert db_cluster.source_proxy_id == tproxy.proxy_id
    assert db_cluster.source_proxy_type == 'topics'
    assert db_cluster.standards.count() == 2

    print('✅ Import topic-based proxy passed')

