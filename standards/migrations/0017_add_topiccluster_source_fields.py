from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('standards', '0016_clusterreport_clusterreportentry_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='topiccluster',
            name='source_proxy_id',
            field=models.CharField(blank=True, help_text='ID of the originating proxy (proxy_id)', max_length=100),
        ),
        migrations.AddField(
            model_name='topiccluster',
            name='source_proxy_type',
            field=models.CharField(blank=True, choices=[('topics', 'Topic-Based Proxy'), ('atoms', 'Atom Clustering Proxy'), ('standards', 'Standard Clustering Proxy')], help_text='Type of the originating proxy', max_length=20),
        ),
        migrations.AddField(
            model_name='topiccluster',
            name='source_run',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='derived_clusters', to='standards.proxyrun'),
        ),
        migrations.AddField(
            model_name='topiccluster',
            name='source_title',
            field=models.CharField(blank=True, help_text='Snapshot of originating proxy title at time of import', max_length=255),
        ),
        migrations.AddIndex(
            model_name='topiccluster',
            index=models.Index(fields=['source_proxy_type', 'source_proxy_id'], name='standards_to_source__2c8be9_idx'),
        ),
    ]

