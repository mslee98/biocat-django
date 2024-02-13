# Generated by Django 4.1 on 2024-02-07 08:30

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Comtnfile',
            fields=[
                ('ATCH_FILE_ID', models.CharField(max_length=20, primary_key=True, serialize=False)),
                ('CREAT_DT', models.DateTimeField()),
                ('USE_AT', models.CharField(default=None, max_length=1, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Comtnfiledetail',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('FILE_SN', models.DecimalField(decimal_places=0, max_digits=10)),
                ('FILE_STRE_COURS', models.CharField(max_length=2000)),
                ('STRE_FILE_NM', models.CharField(max_length=255)),
                ('ORIGNL_FILE_NM', models.CharField(default=None, max_length=255, null=True)),
                ('FILE_EXTSN', models.CharField(max_length=20)),
                ('FILE_CN', models.TextField(default=None, null=True)),
                ('FILE_SIZE', models.DecimalField(decimal_places=0, default=None, max_digits=8, null=True)),
                ('ATCH_FILE_ID', models.ForeignKey(on_delete=django.db.models.deletion.RESTRICT, to='colony_api.comtnfile')),
            ],
            options={
                'unique_together': {('ATCH_FILE_ID', 'FILE_SN')},
            },
        ),
    ]
