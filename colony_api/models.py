from django.db import models

# Create your models here.
class Comtnfile(models.Model):
    ATCH_FILE_ID = models.CharField(max_length=20, primary_key=True)
    CREAT_DT = models.DateTimeField()
    USE_AT = models.CharField(max_length=1, null=True, default=None)

class Comtnfiledetail(models.Model):
    ATCH_FILE_ID = models.ForeignKey(Comtnfile, on_delete=models.RESTRICT)
    FILE_SN = models.DecimalField(max_digits=10, decimal_places=0)
    FILE_STRE_COURS = models.CharField(max_length=2000)
    STRE_FILE_NM = models.CharField(max_length=255)
    ORIGNL_FILE_NM = models.CharField(max_length=255, null=True, default=None)
    FILE_EXTSN = models.CharField(max_length=20)
    FILE_CN = models.TextField(null=True, default=None)
    FILE_SIZE = models.DecimalField(max_digits=8, decimal_places=0, null=True, default=None)

    class Meta:
        unique_together = (('ATCH_FILE_ID', 'FILE_SN'),)
