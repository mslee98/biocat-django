from rest_framework import serializers
from .models import Comtnfile, Comtnfiledetail

class ComtnfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = Comtnfile
        fields = ("__all__")

class ComtnfiledetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = Comtnfiledetail
        fields = ("__all__")
