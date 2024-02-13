from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path("/", views.conoly_api, name="colony_api"),
    path("upload_file/", views.upload_file, name="upload_file"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
