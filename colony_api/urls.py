from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('get_merge_file/<str:upload_file_name>', views.get_merge_file, name='get_merge_file'),
    path('get_predict_file/<str:predict_order>/<str:upload_origin_file_name>/<str:upload_file_name>', views.get_predict_file, name='get_predict_file'),
    path('get_origin_file/<str:upload_file_name>', views.get_origin_file, name='get_origin_file'),
    path('upload_file_list/', views.upload_file_list, name='upload_file_list'),
    path("", views.conoly_api, name="colony_api"),
    path("upload_file/", views.upload_file, name="upload_file"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
