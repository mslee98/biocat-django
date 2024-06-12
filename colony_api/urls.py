from django.urls import path, re_path
from django.conf import settings
from django.conf.urls.static import static
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

from . import views

schema_view = get_schema_view(
   openapi.Info(
      title="Your API",
      default_version='v1',
      description="Test description",
      terms_of_service="https://www.google.com/policies/terms/",
      contact=openapi.Contact(email="contact@yourapi.local"),
      license=openapi.License(name="BSD License"),
   ),
   public=True,
   permission_classes=(permissions.AllowAny,),
)


urlpatterns = [
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
    path('get_predict_file/<str:predict_order>/<str:upload_origin_file_name>/<str:upload_file_name>', views.get_predict_file, name='get_predict_file'),
    path('get_origin_file/<str:upload_file_name>', views.get_origin_file, name='get_origin_file'),
    path('upload_file_list/', views.upload_file_list, name='upload_file_list'),
    path("", views.conoly_api, name="colony_api"),
    path("upload_file/", views.upload_file, name="upload_file"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
