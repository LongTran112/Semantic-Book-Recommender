"""Root URL routing for Django backend."""

from django.urls import include, path
from drf_spectacular.views import SpectacularAPIView, SpectacularRedocView, SpectacularSwaggerView


urlpatterns = [
    path("openapi/schema/", SpectacularAPIView.as_view(), name="openapi-schema"),
    path("openapi/swagger/", SpectacularSwaggerView.as_view(url_name="openapi-schema"), name="openapi-swagger"),
    path("openapi/redoc/", SpectacularRedocView.as_view(url_name="openapi-schema"), name="openapi-redoc"),
    path("", include("rag_api.urls")),
]
