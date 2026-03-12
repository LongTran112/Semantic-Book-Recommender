"""Root URL routing for Django backend."""

from django.urls import include, path


urlpatterns = [
    path("", include("rag_api.urls")),
]
