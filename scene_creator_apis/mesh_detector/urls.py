from django.urls import path
from mesh_detector.views import image_search

urlpatterns = [
    path("search/", image_search, name="image_search"),
]