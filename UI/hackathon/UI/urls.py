from django.urls import path
from .views import search_entity, attach_structured, attach_unstructured

urlpatterns = [
    path("", search_entity, name="search_entity"),
    path("attach-structured/", attach_structured, name="attach_structured"),
    path("attach-unstructured/", attach_unstructured, name="attach_unstructured"),
]
