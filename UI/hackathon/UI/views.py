from django.shortcuts import render
from django.http import JsonResponse
from .models import Entity

def search_entity(request):
    query = request.GET.get("query", "")
    results = Entity.objects.filter(name__icontains=query) if query else []
    return render(request, "search.html", {"results": results, "query": query})

def attach_structured(request):
    return JsonResponse({"message": "Structured data attached (placeholder)"})

def attach_unstructured(request):
    return JsonResponse({"message": "Unstructured data attached (placeholder)"})
