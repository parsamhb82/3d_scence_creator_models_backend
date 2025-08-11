import json
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt  # remove if you use CSRF token
from mesh_detector.search_engine import search_images

@csrf_exempt  # only if calling from external client without CSRF token
@require_POST
def image_search(request):
    try:
        data = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return HttpResponseBadRequest("Invalid JSON body")

    q = (data.get("q") or "").strip()
    if not q:
        return JsonResponse({"error": "Missing field 'q'."}, status=400)

    try:
        k = int(data.get("k", 3))
    except (TypeError, ValueError):
        return JsonResponse({"error": "Field 'k' must be an integer."}, status=400)

    k = max(1, min(k, 50))

    results = search_images(q, top_k=k)
    payload = [{"path": path, "score": float(score)} for path, score in results]
    return JsonResponse({"query": q, "results": payload})