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

    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return JsonResponse({"error": "Missing field 'prompt'."}, status=400)

    try:
        top_k = int(data.get("top_k", 3))
    except (TypeError, ValueError):
        return JsonResponse({"error": "Field 'top_k' must be an integer."}, status=400)

    top_k = max(1, min(top_k, 50))

    results = search_images(prompt, top_k=top_k)
    payload = [{"path": path, "score": float(score)} for path, score in results]
    return JsonResponse({"prompt": prompt, "results": payload})