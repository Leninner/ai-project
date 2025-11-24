from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
from .services import RegistrationService, AuthenticationService


_registration_service = RegistrationService()
_authentication_service = AuthenticationService()


@require_http_methods(["GET", "POST"])
def register_view(request):
    if request.method == "GET":
        return render(request, "authentication/register.html")

    username = request.POST.get("username")
    video_file = request.FILES.get("video")

    if not username or not video_file:
        error_msg = "Username and video are required"
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({"success": False, "error": error_msg}, status=400)
        messages.error(request, error_msg)
        return render(request, "authentication/register.html")

    success, user, message = _registration_service.register_user(
        username=username, video_file=video_file
    )

    if success:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({"success": True, "redirect": "/login/"})
        messages.success(request, message)
        return redirect("login")
    else:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({"success": False, "error": message}, status=400)
        messages.error(request, message)
        return render(request, "authentication/register.html")


@require_http_methods(["GET", "POST"])
def login_view(request):
    if request.method == "GET":
        return render(request, "authentication/login.html")

    username = request.POST.get("username")
    video_file = request.FILES.get("video")

    if not username or not video_file:
        error_msg = "Username and video are required"
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({"success": False, "error": error_msg}, status=400)
        messages.error(request, error_msg)
        return render(request, "authentication/login.html")

    success, user, message = _authentication_service.authenticate(
        username=username, video_file=video_file, request=request
    )

    if success:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({"success": True, "redirect": "/dashboard/"})
        return redirect("dashboard")
    else:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({"success": False, "error": message}, status=400)
        messages.error(request, message)
        return render(request, "authentication/login.html")


def dashboard_view(request):
    user_id = request.session.get("user_id")
    if not user_id:
        return redirect("login")

    from .models import User

    try:
        user = User.objects.get(id=user_id)
        return render(request, "authentication/dashboard.html", {"user": user})
    except User.DoesNotExist:
        request.session.flush()
        return redirect("login")


def logout_view(request):
    request.session.flush()
    return redirect("login")


@require_http_methods(["POST"])
def establish_session_view(request):
    """
    Establish Django session using one-time token from WebSocket authentication.
    """
    import json
    from django.core.cache import cache

    try:
        data = json.loads(request.body)
        token = data.get("token")

        if not token:
            return JsonResponse(
                {"success": False, "error": "Token required"}, status=400
            )

        # Retrieve user_id from cache using token
        cache_key = f"auth_token:{token}"
        user_id = cache.get(cache_key)

        if not user_id:
            return JsonResponse(
                {"success": False, "error": "Invalid or expired token"}, status=401
            )

        # Delete token (one-time use)
        cache.delete(cache_key)

        # Establish session
        request.session["user_id"] = user_id
        request.session.save()

        return JsonResponse({"success": True, "message": "Session established"})

    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Invalid JSON"}, status=400)
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)
