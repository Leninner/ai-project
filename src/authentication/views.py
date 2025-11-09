from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .services import RegistrationService, AuthenticationService


_registration_service = RegistrationService()
_authentication_service = AuthenticationService()


@require_http_methods(["GET", "POST"])
def register_view(request):
    if request.method == 'GET':
        return render(request, 'authentication/register.html')
    
    username = request.POST.get('username')
    video_file = request.FILES.get('video')
    
    if not username or not video_file:
        error_msg = 'Username and video are required'
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'success': False, 'error': error_msg}, status=400)
        messages.error(request, error_msg)
        return render(request, 'authentication/register.html')
    
    success, user, message = _registration_service.register_user(
        username=username,
        video_file=video_file
    )
    
    if success:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'success': True, 'redirect': '/login/'})
        messages.success(request, message)
        return redirect('login')
    else:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'success': False, 'error': message}, status=400)
        messages.error(request, message)
        return render(request, 'authentication/register.html')


@require_http_methods(["GET", "POST"])
def login_view(request):
    if request.method == 'GET':
        return render(request, 'authentication/login.html')
    
    username = request.POST.get('username')
    video_file = request.FILES.get('video')
    
    if not username or not video_file:
        error_msg = 'Username and video are required'
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'success': False, 'error': error_msg}, status=400)
        messages.error(request, error_msg)
        return render(request, 'authentication/login.html')
    
    success, user, message = _authentication_service.authenticate(
        username=username,
        video_file=video_file,
        request=request
    )
    
    if success:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'success': True, 'redirect': '/dashboard/'})
        return redirect('dashboard')
    else:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'success': False, 'error': message}, status=400)
        messages.error(request, message)
        return render(request, 'authentication/login.html')


def dashboard_view(request):
    if not request.user.is_authenticated:
        return redirect('login')
    
    return render(request, 'authentication/dashboard.html', {'user': request.user})


def logout_view(request):
    from django.contrib.auth import logout
    logout(request)
    return redirect('login')

