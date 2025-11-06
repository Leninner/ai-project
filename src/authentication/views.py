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
    
    email = request.POST.get('email')
    password = request.POST.get('password')
    name = request.POST.get('name')
    image_file = request.FILES.get('image')
    voice_file = request.FILES.get('voice')
    
    if not all([email, password, name, image_file, voice_file]):
        messages.error(request, 'All fields are required')
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return render(request, 'authentication/register.html', status=400)
        return render(request, 'authentication/register.html')
    
    success, user, message = _registration_service.register_user(
        email=email,
        password=password,
        name=name,
        voice_file=voice_file,
        image_file=image_file
    )
    
    if success:
        messages.success(request, message)
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return redirect('login')
        return redirect('login')
    else:
        messages.error(request, message)
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return render(request, 'authentication/register.html', status=400)
        return render(request, 'authentication/register.html')


@require_http_methods(["GET", "POST"])
def login_view(request):
    if request.method == 'GET':
        return render(request, 'authentication/login.html')
    
    email = request.POST.get('email')
    password = request.POST.get('password')
    image_file = request.FILES.get('image')
    voice_file = request.FILES.get('voice')
    
    if not all([email, password, image_file, voice_file]):
        messages.error(request, 'All fields are required')
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return render(request, 'authentication/login.html', status=400)
        return render(request, 'authentication/login.html')
    
    success, user, message = _authentication_service.authenticate(
        email=email,
        password=password,
        voice_file=voice_file,
        image_file=image_file,
        request=request
    )
    
    if success:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return redirect('dashboard')
        return redirect('dashboard')
    else:
        messages.error(request, message)
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return render(request, 'authentication/login.html', status=400)
        return render(request, 'authentication/login.html')


def dashboard_view(request):
    if not request.user.is_authenticated:
        return redirect('login')
    
    return render(request, 'authentication/dashboard.html', {'user': request.user})


def logout_view(request):
    from django.contrib.auth import logout
    logout(request)
    return redirect('login')

