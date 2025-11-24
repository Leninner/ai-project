import os
import logging
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR.parent / ".env")

speechbrain_logger = logging.getLogger("speechbrain")
speechbrain_logger.setLevel(logging.CRITICAL)
speechbrain_logger.disabled = True

for module in [
    "speechbrain.utils.fetching",
    "speechbrain.utils.parameter_transfer",
    "speechbrain.dataio.encoder",
    "speechbrain.utils.checkpoints",
]:
    logger = logging.getLogger(module)
    logger.disabled = True


class SkipFaviconFilter(logging.Filter):
    def filter(self, record):
        msg = str(record.getMessage())
        return "/favicon.ico" not in msg


SECRET_KEY = os.environ.get("SECRET_KEY", "django-insecure-change-this-in-production")

DEBUG = True

ALLOWED_HOSTS = []

INSTALLED_APPS = [
    "daphne",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "channels",
    "authentication",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "auth_platform.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "auth_platform.wsgi.application"
ASGI_APPLICATION = "auth_platform.asgi.application"

CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer"
    }
}

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("DB_NAME", "auth_platform"),
        "USER": os.environ.get("DB_USER", "postgres"),
        "PASSWORD": os.environ.get("DB_PASSWORD", "postgres"),
        "HOST": os.environ.get("DB_HOST", "localhost"),
        "PORT": os.environ.get("DB_PORT", "5432"),
    }
}

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True

STATIC_URL = "static/"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "[{levelname:8s}] {asctime} | {name:30s} | {message}",
            "style": "{",
            "datefmt": "%H:%M:%S",
        },
        "detailed": {
            "format": "[{levelname:8s}] {asctime} | {name:30s} | {funcName}:{lineno} | {message}",
            "style": "{",
            "datefmt": "%H:%M:%S",
        },
    },
    "filters": {
        "require_debug_false": {
            "()": "django.utils.log.RequireDebugFalse",
        },
        "skip_favicon": {
            "()": "auth_platform.settings.SkipFaviconFilter",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "DEBUG" if DEBUG else "INFO",
            "filters": ["skip_favicon"],
        },
        "console_detailed": {
            "class": "logging.StreamHandler",
            "formatter": "detailed",
            "level": "DEBUG",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "WARNING",
    },
    "loggers": {
        "django": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False,
        },
        "django.server": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "django.request": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False,
        },
        "django.template": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False,
        },
        "authentication": {
            "handlers": ["console"],
            "level": "DEBUG" if DEBUG else "INFO",
            "propagate": False,
        },
        "speechbrain": {
            "handlers": [],
            "level": "CRITICAL",
            "propagate": False,
        },
        "speechbrain.utils.fetching": {
            "handlers": [],
            "level": "CRITICAL",
            "propagate": False,
        },
        "speechbrain.utils.parameter_transfer": {
            "handlers": [],
            "level": "CRITICAL",
            "propagate": False,
        },
        "speechbrain.dataio.encoder": {
            "handlers": [],
            "level": "CRITICAL",
            "propagate": False,
        },
        "speechbrain.utils.checkpoints": {
            "handlers": [],
            "level": "CRITICAL",
            "propagate": False,
        },
    },
}
