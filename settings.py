import os
from dotenv import load_dotenv
load_dotenv()

SECRET_KEY   = os.getenv("DJANGO_SECRET_KEY", "dev-secret-key-change-in-prod")

# Railway sets NODE_ENV=production — use that to detect environment
IS_PRODUCTION = os.getenv("RAILWAY_ENVIRONMENT") is not None
DEBUG         = not IS_PRODUCTION

# Railway provides the public URL as an env var
RAILWAY_URL = os.getenv("RAILWAY_PUBLIC_DOMAIN", "")
ALLOWED_HOSTS = ["*"] if not IS_PRODUCTION else [
    "localhost",
    "127.0.0.1",
    RAILWAY_URL,
]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "api",
]

MIDDLEWARE = [
    "django.middleware.common.CommonMiddleware",
    "api.middleware.CORSMiddleware",
]

ROOT_URLCONF = "urls"
DATABASES    = {}
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"