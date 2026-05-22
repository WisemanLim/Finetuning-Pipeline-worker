"""FastAPI 라우터 모음."""

from .admin import router as admin_router
from .auth import router as auth_router
from .data import router as data_router
from .jobs import router as jobs_router
from .models_api import router as models_router
from .users import router as users_router

__all__ = [
    "auth_router", "users_router", "models_router",
    "jobs_router", "admin_router", "data_router",
]
