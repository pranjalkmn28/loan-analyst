from django.urls import path
from api.views import analyse_view, health_view

urlpatterns = [
    path("analyse", analyse_view),
    path("health",  health_view),
]