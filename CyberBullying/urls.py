from django.contrib import admin
from django.urls import path
from CyberBullying import views
from django.conf import settings
from django.conf.urls.static import static
import os

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('SendPost', views.SendPost, name='SendPost'),
    path('Register', views.Register, name='Register'),
    path('Admin', views.Admin, name='Admin'),
    path('Login', views.UserLogin, name='Login'),
    path('AddCyberMessages', views.AddCyberMessages, name='AddCyberMessages'),
    path('RunAlgorithms', views.RunAlgorithms, name='RunAlgorithms'),
    path('MonitorPost', views.MonitorPost, name='MonitorPost'),
    path('delete_post/', views.delete_post, name='delete_post'),
    path('Signup', views.Signup, name='Signup'),
    path('AdminLogin', views.AdminLogin, name='AdminLogin'),
    path('ViewUsers', views.ViewUsers, name='ViewUsers'),
    path('ViewUserPost', views.ViewUserPost, name='ViewUserPost'),
    path('AddBullyingWords', views.AddBullyingWords, name='AddBullyingWords'),
    path('RunAlgorithm', views.RunAlgorithm, name='RunAlgorithm'),
    path('PostSent', views.PostSent, name='PostSent'),
]

# Serve static and media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static('/static/photo/', document_root=os.path.join(settings.BASE_DIR, 'CyberBullying', 'static', 'photo'))