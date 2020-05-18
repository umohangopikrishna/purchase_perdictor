from django.urls import path
from ml_app import views
app_name = 'ml_app'
urlpatterns=[
path('homepage/',views.home,name='homepage'),
path('resultpage/',views.result,name='resultpage'),
path('userpage/',views.info_page,name='userpage'),
path('logout/',views.home,name='homepage'),

]
