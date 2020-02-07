from django.urls import path,include
from . import views
from django.contrib import admin

urlpatterns = [
    path('',views.uindex,name='uindex'),
    
    path('classifier/dindex/',views.dindex,name='dindex'),
    path('classifier/dpredictions/<str:imgs>',views.dprediction,name='dprediction'),

    path('classifier/uindex/',views.uindex,name='uindex'),
    path('classifier/upredictions/<str:imgs>',views.uprediction,name='uprediction'),
    
    path('outfit_classification/admin/', admin.site.urls),

    path('classifier/user/',views.user, name='user')
]

