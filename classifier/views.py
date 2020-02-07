from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from classifier.prediction.pred import *
from .models import Add_Images
# Create your views here.

def dindex(request):
    return render(request,'./classifier/dindex.html')

def dprediction(request,imgs) : 
    predicted=db_predict_image(imgs)
    context={
        'title' : 'Predicted Outcome',
        'prediction_class': predicted['prediction_class'],
        'prediction_perc' :  predicted['prediction_perc']
    }
    return render(request,'./classifier/dpredictions.html', context)

def uindex(request) : 
    images=Add_Images.objects.all()
    context={
        'all_images' : images
    }
    return render(request,'./classifier/uindex.html',context)

def uprediction(request,imgs) : 
    predicted=user_predict_image(imgs)
    context={
        'title' : 'Predicted Outcome',
        'prediction_class': predicted['prediction_class'],
        'prediction_perc' :  predicted['prediction_perc']
    }
    return render(request,'./classifier/upredictions.html', context)

def user(request) : 
    all_prediction=[]
    all_images=Add_Images.objects.all()
    for each_image in all_images:
        pre=str(each_image.image)
        predict_label_image = user_image(pre)
        all_prediction.append(predict_label_image)
        predict_label_image={}

    context = {
        'title' : 'hello',
        'all_prediction': all_prediction
    }
    return render(request, './classifier/user.html',context)


