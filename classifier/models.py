from django.db import models

# Create your models here.
class Add_Images(models.Model): 
    image= models.ImageField()
    class Meta:
        verbose_name_plural = "Add_Images"