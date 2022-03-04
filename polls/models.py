from django.db import models
from django.conf import settings

# Create your models here.


class Image(models.Model):
    result = models.CharField(max_length=200)
    image = models.ImageField(upload_to='image/')

    def __str__(self):
        return self.result
