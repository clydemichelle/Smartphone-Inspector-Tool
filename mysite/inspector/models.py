import datetime

from django.db import models
from django.utils import timezone

# Create your models here.
class Question(models.Model):
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date asked')

    def __str__(self):
        return self.question_text
    
    def was_published_recently(self):
        return self.pub_date >= timezone.now() - datetime.timedelta(days=1)

class Info(models.Model):
    brand_name = models.CharField(max_length=200)
    price = models.IntegerField()

    def __str__(self):
        return  '{}-{}'.format(self.brand_name, self.price) 


