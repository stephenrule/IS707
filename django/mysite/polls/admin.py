from django.contrib import admin

# Edit questions table from Admin page
from .models import Question

# Register your models here.

# Edit questions table from Admin page
admin.site.register(Question)
