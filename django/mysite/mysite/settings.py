"""
Django settings for mysite project.

Generated by 'django-admin startproject' using Django 4.0.3.

For more information on this file, see
https://docs.djangoproject.com/en/4.0/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/4.0/ref/settings/
"""

from pathlib import Path
import os.path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure--h$$%0a9iaqm5u1%9u1937+c@q*pcaqz*99!@eqyfg4b+%n4z1'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []


# Application definition

INSTALLED_APPS = [
    'hello_world', # hello_world app
    'polls.apps.PollsConfig', # New Polls config (After model update)
    'django.contrib.admin', # Admin Site
    'django.contrib.auth', # Authentication System
    'django.contrib.contenttypes', # Framework for content types
    'django.contrib.sessions', # Session framework
    'django.contrib.messages', # Messaging framework
    'django.contrib.staticfiles', # Framework for managing static files. 
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'mysite.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': ["mysite/templates/"],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'mysite.wsgi.application'


# Database
# https://docs.djangoproject.com/en/4.0/ref/settings/#databases

DATABASES = {
    'default': {
        # SQLITE3 - WORKING
        #'ENGINE': 'django.db.backends.sqlite3',
        #'NAME': BASE_DIR / 'db.sqlite3',

        # DJANGO - WORKING (LAB Remote Connection and Local on Linux Server - Host changed to localhost)
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'djangodatabase',
        'USER': 'chatbot_user',
        'PASSWORD': 'password',
        'HOST': '172.16.1.15',
        'PORT': '3306',
        
    }
}
################################################################
# HOLD - Change database over to another type
# django.db.backends.sqlite3
# django.db.backends.postgresql
# django.db.bakends.mysql
# django.db.bakends.oracle

# HOLD - name
# BASE_DIR / 'db.sqlite3'  (Name of file and location)

# Additional Settings
# 'USER': 'mydatabaseuser',
# 'PASSWORD': 'mydatabasepassword',
# 'HOST': '/var/run/mysql' or 'localhost' or 'x.x.x.x'
# 'PORT': '3306',
################################################################

# Password validation
# https://docs.djangoproject.com/en/4.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/4.0/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'America/New_York'

# Specifies whether Django's tranlsation system should be enabled. 
USE_I18N = True

# Localized formatting: Display numbers and dates using the format of the current locale.
# Note: Added
USE_L10N = True

# True = default time zone Django will use. False: Will use naive datetimes in local time. 
USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.0/howto/static-files/

#STATIC_ROOT = 'D:\A_anaconda\IS707\django\myproject\static'
STATIC_ROOT = os.path.join(BASE_DIR, 'static')
STATIC_URL = '/static/'

#STATICFILES_DIRS = 'D:\A_anaconda\IS707\django\myproject\static'
STATICFILES_DIRS = (os.path.join(BASE_DIR, 'static'), )

# Default primary key field type
# https://docs.djangoproject.com/en/4.0/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
