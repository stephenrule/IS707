from django.test import Client, TestCase
from .views import home
from django.urls import resolve

# Create your tests here.

class HomeTests(TestCase):
    def test_home_view_status_code(self):
        response = self.client.get('', follow=True)
        self.assertEqual(response.status_code, 200)

    def test_home_url_resolves_home_view(self):
        view = resolve('/')
        self.assertEquals(view.func, home)
    