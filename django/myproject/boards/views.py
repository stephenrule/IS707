from django.shortcuts import render
from django.http import HttpResponse
from .models import Board

####################################################################
import json
from django.views.generic.base import TemplateView
from django.views.generic import View
from django.http import JsonResponse
#from chatterbot import ChatBot
#from chatterbot.ext.django_chatterbot import settings

# Create your views here.

class chatterbot:
    def __init__(self, name):
        self.name = 'John'
    
    def get_response(self, i):
        self.inp = i


class ChatterBotAppView(TemplateView):
    template_name = 'chat.html'


class ChatterBotApiView(View):
    """
    Provide an API endpoint to interact with ChatterBot.
    """

    #chatterbot = ChatBot(**settings.CHATTERBOT)
    chatterbot = 'John'

    def post(self, request, *args, **kwargs):
        """
        Return a response to the statement in the posted data.
        * The JSON data should contain a 'text' attribute.
        """
        input_data = json.loads(request.body.decode('utf-8'))

        if 'text' not in input_data:
            return JsonResponse({
                'text': [
                    'The attribute "text" is required.'
                ]
            }, status=400)

        response = self.chatterbot.get_response(input_data)

        response_data = response.serialize()

        return JsonResponse(response_data, status=200)

    def get(self, request, *args, **kwargs):
        """
        Return data corresponding to the current conversation.
        """
        return JsonResponse({
            'name': self.chatterbot.name
        })

####################################################################

def home(request):
    #return HttpResponse('Hello, World!')
    boards = Board.objects.all()
    return render(request, 'home.html', {'boards':boards})
    #boards_names = list()

    #for board in boards:
        #boards_names.append(board.name)
    
    #response_html = '<br>'.join(boards_names)

    #return HttpResponse(response_html)

def board_topics(request, pk):
    board = Board.objects.get(pk=pk)
    #board = Board.objects.get(pk=self.kwargs.get('pk'))
    return render(request, 'topics.html', {'board': board})

#def chat(request):
#    chat = Board.objects.all()
#    return render(request, 'chat.html', {'chat':chat})

