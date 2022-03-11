from flask import render_template
from app import app
from app.forms import LoginForm

@app.route('/')
@app.route('/index')
def index():
    #return "Hello, World!"
    user = {'username': 'user'}
    posts = [
        {
        'author': {'username': 'John'},
        'body': 'Beautiful day in portland!'
        },
        {
            'author': {'username': 'Susan'},
            'body': 'The Avengers movie was so cool!'
        }

    ]
    return render_template('index.html', title='Home', user=user, posts=posts)



########################## CHAT ###########################
#### long_responses.py
import random

R_EATING = "I don't like eating anything because I'm a bot obviously!"
R_ADVICE = "If I were you, I would go to the internet and type exactly what you wrote there!"

def unknown():
    response = ['Could you please re-phrase that?', 
                "...", 
               "Sounds about right",
               "What does that mean?"][random.randrange(4)]
    return response


#### Chat
from flask import Flask, render_template, request
import re
# Not pulling in long_responses for some reason. Run flask shell and error popped up
# Fix: Copied contents from long_responses into this file. 
#import long_responses as long

def message_probability(user_message, recognised_words, single_response=False, required_words=[]):
    message_certainty = 0
    has_required_words = True

    # Counts how many words are present in each predefined message
    for word in user_message:
        if word in recognised_words:
            message_certainty += 1

    # Calculates the percent of recognised words in a user message
    percentage = float(message_certainty) / float(len(recognised_words))

    # Checks that the required words are in the string
    for word in required_words:
        if word not in user_message:
            has_required_words = False
            break

    # Must either have the required words, or be a single response
    if has_required_words or single_response:
        return int(percentage * 100)
    else:
        return 0
    
def check_all_messages(message):
    highest_prob_list = {}

    # Simplifies response creation / adds it to the dict
    def response(bot_response, list_of_words, single_response=False, required_words=[]):
        nonlocal highest_prob_list
        highest_prob_list[bot_response] = message_probability(message, list_of_words, single_response, required_words)

    #login = "Hackers Storage Unit Login Page - <a href=https://www.google.com>Login Page</a>"
    login = "UMBC Storage: <a href=/login>Login Page</a>"
    image = "<img src=\"static/one.jpg\" alt=\"Test Image\">"

    # Responses -------------------------------------------------------------------------------------------------------
    response('Hello!', ['hello', 'hi', 'hey', 'sup', 'heyo'], single_response=True)
    response('See you!', ['bye', 'goodbye'], single_response=True)
    response('I\'m doing fine, and you?', ['how', 'are', 'you', 'doing'], required_words=['how'])
    response('You\'re welcome!', ['thank', 'thanks'], single_response=True)
    response('Thank you!', ['i', 'love', 'code', 'palace'], required_words=['code', 'palace'])
    response(login, ['website', 'login'], single_response=True)  
    response(image, ['image', 'logo'], single_response=True)
    response('Small Storage Unit Price: $50', ['small', 'storage', 'unit', 'price'], required_words=['small', 'price'])
    response('Medium Storage Unit Price: $75', ['medium', 'storage', 'unit', 'price'], required_words=['medium', 'price'])
    response('Large Storage Unit Price: $95', ['large', 'storage', 'unit', 'price'], required_words=['large', 'price'])


    # Longer responses
    response(R_ADVICE, ['give', 'advice'], required_words=['advice'])
    response(R_EATING, ['what', 'you', 'eat'], required_words=['you', 'eat'])

    best_match = max(highest_prob_list, key=highest_prob_list.get)
    # DEBUG - Shows us the probability
    print(highest_prob_list)
    
    #return best_match
    return unknown() if highest_prob_list[best_match] < 1 else best_match


# Used to get the response
def get_response(user_input):
    split_message = re.split(r'\s+|[,;?!.-]\s*', user_input.lower())
    response = check_all_messages(split_message)
    return response


# Testing the response system
#while True:
#    print('Bot: ' + get_response(input('You: ')))

@app.route('/get')
def get_bot_response():
    userText = request.args.get('msg')
    return str(get_response(userText))

@app.route('/chat')
def home():
    return render_template("chat.html")

@app.route('/login')
def login():
    form = LoginForm()
    return render_template('login.html', title='Sign In', form=form)
