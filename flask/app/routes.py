from turtle import hideturtle
from flask import request, render_template, flash, redirect
from app import app
from app.forms import LoginForm

# Import local python files
import db
import chat

# Main Webpage - Splash Page
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

# Called from the chatbot page (chat.html - "/get") - $.get("/get", { msg: rawText }).done(function(data) {
# Takes user input and gets chatbot response from chat (chat.py) to be displayed on Chatbot Page
@app.route('/get')
def get_bot_response():
    userText = request.args.get('msg')
    return str(chat.get_response(userText))

# Chatbot Page
@app.route('/chat')
def home():
    return render_template("chat.html")

# Testing Page - Temporary and used for testing things. 
# Note: Remove link in base.html and delete test.html when complete
@app.route('/test')
def test1():
    return render_template("test.html")

# User Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        flash('Login requested for user {}, remember_me={}'.format(
            form.username.data, form.remember_me.data))
        return redirect('/index')
    return render_template('login.html', title='Sign In', form=form)