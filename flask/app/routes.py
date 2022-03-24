from turtle import hideturtle
from flask import request, render_template, flash, redirect, url_for
from app import app, db
from app.forms import LoginForm, RegistrationForm
from flask_login import current_user, login_user, logout_user, login_required
from app.models import User
from werkzeug.urls import url_parse

# Import local python files
#import db as local_db
import chat

# Main Webpage - Splash Page
@app.route('/')
@app.route('/index')
@login_required
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
    return render_template('index.html', title='Home', posts=posts)

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
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
        #return redirect(url_for('index'))
    return render_template('login.html', title='Sign In', form=form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        #print(user)
        # Issue here where its not recognizing session from db connection
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)