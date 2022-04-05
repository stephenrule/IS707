from turtle import hideturtle
from flask import request, render_template, flash, redirect, url_for
from app import app, db
from app.forms import LoginForm, RegistrationForm
from flask_login import current_user, login_user, logout_user, login_required
from app.models import User
from werkzeug.urls import url_parse
import re

# Import local python files
import db as local_db
import chat

# Main Webpage - Splash Page
@app.route('/')
@app.route('/index')
#@login_required
def index():
    #return "Hello, World!"
   
    return render_template('index.html', title='Home')

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

# Profile Page - Used to make purchases and update information
# Thought: Display current information and have a form below to update the information
#@app.route('/profile')
@app.route('/user/<username>', methods=['GET', 'POST'])
@login_required
#def profile():
def user(username):
    # Post - For Buttons
    if request.method == "POST":
        if request.form["submit_button"] == "Add 1 Small Unit":
            print('Add 1 Small Unit')
            # Database Code (Delete Print Above)
            return redirect(url_for('user', username=current_user.username))
        elif request.form["submit_button"] == "Remove 1 Small Unit":
            print('Remove 1 Small Unit')
            # Database Code (Delete Print Above)
            return redirect(url_for('user', username=current_user.username))
        elif request.form["submit_button"] == "Add 1 Medium Unit":
            print('Add 1 Medium Unit')
            # Database Code (Delete Print Above)
            return redirect(url_for('user', username=current_user.username))
        elif request.form["submit_button"] == "Remove 1 Medium Unit":
            print('Remove 1 Medium Unit')
            # Database Code (Delete Print Above)
            return redirect(url_for('user', username=current_user.username))
        elif request.form["submit_button"] == "Add 1 Large Unit":
            print('Add 1 Large Unit')
            # Database Code (Delete Print Above)
            return redirect(url_for('user', username=current_user.username))
        elif request.form["submit_button"] == "Remove 1 Large Unit":
            print('Remove 1 Large Unit')
            # Database Code (Delete Print Above)
            return redirect(url_for('user', username=current_user.username)) 
        else: 
            print('Everything Else')
            return redirect(url_for('user', username=current_user.username))

    # GET - Display Page
    elif request.method == 'GET':
        user = User.query.filter_by(username=username).first_or_404()

        # Database queries to display data on profile page
        ## Get units available for each unit size
        small_units = local_db.getUnitAvailability('catonsville', 'SMALL')
        medium_units = local_db.getUnitAvailability('catonsville', 'MEDIUM')
        large_units = local_db.getUnitAvailability('catonsville', 'LARGE')
        
        ## Get total number of units the user owns for each unit size
        get_user_small = len(local_db.getUserUnits(user.username, 'SMALL'))
        get_user_medium = len(local_db.getUserUnits(user.username, 'MEDIUM'))
        get_user_large = len(local_db.getUserUnits(user.username, 'LARGE'))

        ## Get the unit numbers for each unit the user owns for each unit size
        get_units_small = str(local_db.getUserUnits(user.username, 'SMALL'))
        get_units_medium = str(local_db.getUserUnits(user.username, 'MEDIUM'))
        get_units_large = str(local_db.getUserUnits(user.username, 'LARGE'))
        #### Remove the [ ] ( ) ,
        # Debug
        #print(type(get_units_small))
        #print(get_units_small)
        get_units_small = re.sub(r'[\(\)\[\]\,]','',get_units_small)
        # Debug
        #print(get_units_small)
        get_units_medium = re.sub(r'[\(\)\[\]\,]','',get_units_medium)
        get_units_large = re.sub(r'[\(\)\[\]\,]','',get_units_large)

        # Render profile.html and pas all variables
        return render_template("profile.html", user=user,
                            smallUnits=small_units, mediumUnits=medium_units, largeUnits=large_units,
                            userSmall=get_user_small, userMedium=get_user_medium, userLarge=get_user_large,
                            userUnitSmall=get_units_small, userUnitMedium=get_units_medium, userUnitLarge=get_units_large)

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
            #next_page = url_for('profile')
            next_page = url_for('user', username=current_user.username)
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

        # 2nd Database
        local_db.addSimpleUserAccount(form.username.data)

        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)