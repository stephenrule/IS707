# Database: db.py
import db as local_db

# Responses: long_responses.py
import long_responses as long

# Imports
import random
from flask import Flask, render_template, request
import re
import BiDAF
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet


#### Determines probability based on number of words that match and returns highest value (Percentage)
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

def getSynSet(phrase):
    synonyms = []

    for syn in wordnet.synsets(phrase):
        for i in syn.lemmas():
            synonyms.append(i.name())
    print(synonyms)
    return synonyms


# Take user input and see if any word matches predefined text
def check_all_messages(message, ai_string):
    highest_prob_list = {}

    # Simplifies response creation / adds it to the dict
    def response(bot_response, list_of_words, single_response=False, required_words=[]):
        nonlocal highest_prob_list
        highest_prob_list[bot_response] = message_probability(message, list_of_words, single_response, required_words)

    # Images and Links
    #login = "Hackers Storage Unit Login Page - <a href=https://www.google.com>Login Page</a>"
    login = "UMBC Storage: <a href=/login>Login Page</a>"
    site_map = "<img src=\"static/site_map.png\" alt=\"Site Map Image\">"
    contact = "<img src=\"static/contact.png\" alt=\"Contact Image\">"

    # Responses -------------------------------------------------------------------------------------------------------
    # NOTE: Look into stemming and **lemmatizing
    # NOTE: NLTK corpus (For narrowing hello, hi, hey, etc): from nltk.corpus import wordnet
    # NOTE: syns = wordnet.synsets
    # NOTE: Do a print statement of the results after lemmatizing and synsets just to see everything before applying rules
    response('Hello!', ['hello', 'hi', 'hey', 'sup', 'heyo'], single_response=True)
    #response('See you!', ['bye', 'goodbye'], single_response=True)
    response('See you!', getSynSet('bye'), single_response=True)
    response('I\'m doing fine, and you?', ['how', 'are', 'you', 'doing'], required_words=['how'])
    response('You\'re welcome!', ['thank', 'thanks'], single_response=True)
    response(login, ['website', 'login'], single_response=True)  
    response(site_map, ['sitemap', 'site', 'map'], single_response=True)
    response(contact, ['contact', 'number', 'address', 'phone'], single_response=True)
    response('Small Storage Unit Price: $50', ['small', 'storage', 'unit', 'price'], required_words=['small', 'price'])
    response('Medium Storage Unit Price: $75', ['medium', 'storage', 'unit', 'price'], required_words=['medium', 'price'])
    response('Large Storage Unit Price: $95', ['large', 'storage', 'unit', 'price'], required_words=['large', 'price'])
        
    # Database responses
    response('Small Units Available: ' + str(local_db.getUnitAvailability('catonsville', 'SMALL')), ['small', 'units', 'available'], required_words=['small', 'units', 'available'])
    response('Medium Units Available: ' + str(local_db.getUnitAvailability('catonsville', 'MEDIUM')), ['medium', 'units', 'available'], required_words=['medium', 'units', 'available'])
    response('Medium Units Available: ' + str(local_db.getUnitAvailability('catonsville', 'LARGE')), ['large', 'units', 'available'], required_words=['large', 'units', 'available'])
    response('Small Units Available: ' + str(local_db.getUnitAvailability('catonsville', 'SMALL')) + 
             ', Medium Units Available: ' + str(local_db.getUnitAvailability('catonsville', 'MEDIUM')) +
             ', Large Units Available: ' + str(local_db.getUnitAvailability('catonsville', 'LARGE'))
             , ['units', 'available'], required_words=['units', 'available'])

    # Longer responses - long_responses.py
    response(long.R_ADVICE, ['give', 'advice'], required_words=['advice'])
    response(long.R_EATING, ['what', 'you', 'eat'], required_words=['you', 'eat'])
    

    ##########
    best_match = max(highest_prob_list, key=highest_prob_list.get)
    # DEBUG - Shows us the probability for each response in a new line
    for key, value in highest_prob_list.items():
        print(key, ' : ', value)
    # DEBUG - Shows us the probability all together
    #print(highest_prob_list)

    # FUTURE: AI Approach
    print("####################################################")
    print("AI String: " + str(ai_string))
    print("Higest Prob List: " + str(highest_prob_list))
    if highest_prob_list[best_match] < 1: 
        #AI Approach (Variables: message (Tokenized), ai_string (String)) Note: message is tokenized list of lowercase words
        print("#### AI APPROACH ####")
        bot_response = BiDAF.chatbot_ai(ai_string)
        print(bot_response)

        # bot_response will probably never be empty but if doesn't know how to answer a question it usually responds with the entire passage.
        ## if len is greater than 100 then we can assume its the passage and the bot doesn't know how to respond which puts is to the unknown responses. 
        if bot_response != '' and len(bot_response) < 100:
            highest_prob_list[bot_response] = 15
            print("#### Added bot_response to highest_prob_list ####")
            print(highest_prob_list)

        #bot_response = "AI Approach Returned Message"
        # Logic: If AI returns a message then assign 1. If AI returns nothing then return nothing 
        #if bot_response != "Nothing":
            #highest_prob_list[bot_response] = 1


    #return best_match
    best_match = max(highest_prob_list, key=highest_prob_list.get)
    return long.unknown() if highest_prob_list[best_match] < 1 else best_match


# Used to get the response
def get_response(user_input):
    split_message = re.split(r'\s+|[,;?!.-]\s*', user_input.lower())
    ####################
    # Tokenize using NLTK (If works comment line above "split_message....")
    #split_new = NLTK SPLITTING

    # Remove stop words - Use nltk
    #split_new = NLTK stop word removal
    
    #Lemmatize
    #split_new = Lemmatized words
    #####################
    response = check_all_messages(split_message, user_input)

    # Add user_input and respone to database to view later.
    local_db.addResponse(user_input, response)
    return response
