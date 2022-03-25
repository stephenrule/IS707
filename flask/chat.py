# Database: db.py
import db as local_db

# Responses: long_responses.py
import long_responses as long

# Imports
import random
from flask import Flask, render_template, request
import re


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

# Take user input and see if any word matches predefined text
def check_all_messages(message):
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
    # NOTE: Look into stemming and lemmatizing
    # NOTE: NLTK corpus (For narrowing hello, hi, hey, etc): from nltk.corpus import wordnet
    # NOTE: syns = wordnet.synsets
    response('Hello!', ['hello', 'hi', 'hey', 'sup', 'heyo'], single_response=True)
    response('See you!', ['bye', 'goodbye'], single_response=True)
    response('I\'m doing fine, and you?', ['how', 'are', 'you', 'doing'], required_words=['how'])
    response('You\'re welcome!', ['thank', 'thanks'], single_response=True)
    response('Thank you!', ['i', 'love', 'code', 'palace'], required_words=['code', 'palace'])
    response(login, ['website', 'login'], single_response=True)  
    response(site_map, ['sitemap', 'site', 'map'], single_response=True)
    response(contact, ['contact', 'number', 'address', 'phone'], single_response=True)
    response('Small Storage Unit Price: $50', ['small', 'storage', 'unit', 'price'], required_words=['small', 'price'])
    response('Medium Storage Unit Price: $75', ['medium', 'storage', 'unit', 'price'], required_words=['medium', 'price'])
    response('Large Storage Unit Price: $95', ['large', 'storage', 'unit', 'price'], required_words=['large', 'price'])
    response('Small Units Available: ' + str(local_db.getUnitAvailability('SMALL', 'catonsville')), ['small', 'unit', 'units', 'available'], single_response=True)
    
    # Database responses
    response(local_db.getUnitAvailability('MEDIUM', 'catonsville'), ['medium', 'unit', 'units', 'available'], single_response=True)
    response('Small Units Available: ' + str(local_db.getUnitAvailability('SMALL', 'catonsville')) + 
             ', Medium Units Available: ' + str(local_db.getUnitAvailability('MEDIUM', 'catonsville')) +
             ', Large Units Available: ' + str(local_db.getUnitAvailability('Large', 'catonsville'))
             , ['test', 'unit', 'units', 'available'], single_response=True)

    # Longer responses - long_responses.py
    response(long.R_ADVICE, ['give', 'advice'], required_words=['advice'])
    response(long.R_EATING, ['what', 'you', 'eat'], required_words=['you', 'eat'])
    
    best_match = max(highest_prob_list, key=highest_prob_list.get)
    # DEBUG - Shows us the probability for each response in a new line
    for key, value in highest_prob_list.items():
        print(key, ' : ', value)
    # DEBUG - Shows us the probability all together
    #print(highest_prob_list)

    # FUTURE: AI Approach
    #if highest_prob_list[best_match] < 1: 
        #DO AI Approach (Variable = message) Note: message is tokenized list of lowercase words

        #bot_response = "AI Approach Returned Message"
        # Logic: If AI returns a message then assign 1. If AI returns nothing then return nothing 
        #if bot_response != "Nothing":
            #highest_prob_list[bot_response] = 1


    #return best_match
    return long.unknown() if highest_prob_list[best_match] < 1 else best_match


# Used to get the response
def get_response(user_input):
    split_message = re.split(r'\s+|[,;?!.-]\s*', user_input.lower())
    ####################
    # Remove stop words - Use nltk
    #####################
    response = check_all_messages(split_message)
    return response
