#### Chat Bot Project - IS 707
#
# Requires:
#### Python: 3.4 - 3.8
#### Pip: chatterbot, chatterbot_corpus
# 

# Imports
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

# Create and Train Chatbot
# Note: Following: https://www.upgrad.com/blog/how-to-make-chatbot-in-python/
my_bot = ChatBot(name='PyBot', read_only=True,
                  logic_adapters=[
                  'chatterbot.logic.MathematicalEvaluation',
                  'chatterbot.logic.BestMatch'])
                  
small_talk = ['hi there!', 
               'hi!',
               'how are you?',
               'i\'m cool.',
               'fine, you?',
               'always cool.',
               'i\'m ok',
               'glad to hear that.',
               'i\'m fine',
               'glad to hear that.',
               'excellent, glad to hear that.',
               'not so good',
               'sorry to hear that.',
               'what\'s your name?',
               'i\'m pybot. ask me a math question, please.']
               
math_talk_1 = ['pythagorean theorem',
               'a squared plus b suqred equals c squard.']

math_talk_2 = ['law of cosines',
                'c**2 = a**2 + b**2 - 2 * a * b * cos(gamma)']
                
# Train the bot
list_trainer = ListTrainer(my_bot)

for item in (small_talk, math_talk_1, math_talk_2):
    list_trainer.train(item)
    
print(my_bot.get_resposne("hi"))
print(my_bot.get_response("i fell awesome today"))
# Do a loop here to get input with q exiting

# Train chat bot with a corpus of data
#from chatterbot.trainers import ChatterBotCorpusTrainer

#corpus_trainer = ChatterBotCorpusTrainer(my_bot)
#corpus_trainer.train('chatterbot.corpus.english')