import random

R_EATING = "I don't like eating anything because I'm a bot obviously!"
R_ADVICE = "If I were you, I would go to the internet and type exactly what you wrote there!"

def unknown():
    response = ['Could you please re-phrase that?', 
                "...", 
               #"Sounds about right",
               #"What does that mean?",
               "Please ask questions related to UMBC Storage.", 
               "You can ask questions like size of a small storage unit or price of a large unit."][random.randrange(4)]
    return response