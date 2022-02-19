# IS707
IS 707 Course


## Troubleshoot/Errors:

1. OS Error: [E941] Can't find model 'en'. nlp = spacy.load("en_core_web_sm")

- Link: https://stackoverflow.com/questions/66087475/chatterbot-error-oserror-e941-cant-find-model-en
- navigate to (Linux) chatbot/lib64/python3.6/site-packages/chatterbot/tagging.py and add the below lines in place of: self.nlp = spacy.load(self.language.ISO_639_1.lower()) 
- if self.language.ISO_639_1.lower() == 'en':
-   self.nlp = spacy.load('en_core_web_sm')
- else:
-   self.nlp = spacy.load(self.language.ISO_639_1.lower()) 