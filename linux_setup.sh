#!/bin/bash

#### To run this script
#bash ./linux_setup.sh

#### Assumptions
# 1. You are running a RHEL based system
# 2. You are in the IS707 directory when running this script
# 3. You already have git installed. Required to clone this repository
# 4. Python is installed. Version should be between 3.4 - 3.8

#### Clone Repository
# Note: You should already have this file by the below command
# git clone https://github.com/stephenrule/IS707.git

#### Create Python Virtual Environment - Should be done with the repository
#python3 -m venv chatbot

#### Enter Python Virtual Environment (chatbot)
source chatbot/bin/activate
# Note: You will see (chatbot) in your CLI...

#### Update and install dependencies
python3 -m pip install --upgrade pip

################ Jupyter Notebook Specific - OPTIONAL ###################
#### Install Jupyter Notebook (Don't do this on production server)
#pip3 install jupyter

#### Open ports for Jupyter Notebook (Don't do this on production server)
#firewall-cmd --permanent --add-port=8888/tcp
#firewall-cmd --permanent --add-service={http,https}
#firewall-cmd --reload

#### Start Jupyter Notebook (Don't do this on production server)
#jupyter notebook
##########################################################################

#### Install project dependencies
pip3 install chatterbot
pip3 install chatterbot_corpus
pip3 install spacy
python3 -m spacy download en_core_web_sm

#### Start project code
#python3 ./linux_chatbot.py