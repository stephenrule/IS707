#sudo yum update -y
#sudo yum install git -y
#git clone https://github.com/stephenrule/IS707.git
#cd IS707
#sudo rm -rf Archive django Storage\ Facility.png
#cd flask

# Assumption: Above is already complete: bash aws_commands.py
sudo yum install gcc openssl-devel bzip2-devel libffi-devel -y
cd /opt
sudo wget https://www.python.org/ftp/python/3.9.10/Python-3.9.10.tgz
sudo tar xvf Python-3.9.10.tgz
cd Python-*/
sudo ./configure --enable-optimizations
sudo make altinstall
python3.9 --version
pip3.9 --version
sudo rm -f /opt/Python-3.9.10.tgz
/usr/local/bin/python3.9 -m pip install --upgrade pip
pip3.9 install awscli --user
cd ~/IS707/flask
sudo yum install gcc-c++ python3-devel -y
sudo yum groupinstall "Development Tools" -y
# no-cache-dir was doen to stop AWS from killing the process due to low memory
## Need to comment out the last two lines for allennlp and modules.  Need other things to get installed
# Might need scikit-learn==0.20.3 
pip3.9 install -r requirements.txt --no-cache-dir
pip3.9 install allennlp --no-cache-dir
pip3.9 install allennlp_models --no-cache-dir
sudo yum install python3-tkinter -y

## Note: I had to comment the line about tkinter from routes.py because there is no gui to use that command. Not sure what the import is for yet.

# Get sqlite3
cd /opt/Python-3.9.10
sudo yum install sqlite-devel -y
sudo yum install xz-devel -y
sudo ./configure
sudo make altinstall

# Build DB 1
cd ~/IS707/flask
rm -rf migrations app.db chat.db
flask db init
flask db migrate -m "init"
flask db upgrade

# Build DB 2
python3.9 standup.py

# Firewall - AWS Web Interface opens the port for us
#sudo firewall-cmd --permanent --add-service=http
#sudo firewall-cmd --reload
#sudo iptables -L
#sudo iptables -I INPUT -i eth0 -p tcp --dport 80 -m comment --comment "# Flask WebApp #" -j ACCEPT_RETRY_DELAY
#sudo iptables -L -n
#sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 5000


# Run Flask
sudo yum install screen -y
screen
flask run --host 0.0.0.0
## Command: Ctrl + A, D
## Command: screen -ls
## Command: screen -r

## Note: 80 is denied on ec2 but 5000 is save. Make sure to open 5000 from 
