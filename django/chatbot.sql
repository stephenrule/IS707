--ENTER MYSQL
--sudo mysql

--HELPFUL COMMANDS
SHOW DATABASES;

--CREATE DATABASE 
CREATE DATABASE djangodatabase;

--ADMINISTRATION
CREATE USER 'chatbot_user'@'localhost' IDENTIFIED WITH mysql_native_password BY 'password';
GRANT ALL ON djangodatabase.* to 'chatbot_user'@'localhost';
FLUSH PRIVILEGES;

--OPTIONS FILE
--sudo vim /etc/mysql/my.cnf
----[client]
----database = storage
----user = chatbot_user
----password = password
----default-character-set = utf8
--sudo systemctl daemon-reload
--sudo systemctl restart mysql




