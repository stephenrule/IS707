# Set base image (Host OS)
FROM python:3.9-alpine

# By default, listen on port 5000 (Docker Port)
EXPOSE 5000/tcp

# Set the working directory in the container
WORKDIR /app

# Copy project files to the working directory
COPY ./flask/ ./

# Install any dependencies
RUN apk add build-base   # Installs GCC for python package
RUN apk upgrade
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

# Build Flask Database
RUN flask db init
RUN flask db migrate -m "Init"
RUN flask db upgrade

# Build Python Database
RUN bash standup.py

# Specify the command to run on container start
CMD "flask run --host=0.0.0.0 --port=80"