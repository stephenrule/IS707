#!/bin/bash

echo 'Starting to Deploy...'
ssh ec2-user@ec2-34-204-48-212.compute-1.amazonaws.com bash /home/ec2-user/IS707/flask/aws_commands.sh

echo 'Deployment completed successfully'
