#!/bin/bash

echo 'Starting to Deploy...'
ssh ${{ secrets.USERNAME }}@${{ secrets.HOST_DNS }}

cd /home/ec2-user/IS707/flask
bash aws_commands.sh

echo 'Deployment completed successfully'
