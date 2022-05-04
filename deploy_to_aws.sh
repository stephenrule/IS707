#!/bin/bash

echo 'Starting to Deploy...'
ssh ${{ secrets.USERNAME }}@${{ secrets.HOST_DNS }}

cd ${{ secrets.TARGET }}/flask
bash aws_commands.sh

echo 'Deployment completed successfully'
