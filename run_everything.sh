#!/bin/bash
source /home/ubuntu/cowrie/cowrie-env/bin/activate

# Combine JSON logs
python3 /home/ubuntu/cowrie/var/log/AppendToFullList_FINAL.py

# Parse combined JSON and generate CSV, then upload
python3 /home/ubuntu/cowrie/var/log/cowrie/parse_cowrie_logs.py
aws s3 cp /home/ubuntu/cowrie/var/log/cowrie/events_new.csv s3://honeyai-cowrie-logs/events_new.csv

