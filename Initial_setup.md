# Cowrie Honeypot Setup Instructions

This document outlines the manual steps and commands needed to set up the Cowrie honeypot on an Ubuntu EC2 instance, including system dependencies, configuration changes, and network setup.

---

## 1. Install System Dependencies

## Update package lists and install required packages:


sudo apt-get update
sudo apt-get install -y git python3 python3-venv libssl-dev libffi-dev build-essential iptables


git clone https://github.com/cowrie/cowrie.git
cd cowrie

cp etc/cowrie.cfg.dist etc/cowrie.cfg


##Change SSH port
sudo nano /etc/ssh/sshd_config

## Find line "#Port 22" and change it to "Port 2223"; save and exit editor
sudo systemctl restart sshd


##IP redirection (SSH traffic reroute)

sudo iptables -t nat -A PREROUTING -p tcp --dport 22 -j REDIRECT --to-port 2222
sudo iptables -t nat -A OUTPUT -p tcp --dport 22 -j REDIRECT --to-port 2222


##Adjust Server Inbound settings (Eg. AWS EC2 Inbound settings)

## Start Cowrie

cd ~/cowrie
python3 -m venv cowrie-env
source cowrie-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
bin/cowrie start
