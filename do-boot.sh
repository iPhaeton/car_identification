#!/bin/sh
# Before running, run 'chmod +x digital-ocean-boot.sh' on the server

echo -----------------apt-get update-----------------
apt-get update
echo -----------------apt-get install----------------
apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common -y
echo -----------------download docker----------------
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
apt-key fingerprint 0EBFCD88
echo --------------add docker repository-------------
add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
echo -----------------apt-get update-----------------
apt-get update
echo -----------------install docker-----------------
apt-get install docker-ce -y
echo ---------------apt-get autoremove---------------
apt-get autoremove -y

#docker swarm init

#docker node update --label-add tags=dev,noviopus hf0jiv332mky5l7m45edsdnkv
#docker login
#deploy stack, see deploy.sh
