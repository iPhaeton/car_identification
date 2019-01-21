scp -o "StrictHostKeyChecking no" docker-stack.yml root@$DO_IP:docker-stack.yml
ssh -o "StrictHostKeyChecking no" root@$DO_IP "docker stack deploy -c ~/docker-stack.yml --with-registry-auth car-identification"
ssh -o "StrictHostKeyChecking no" root@$DO_IP "yes y | docker system prune"