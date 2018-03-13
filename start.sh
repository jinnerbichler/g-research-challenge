#!/usr/bin/env bash

INSTANCE_NAME="g-research-runner"

set -e    # Exits immediately if a command exits with a non-zero status.

if [ "$1" == "create" ]; then

    gcloud compute instances create ${INSTANCE_NAME} \
    --machine-type n1-standard-2 --zone us-east1-d \
    --accelerator type=nvidia-tesla-k80,count=1 \
    --boot-disk-size=200GB \
    --image-family ubuntu-1604-lts --image-project ubuntu-os-cloud \
    --maintenance-policy TERMINATE --restart-on-failure \
    --preemptible

    while [ -n "$(gcloud compute ssh ${INSTANCE_NAME} --command "echo ok" --zone us-east1-d 2>&1 > /dev/null)" ]; do
        echo "Waiting for VM to be available"
        sleep 1.0
    done

    # Sleep to be sure
    sleep 1.0

elif [ "$1" == "delete" ]; then

    gcloud compute instances delete --zone us-east1-d ${INSTANCE_NAME} --quiet

elif [ "$1" == "init" ]; then

    gcloud compute scp ./start.sh ./daemon.json ${INSTANCE_NAME}:~/ --zone us-east1-d
    gcloud compute ssh ${INSTANCE_NAME} --command="~/start.sh init-remote" --zone us-east1-d

elif [ "$1" == "update-data" ]; then

    gcloud compute scp ./data/ ${INSTANCE_NAME}:~/ --recurse --zone us-east1-d

elif [ "$1" == "init-remote" ]; then

    echo "Checking for CUDA and installing."
    # Check for CUDA and try to install.
    if ! dpkg-query -W cuda-9-1; then
      curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
      sudo dpkg -i ./cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
      sudo apt-get update
      sudo apt-get install cuda-9-1 -y --allow-unauthenticated
    fi
    sudo nvidia-smi -pm 0
    sudo nvidia-smi -ac 2505,875
    # On instances with NVIDIA® Tesla® K80 GPU: disable autoboost
    sudo nvidia-smi --auto-boost-default=DISABLED
    nvidia-smi

    # Installing Docker
    echo "Installing Docker and Docker Compose"
    sudo apt-get -y install apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    sudo apt-get update
    sudo apt-get -y install docker-ce
    docker --version
    sudo curl -L https://github.com/docker/compose/releases/download/1.19.0/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    docker-compose --version

    # Installing Nvidia Docker
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
    sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    sudo pkill -SIGHUP dockerd

    # https://github.com/NVIDIA/nvidia-docker/issues/262
    sudo nvidia-modprobe -u -c=0

    # Set default runtime
    sudo mv ~/daemon.json /etc/docker/daemon.json
    sudo service docker restart

    # Test nvidia-smi with the latest official CUDA image
    sudo docker run --rm nvidia/cuda nvidia-smi

elif [ "$1" == "deploy" ]; then

    gcloud compute scp  \
        ./docker-compose.yml \
        ./Dockerfile \
        main.py \
        genetic.py \
        requirements-gpu.txt \
        ${INSTANCE_NAME}:~/ --zone us-east1-d
    gcloud compute ssh ${INSTANCE_NAME} --command="sudo docker-compose -f ~/docker-compose.yml up -d --build --force-recreate" --zone us-east1-d
    gcloud compute ssh ${INSTANCE_NAME} --command="sudo docker-compose -f ~/docker-compose.yml logs -f" --zone us-east1-d

elif [ "$1" == "reset" ]; then

    gcloud compute ssh ${INSTANCE_NAME} --command="sudo docker-compose -f ~/docker-compose.yml down -v" --zone us-east1-d

elif [ "$1" == "logs" ]; then

    gcloud compute ssh ${INSTANCE_NAME} --command="sudo docker-compose -f ~/docker-compose.yml logs -f" --zone us-east1-d

elif [ "$1" == "submissions" ]; then

    gcloud compute scp ${INSTANCE_NAME}:~/submissions/* ./submissions/ --zone us-east1-d

fi
