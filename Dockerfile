FROM pytorch
COPY requirements.txt .
RUN apt update && apt install -y git && pip install -r requirements.txt
# docker run -dit --name=maml --shm-size=2gb --device nvidia.com/gpu=all --network=host -v ~/few-shot-hypernets-public:/workspace gmum-fewshot bash
