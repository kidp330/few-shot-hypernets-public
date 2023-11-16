CONTAINER_CMD=$2
if [ "$3"="--resume" ]; then
	CONTAINER_CMD+=' --resume'
fi

docker run -d --rm \
	--name=$1 --network=host \
	-v $PWD:/workspace/ --env-file=neptune.env \
	--shm-size=2gb --device nvidia.com/gpu=all \
	--log-driver=journald -- \
       	gmum-fewshot $CONTAINER_CMD
