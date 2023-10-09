

# 使用nvidia-docker构建GPU镜像
docker-gpu:
	docker build -t my_gpu_image -f Dockerfile.gpu .

# 使用Docker运行GPU容器
run-gpu:
	docker run -it --runtime=nvidia --rm --network host --ipc=host --name my_gpu_container -v $(PWD):/app my_gpu_image python test2.py

