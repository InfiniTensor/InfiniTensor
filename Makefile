.PHONY : build clean format install-python test-cpp test-onnx

TYPE ?= Debug
CUDA ?= OFF
BANG ?= OFF
KUNLUN ?= OFF
ASCEND ?= OFF
INTELCPU ?= off
BACKTRACE ?= ON
TEST ?= ON
DIST ?= OFF
NNET ?= OFF
DIST ?= OFF
FORMAT_ORIGIN ?=
# Docker build options
DOCKER_NAME ?= infinitensor
DOCKER_IMAGE_NAME ?= infinitensor
DOCKER_FILE ?= infinitensor_ubuntu_22.04.dockerfile
DOCKER_RUN_OPTION ?=

# CUDA option.
ifeq ($(CUDA), ON)
	DOCKER_IMAGE_NAME = infinitensor_cuda
	DOCKER_NAME = infinitensor_cuda
	DOCKER_FILE = infinitensor_ubuntu_22.04_CUDA.dockerfile
	DOCKER_RUN_OPTION += --gpus all -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v `pwd`:`pwd` -w `pwd`
endif

CMAKE_OPT = -DCMAKE_BUILD_TYPE=$(TYPE)
CMAKE_OPT += -DUSE_CUDA=$(CUDA)
CMAKE_OPT += -DUSE_BANG=$(BANG)
CMAKE_OPT += -DUSE_KUNLUN=$(KUNLUN)
CMAKE_OPT += -DUSE_ASCEND=$(ASCEND)
CMAKE_OPT += -DUSE_BACKTRACE=$(BACKTRACE)
CMAKE_OPT += -DBUILD_TEST=$(TEST)
CMAKE_OPT += -DBUILD_DIST=$(DIST)
CMAKE_OPT += -DBUILD_NNET=$(NNET)

ifeq ($(INTELCPU), ON)
	CMAKE_OPT += -DUSE_INTELCPU=ON -DCMAKE_CXX_COMPILER=dpcpp
endif

build:
	mkdir -p build/$(TYPE)
	cd build/$(TYPE) && cmake $(CMAKE_OPT) ../.. && make -j8

clean:
	rm -rf build

format:
	@python3 scripts/format.py $(FORMAT_ORIGIN)

install-python: build
	cp build/$(TYPE)/backend*.so pyinfinitensor/src/pyinfinitensor
	pip install -e pyinfinitensor/

test-cpp:
	@echo
	cd build/$(TYPE) && make test

test-onnx:
	@echo
	python3 pyinfinitensor/tests/test_onnx.py

test-api:
	@echo
	python3 pyinfinitensor/tests/test_api.py

docker-build:
	docker build -f scripts/dockerfile/$(DOCKER_FILE) -t $(DOCKER_NAME) .

docker-run:
	docker run -t --name $(DOCKER_IMAGE_NAME) -d $(DOCKER_NAME) $(DOCKER_RUN_OPTION)

docker-start:
	docker start $(DOCKER_IMAGE_NAME)

docker-exec:
	docker exec -it $(DOCKER_IMAGE_NAME) bash
