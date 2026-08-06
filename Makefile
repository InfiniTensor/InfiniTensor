.PHONY : build clean format install-python test-cpp test-onnx test-api

TYPE ?= Release
INFINI ?= ON
ATEN ?= ON
INFINIOPS_ROOT ?=
INFINIRT_ROOT ?=
INFINIOPS_CXX11_ABI ?=
BACKTRACE ?= ON
TEST ?= ON
DIST ?= OFF
INFINICCL ?= $(DIST)
INFINICCL_ROOT ?=
NNET ?= OFF
FORMAT_ORIGIN ?=
# Docker build options
DOCKER_NAME ?= infinitensor
DOCKER_IMAGE_NAME ?= infinitensor
DOCKER_FILE ?= infinitensor_ubuntu_22.04.dockerfile
DOCKER_RUN_OPTION ?=

CMAKE_OPT = -DCMAKE_BUILD_TYPE=$(TYPE)
CMAKE_OPT += -DUSE_INFINIOPS_KERNELS=$(INFINI)
CMAKE_OPT += -DUSE_INFINIOPS_ATEN_KERNELS=$(ATEN)
CMAKE_OPT += -DINFINIOPS_ROOT=$(INFINIOPS_ROOT)
CMAKE_OPT += -DINFINIRT_ROOT=$(INFINIRT_ROOT)
CMAKE_OPT += -DINFINIOPS_CXX11_ABI=$(INFINIOPS_CXX11_ABI)
CMAKE_OPT += -DUSE_BACKTRACE=$(BACKTRACE)
CMAKE_OPT += -DBUILD_TEST=$(TEST)
CMAKE_OPT += -DBUILD_DIST=$(DIST)
CMAKE_OPT += -DUSE_INFINICCL=$(INFINICCL)
CMAKE_OPT += -DINFINICCL_ROOT=$(INFINICCL_ROOT)
CMAKE_OPT += -DBUILD_NNET=$(NNET)

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
	python3 pyinfinitensor/tests/test_onnxstub.py
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
