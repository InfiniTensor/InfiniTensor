.PHONY : build clean format install-python test-cpp test-onnx test-api

PYTHON ?= python3
JOBS ?= 8
BUILD_DIR ?= build/$(TYPE)
PIP_INSTALL_ARGS ?=
TYPE ?= Release
INFINI ?= ON
ATEN ?= ON
INFINIOPS_ROOT ?=
INFINIRT_ROOT ?=
INFINIOPS_CXX11_ABI ?=
INFINIOPS_PROVIDER_LIBRARY_DIRS ?=
BACKTRACE ?= OFF
TEST ?= ON
PROVIDER_MODULES ?=
CHECK_BACKEND ?=
CHECK_DEVICE ?= 0
DIST ?= OFF
NNET ?= OFF
FORMAT_ORIGIN ?=
# Docker build options
DOCKER_NAME ?= infinitensor
DOCKER_IMAGE_NAME ?= infinitensor
DOCKER_FILE ?= infinitensor_ubuntu_22.04.dockerfile
DOCKER_RUN_OPTION ?=

CMAKE_OPT = -DCMAKE_BUILD_TYPE=$(TYPE)
CMAKE_OPT += "-DPython_EXECUTABLE=$(PYTHON)"
CMAKE_OPT += -DUSE_INFINIOPS_KERNELS=$(INFINI)
CMAKE_OPT += -DUSE_INFINIOPS_ATEN_KERNELS=$(ATEN)
CMAKE_OPT += -DINFINIOPS_ROOT=$(INFINIOPS_ROOT)
CMAKE_OPT += -DINFINIRT_ROOT=$(INFINIRT_ROOT)
CMAKE_OPT += -DINFINIOPS_CXX11_ABI=$(INFINIOPS_CXX11_ABI)
CMAKE_OPT += "-DINFINIOPS_PROVIDER_LIBRARY_DIRS=$(INFINIOPS_PROVIDER_LIBRARY_DIRS)"
CMAKE_OPT += -DUSE_BACKTRACE=$(BACKTRACE)
CMAKE_OPT += -DBUILD_TEST=$(TEST)
CMAKE_OPT += -DBUILD_DIST=$(DIST)
CMAKE_OPT += -DBUILD_NNET=$(NNET)

build:
	mkdir -p $(BUILD_DIR)
	cmake -S . -B $(BUILD_DIR) $(CMAKE_OPT)
	cmake --build $(BUILD_DIR) --parallel $(JOBS)

clean:
	rm -rf build

format:
	@$(PYTHON) scripts/format.py $(FORMAT_ORIGIN)

install-python: build
	install -m 0755 $(BUILD_DIR)/backend*.so pyinfinitensor/src/pyinfinitensor/
	install -m 0755 $(BUILD_DIR)/libInfiniTensor.so pyinfinitensor/src/pyinfinitensor/
	$(PYTHON) -m pip install $(PIP_INSTALL_ARGS) -e pyinfinitensor/
	INFINIOPS_PROVIDER_MODULES="$(PROVIDER_MODULES)" $(PYTHON) scripts/check_install.py \
		$(if $(strip $(PROVIDER_MODULES)),--provider-modules "$(PROVIDER_MODULES)",) \
		$(if $(strip $(CHECK_BACKEND)),--backend "$(CHECK_BACKEND)" --device "$(CHECK_DEVICE)",)

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
