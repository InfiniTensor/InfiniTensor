.PHONY : build clean format install-python test-cpp test-onnx install-test clean-test

TYPE ?= release
CUDA ?= OFF
BANG ?= OFF
INTELCPU ?= off
BACKTRACE ?= ON
TEST ?= ON
FORMAT_ORIGIN ?=

CMAKE_OPT = -DCMAKE_BUILD_TYPE=$(TYPE)
CMAKE_OPT += -DUSE_CUDA=$(CUDA)
CMAKE_OPT += -DUSE_BANG=$(BANG)
CMAKE_OPT += -DUSE_BACKTRACE=$(BACKTRACE)
CMAKE_OPT += -DBUILD_TEST=$(TEST)

ifeq ($(INTELCPU), ON)
	CMAKE_OPT += -DUSE_INTELCPU=ON -DCMAKE_CXX_COMPILER=dpcpp
endif

build:
	mkdir -p build/$(TYPE)
	cd build/$(TYPE) && cmake $(CMAKE_OPT) ../.. && make -j8

clean: clean-test
	rm -rf build

format:
	@python3 scripts/format.py $(FORMAT_ORIGIN)

install-python: build
	cp build/$(TYPE)/backend*.so pyinfinitensor/src/pyinfinitensor
	pip install pyinfinitensor/

test-cpp:
	@echo
	cd build/$(TYPE) && make test

test-onnx:
	@echo
	python3 pyinfinitensor/tests/test_onnx.py

build-test:
	cd InfiniTest/src/InfiniTest && protoc --python_out=./ operator.proto
ifeq ($(CUDA), ON)
	@echo "CUDA_HOME: ${CUDA_HOME}"
	@c++ -O3 -Wall -shared -std=c++11 -fPIC $$(python3 -m pybind11 --includes) -I${CUDA_HOME}/include -DCUDA ./InfiniTest/Device/device.cpp -o ./InfiniTest/src/InfiniTest/device$$(python3-config --extension-suffix) -L$(CUDA_HOME)/lib64 -lcudart
endif
ifeq ($(BANG), ON)
	@echo "NEUWARE_HOME: ${NEUWARE_HOME}"
	@c++ -O3 -Wall -shared -std=c++11 -fPIC $$(python3 -m pybind11 --includes) -I${NEUWARE_HOME}/include -DBANG ./InfiniTest/Device/device.cpp -o ./InfiniTest/src/InfiniTest/device$$(python3-config --extension-suffix) -L$(NEUWARE_HOME)/lib64 -lcnrt
endif

install-test: build-test
	pip install InfiniTest/

clean-test:
	rm -rf InfiniTest/src/InfiniTest.egg-info
	rm -rf InfiniTest/src/InfiniTest/operator_pb2.py
	rm -rf InfiniTest/src/InfiniTest/device$$(python3-config --extension-suffix)
