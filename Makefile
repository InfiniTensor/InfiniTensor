.PHONY : build clean install-python test-cpp test-onnx

TYPE ?= release
CUDA ?= OFF
BANG ?= OFF
INTELCPU ?= off
BACKTRACE ?= ON
TEST ?= OFF
BENCH ?= ON

CMAKE_OPT = -DCMAKE_BUILD_TYPE=$(TYPE)
CMAKE_OPT += -DUSE_CUDA=$(CUDA)
CMAKE_OPT += -DUSE_BANG=$(BANG)
CMAKE_OPT += -DUSE_BACKTRACE=$(BACKTRACE)
CMAKE_OPT += -DBUILD_TEST=$(TEST)
CMAKE_OPT += -BENCH=$(BENCH)

ifeq ($(INTELCPU), ON)
	CMAKE_OPT += -DUSE_INTELCPU=ON -DCMAKE_CXX_COMPILER=dpcpp
endif

build:
	mkdir -p build/$(TYPE)
	cd build/$(TYPE) && cmake $(CMAKE_OPT) ../.. && make -j8

clean:
	rm -rf build

install-python: build
	cp build/$(TYPE)/backend*.so pyinfinitensor/src/pyinfinitensor
	pip install pyinfinitensor/

test-cpp:
	@echo
	cd build/$(TYPE) && make test

test-onnx:
	@echo
	python3 pyinfinitensor/tests/test_onnx.py
