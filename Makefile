.PHONY : build clean install-python test-cpp test-onnx

TYPE ?= release

build:
	mkdir -p build/$(TYPE)
	cd build/$(TYPE) && cmake -DCMAKE_BUILD_TYPE=$(TYPE) ../.. && make -j8

clean:
	rm -rf build

install-python: build
	cp build/$(TYPE)/backend*.so pyinfinitensor/src/pyinfinitensor
	pip install pyinfinitensor/

test-cpp: build
	@echo
	cd build/$(TYPE) && make test

test-onnx:
	@echo
	python3 pyinfinitensor/tests/test_onnx.py
