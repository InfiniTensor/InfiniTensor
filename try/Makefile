.PHONY : build clean install-python test-cpp test-onnx

TYPE ?= release

build:
	mkdir -p build/$(TYPE)
	cd build/$(TYPE) && cmake $(CMAKE_OPT) ../.. && make -j8

test: build
	build/$(TYPE)/test
	@echo Done

clean:
	rm -rf build
