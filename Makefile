.PHONY: all test

all:
	mkdir -p build
	cd build && cmake ..
	cd build && make

test:
	mkdir -p build
	cd build && cmake ..
	cd build && make
	cd build && make test