run: build
	./simple_nn
build:
	gcc -Wall -W -O3 -march=native -o simple_nn main.c nnLayer.c nnNetwork.c -lm

clean:
	del simple_nn.exe