run: build
	./simple_nn
build:
	gcc -o simple_nn main.c nnLayer.c nnNetwork.c -lm

clean:
	del simple_nn.exe