.PHONY: all clean run
CC=g++
CFLAGS=-std=c++11 -O2
LDFLAGS=-lm
TARGET=train test
TRAIN_ITER=100
INIT_PATH=model_init.txt

all: $(TARGET)

train: src/train.cpp
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) -Iinc

test: src/test.cpp
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) -Iinc

run: train
	./train $(TRAIN_ITER) $(INIT_PATH) data/train_seq_01.txt model_01.txt
	./train $(TRAIN_ITER) $(INIT_PATH) data/train_seq_02.txt model_02.txt
	./train $(TRAIN_ITER) $(INIT_PATH) data/train_seq_03.txt model_03.txt
	./train $(TRAIN_ITER) $(INIT_PATH) data/train_seq_04.txt model_04.txt
	./train $(TRAIN_ITER) $(INIT_PATH) data/train_seq_05.txt model_05.txt

clean:
	rm -f $(TARGET)

