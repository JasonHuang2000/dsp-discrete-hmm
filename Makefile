.PHONY: all clean run
CC=g++
CFLAGS=-std=c++11 -O2
LDFLAGS=-lm
TARGET=train 
TRAIN_ITER=100
INIT_PATH=model_init.txt
RESULT_PATH=result.txt

all: $(TARGET)

train: src/train.cpp
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) -Iinc

test: src/test.cpp
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) -Iinc

run: train
	./train $(TRAIN_ITER) $(INIT_PATH) data/train_seq_01.txt $(RESULT_PATH)

clean:
	rm -f $(TARGET)

