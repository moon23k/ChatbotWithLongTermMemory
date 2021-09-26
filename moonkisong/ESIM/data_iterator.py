import numpy
import random
import math


class TextIterator:
    def __init__(self, input_file, token_to_idx,
                 batch_size=128, vocab_size=-1, shuffle=True, factor=20):
        self.input_file = open(input_file, 'r')
        self.token_to_idx = token_to_idx
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.shuffle = shuffle
        self.end_of_data = False
        self.instance_buffer = []
        self.max_buffer_size = batch_size * factor

    def __iter__(self):
        return self

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.input_file.seek(0)
            raise StopIteration

        instance = []

        if len(self.instance_buffer) == 0:
            for _ in range(self.max_buffer_size):
                line = self.input_file.readline()
                if line == "":
                    break
                arr = line.strip().split('\t')
                assert len(arr) == 3
                self.instance_buffer.append(
                    [arr[0], arr[1].split(' '), arr[2].split(' ')])

            if self.shuffle:
                length_list = []
                
                for ins in self.instance_buffer:
                    current_length = len(ins[1]) + len(ins[2])
                    length_list.append(current_length)

                length_array = numpy.array(length_list)
                length_idx = length_array.argsort()
                tindex = []
                small_index = range(int(math.ceil(len(length_idx) * 1. / self.batch_size)))
                random.shuffle(small_index)

                for i in small_index:
                    if (i + 1) * self.batch_size > len(length_idx):
                        tindex.extend(length_idx[i * self.batch_size:])
                    else:
                        tindex.extend(
                            length_idx[i * self.batch_size:(i + 1) * self.batch_size])

                _buf = [self.instance_buffer[i] for i in tindex]
                self.instance_buffer = _buf

        if len(self.instance_buffer) == 0:
            self.end_of_data = False
            self.input_file.seek(0)
            raise StopIteration

        try:
            while True:
                try:
                    current_instance = self.instance_buffer.pop(0)
                except IndexError:
                    break

                label = current_instance[0]
                sent1 = current_instance[1]
                sent2 = current_instance[2]

                sent1.insert(0, '_BOS_')
                sent1.append('_EOS_')
                sent1 = [self.token_to_idx[w]
                         if w in self.token_to_idx else 1 for w in sent1]
                if self.vocab_size > 0:
                    sent1 = [w if w < self.vocab_size else 1 for w in sent1]

                sent2.insert(0, '_BOS_')
                sent2.append('_EOS_')
                sent2 = [self.token_to_idx[w] if w in self.token_to_idx else 1
                         for w in sent2]
                if self.vocab_size > 0:
                    sent2 = [w if w < self.vocab_size else 1 for w in sent2]

                instance.append([label, sent1, sent2])

                if len(instance) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(instance) <= 0:
            self.end_of_data = False
            self.input_file.seek(0)
            raise StopIteration

        return instance