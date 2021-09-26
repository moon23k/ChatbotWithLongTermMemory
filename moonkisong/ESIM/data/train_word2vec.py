import os
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


if __name__ == '__main__':
    input_file = 'ubuntu_data/train.txt'
    output_file = 'embedding_w2v_d300.txt'
    sentences = []
    for line in open(input_file):
        texts = line.replace("\n", "").split("\t")[1:]
        for uter in texts:
            sentences.append(uter.split())

    model = Word2Vec(sentences, vector_size=300, window=5, min_count=5, sg=1,
                     workers=multiprocessing.cpu_count())

    model.wv.save_word2vec_format(output_file, binary=False)