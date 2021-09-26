import os
from collections import OrderedDict
import numpy


def build_dictionary(org_path, dst_path, is_lowercase=False):
    token_to_freqs = OrderedDict()
    count = 0
    with open(org_path, 'r') as f:
        for line in f:
            if is_lowercase:
                line = line.lower()
            arr = line.strip().split('\t')
            assert len(arr) == 3

            for text in arr[1:]:
                tokens = text.split(' ')
                for w in tokens:
                    if w in token_to_freqs:
                        token_to_freqs[w] += 1
                    else:
                        token_to_freqs[w] = 1
            if count % 10000 == 0:
                print(count)
            count += 1

    tokens = list(token_to_freqs.keys())
    freqs = token_to_freqs.values()

    sorted_idx = numpy.argsort(freqs)
    sorted_tokens = [tokens[i] for i in sorted_idx[::-1]]

    token_to_idx = OrderedDict()
    token_to_idx['_PAD_'] = 0  # default, padding
    token_to_idx['_UNK_'] = 1  # out-of-vocabulary
    token_to_idx['_BOS_'] = 2  # begin of sentence token
    token_to_idx['_EOS_'] = 3  # end of sentence token

    for i, t in enumerate(sorted_tokens):
        token_to_idx[t] = i + 4

    with open(dst_path, 'w') as f:
        for t in token_to_idx.keys():
            f.write(t + '\n')

    print('Dict size', len(token_to_idx))


def concat_context(org_file, dst_file):
    with open(org_file, 'r') as fi:
        with open(dst_file, 'w') as fo:
            for idx, line in enumerate(fi):
                arr = line.strip().split('\t')
                label = arr[0]
                context = ' __eou__ __eot__ '.join(
                    arr[1:-1]) + ' __eou__ __eot__ '
                response = arr[-1]
                fo.write('\t'.join([label, context, response]) + '\n')


def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


if __name__ == '__main__':
    base_dir = '/content/drive/MyDrive/Data/ESIM_data/'
    org_dir = os.path.join(base_dir, 'ubuntu_data/')
    dst_dir = os.path.join(base_dir, 'ubuntu_data_concat/')
    make_dirs([dst_dir])

    print("***** Concatenate Context ***** ")
    concat_context(os.path.join(org_dir, 'test.txt'),
                   os.path.join(dst_dir, 'test.txt'))
    concat_context(os.path.join(org_dir, 'valid.txt'),
                   os.path.join(dst_dir, 'valid.txt'))
    concat_context(os.path.join(org_dir, 'train.txt'),
                   os.path.join(dst_dir, 'train.txt'))

    print("***** Obtain Dictionary ***** ")
    build_dictionary(os.path.join(dst_dir, 'train.txt'),
                     os.path.join(dst_dir, 'vocab.txt'),
                     is_lowercase=False)