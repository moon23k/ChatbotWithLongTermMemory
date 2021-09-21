from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import time
import random
import numpy as np


class BERTDataset(Dataset):
    def __init__(self, dataset, sampling_size, bert_tokenizer, max_len, pad, pair):
        '''
        형식은 [context_input, response_input, lbael]
        구조는 [총데이터/샘플링사이즈, 샘플링사이즈^2, 3]
        '''
        self.transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        self.sampling_size = sampling_size
        self.dataset = dataset
        self.dataset_index = self.negative_sampling_indexing(self.sampling_size,self.dataset)

        self.transformed_sentences = self.transformation(self.dataset.Q)
        self.transformed_responses = self.transformation(self.dataset.A)
        self.sentences = self.get_utter(self.transformed_sentences,self.dataset_index[:,:,0])
        self.responses = self.get_utter(self.transformed_responses,self.dataset_index[:,:,1])
        self.labels = self.dataset_index[:,:,2].flatten()

    def get_utter(self, transformed_data, dataset_index):
        '''
        변환된 데이터에 대해서 utterance 리스트로 반환
        '''
        return [transformed_data[ins] for row in dataset_index for ins in row]

    def negative_sampling_indexing(self, num, df):
        '''
        네거티브 샘플링할 데이터를 인덱싱하는 과정
        '''
        container=[]
        total_data = list(df.index)
        random.shuffle(total_data)
        eye_matrix = np.eye(num)
        start = time.time()
        for i in range(int(len(total_data)/num)):
            inner_container=[]
            sampled = total_data[i*num:(i+1)*num]
            for s1,r in zip(sampled,eye_matrix):
                for s2,v in zip(sampled,r):
                    inner_container.append([s1,s2,v])
            end = time.time()
            container.append(inner_container)
        print(f"shuffle and batch : {round(time.time()-start,3)}s")
        return np.array(container,dtype='int32')

    def transformation(self,data_to_transform):
        '''
        토크나이징
        '''
        return [self.transform([d]) for d in data_to_transform]

    def re_initializing(self):
        '''
        훈련시 네거티브샘플링을 랜덤하게 뽑기위한 초기화
        '''
        start = time.time()
        self.dataset_index = self.negative_sampling_indexing(self.sampling_size,self.dataset)
        self.sentences = self.get_utter(self.transformed_sentences,self.dataset_index[:,:,0])
        self.responses = self.get_utter(self.transformed_responses,self.dataset_index[:,:,1])
        print(f"reinitialized! {round(time.time()-start,3)}s")

    def __getitem__(self, i):
        '''
        데이터로더에 뽑히는 형태
        '''
        return (self.sentences[i], self.responses[i], (self.labels[i]))

    def __len__(self):
        return (len(self.labels))

class InferenceSet(Dataset):
    def __init__(self, input_sentence, dataset, bert_tokenizer, max_len, pad, pair):
      self.transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
      self.dataset = dataset
      self.input_sentence = input_sentence
      self.transformed_sentences = self.transformation(self.dataset)
      self.transformed_input = self.transformation([self.input_sentence])

    def transformation(self,data_to_transform):
      print(f"transforming...")
      return [self.transform([d]) for d in data_to_transform]


    def __getitem__(self, i):
        return (self.transformed_input, self.transformed_sentences[i])

    def __len__(self):
        return (len(self.transformed_sentences))



