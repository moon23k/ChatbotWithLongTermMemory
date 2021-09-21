''' 설치패키지
!pip install mxnet
!pip install gluonnlp
!pip install transformers
!pip install sentencepiece
!pip install git+https://git@github.com/SKTBrain/KoBERT.git@master
'''
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from DataLoader import BERTDataset
from Encdoer import BiEncoder



def calc_accuracy(X,Y):
    '''
    accuracy metric, 이 훈련에서 큰 의미는 없을 것으로 예상되나 모니터링을 위한용도로 사용
    '''
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc



if __name__=="main":
    # load data
    train_data = "your data" # pandas dataframe이 아닌경우 일부 코드 수정필요

    # set device
    device = torch.device("cuda:0")
    print(f"CUDA available {torch.cuda.is_available()}")

    # load pretrained bertmodel(KoBERT)
    context_bert, vocab = get_pytorch_kobert_model()
    response_bert, _ = get_pytorch_kobert_model()
    model = BiEncoder(context_bert, response_bert, dr_rate=0.5).to(device)

    # load tokenizer
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    # set parameters
    max_len = 64
    warmup_ratio = 0.1
    num_epochs = 5
    max_grad_norm = 1
    log_interval = 100
    batch_size=10
    learning_rate =  0.01

    # load dataset
    train_dataset = BERTDataset(train_data, 10, tok, max_len, True, False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    # set optimizer & schedule
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = nn.BCELoss()  # Bianry Cross Entropy
    t_total = len(train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    # trainstep
    for e in range(num_epochs):
        train_dataset.re_initializing() # negative sample 재배치
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size) 
        train_acc = 0.0
        test_acc = 0.0

        model.train()
        for batch_id, (context_input, response_input, label) in enumerate(tqdm_notebook(train_dataloader)):
            # get input
            token_ids_cnt, valid_length_cnt, segment_ids_cnt = context_input
            token_ids_rsp, valid_length_rsp, segment_ids_rsp = response_input

            optimizer.zero_grad()

            # to device
            token_ids_cnt = token_ids_cnt.long().to(device)
            segment_ids_cnt = segment_ids_cnt.long().to(device)
            valid_length_cnt = valid_length_cnt
            token_ids_rsp = token_ids_rsp.long().to(device)
            segment_ids_rsp = segment_ids_rsp.long().to(device)
            valid_length_rsp= valid_length_rsp
            label = label.float().to(device)
            
            out = model(token_ids_cnt, segment_ids_cnt, valid_length_cnt, token_ids_rsp, segment_ids_rsp, valid_length_rsp) # output
            
            # get loss & backprop
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            # monitoring
            train_acc += calc_accuracy(out, label)
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
        print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))


