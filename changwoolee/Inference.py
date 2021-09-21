import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
from transformers import AdamW
from transformers import BertModel
from transformers.optimization import get_cosine_schedule_with_warmup
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from DataLoader import InferenceSet
def inference(input_sentence,transform,model):
  scoring = {"index":[], "score":[]}

  print("constructing....")
  inference_set = InferenceSet(cnt, df_train_du.iloc[:1000], tok, max_len, True, False)
  inference_dataloader = torch.utils.data.DataLoader(inference_set, batch_size=batch_size)
  print("complete!")

  print("inference...")
  model.eval()

  for batch_id, (context_input, response_input) in enumerate(tqdm_notebook(inference_dataloader)):
      token_ids_cnt, valid_length_cnt, segment_ids_cnt = context_input[0]
      token_ids_rsp, valid_length_rsp, segment_ids_rsp = response_input
      token_ids_rsp = token_ids_rsp.long().to(device)
      segment_ids_rsp = segment_ids_rsp.long().to(device)
      valid_length_rsp= valid_length_rsp
      token_ids_cnt = token_ids_cnt.long().to(device)
      segment_ids_cnt = segment_ids_cnt.long().to(device)
      valid_length_cnt = valid_length_cnt

      out = model(token_ids_cnt, segment_ids_cnt, valid_length_cnt,
                  token_ids_rsp, segment_ids_rsp, valid_length_rsp)
      scoring['index'].append(batch_id)
      scoring['score'].append(out)
      torch.cuda.empty_cache()
  return scoring

if __name__=="main":
    # set parameters
    max_len = 64
    batch_size=10

    # load model
    context_bert = BertModel.from_pretrained("your/pretrained_dir")
    response_bert = BertModel.from_pretrained("your/pretrained_dir")

    # load tokenizer
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    sentence_transform = nlp.data.BERTSentenceTransform(tok, max_seq_length=max_len, pad=True, pair=False)

    # load data
    reference_data = ("your data")
    input_text = input("talk to chatbot") # input text
    inference_set = InferenceSet(input_text, reference_data, tok, max_len, True, False)
    inference_dataloader = torch.utils.data.DataLoader(inference_set, batch_size=batch_size)
    model = BiEncoder(context_bert, response_bert, dr_rate=0.5).to(device) # from
    
    result = inference(input_text, sentence_transform, model)
    # 추후 결과는 top k로 제공



    