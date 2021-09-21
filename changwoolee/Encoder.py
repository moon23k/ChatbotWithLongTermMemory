import torch
from torch import nn

class BiEncoder(nn.Module):
    def __init__(self, context_bert, response_bert, num_classes=1, dr_rate=None, params=None):
        super(BiEncoder, self).__init__()
        self.context_bert = context_bert
        self.response_bert = response_bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(self.context_bert.pooler.dense.out_features + self.response_bert.pooler.dense.out_features, 
                                    num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
        self.sigmoid = nn.Sigmoid()
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids_cnt, segment_ids_cnt, valid_length_cnt,
                      token_ids_rsp, segment_ids_rsp, valid_length_rsp):
        attention_mask_cnt = self.gen_attention_mask(token_ids_cnt, valid_length_cnt)
        attention_mask_rsp = self.gen_attention_mask(token_ids_rsp, valid_length_rsp)
        
        _, pooler_cnt =  self.context_bert(input_ids = token_ids_cnt, 
                                           token_type_ids = segment_ids_cnt.long(), 
                                           attention_mask = attention_mask_cnt.float().to(token_ids_cnt.device))
        _, pooler_rsp = self.response_bert(input_ids = token_ids_rsp, 
                                           token_type_ids = segment_ids_rsp.long(), 
                                           attention_mask = attention_mask_rsp.float().to(token_ids_rsp.device))
        output = torch.cat((pooler_cnt,pooler_rsp),1)
        # output = torch.flatten(output)
        if self.dr_rate:
            output = self.dropout(output)
        output = self.classifier(output)
        output = self.sigmoid(output)
        return output
