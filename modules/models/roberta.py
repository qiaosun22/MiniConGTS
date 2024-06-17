import torch
from transformers import RobertaModel


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        self.bert = RobertaModel.from_pretrained(args.model_name_or_path)
        self.norm0 = torch.nn.LayerNorm(args.bert_feature_dim)
        self.drop_feature = torch.nn.Dropout(0.1)

        self.linear1 = torch.nn.Linear(args.bert_feature_dim*2, self.args.max_sequence_len)
        self.norm1 = torch.nn.LayerNorm(self.args.max_sequence_len)
        self.cls_linear = torch.nn.Linear(self.args.max_sequence_len, args.class_num)
        self.cls_linear1 = torch.nn.Linear(self.args.max_sequence_len, 2)
        self.gelu = torch.nn.GELU()

    def forward(self, tokens, masks):
        bert_feature, _ = self.bert(tokens, masks, return_dict=False)
        bert_feature = self.norm0(bert_feature)
        # bert_feature = self.drop_feature(bert_feature)  # 对 bert 后的特征表示做 dropout
        bert_feature = bert_feature.unsqueeze(2).expand([-1, -1, self.args.max_sequence_len, -1])
        bert_feature_T = bert_feature.transpose(1, 2)
        
        features = torch.cat([bert_feature, bert_feature_T], dim=3)
        
        
        sim_matrix = torch.nn.functional.cosine_similarity(bert_feature, bert_feature_T, dim=3)
        # print(sim_matrix.shape)
        sim_matrix = sim_matrix * masks

        # print(sim_matrix.shape, masks.shape)
        
        hidden = self.linear1(features)
        # hidden = self.drop_feature(hidden)
        hidden = self.norm1(hidden)
        hidden = self.gelu(hidden)

        logits = self.cls_linear(hidden)
        logits1 = self.cls_linear1(hidden)
        
        masks0 = masks.unsqueeze(3).expand([-1, -1, -1, self.args.class_num])#.shape
        masks1 = masks.unsqueeze(3).expand([-1, -1, -1, 2])#.shape
        
        logits = masks0 * logits
        logits1 = masks1 * logits1
        
        return logits, logits1, sim_matrix
    
