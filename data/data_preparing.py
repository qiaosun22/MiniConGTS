import math
import torch
import json
import os
import copy
import torch.nn.functional as F


class Instance(object):
    '''
    Re-organiztion for a single sentence;
    Input is in the formulation of: 
        {
        'id': '3547',
        'sentence': 'Taj Mahal offeres gret value and great food .',
        'triples': [
                    {'uid': '3547-0',
                    'target_tags': 'Taj\\O Mahal\\O offeres\\O gret\\O value\\B and\\O great\\O food\\O .\\O',
                    'opinion_tags': 'Taj\\O Mahal\\O offeres\\O gret\\O value\\O and\\O great\\B food\\O .\\O',
                    'sentiment': 'positive'},
                    {'uid': '3547-1',
                    'target_tags': 'Taj\\O Mahal\\O offeres\\O gret\\O value\\O and\\O great\\O food\\B .\\O',
                    'opinion_tags': 'Taj\\O Mahal\\O offeres\\O gret\\O value\\O and\\O great\\B food\\O .\\O',
                    'sentiment': 'positive'}
                    ]
        }
    Usage example:
    # sentence_packs = json.load(open(args.prefix + args.dataset + '/test.json'))
    # instances = load_data_instances(sentence_packs, args)
    # testset = DataIterator(instances, args)
    '''
    def __init__(self, tokenizer, single_sentence_pack, args):
        self.args = args
        self.sentence = single_sentence_pack['sentence']
        self.tokens = tokenizer.tokenize(self.sentence, add_prefix_space=True)  # ['ĠL', 'arg', 'est', 'Ġand', 'Ġfres', 'hest', 'Ġpieces', 'Ġof', 'Ġsushi', 'Ġ,', 'Ġand', 'Ġdelicious', 'Ġ!']
        self.L_token = len(self.tokens)
        self.word_spans = self.get_word_spans()  # '[[0, 2], [3, 3], [4, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12]]'
        self._word_spans = copy.deepcopy(self.word_spans)
        self.id = single_sentence_pack['id']
        
        self.triplets = single_sentence_pack['triples']

        self.triplets_in_spans = self.get_triplets_in_spans()

        self.token_classes = self.get_token_classes()

        self.cl_mask = self.get_cl_mask()
        

        assert len(self.sentence.strip().split(' ')) == len(self.word_spans)
        

        self.bert_tokens = tokenizer.encode(self.sentence, add_special_tokens=False, add_prefix_space=True)
        self.bert_tokens_padded = torch.zeros(args.max_sequence_len).long()

        self.mask = self.get_mask()

        if len(self.bert_tokens) != self._word_spans[-1][-1] + 1:
            print(self.sentence, self._word_spans)
            
        for i in range(len(self.bert_tokens)):
            self.bert_tokens_padded[i] = self.bert_tokens[i]
        
        self.tagging_matrix = self.get_tagging_matrix()
        self.tagging_matrix = (self.tagging_matrix + self.mask - torch.tensor(1)).long()

    def get_mask(self):
        mask = torch.ones((self.args.max_sequence_len, self.args.max_sequence_len))
        mask[:, len(self.bert_tokens):] = 0
        mask[len(self.bert_tokens):, :] = 0
        for i in range(len(self.bert_tokens)):
            mask[i][i] = 0
        return mask
        
    def get_word_spans(self):
        '''
        get roberta-token-spans of each word in a single sentence
        according to the rule: each 'Ġ' maps to a single word
        required: tokens = tokenizer.tokenize(sentence, add_prefix_space=True)
        '''

        l_indx = 0
        r_indx = 0
        word_spans = []
        while r_indx + 1 < len(self.tokens):
            if self.tokens[r_indx+1][0] == 'Ġ':
                word_spans.append([l_indx, r_indx])
                r_indx += 1
                l_indx = r_indx
            else:
                r_indx += 1
        word_spans.append([l_indx, r_indx])
        return word_spans

    def get_triplets_in_spans(self):
        triplets_in_spans = []
        for triplet in self.triplets:
            sentiment2id = {'negative': 2, 'neutral': 3, 'positive': 4}

            aspect_tags = triplet['target_tags']
            opinion_tags = triplet['opinion_tags']
            sentiment = triplet['sentiment']

            aspect_spans = self.get_spans_from_BIO(aspect_tags)
            opinion_spans = self.get_spans_from_BIO(opinion_tags)

            triplets_in_spans.append((aspect_spans, opinion_spans, sentiment2id[sentiment]))

        return triplets_in_spans
    
    
    def get_spans_from_BIO(self, tags):
        '''for BIO tag'''
        token_ranges = copy.deepcopy(self.word_spans)
        tags = tags.strip().split()
        length = len(tags)
        spans = []
        # start = -1
        for i in range(length):
            # print(i)
            if tags[i].endswith('B'):
                spans.append(token_ranges[i])
                # 遇到一个 B，意味着开启一个新的三元组
                # 接下来需要看有没有跟 I，如果一直跟着 I，则一直扩大这个三元组的span范围，直到后面是 O 或下一个 B为止，此时则重新继续找下一个 B
                # 其实如果一旦找到一个 B 然后直到这个 B 终止于一个 O或下一个 B，这个刚刚被找到的三元组 BI··O/B的范围内就不再会有其他 B，因此不需要回溯，直接 go on
            elif tags[i].endswith('I'):
                spans[-1][-1] = token_ranges[i][-1]
            else:  # endswith('O')
                pass
        return spans

    def get_token_classes(self):
        # 0: NULL
        # 1: Aspect
        # 2: Opinion negative
        # 3: Opinion neutral
        # 4: Opinion positive
        token_classes = [0] * self.L_token
        # sentiment2id = {'negative': 2, 'neutral': 3, 'positive': 4}
        for aspect_spans, opinion_spans, sentiment in self.triplets_in_spans:
            # print(aspect_spans, opinion_spans, sentiment)
            for a in aspect_spans:
                _a = copy.deepcopy(a)
                token_classes[_a[0]: _a[-1]+1] = [1] * (_a[-1]+1 - _a[0])
            for o in opinion_spans:
                _o = copy.deepcopy(o)
                token_classes[_o[0]: _o[-1]+1] = [sentiment] * (_o[-1]+1 - _o[0])
        return token_classes

    def get_cl_mask(self):
        token_classes = torch.tensor(self.token_classes).unsqueeze(0).expand(self.L_token, -1)
        eq = (token_classes == token_classes.T)
        mask01 = ((torch.tril(torch.ones(self.L_token, self.L_token)) - 1) * (-1))
        m = (eq * 2 - 1) * mask01
        pad_len = self.args.max_sequence_len - self.L_token
        return F.pad(m, (0, pad_len, 0, pad_len), "constant", 0)
        
    def get_tagging_matrix(self):
        '''
        mapping the tags to a Matrix Tagginh scheme
        '''
        tagging_matrix = torch.zeros((self.args.max_sequence_len, self.args.max_sequence_len))
        '''
        tagging_matrix                      O   
        array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        A      [0., 0., 0., 0., 0., 0., 0., S , 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        '''

        for triplet in self.triplets_in_spans:
            # triplets = [{'uid': '2125-0',
            #   'target_tags': 'Largest\\O and\\O freshest\\O pieces\\B of\\I sushi\\I ,\\O and\\O delicious\\O !\\O',
            #   'opinion_tags': 'Largest\\B and\\O freshest\\B pieces\\O of\\O sushi\\O ,\\O and\\O delicious\\B !\\O',
            #   'sentiment': 'positive'}]

            # print(aspect_tags)
            # print(opinion_tags)
            # print(sentiment_tags)
            # Largest\O and\O freshest\O pieces\B of\I sushi\I ,\O and\O delicious\O !\O
            # Largest\B and\O freshest\B pieces\O of\O sushi\O ,\O and\O delicious\B !\O
            # positive
            # break
            sentiment2id = {'negative': 2, 'neutral': 3, 'positive': 4}
            
            # if len(self.triplets_in_spans) != 3:
            #     print(self.triplets_in_spans)

            aspect_spans, opinion_spans, sentiment = triplet

            # aspect_spans = self.get_spans_from_BIO(aspect_tags, self.word_spans)
            # opinion_spans = self.get_spans_from_BIO(opinion_tags, self.word_spans)

            '''set tag for sentiment'''
            for aspect_span in aspect_spans:
                for opinion_span in opinion_spans:
                    # print(aspect_span)
                    # print(opinion_span)
                    al = aspect_span[0]
                    ar = aspect_span[1]
                    pl = opinion_span[0]
                    pr = opinion_span[1]
                    for i in range(al, ar+1):
                        for j in range(pl, pr+1):
                            # print(al, ar, pl, pr)
                            # print(i, j)
                            # print(i==al and j==pl)
                            if i==al and j==pl:
                                tagging_matrix[i][j] = sentiment  # 3 4 5
                            else:
                                tagging_matrix[i][j] = 1  # 1: ctd

        return tagging_matrix

    


class DataIterator(object):
    def __init__(self, instances, args):
        self.instances = instances
        self.args = args
        self.batch_count = math.ceil(len(instances)/args.batch_size)

    def get_batch(self, index):
        sentence_ids = []
        word_spans = []
        bert_tokens = []
        masks = []
        tagging_matrices = []
        tokenized = []
        cl_masks = []
        token_classes = []
        
        for i in range(index * self.args.batch_size, min((index + 1) * self.args.batch_size, len(self.instances))):
            sentence_ids.append(self.instances[i].id)
            word_spans.append(self.instances[i].word_spans)
            bert_tokens.append(self.instances[i].bert_tokens_padded)
            masks.append(self.instances[i].mask)
            # aspect_tags.append(self.instances[i].aspect_tags)
            # opinion_tags.append(self.instances[i].opinion_tags)
            tagging_matrices.append(self.instances[i].tagging_matrix)
            tokenized.append(self.instances[i].tokens)
            cl_masks.append(self.instances[i].cl_mask)
            token_classes.append(self.instances[i].token_classes)

        if len(bert_tokens) == 0:
            print(bert_tokens)

        bert_tokens = torch.stack(bert_tokens).to(self.args.device)
        # lengths = torch.tensor(lengths).to(self.args.device)
        masks = torch.stack(masks).to(self.args.device)
        # aspect_tags = torch.stack(aspect_tags).to(self.args.device)
        # opinion_tags = torch.stack(opinion_tags).to(self.args.device)
        tagging_matrices = torch.stack(tagging_matrices).long().to(self.args.device)
        cl_masks = torch.stack(cl_masks).long().to(self.args.device)
        return sentence_ids, bert_tokens, masks, word_spans, tagging_matrices, tokenized, cl_masks, token_classes


if __name__ == "__main__":

    import sys
    from transformers import RobertaTokenizer, RobertaModel
    import argparse
    # import os
    
    # 获取当前文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 获取上一级目录
    parent_dir = os.path.dirname(current_dir)
    
    # 将上一级目录添加到sys.path
    sys.path.append(parent_dir)
    from utils.data_utils import load_data_instances

    # Load Dataset
    train_sentence_packs = json.load(open(os.path.abspath('D1/res14/train.json')))
    # # random.shuffle(train_sentence_packs)
    # dev_sentence_packs = json.load(open(os.path.abspath(args.prefix + args.data_version + '/' + args.dataset + '/dev.json')))
    # test_sentence_packs = json.load(open(os.path.abspath(args.prefix + args.data_version + '/' + args.dataset + '/test.json')))


    #加载预训练字典和分词方法
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base",
        cache_dir="../modules/models/",  # 将数据保存到的本地位置，使用cache_dir 可以指定文件下载位置
        force_download=False,  # 是否强制下载
    )

    # 创建一个TensorBoard写入器
    torch.cuda.set_device(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_sequence_len', type=int, default=100, help='max length of the tagging matrix')
    parser.add_argument('--sentiment2id', type=dict, default={'negative': 2, 'neutral': 3, 'positive': 4}, help='mapping sentiments to ids')
    parser.add_argument('--model_cache_dir', type=str, default='./modules/models/', help='model cache path')
    parser.add_argument('--model_name_or_path', type=str, default='roberta-base', help='reberta model path')
    parser.add_argument('--batch_size', type=int, default=16, help='json data path')
    parser.add_argument('--device', type=str, default="cuda", help='gpu or cpu')
    parser.add_argument('--prefix', type=str, default="./data/", help='dataset and embedding path prefix')

    parser.add_argument('--data_version', type=str, default="D1", choices=["D1", "D2"], help='dataset and embedding path prefix')
    parser.add_argument('--dataset', type=str, default="res14", choices=["res14", "lap14", "res15", "res16"], help='dataset')

    parser.add_argument('--bert_feature_dim', type=int, default=768, help='dimension of pretrained bert feature')
    parser.add_argument('--epochs', type=int, default=2000, help='training epoch number')
    parser.add_argument('--class_num', type=int, default=5, help='label number')
    parser.add_argument('--task', type=str, default="triplet", choices=["pair", "triplet"], help='option: pair, triplet')
    parser.add_argument('--model_save_dir', type=str, default="./modules/models/saved_models/", help='model path prefix')
    parser.add_argument('--log_path', type=str, default="log.log", help='log path')


    args = parser.parse_known_args()[0]

    
    train_instances = load_data_instances(tokenizer, train_sentence_packs, args)
    # dev_instances = load_data_instances(tokenizer, dev_sentence_packs, args)
    # test_instances = load_data_instances(tokenizer, test_sentence_packs, args)

    trainset = DataIterator(train_instances, args)
    # devset = DataIterator(dev_instances, args)
    # testset = DataIterator(test_instances, args)
