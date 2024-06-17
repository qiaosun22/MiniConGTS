import math
import torch
import json
import os

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
        self.id = single_sentence_pack['id']
        self.sentence = single_sentence_pack['sentence']
        self.tokens = tokenizer.tokenize(self.sentence, add_prefix_space=True)  # ['ĠL', 'arg', 'est', 'Ġand', 'Ġfres', 'hest', 'Ġpieces', 'Ġof', 'Ġsushi', 'Ġ,', 'Ġand', 'Ġdelicious', 'Ġ!']
        self.word_spans = self.get_word_spans(tokens=self.tokens)  # '[[0, 2], [3, 3], [4, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12]]'
        assert len(self.sentence.strip().split(' ')) == len(self.word_spans)
        self.triplets = single_sentence_pack['triples']

        self.bert_tokens = tokenizer.encode(self.sentence, add_special_tokens=False, add_prefix_space=True)
        self.bert_tokens_padded = torch.zeros(args.max_sequence_len).long()
        
        self.mask = torch.ones((args.max_sequence_len, args.max_sequence_len))
        self.mask[:, len(self.bert_tokens):] = 0
        self.mask[len(self.bert_tokens):, :] = 0
        for i in range(len(self.bert_tokens)):
            self.mask[i][i] = 0

        if len(self.bert_tokens) != self.word_spans[-1][-1] + 1:
            print(self.sentence, self.word_spans)
            
        for i in range(len(self.bert_tokens)):
            self.bert_tokens_padded[i] = self.bert_tokens[i]
        
        self.tagging_matrix = self.get_tagging_matrix(triplets=self.triplets, word_spans=self.word_spans, sentiment2id=args.sentiment2id)
        self.tagging_matrix = (self.tagging_matrix + self.mask - torch.tensor(1)).long()

    # def get_mask(self):
        
        
    def get_word_spans(self, tokens):
        '''
        get roberta-token-spans of each word in a single sentence
        according to the rule: each 'Ġ' maps to a single word
        required: tokens = tokenizer.tokenize(sentence, add_prefix_space=True)
        '''

        l_indx = 0
        r_indx = 0
        word_spans = []
        while r_indx + 1 < len(tokens):
            if tokens[r_indx+1][0] == 'Ġ':
                word_spans.append([l_indx, r_indx])
                r_indx += 1
                l_indx = r_indx
            else:
                r_indx += 1
        word_spans.append([l_indx, r_indx])
        return word_spans
    
    def get_spans_from_BIO(self, tags, token_ranges):
        '''for BIO tag'''
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
        
    def get_tagging_matrix(self, triplets, word_spans, sentiment2id = {'negative': 2, 'neutral': 3, 'positive': 4}):
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

        for triplet in triplets:
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
            # sentiment2id = {'negative': 2, 'neutral': 3, 'positive': 4}

            aspect_tags = triplet['target_tags']
            opinion_tags = triplet['opinion_tags']
            sentiment = triplet['sentiment']

            aspect_spans = self.get_spans_from_BIO(aspect_tags, self.word_spans)
            opinion_spans = self.get_spans_from_BIO(opinion_tags, self.word_spans)

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
                                tagging_matrix[i][j] = sentiment2id[sentiment]  # 3 4 5
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

        for i in range(index * self.args.batch_size, min((index + 1) * self.args.batch_size, len(self.instances))):
            sentence_ids.append(self.instances[i].id)
            word_spans.append(self.instances[i].word_spans)
            bert_tokens.append(self.instances[i].bert_tokens_padded)
            masks.append(self.instances[i].mask)
            # aspect_tags.append(self.instances[i].aspect_tags)
            # opinion_tags.append(self.instances[i].opinion_tags)
            tagging_matrices.append(self.instances[i].tagging_matrix)
            tokenized.append(self.instances[i].tokens)

        bert_tokens = torch.stack(bert_tokens).to(self.args.device)
        # lengths = torch.tensor(lengths).to(self.args.device)
        masks = torch.stack(masks).to(self.args.device)
        # aspect_tags = torch.stack(aspect_tags).to(self.args.device)
        # opinion_tags = torch.stack(opinion_tags).to(self.args.device)
        tagging_matrices = torch.stack(tagging_matrices).long().to(self.args.device)
        return sentence_ids, bert_tokens, masks, word_spans, tagging_matrices, tokenized


