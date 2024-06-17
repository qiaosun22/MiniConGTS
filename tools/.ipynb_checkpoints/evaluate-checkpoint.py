import torch
import torch.nn.functional as F
from utils.common_utils import Logging
from tools.metric import Metric

from utils.eval_utils import get_triplets_set



def evaluate(model, dataset, stop_words, logging, args):
    model.eval()
    with torch.no_grad():
        all_ids = []
        all_preds = []
        all_labels = []
        # all_lengths = []
        all_sens_lengths = []
        all_token_ranges = []
        all_tokenized = []

        for i in range(dataset.batch_count):
            sentence_ids, tokens, masks, token_ranges, tags, tokenized, _ = dataset.get_batch(i)
            # sentence_ids, bert_tokens, masks, word_spans, tagging_matrices = trainset.get_batch(i)
            preds, _, _ = model(tokens, masks) #1
            preds = torch.argmax(preds, dim=3) #2
            all_preds.append(preds) #3
            all_labels.append(tags) #4
            # all_lengths.append(lengths) #5
            sens_lens = [len(token_range) for token_range in token_ranges]
            all_sens_lengths.extend(sens_lens) #6
            all_token_ranges.extend(token_ranges) #7
            all_ids.extend(sentence_ids) #8
            all_tokenized.extend(tokenized)

        all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
        all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
        # all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()
        
        # 引入 metric 计算评价指标
        metric = Metric(args, stop_words, all_tokenized, all_ids, all_preds, all_labels, all_sens_lengths, all_token_ranges, ignore_index=-1, logging=logging)
        predicted_set, golden_set = metric.get_sets()
        
        
        aspect_results = metric.score_aspect(predicted_set, golden_set)
        opinion_results = metric.score_opinion(predicted_set, golden_set)
        pair_results = metric.score_pairs(predicted_set, golden_set)
        
        precision, recall, f1 = metric.score_triplets(predicted_set, golden_set)
        
        
        logging('Aspect term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(aspect_results[0], aspect_results[1],
                                                                  aspect_results[2]))
        logging('Opinion term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(opinion_results[0], opinion_results[1], 
                                                                     opinion_results[2]))
        logging(args.task + '\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\n'.format(precision, recall, f1))

    model.train()
    return precision, recall, f1
