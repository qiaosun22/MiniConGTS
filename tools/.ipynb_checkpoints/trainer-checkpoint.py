import torch
from torch.utils.tensorboard import SummaryWriter
import datetime

from utils.common_utils import stop_words
from tools.evaluate import evaluate

from tqdm import trange


class Trainer():
    def __init__(self, model, trainset, devset, testset, optimizer, criterion, lr_scheduler, args, logging):
        self.model = model
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.optimizer = optimizer
        # self.criterion = criterion
        self.f_loss = criterion[0]
        self.f_loss1 = criterion[1]
        self.lr_scheduler = lr_scheduler
        self.best_joint_f1 = 0
        self.best_joint_f1_test = 0
        self.best_joint_epoch = 0
        self.best_joint_epoch_test = 0

        self.writer = SummaryWriter()
        self.args = args
        self.logging = logging

        self.evaluate = evaluate
        self.stop_words = stop_words

    def train(self):
        for i in range(self.args.epochs):
            self.logging('\n\nEpoch:{}'.format(i+1))
            epoch_sum_loss = []
            for j in trange(self.trainset.batch_count):
                self.model.train()
                sentence_ids, bert_tokens, masks, word_spans, tagging_matrices, tokenized, cl_masks = self.trainset.get_batch(j)
                
                mask_cl = ((torch.tril(torch.ones(self.args.max_sequence_len, self.args.max_sequence_len)) - 1) * (-1)).unsqueeze(0).expand(len(sentence_ids), -1, -1).to(self.args.device)

                # bert_tokens = bert_tokens.half().to(args.device)
                # masks = masks.half().to(args.device)

                logits, logits1, sim_matrices = self.model(bert_tokens, masks)

                logits_flatten = logits.reshape([-1, logits.shape[3]])
                tagging_matrices_flatten = tagging_matrices.reshape([-1])
                
                loss0 = self.f_loss(logits_flatten, tagging_matrices_flatten)#, logits1, sim_matrix

                tags1 = tagging_matrices.clone()#[tags>0]
                tags1[tags1>0] = 1
                logits1_flatten = logits1.reshape([-1, logits1.shape[3]])
                tags1_flatten = tags1.reshape([-1]).to(self.args.device)
                loss1 = self.f_loss1(logits1_flatten.float(), tags1_flatten)
                
                # tags2 = tags1.clone()#[tags>0]
                # '''transpose so that only a triangle is used'''
                # tags2 = tags2 + tags2.transpose(1, 2)
                
                # tags2 = 1.5 * tags2**2 + 0.5 * tags2 - 1

                # print(tags2.shape, sim_matrix.shape, mask_cl.shape)

                
                loss_cl = - (sim_matrices * cl_masks).mean()

                print(loss0.item(), loss1.item(), loss_cl.item())
                
                loss = loss0 + loss1 + 1e-3 * loss_cl
                epoch_sum_loss.append(loss)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar('train loss', loss, i*self.trainset.batch_count+j+1)
                self.writer.add_scalar('train loss0', loss0, i*self.trainset.batch_count+j+1)
                self.writer.add_scalar('train loss1', loss1, i*self.trainset.batch_count+j+1)
                self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], i*self.trainset.batch_count+j+1)
                # writer.add_scalar('lr1', optimizer.param_groups[1]['lr'], i*trainset.batch_count+j+1)
                # writer.add_scalar('lr2', optimizer.param_groups[2]['lr'], i*trainset.batch_count+j+1)
                # writer.add_scalar('lr3', optimizer.param_groups[3]['lr'], i*trainset.batch_count+j+1)
                
            epoch_avg_loss = sum(epoch_sum_loss) / len(epoch_sum_loss)
            

            self.logging('{}\tAvg loss: {:.10f}'.format(str(datetime.datetime.now()), epoch_avg_loss))
                
            joint_precision, joint_recall, joint_f1 = self.evaluate(self.model, self.devset, self.stop_words, self.logging, self.args)
            joint_precision_test, joint_recall_test, joint_f1_test = self.evaluate(self.model, self.testset, self.stop_words, self.logging, self.args)
            
            if joint_f1 >= self.best_joint_f1:
                best_joint_f1 = joint_f1
                self.best_joint_epoch = i
                if joint_f1_test > self.best_joint_f1_test:
                    model_path = self.args.model_save_dir + self.args.data_version + "-" + self.args.dataset + "-" + str(round(100*joint_f1_test, 4)) + "-" + 'epoch' + str(i) + '.pt'
                    torch.save(self.model, model_path)
                    self.best_joint_f1_test = joint_f1_test
                    self.best_joint_epoch_test = i

            self.writer.add_scalar('dev f1', joint_f1, i+1)
            self.writer.add_scalar('test f1', joint_f1_test, i+1)
            self.writer.add_scalar('dev precision', joint_precision, i+1)
            self.writer.add_scalar('test precision', joint_precision_test, i+1)
            self.writer.add_scalar('dev recall', joint_recall, i+1)
            self.writer.add_scalar('test recall', joint_recall_test, i+1)
            
            self.lr_scheduler.step()


            self.logging('best epoch: {}\tbest dev {} f1: {:.5f}'.format(self.best_joint_epoch+1, self.args.task, best_joint_f1))
            self.logging('best epoch: {}\tbest test {} f1: {:.5f}'.format(self.best_joint_epoch_test+1, self.args.task, self.best_joint_f1_test))


        # 关闭TensorBoard写入器
        self.writer.close()
