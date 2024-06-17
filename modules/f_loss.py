import torch
import torch.nn.functional as F

# Loss Func
class FocalLoss(torch.nn.Module):
    def __init__(self, weight, alpha=0.99, gamma=4., size_average=True, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.weight = weight

    def forward(self, inputs, targets):
        # F.cross_entropy(x,y)工作过程就是(Log_Softmax+NllLoss)：①对x做softmax,使其满足归一化要求，结果记为x_soft;②对x_soft做对数运算
        # 并取相反数，记为x_soft_log;③对y进行one-hot编码，编码后与x_soft_log进行点乘，只有元素为1的位置有值而且乘的是1，
        # 所以点乘后结果还是x_soft_log
        # 总之，F.cross_entropy(x,y)对应的数学公式就是CE(pt)=-1*log(pt)
        # inputs = inputs.long()
        # targets = targets.long()

        # print(inputs.dtype)
        # inputs = inputs.float()

        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)  # pt是预测该类别的概率，要明白F.cross_entropy工作过程就能够理解
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

