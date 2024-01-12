'''
Maximal Entropy Loss
    Author:  Rafael Henrique Vareto
    Paper A: Open-set Face Recognition with Neural Ensemble, Maximal Entropy Loss and Feature Augmentation, in Conference on Graphics, Patterns and Images (SIBGRAPI'23)
    Paper B: Open-Set Face Recognition with Maximal Entropy and Objectosphere Loss, in Imave and Vision Computing Journal (IMAVIS'24)
'''

import torch

__all__ = [ 'MaximalEntropyLoss']


class MaximalEntropyLoss(torch.nn.Module):
    '''
    This criterion produces more rigorous decision boundaries for known classes (target >= 0) and also increases the entropy for negative training samples (target < 0).

    It is useful when training a classification problem with `C` classes.

    The `logits` is expected to contain the unnormalized logits for each class (which do `not` need to be positive or sum to 1, in general).
    The `targets` that this criterion expects should contain class indices in the range :math:`[0, C)` where :math:`C` is the number of classes
    '''
    def __init__(self, num_classes:int, margin:float=0.35, reduction:str='mean', weight:torch.Tensor=None):
        super(MaximalEntropyLoss, self).__init__()

        self.margin = margin
        self.num_classes = num_classes
        self.reduction = reduction
        self.weight = weight

        self.eye = None
        self.ones = None
        self.unknowns_multiplier = 1.0 / self.num_classes

    def forward(self, logits:torch.Tensor, targets:torch.Tensor):
        # guarantee all tensors run on same device
        assert logits.device == targets.device, 'indices should be either on cpu or on the same device as the indexed tensor (cpu)'
        if (self.eye is None) and (self.ones is None):
            self.eye = torch.eye(self.num_classes, device=logits.device)
            self.ones = torch.ones(self.num_classes, device=logits.device)
        # initialize variables with zeros
        categorical_targets = torch.zeros_like(logits)
        margin_logits = torch.zeros_like(logits)
        # get boolean tensor (true/false) indicating elements satisfying criteria
        neg_indexes = (targets  < 0)
        pos_indexes = (targets >= 0)
        # convert known targets to categorical: "target 0 to [1 0 0]", "target 1 to [0 1 0]", "target 2 to [0 0 1]" (ternary example)
        categorical_targets[pos_indexes, :] = self.eye[targets[pos_indexes]]
        # expand self.ones matrix considering unknown_indexes to obtain maximum entropy: "[0.5  0.5] for two classes" and "[0.33  0.33 0.33] for three classes"
        categorical_targets[neg_indexes, :] = (self.ones.expand(neg_indexes.count_nonzero().item(), self.num_classes) * self.unknowns_multiplier)
        # applying margin to target classes to force network to learn better parameters
        target_logits = logits - self.margin
        margin_logits[pos_indexes] = logits[pos_indexes] * (1 - self.eye[targets[pos_indexes]]) + target_logits[pos_indexes] * self.eye[targets[pos_indexes]]
        margin_logits[neg_indexes] = logits[neg_indexes]
        # obtain negative log softmax in range [0, +inf)
        negative_log_values = (-1) * torch.nn.functional.log_softmax(margin_logits, dim=1)
        # obtain ground-truth loss for knowns and distributed loss for unknown classes (element wise)
        loss = negative_log_values * categorical_targets
        # get loss for each sample in batch
        loss = torch.sum(loss, dim=1)
        # compute weighted loss
        if self.weight is not None:
            loss = loss * self.weight
        # return batch loss
        if   self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum':  return loss.sum()
        else: return loss


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    num_classes, num_samples = 11, 100

    values = torch.randn(num_samples, num_classes, requires_grad=True).to(device)
    labels = torch.randint(num_classes, (num_samples,), dtype=torch.int64).to(device) - 1
    print('MEL', device, values.shape, labels.shape)

    criterion = MaximalEntropyLoss(num_classes=num_classes, reduction='sum')
    loss_score = criterion(values, labels)
    loss_score.backward()
    print('MEL', loss_score)
