'''
ObjectoSphere Loss
    Author:  Akshay Raj Dhamija, Manuel Gunther, and Terrance E. Boult
    Paper A: Reducing Network Agnostophobia, in Advances in Neural Information Processing Systems (ANIPS'18)
    Paper B: Watchlist adaptation: protecting the innocent, in International Conference of the Biometrics Special Interest Group (BIOSIG'20)
'''

import torch

__all__ = [ 'ObjectoSphereLoss']


class ObjectoSphereLoss(torch.nn.Module):
    '''
    This criterion attempts to maintain the magnitude of known samples higher than an specified threshold as well as push the magnitude of unknown samples towards 0 (zero).

    It is useful when training a classification problem with `C` classes.

    The `features` is expected to contain the latent representation vector of each sample.
    The `targets` that this criterion expects should contain class indices in the range :math:`[0, C)` where :math:`C` is the number of classes
    '''
    def __init__(self, min_magnitude:float=10.0, reduction:str='mean'):
        super(ObjectoSphereLoss, self).__init__()

        self.knownsMinimumMag = min_magnitude
        self.reduction = reduction

    def forward(self, features:torch.Tensor, targets:torch.Tensor, sample_weights:torch.Tensor=None):
        # guarantee all tensors run on same device
        assert features.device == targets.device, 'indices should be either on cpu or on the same device as the indexed tensor (cpu)'
        # get boolean tensor (true/false) indicating elements satisfying criteria
        neg_indexes = (targets  < 0)
        pos_indexes = (targets >= 0)
        # compute feature magnitude
        mag = features.norm(p=2, dim=1)
        # for knowns we want a certain minimum magnitude
        mag_diff_from_ring = torch.clamp(self.knownsMinimumMag - mag, min=0.0)
        # create container to store loss per sample
        loss = torch.zeros(features.shape[0], device=features.device)
        # knowns: punish if magnitude is inside of ring
        loss[pos_indexes] = mag_diff_from_ring[pos_indexes]
        # unknowns: punish any magnitude
        loss[neg_indexes] = mag[neg_indexes]
        # exponentiate each element and remove negative values
        loss = torch.pow(loss, 2)
        # compute weighted loss
        if sample_weights is not None:
            loss = sample_weights * loss
        # return batch loss
        if   self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum':  return loss.sum()
        else: return loss


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    feat_size, num_classes, num_samples = 128, 11, 100

    features = torch.randn(num_samples, feat_size, requires_grad=True).to(device)
    labels = torch.randint(num_classes, (num_samples,), dtype=torch.int64).to(device) - 1
    print(device, features.shape, labels.shape)

    criterion = ObjectoSphereLoss(min_magnitude=5.0, reduction='sum')
    loss_score = criterion(features, labels)
    loss_score.backward()
    print(loss_score)