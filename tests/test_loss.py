import torch

from maximal_entropy_loss.src.openloss import EntropicOpenSetLoss, MaximalEntropyLoss, ObjectoSphereLoss

torch.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def test_eos():
    num_classes, num_samples = 11, 100

    values = torch.randn(num_samples, num_classes, requires_grad=True).to(device)
    labels = torch.randint(num_classes, (num_samples,), dtype=torch.int64).to(device) - 1

    criterion = EntropicOpenSetLoss(num_classes=num_classes, reduction='sum')
    loss_score = criterion(values, labels)
    loss_score.backward()

    assert round(loss_score.item(), 3) == 312.6840


def test_mel():
    num_classes, num_samples = 11, 100

    values = torch.randn(num_samples, num_classes, requires_grad=True).to(device)
    labels = torch.randint(num_classes, (num_samples,), dtype=torch.int64).to(device) - 1

    criterion = MaximalEntropyLoss(num_classes=num_classes, reduction='sum')
    loss_score = criterion(values, labels)
    loss_score.backward()

    assert round(loss_score.item(), 3) == 313.717


def test_osl():
    feat_size, num_classes, num_samples = 128, 11, 100

    features = torch.randn(num_samples, feat_size, requires_grad=True).to(device)
    labels = torch.randint(num_classes, (num_samples,), dtype=torch.int64).to(device) - 1

    criterion = ObjectoSphereLoss(min_magnitude=5.0, reduction='sum')
    loss_score = criterion(features, labels)
    loss_score.backward()

    assert round(loss_score.item(), 3) == 1206.067
