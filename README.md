# Maximal-Entropy Loss
This repository discloses a novel cost function implemmentation, entitled Maximal Entropy Loss (MEL), that increases the entropy for negative samples and attaches a penalty to known target classes in pursuance of gallery specialization.

MEL was initially designed to address open-set face recognition applications; however, it can also be employed in many classification tasks.
The proposed approach usage models the conventional cost functions available under the PyTorch framework so that it can be invoked as follows:

```python
    import torch

    num_classes, num_samples = 11, 100
    values = torch.randn(num_samples, num_classes, requires_grad=True)
    labels = torch.randint(num_classes, (num_samples,), dtype=torch.int64) - 1

    criterion = MaximalEntropyLoss(num_classes=num_classes, reduction='sum')
    loss_score = criterion(values, labels)
    loss_score.backward()
```

There is a [notebook tutorial](tests/tutorial.ipynb) implemented with the ```PyTorch``` framework as well as the ```BOB.measure``` library demonstrating how to use MEL in open-set applications. 
The tutorial relies on E-MNIST and K-MNIST datasets in which letters make up the gallery and probe sets, numbers constitute the negative training set whereas Korean digits compose the set of unknown probe samples. 
If you have further questions about the proposed approach, you can reach the corresponding author though the contact form available at [www.vareto.com.br](https://vareto.com.br/index.html#contact).

If you make use of this criterion, please cite both or one of the following works:

> @article{vareto2024open,title={Open-set face recognition with maximal entropy and Objectosphere loss},author={Vareto, Rafael Henrique and Linghu, Yu and Boult, Terrance Edward and Schwartz, William Robson and G{\"u}nther, Manuel},journal={Image and Vision Computing},volume={141},pages={104862},year={2024},publisher={Elsevier}}

> @inproceedings{vareto2023open,title={Open-set Face Recognition with Neural Ensemble, Maximal Entropy Loss and Feature Augmentation},author={Vareto, Rafael Henrique and G{\"u}nther, Manuel and Schwartz, William Robson},
booktitle={2023 36th SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI)},pages={55--60},year={2023},organization={IEEE}}
