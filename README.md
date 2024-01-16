# Open-set Learning Cost Functions

This project encompasses a list of three cost functions recently designed for open-set learning and classification, available through the [Python Package Index](https://pypi.org/project/openloss/) so that it can be easily deployed to your environment:

* ```EntropicOpenSetLoss(num_classes, reduction='mean', weight=None)``` *[1,3]*
* ```MaximalEntropyLoss(num_classes, margin=0.5, reduction='mean', weight=None)``` *[1,2]*
* ```ObjectoSphereLoss(min_magnitude=10.0, reduction='mean')``` *[1,3]*

Our most recent contribution, the Maximal Entropy Loss (MEL), increases the entropy for negative samples and attaches a penalty to known target classes in pursuance of gallery specialization.
MEL was initially designed to address open-set face recognition applications; however, it can also be employed in any classification task.

The proposed functions' API models the conventional cost functions available under the PyTorch framework so that it can be invoked according to the code block below:

```python
    import openloss
    import torch

    # Generate 10 positive classes and one negative class
    num_classes, num_samples = 11, 100
    values = torch.randn(num_samples, num_classes, requires_grad=True)
    labels = torch.randint(num_classes, (num_samples,), dtype=torch.int64) - 1

    # Feed criterion with both values and labels and compute loss
    criterion = openloss.MaximalEntropyLoss(num_classes=num_classes, margin=0.5)
    loss_score = criterion(values, labels)
    loss_score.backward()
```

There is a [notebook tutorial](https://gist.github.com/rafaelvareto/b92d59d5dab1d0bfcf1e495fdcc4eb26) implemented with the ```PyTorch``` framework as well as the ```BOB.measure``` library demonstrating how to use MEL in open-set applications. 
The tutorial relies on E-MNIST and K-MNIST datasets in which letters make up the gallery and probe sets, numbers constitute the negative training set whereas Korean digits compose the set of unknown probe samples. 
If you have further questions about the proposed approach, you can reach the corresponding author though the contact form available at [www.vareto.com.br](https://vareto.com.br/index.html#contact).

If you make use of the aforementioned cost functions, please cite all or one of the following works:

1. @article{vareto2024open,title={Open-set face recognition with maximal entropy and Objectosphere loss},author={Vareto, Rafael Henrique and Linghu, Yu and Boult, Terrance Edward and Schwartz, William Robson and G{\"u}nther, Manuel},journal={Image and Vision Computing},volume={141},pages={104862},year={2024},publisher={Elsevier}}
2. @inproceedings{vareto2023open,title={Open-set Face Recognition with Neural Ensemble, Maximal Entropy Loss and Feature Augmentation},author={Vareto, Rafael Henrique and G{\"u}nther, Manuel and Schwartz, William Robson},booktitle={2023 36th SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI)},pages={55--60},year={2023},organization={IEEE}}
3. @inproceedings{gunther2020watchlist, title={Watchlist adaptation: protecting the innocent},author={G{\"u}nther, Manuel and Dhamija, Akshay Raj and Boult, Terrance E},
  booktitle={2020 International Conference of the Biometrics Special Interest Group (BIOSIG)},pages={1--7},year={2020},organization={IEEE}}
