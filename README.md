# DCSGD
This repo contains the source code of DCSGDE and DCSGDP: two DPSGD (Differentially Private Stochastic Gradient Descent, which is the most widely used method to train a differentially private deep learning model) variants that can adaptively set the clipping threshold C based on the differentially private estimation of gradient norm distribution.



##Files
.
├── Adaptive.py        Code for percentile method and privacy accountant.
├── DCSGDE.py          DCSGDE for SVHN, CIFAR10, MNIST on Resnet18, Resnet34, linear regression.

DCSGDP.py          DCSGDP for SVHN, CIFAR10, MNIST on Resnet18, Resnet34, linear regression.
bertDCSGDE.py      DCSGDE for SNLI on BERT-base
bertDCSGDP.py      DCSGDP for SNLI on BERT-base.
bert.py            Code for downloading SNLI dataset.

