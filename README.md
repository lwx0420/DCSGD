# DCSGD
This repo contains the source code of DCSGDE and DCSGDP: two DPSGD (Differentially Private Stochastic Gradient Descent, which is the most widely used method to train a differentially private deep learning model) variants that can adaptively set the clipping threshold C based on the differentially private estimation of gradient norm distribution.

## Datasets and Models:
Three image datasets: SVHN CIFAR10 MNIST, can be downloaded from pytorch in the code for DCSGDE and DCSGDP.

One text datasets: SNLI, can be downloaded by runing bert.py from Stanford NLP mirror.

Three models for CV task: ResNet18 ResNet34 linear regression, we use the model architecture pytorch provides, and use group normalization to replace the batch normalization in ResNet since batch normalization is unsupported under DP.

One model for NLP task: Bert-base, we use the pretrained bert base model in [huggingface transformers repo](https://github.com/huggingface/transformers).

## Files:
    .
    ├── Adaptive.py        #Code for percentile method and privacy accountant.
    ├── DCSGDE.py          #DCSGDE for SVHN, CIFAR10, MNIST on Resnet18, Resnet34, linear regression.
    ├── DCSGDP.py          #DCSGDP for SVHN, CIFAR10, MNIST on Resnet18, Resnet34, linear regression.
    ├── bertDCSGDE.py      #DCSGDE for SNLI on BERT-base
    ├── bertDCSGDP.py      #DCSGDP for SNLI on BERT-base.
    ├── bert.py            #Code for downloading SNLI dataset.
    
To run experiments on CV tasks, run DCSGDE.py and DCSGDP.py directly, it will download the dataset and train the model according to the prarameters.

To run experiments on NLP task, firts run bert.py to download the dataset, and then run bertDCSGDP.py and bertDCSGDE.py.
    
## Dependecies:
    python 3.7
    opacus 1.0.0
    pytorch 1.11.0
    torchvision 0.12.0
    transformers 3.2.0

## Parameters:
    ├── epochs              #the number of training epoch
    ├── batchsize_test      #batch size for testing       
    ├── batchsize_traing    #batch size for training
    ├── lr                  #learning rate
    ├── model_type          #which model to use
    ├── dataset             #which dataset to use
    ├── dp_able             #if train with DP
    ├── sigma               #the noise multiplier for DP training
    ├── data_path           #the directory of dataset
    ├── device              #use CPU or GPU to train
    ├── clip_bound          #the initial clipping threshold C
    ├── optim               #the type of optimizer
    ├── resample_batchsize  #batch size for building histogram
    ├── delta               #privacy parameter $\delta$
    ├── target_eps          #target privacy parameter $\epsilon$
    ├── adaptive            #if using a adaptive clipping threshold
    ├── percentile          #the percentile p of DCSGDP
    ├── resample_num        #the interval of updating C (the number of examples)
    ├── save_path           #the directory to save the result and record
    ├── sigma_t             #the noise multiplier for building histogram
    ├── bins                #the number of bins in histogram
    ├── pretrain            #if use the pretrained model pytorch provide
    
