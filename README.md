# CADA-VAE
A PyTorch implementation of "Generalized Zero-and Few-Shot Learning via Aligned Variational Autoencoders" (CVPR 2019).

Paper: https://arxiv.org/pdf/1812.01784.pdf

<p align="center">
  <img width="600" src="model.jpg">
</p>
<p align="justify">
  
### Requirements
The code was implemented using Python 3.5.6 and the following packages:
```
torch==0.4.1
numpy==1.14.3
scipy==1.1.0
scikit_learn==0.20.3
networkx==1.11
```

### Data
Download the following folder https://www.dropbox.com/sh/btoc495ytfbnbat/AAAaurkoKnnk0uV-swgF-gdSa?dl=0
and put it in this repository.
Next to the folder "model", there should be a folder "data".

### Experiments

To run the experiments from the paper, execute the following:
```
python single_experiment.py --dataset CUB --num_shots 0 --generalized True
```
The choices for the parameters are:
```
datasets: CUB, SUN, AWA1, AWA2
num_shots: any number 
generalized: True, False
```
More hyperparameters can be adjusted in the file single_experiment.py directly. The results can vary by 1-2% between identical runs.
