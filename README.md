# CADA-VAE
Original PyTorch implementation of "Generalized Zero-and Few-Shot Learning via Aligned Variational Autoencoders" (CVPR 2019).

Paper: https://arxiv.org/pdf/1812.01784.pdf

<p align="center">
  <img width="600" src="Model.jpg">
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
Using Python 2 is not recommended.

### Data
Download the following folder https://www.dropbox.com/sh/btoc495ytfbnbat/AAAaurkoKnnk0uV-swgF-gdSa?dl=0
and put it in this repository.
Next to the folder "model", there should be a folder "data".

### Experiments

To run the experiments from the paper, navigate to the model folder and execute the following:
```
python single_experiment.py --dataset CUB --num_shots 0 --generalized True
```
The choices for the input arguments are:
```
datasets: CUB, SUN, AWA1, AWA2
num_shots: any number 
generalized: True, False
```
More hyperparameters can be adjusted in the file single_experiment.py directly. The results vary by 1-2% between identical runs.

### Citation
If you use this work please cite
```
@inproceedings{schonfeld2019generalized,
  title={Generalized zero-and few-shot learning via aligned variational autoencoders},
  author={Schonfeld, Edgar and Ebrahimi, Sayna and Sinha, Samarth and Darrell, Trevor and Akata, Zeynep},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={8247--8255},
  year={2019}
}
```
### Contact 
For questions or help, feel welcome to write an email to edgarschoenfeld@live.de

