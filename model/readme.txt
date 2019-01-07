
To run the experiments from the paper, execute the following:

python single_experiment.py --dataset CUB --num_shots 0 --generalized True

The choices for the parameters are:
datasets    --> CUB, SUN, AWA1, AWA2
num_shots   --> any number 
generalized --> True, False

More hyperparameters can be adjusted in the file single_experiment.py directly.

The code is tested for Pytorch 0.4.1

