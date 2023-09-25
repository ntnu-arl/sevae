# Semantically-enhanced Deep Collision Prediction for Autonomous Navigation using Aerial Robots

[![License: BSD3](https://img.shields.io/badge/License-BSD3-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


## About

This repository contains the code for the work described in [Semantically-enhanced Variational Autoencoder for Deep Collision Prediction](https://arxiv.org/abs/2307.11522). At its current state, this repository is expected to be used with the code in [ORACLE](https://github.com/ntnu-arl/ORACLE) for inference.


The work is explained in the following videos:

1. Overview: https://www.youtube.com/watch?v=Ni4VywUQCPw
1. Detailed Explanation: https://www.youtube.com/watch?v=yoO5MqSPfKw
1. Field Experiments: https://www.youtube.com/watch?v=9NZvVPvUrPo



## Setup and Installation

Recommended: [Miniconda](https://docs.conda.io/en/latest/miniconda.html)


Clone the repository:

```bash
git clone git@github.com:ntnu-arl/sevae.git
```

To install the repository, run the following commands:
```bash
cd sevae
pip3 install -e .
```

## Folder Description
The folders contain the following:

1. **networks**: Contains the VAE network, and the loss functions for training the VAE
1. **datasets**: Contain scripts that utilize pytorch's dataset class to read from a tfrecord file
1. **utils**: Contains utility scripts for creating the tfrecord files, and other utilities
1. **weights**: Contains the weights for the VAE
1. **inference**: Contains the scripts for running the VAE node, and the scripts for interfacing with the VAE node with ROS-based simulators
1. **baselines**: Contains the scripts for running the baseline compression methods (FFT and Wavelets)


## Usage

### Inference

To run the seVAE node for inference to obtain the latent space, run the following command:
```
cd sevae/inference/src
python3 vae_node.py --sim=True
```

### Training
The file `train_seVAE.py` contains the code used to train the Semantically-enhanced Variational Autoencoder for Deep Collision Prediction as described in [this paper](https://arxiv.org/abs/2307.11522). Currently this is not supported with datasets but support for training yourselves will be added soon!

## Citing

If you use this work in your research, please cite the following paper:

```
@misc{kulkarni2023semanticallyenhanced,
      title={Semantically-enhanced Deep Collision Prediction for Autonomous Navigation using Aerial Robots}, 
      author={Mihir Kulkarni and Huan Nguyen and Kostas Alexis},
      year={2023},
      eprint={2307.11522},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

## Contact:
* [Mihir Kulkarni](mailto:mihir.kulkarni@ntnu.no)
* [Huan Nguyen](mailto:ndhuan93@gmail.com)
* [Kostas Alexis](mailto:konstantinos.alexis@ntnu.no)
