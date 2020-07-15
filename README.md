# AdvFlow

This repository is the official implementation of [_AdvFlow: Inconspicuous Black-box Adversarial Attacks using Normalizing Flows_]().
A small part of this work, the Greedy AdvFlow, has been publish in [ICML Workshop on Invertible Neural Networks, Normalizing Flows, and Explicit Likelihood Models](https://invertibleworkshop.github.io/accepted_papers/pdfs/36.pdf).

<p align="center">
  <img width="640" height="307" src="./images/CelebA.png">
</p>
<p align="center">
    <em>Adversarial attack on VGG19 classifier trained to detect smiles in CelebA faces.</em>
</p>

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training Normalizing Flows

To train the a flow-based model, first set `mode = 'pre_training'`, and specify all relevant variables in `config.py`. Once specified, run this command:

```train
python train.py
```

## Attack Evaluation

To perform AdvFlow black-box adversarial attack, first set the `mode = 'attack'` in `config.py`.
Also, specify the dataset, target model architecture and path by setting the `dataset`, `target_arch`, 
and `target_weight_path` variables in `config.py`, respectively. Once specified, run:

```eval
python attack.py
```

for CIFAR-10, SVHN, and CelebA. For ImageNet, however, you need to run:

```eval
python attack_imagenet.py
```

Finally, you can run the Greedy AdvFlow by:

```eval
python attack_greedy.py
```

## Pre-trained Models

Pre-trained flow-based models as well as some target classifiers can be found [here]().

## Acknowledgement

This repository is mainly built upon [FrEIA, the Framework for Easily Invertible Architectures](https://github.com/VLL-HD/FrEIA), and [NATTACK](https://github.com/Cold-Winter/Nattack).
We thank the authors of these two repositories.

## Citation

If you have found our code or paper beneficial to your research, please consider citing them as:
```bash
@article{dolatabadi2020advflow,
  title={AdvFlow: Inconspicuous Black-box Adversarial Attacks using Normalizing Flows},
  author={Dolatabadi, Hadi M. and Erfani, Sarah and Leckie, Christopher},
  year={2020},
}
```

