# This code is taken from mister_ed repository for PyTorch adversarial attack
# https://github.com/revbucket/mister_ed
import torch
from wide_resnets import Wide_ResNet
from resnet import ResNet50
import os
import re
from torch import nn
from torchvision.models import inception_v3
import configs as c

WEIGHT_PATH = './target_models/'

##############################################################################
#                                                                            #
#                               MODEL LOADER                                 #
#                                                                            #
##############################################################################


def load_pretrained_wide_resnet():
    """ Helper fxn to initialize/load a pretrained wideresnet """

    state_dict     = torch.load(c.target_weight_path)['state_dict']
    classifier_net = Wide_ResNet(depth=34, widen_factor=10, num_classes=10, dropout_rate=0.1)
    classifier_net = torch.nn.DataParallel(classifier_net)
    classifier_net.apply(weights_init)
    classifier_net.load_state_dict(state_dict, strict=True)

    return classifier_net


def load_pretrained_cifar_resnet50():
    """ Helper fxn to initialize/load a pretrained resnet-50 """

    state_dict     = torch.load(c.target_weight_path)['state_dict']
    classifier_net = ResNet50()
    classifier_net = torch.nn.DataParallel(classifier_net)
    classifier_net.apply(weights_init)
    classifier_net.load_state_dict(state_dict, strict=True)

    return classifier_net


def load_pretrained_ImageNet():

    classifier_net = inception_v3(pretrained=True)
    classifier_net = torch.nn.DataParallel(classifier_net)

    return classifier_net

##############################################################################
#                                                                            #
#                               Normalizer                                   #
#                                                                            #
##############################################################################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and m.affine:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



