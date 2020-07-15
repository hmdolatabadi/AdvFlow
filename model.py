import torch.optim
import torch.nn as nn
import numpy as np

from FrEIA.framework import *
from FrEIA.modules import *
import config as c
import classifier_loader as cld

nodes = [InputNode(*c.img_dims, name='inp')]

if c.dataset == 'cifar-10' or c.dataset == 'svhn' or c.dataset == 'CelebA' or c.dataset == 'ImageNet':

    ndim_x = c.output_dim

    def random_orthog(n):
        w = np.random.randn(n, n)
        w = w + w.T
        w, S, V = np.linalg.svd(w)
        return torch.FloatTensor(w)

    # Higher resolution convolutional part
    for k in range(c.high_res_blocks):
        nodes.append(Node(nodes[-1],
                          glow_coupling_layer, {'clamp': c.clamping, 'F_class': F_conv,
                          'F_args': {'channels_hidden': c.channels_hidden, 'batch_norm': c.batch_norm}},
                          name=F'conv_high_res_{k}'))

        nodes.append(Node(nodes[-1], permute_layer, {'seed': k}, name=F'permute_high_res_{k}'))

    nodes.append(Node(nodes[-1], i_revnet_downsampling, {}))

    # Lower resolution convolutional part
    for k in range(c.low_res_blocks):

        nodes.append(Node(nodes[-1], conv_1x1, {'M': random_orthog(3 * 4)}, name=F'conv_1x1_{k}'))

        nodes.append(Node(nodes[-1], glow_coupling_layer, {'clamp': c.clamping, 'F_class': F_conv,
                          'F_args': {'channels_hidden': c.channels_hidden, 'batch_norm': c.batch_norm}},
                          name=F'conv_low_res_{k}'))

        nodes.append(Node(nodes[-1], permute_layer, {'seed': k}, name=F'permute_low_res_{k}'))

    # Make the outputs into a vector, then split off 1/4 of the outputs for the
    # fully connected part
    nodes.append(Node(nodes[-1], flattening_layer, {}, name='flatten'))

    split_node = Node(nodes[-1],
                      Split1D, {'split_size_or_sections': (ndim_x // 4, 3 * ndim_x // 4), 'dim': 0}, name='split')

    nodes.append(split_node)

    # Fully connected part
    for k in range(c.n_blocks):
        nodes.append(Node(nodes[-1],
                          glow_coupling_layer, {'clamp': c.clamping,'F_class': F_fully_connected,
                          'F_args':{'dropout': c.fc_dropout, 'internal_size': c.internal_width}},
                          name=F'fully_connected_{k}'))

        nodes.append(Node(nodes[-1], permute_layer, {'seed': k}, name=F'permute_{k}'))

    # Concatenate the fully connected part and the skip connection to get a single output
    nodes.append(Node([nodes[-1].out0, split_node.out1], Concat1d, {'dim': 0}, name='concat'))

elif c.dataset == 'mnist':

    ndim_x = c.output_dim

    nodes.append(Node([nodes[-1].out0], flattening_layer, {}, name='flatten'))
    for i in range(c.n_blocks):
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': i}, name=F'permute_{i}'))
        nodes.append(Node([nodes[-1].out0], glow_coupling_layer,
                          {'clamp': c.clamping, 'F_class': F_fully_connected,
                           'F_args': {'dropout': c.fc_dropout, 'internal_size': c.internal_width}}, name=F'fc_{i}'))
else:
    raise ValueError('Dataset {} is not defined!'.format(c.dataset))

nodes.append(OutputNode([nodes[-1].out0], name='out'))


def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).cuda()
            # if len(split) > 3 and split[3][-1] == '3': # last convolution in the coeff func
            #     param.data.fill_(0.)

def save(name):
    save_dict = {'net': model.state_dict()}
    torch.save(save_dict, name)


def load(name):
    state_dicts = torch.load(name)
    model.load_state_dict(state_dicts['net'])


model = ReversibleGraphNet(nodes, verbose=False)
model.cuda()
init_model(model)

if c.train_from_scratch == True:

    params_trainable = list(filter(lambda p: p.requires_grad, model.parameters()))

    gamma = c.decay_by ** (1. / c.n_epochs)
    optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=gamma)

    def optim_step():
        optim.step()
        optim.zero_grad()


else:
    for param in model.parameters():
        param.requires_grad = False

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and m.affine:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def load_target_model(type='resnet50'):

    if type == 'resnet50':
        black_box_model = cld.load_pretrained_resnet50()
    elif type == 'wideresnet':
        black_box_model = cld.load_pretrained_wide_resnet()
    elif type == 'ImageNet':
        black_box_model = cld.load_pretrained_inception()
    else:
        raise ValueError('Undefined target model! Please check your input again!')

    for param in black_box_model.parameters():
        param.requires_grad = False

    black_box_model.cuda()
    black_box_model.eval()

    return black_box_model

