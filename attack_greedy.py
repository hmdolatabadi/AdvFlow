#!/usr/bin/env python
import sys

import torch
import torch.nn
import torch.optim
from torch.nn.functional import avg_pool2d, interpolate, softmax
from torch.autograd import Variable
import numpy as np
import tqdm
import matplotlib.pyplot as plt

import config as c
import opts

opts.parse(sys.argv)
config_str = ""
config_str += "==="*30 + "\n"
config_str += "Config options:\n\n"

for v in dir(c):
    if v[0]=='_': continue
    s=eval('c.%s'%(v))
    config_str += "  {:25}\t{}\n".format(v,s)

config_str += "==="*30 + "\n"

print(config_str)

import model
import data


if c.load_file:
    model.load(c.load_file)
    black_box_target = model.load_target_model(type=c.target_arch)
    model.model.eval()

with torch.no_grad():
    for i_epoch in range(c.n_epochs):

        total_imgs = 0
        succ_imgs  = 0
        fail_list  = []
        succ_list  = []
        print_list = []
        data_iter  = iter(data.test_loader)

        for i_batch, data_tuple in tqdm.tqdm(enumerate(data_iter),
                                             total=len(data.test_loader),
                                             leave=False,
                                             mininterval=1.,
                                             disable=(not c.progress_bar),
                                             ncols=83):
            success = False
            print('\nEvaluating {:d}'.format(i_batch), flush=True)

            x, y = data_tuple
            x    = x.cuda()
            y    = y.cuda()
            z    = model.model(x)

            mu     = 0.001 * torch.randn([1, c.output_dim]).cuda()
            logits = black_box_target((x - data.means)/data.stds)
            probs  = softmax(logits, dim=1)

            if torch.argmax(probs[0]) != y:
                print('\nSkipping the wrong example ', i_batch)
                continue

            total_imgs += 1

            for run_step in range(c.n_iter):

                z_sample   = torch.randn([c.n_pop, c.output_dim]).cuda()
                modify_try = mu.repeat(c.n_pop, 1) + c.sigma * z_sample
                x_hat_s    = torch.clamp(model.model(z + modify_try, rev=True), 0., 1.)

                dist       = x_hat_s - x
                clip_dist  = torch.clamp(dist, -c.epsi, c.epsi)
                clip_input = torch.clamp((clip_dist + x).view(c.n_pop, 3, 32, 32), 0., 1.)

                target_onehot = torch.zeros((1, 10)).cuda()
                target_onehot[0][y] = 1.

                clip_input = clip_input.squeeze()
                outputs    = black_box_target((clip_input - data.means)/data.stds)
                outputs    = softmax(outputs, dim=1)

                target_onehot = target_onehot.repeat(c.n_pop, 1)

                real  = torch.log((target_onehot * outputs).sum(1) + 1e-10)
                other = torch.log(((1. - target_onehot) * outputs - target_onehot * 10000.).max(1)[0] + 1e-10)
                loss1 = torch.clamp(real - other, 0., 1000.)
                idx   = torch.topk(loss1, k=4, largest=False)[1]
                mu    = torch.mean(modify_try[idx], dim=0, keepdim=True)

                if loss1[idx[0]] == 0.0:

                    real_input_img = torch.clamp(model.model(z + modify_try[idx[0]], rev=True), 0., 1.)
                    real_dist      = real_input_img - x

                    real_clip_dist  = torch.clamp(real_dist, -c.epsi, c.epsi)
                    real_clip_input = real_clip_dist + x

                    outputs_real = black_box_target((real_clip_input - data.means)/data.stds)
                    outputs_real = softmax(outputs_real, dim=1)

                    if (torch.argmax(outputs_real) != y) and (torch.abs(real_clip_dist).max() <= c.epsi):

                        succ_imgs += 1
                        success    = True
                        print('\nClip image success images: ' + str(succ_imgs) + '  total images: ' + str(total_imgs))
                        succ_list.append(i_batch)
                        print_list.append(run_step)
                        diff = torch.max(torch.abs(model.model(real_clip_input) - model.model(x)))

                        if succ_imgs == 1:
                            clean_data_tot = x.clone().data.cpu()
                            adv_data_tot   = real_clip_input.clone().cpu()
                            label_tot      = y.clone().data.cpu()
                        else:
                            clean_data_tot = torch.cat((clean_data_tot, x.clone().data.cpu()), 0)
                            adv_data_tot   = torch.cat((adv_data_tot, real_clip_input.clone().cpu()), 0)
                            label_tot      = torch.cat((label_tot, y.clone().data.cpu()), 0)

                        break

            if not success:
                fail_list.append(i_batch)
                print('\nFailed!', flush=True)
            else:
                print('\nSucceed!', flush=True)

        print(fail_list)
        success_rate = succ_imgs/float(total_imgs)
        
        # note that this is the number of steps.
        # to get number of queries you have to multiply the vector by c.n_pop
        print('\nRun steps: ', print_list, flush=True)
        np.savez('runstep', print_list)

        print('\nAttack success rate: ', success_rate, flush=True)

        torch.save(clean_data_tot,'%s/clean_data_%s_%s_%s.pth' % ('adv_output', c.model, c.dataset, 'AdvFlow_Greedy'))
        torch.save(adv_data_tot, '%s/adv_data_%s_%s_%s.pth' % ('adv_output', c.model, c.dataset, 'AdvFlow_Greedy'))
        torch.save(label_tot, '%s/label_%s_%s_%s.pth' % ('adv_output', c.model, c.dataset, 'AdvFlow_Greedy'))
