#!/usr/bin/env python
import sys

import torch
import torch.nn
import torch.optim
from torch import autograd
from torch.nn.functional import avg_pool2d, interpolate, softmax
from torch.autograd import Variable
import numpy as np
import tqdm
import matplotlib.pyplot as plt

import config as c
import opts
import time

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

# Loading the flow-based model, as well as the target classifier
if c.load_file:
    model.load(c.load_file)
    black_box_target = model.load_target_model(type=c.target_arch)
    model.model.eval()

with torch.no_grad():
    for i_epoch in range(c.n_epochs):
        
        # Initializiation
        total_imgs = 0
        succ_imgs  = 0
        fail_list  = []
        succ_list  = []
        print_list = []
        data_iter  = iter(data.test_loader)
        L2_mean    = 0
        
        for i_batch, data_tuple in tqdm.tqdm(enumerate(data_iter),
                                             total=len(data.test_loader),
                                             leave=False,
                                             mininterval=1.,
                                             disable=True,
                                             ncols=83):
            success = False
            print('\nEvaluating {:d}'.format(i_batch), flush=True)
            
            # Getting the data and moving them to the GPU device
            x, y = data_tuple
            x    = x.cuda()
            y    = y.cuda()
            
            # Mapping the clean image to the flow-based model latent space (line 2 of Algorithm 1)
            z = model.model(x)
            
            # Initializing the shift vector \mu (line 1 of Algorithm 1)
            mu     = 0.001 * torch.randn([1, c.output_dim]).cuda()
            
            # Getting the target classifier prediction for the current image
            logits = black_box_target((x - data.means)/data.stds)
            probs  = softmax(logits, dim=1)
            
            # Check if the classifier is predicting the label correctly.
            # Otherwise, we skip the sample as it is already classified wrong.
            if torch.argmax(probs[0]) != y:
                print('\nSkipping the wrong example ', i_batch)
                continue
            
            # Adding this image to the total number of images
            total_imgs += 1
            
            # Main adversarial example generation loop
            for run_step in range(c.n_iter):
                
                # Generating a bunch of candidate points in the latent space based 
                # on a normal distribution with mean \mu and variance \simga^2 (line 4 of Algorithm 1)
                z_sample   = torch.randn([c.n_pop, c.output_dim]).cuda()
                modify_try = mu.repeat(c.n_pop, 1) + c.sigma * z_sample
                
                # Mapping the latent points back to the original image space (lines 5 and 6 of Algorithm 1 (f(z_k) part))
                x_hat_s    = torch.clamp(model.model(z + modify_try, rev=True), 0., 1.)
                
                # Checking whether the classifier is already fooled.
                if run_step % 10 == 0:
                    
                    # Mapping the candidate adversarial image to the image space
                    real_input_img = torch.clamp(model.model(z + mu, rev=True), 0., 1.)
                    
                    # Computing the adversarial and clean images distance
                    real_dist = real_input_img - x
                    
                    # Making sure that the perturbation lies within the deifned boundary
                    # Here, we use \ell_\inf, hence the torch.clamp function.
                    # One can easily extend this to other norms such as \ell_2.
                    real_clip_dist  = torch.clamp(real_dist, -c.epsi, c.epsi)
                    
                    # Adding the perturbation to the clean image
                    real_clip_input = real_clip_dist + x
                    
                    # Querying the target classifier
                    outputs_real = black_box_target((real_clip_input - data.means)/data.stds)
                    outputs_real = softmax(outputs_real, dim=1)
                    
                    # Checking whether the classifier is fooled and the perturbations are within the defined boundary.
                    if (torch.argmax(outputs_real) != y) and (torch.abs(real_clip_dist).max() <= c.epsi):
                        
                        # Adding the current image to the list of successfully attacked ones. 
                        succ_imgs += 1
                        success    = True
                        
                        print('\nClip image success images: ' + str(succ_imgs) + '  total images: ' + str(total_imgs))
                        
                        succ_list.append(i_batch)
                        print_list.append(run_step)
                        L2_mean += torch.sqrt(torch.mean(real_clip_dist ** 2))
                        
                        # Appending the successfully attacked images to a torch array for logging purposes.
                        if succ_imgs == 1:
                            clean_data_tot = x.clone().data.cpu()
                            adv_data_tot   = real_clip_input.clone().cpu()
                            label_tot      = y.clone().data.cpu()
                        else:
                            clean_data_tot = torch.cat((clean_data_tot, x.clone().data.cpu()), 0)
                            adv_data_tot   = torch.cat((adv_data_tot, real_clip_input.clone().cpu()), 0)
                            label_tot      = torch.cat((label_tot, y.clone().data.cpu()), 0)

                        break
                
                # Computing the perturbation (line 5 of Algorithm 1 (the proj. function part))
                dist       = x_hat_s - x
                
                # Ensuring that the perturbation lies within the defined boundary (line 6 of Algorithm 1 (the proj. function part))
                clip_dist  = torch.clamp(dist, -c.epsi, c.epsi)
                
                # Adding the correctly clipped perturbation to the original image (line 6 of Algorithm 1 (the proj. function part))
                clip_input = (clip_dist + x).view(c.n_pop, c.img_dims[0], c.img_dims[1], c.img_dims[2])
                
                # Initializing the one hot code for the correct label
                target_onehot = torch.zeros((1, c.num_classes)).cuda()
                target_onehot[0][y] = 1.
                target_onehot = target_onehot.repeat(c.n_pop, 1)
                
                # Querying the classifier with adversarial image candidates
                clip_input = clip_input.squeeze()
                outputs    = black_box_target((clip_input - data.means)/data.stds)
                outputs    = softmax(outputs, dim=1)
                
                # Computing the C&W loss for all candidate images (line 6 of Algorithm 1)
                real  = torch.log((target_onehot * outputs).sum(1) + 1e-10)
                other = torch.log(((1. - target_onehot) * outputs - target_onehot * 10000.).max(1)[0] + 1e-10)
                loss1 = torch.clamp(real - other, 0., 1000.)
                
                # Updating the shift vector \mu (lines 7, 8, and 9 of Algorithm 1)
                Reward = - 0.5 * loss1
                A      = (Reward - torch.mean(Reward))/(torch.std(Reward) + 1e-10) # (line 7 of Algorithm 1)
                mu    += (c.lr / (c.n_pop * c.sigma))*(torch.matmul(z_sample.view(c.n_pop, -1).t(), A.view(-1, 1))).view(1, -1)  # (lines 8 and 9 of Algorithm 1)
            
            # Logging
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

        print('\nAvg L2 norm: ', L2_mean/float(succ_imgs), flush=True)

        torch.save(clean_data_tot,'%s/clean_data_%s_%s_%s.pth' % ('adv_output', c.model, c.dataset, 'AdvFlow'))
        torch.save(adv_data_tot, '%s/adv_data_%s_%s_%s.pth' % ('adv_output', c.model, c.dataset, 'AdvFlow'))
        torch.save(label_tot, '%s/label_%s_%s_%s.pth' % ('adv_output', c.model, c.dataset, 'AdvFlow'))
