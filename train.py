#!/usr/bin/env python
import sys

import torch
import torch.nn
import torch.optim
from torch.nn.functional import avg_pool2d, interpolate
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


class dummy_loss(object):
    def item(self):
        return 1.


def sample_outputs(sigma):
    return sigma * torch.cuda.FloatTensor(c.batch_size, c.output_dim).normal_()

def img_tile(imgs, row_col = None, transpose = False, channel_first=True, channels=3):
    '''
    tile a list of images to a large grid.
    imgs:       iterable of images to use
    row_col:    None (automatic), or tuple of (#rows, #columns)
    transpose:  Wheter to stitch the list of images row-first or column-first
    channel_first: if true, assume images with CxWxH, else WxHxC
    channels:   3 or 1, number of color channels
    '''

    if row_col == None:
        sqrt = np.sqrt(len(imgs))
        rows = np.floor(sqrt)
        delt = sqrt - rows
        cols = np.ceil(rows + 2*delt + delt**2 / rows)
        rows, cols = int(rows), int(cols)
    else:
        rows, cols = row_col

    if channel_first:
        h, w = imgs[0].shape[1], imgs[0].shape[2]
    else:
        h, w = imgs[0].shape[0], imgs[0].shape[1]

    show_im = np.zeros((rows*h, cols*w, channels))

    if transpose:
        def iterator():
            for i in range(rows):
                for j in range(cols):
                    yield i, j

    else:
        def iterator():
            for j in range(cols):
                for i in range(rows):
                    yield i, j

    k = 0
    for i, j in iterator():

            im = imgs[k]
            if channel_first:
                im = np.transpose(im, (1, 2, 0))

            show_im[h*i:h*i+h, w*j:w*j+w] = im

            k += 1
            if k == len(imgs):
                break

    return np.squeeze(show_im)

try:

    fixed_noise = sample_outputs(1.0)

    for i_epoch in range(-c.pre_low_lr, c.n_epochs):

        loss_history = []
        data_iter = iter(data.train_loader)

        if i_epoch < 0:
            for param_group in model.optim.param_groups:
                param_group['lr'] = c.lr * 2e-2

        for i_batch, data_tuple in tqdm.tqdm(enumerate(data_iter),
                                             total=min(len(data.train_loader), c.n_its_per_epoch),
                                             leave=False,
                                             mininterval=1.,
                                             disable=(not c.progress_bar),
                                             ncols=83):


            x, y = data_tuple
            x    = x.cuda()
            x   += c.add_image_noise * torch.cuda.FloatTensor(x.shape).normal_()

            output = model.model(x)

            if c.do_fwd:
                zz  = torch.sum(output**2, dim=1)
                jac = model.model.log_jacobian(run_forward=False)

                neg_log_likeli = 0.5 * zz - jac

                l = torch.mean(neg_log_likeli)
                l.backward(retain_graph=c.do_rev)
            else:
                l = dummy_loss()

            if c.do_rev:
                samples_noisy = sample_outputs(c.latent_noise) + output.data

                x_rec = model.model(samples_noisy, rev=True)
                l_rev = torch.mean((x-x_rec)**2)
                l_rev.backward()
            else:
                l_rev = dummy_loss()

            model.optim_step()
            loss_history.append([l.item(), l_rev.item()])

            if i_batch+1 >= c.n_its_per_epoch:
                # somehow the data loader workers don't shut down automatically
                try:
                    data_iter._shutdown_workers()
                except:
                    pass

                break

        model.weight_scheduler.step()

        epoch_losses    = np.mean(np.array(loss_history), axis=0)
        epoch_losses[0] = min(epoch_losses[0], 0)

        if i_epoch > 1 - c.pre_low_lr:
            print(epoch_losses, flush=True)

        model.model.zero_grad()

        if (i_epoch % c.checkpoint_save_interval) == 0:
            model.save(c.filename + '_checkpoint_%.4i' % (i_epoch * (1-c.checkpoint_save_overwrite)))
            with torch.no_grad():
                rev_imgs    = model.model(fixed_noise, rev=True)
                rev_imgs    = torch.clamp(rev_imgs, 0., 1.)
                imgs        = [rev_imgs[i].cpu().data.numpy() for i in range(c.batch_size)]
                imgs        = img_tile(imgs, (8, 8), transpose=False, channel_first=True, channels=3)
                plt.imsave(F'./training_images/random_samples_{i_epoch}.png', imgs, vmin=0, vmax=1, dpi=300)

    model.save(c.filename)

except:
    if c.checkpoint_on_error:
        model.save(c.filename + '_ABORT')

    raise
