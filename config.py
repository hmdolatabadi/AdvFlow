#####################
# Which experiment: #
#####################

# Dataset to train
dataset = 'cifar-10'
mode    = 'attack'

#########
# Data: #
#########

if dataset == 'cifar-10' or dataset == 'svhn':
    img_dims   = (3, 32, 32)
    output_dim = img_dims[0] * img_dims[1] * img_dims[2]
elif dataset == 'ImageNet':
    img_dims   = (3, 64, 64)
    output_dim = img_dims[0] * img_dims[1] * img_dims[2]
elif dataset == 'CelebA':
    img_dims   = (3, 128, 128)
    output_dim = img_dims[0] * img_dims[1] * img_dims[2]
else:
    raise ValueError('Dataset {} is not defined!'.format(dataset))

add_image_noise = 0.02

#################
# Architecture: #
#################

# For Vanilla Glow Image generation:
if dataset == 'cifar-10' or dataset == 'svhn':

    high_res_blocks = 4
    n_blocks        = 6
    low_res_blocks  = 6
    internal_width  = 128
    clamping        = 1.5
    fc_dropout      = 0.0
    channels_hidden = 128
    num_classes     = 10
    batch_norm      = False

elif dataset == 'ImageNet':

    high_res_blocks = 4
    n_blocks        = 6
    low_res_blocks  = 6
    internal_width  = 128
    clamping        = 1.5
    fc_dropout      = 0.0
    channels_hidden = 128
    num_classes     = 1000
    batch_norm      = False
    org_size        = 299

elif dataset == 'CelebA':

    high_res_blocks = 4
    n_blocks        = 6
    low_res_blocks  = 6
    internal_width  = 128
    clamping        = 1.5
    fc_dropout      = 0.1
    channels_hidden = 128
    num_classes     = 2
    batch_norm      = False

else:
    raise ValueError('Dataset {} is not defined!'.format(dataset))

####################
# Logging/preview: #
####################

loss_names           = ['L', 'L_rev']
preview_upscale      = 3                    # Scale up the images for preview
sampling_temperature = 1.0                  # Sample at a reduced temperature for the preview
progress_bar         = True                 # Show a progress bar of each epoch

##############
# Attack mode:
##############
if mode == 'attack':

    train_from_scratch = False

    workers            = 0
    lr                 = 2e-2
    batch_size         = 1
    n_pop              = 20
    n_iter             = 500
    sigma              = 0.1
    epsi               = 8 / 255.
    n_epochs           = 1
    latent_noise       = 0.1
    init_scale         = 0.01
    model              = 'Free_Adv_Train'

    # Load pre-trained flow-based network
    load_file = './flows/{}_{}_blocks_Simple_Gaussian.pt'.format(dataset, n_blocks)

    # Load the target network
    target_arch        = 'wideresnet'
    target_weight_path = './target_models/cifar-10_wide_resnet_free.pth'

##############
# Train mode:
##############
elif mode == 'pre_training':

    train_from_scratch = True

    lr              = 1e-4
    batch_size      = 64
    decay_by        = 0.01
    weight_decay    = 1e-5
    betas           = (0.9, 0.999)
    do_rev          = False
    do_fwd          = True
    n_epochs        = 350
    n_its_per_epoch = 2 ** 10
    init_scale      = 0.03
    pre_low_lr      = 1
    latent_noise    = 0.1

    # Save parameters under this name
    filename = './flows/{}_{}_blocks_Simple_Gaussian_New.pt'.format(dataset, n_blocks)

    checkpoint_save_interval  = 20
    checkpoint_save_overwrite = True  # Overwrite each checkpoint with the next one
    checkpoint_on_error       = True  # Write out a checkpoint if the training crashes
