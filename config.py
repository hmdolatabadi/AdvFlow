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
elif dataset == 'ImageNet':
    img_dims   = (3, 64, 64)
elif dataset == 'CelebA':
    img_dims   = (3, 128, 128)
else:
    raise ValueError('Dataset {} is not defined!'.format(dataset))

output_dim      = img_dims[0] * img_dims[1] * img_dims[2]
add_image_noise = 0.02

#################
# Architecture: #
#################

# Flow-based model architecture
if dataset == 'cifar-10' or dataset == 'svhn':

    high_res_blocks = 4     # Number of high-resolution, convolutional blocks
    low_res_blocks  = 6     # Number of low-resolution, convolutional blocks
    channels_hidden = 128   # Number of hidden channels for the convolutional blocks
    batch_norm      = False # Batch normalization?
    
    n_blocks        = 6     # Number of fully-connected blocks
    internal_width  = 128   # Internal width of the FC blocks
    fc_dropout      = 0.0   # Dropout for FC blocks
    
    clamping        = 1.5   # Clamping parameter for avoiding exploding exponentiation
    
    num_classes     = 10    # Number of classes

elif dataset == 'ImageNet':
    
    high_res_blocks = 4     # Number of high-resolution, convolutional blocks
    low_res_blocks  = 6     # Number of low-resolution, convolutional blocks
    channels_hidden = 128   # Number of hidden channels for the convolutional blocks
    batch_norm      = False # Batch normalization?
    
    n_blocks        = 6     # Number of fully-connected blocks
    internal_width  = 128   # Internal width of the FC blocks
    fc_dropout      = 0.0   # Dropout for FC blocks
    
    clamping        = 1.5   # Clamping parameter for avoiding exploding exponentiation
    
    num_classes     = 1000  # Number of classes
    org_size        = 299   # Image-size to re-shape the ImageNet data

elif dataset == 'CelebA':

    high_res_blocks = 4     # Number of high-resolution, convolutional blocks
    low_res_blocks  = 6     # Number of low-resolution, convolutional blocks
    channels_hidden = 128   # Number of hidden channels for the convolutional blocks
    batch_norm      = False # Batch normalization?
    
    n_blocks        = 6     # Number of fully-connected blocks
    internal_width  = 128   # Internal width of the FC blocks
    fc_dropout      = 0.1   # Dropout for FC blocks
    
    clamping        = 1.5   # Clamping parameter for avoiding exploding exponentiation
    
    num_classes     = 10    # Number of classes

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

# Attack mode hyperparameters
if mode == 'attack':

    train_from_scratch = False
    
    workers            = 1                 # Dataloader workers
    batch_size         = 1                 # Batch-size (It has to be 1 for the current version. We plan to release the parallel attack in the near future.)
    n_epochs           = 1                 # Number of epochs to go over the test set
    
    lr                 = 2e-2              # Attack learning rate                 
    n_pop              = 20                # The population size of the number of adversarial images generated in the latent space
    n_iter             = 500               # The number of iterations to query the classifier. To get the total number of queries, this number is multiplied by n_pop.
    sigma              = 0.1               # Standard deviation of the latent space Gaussian
    epsi               = 8 / 255           # Maximum pixel deviation (\ell_\inf)
    K                  = 4                 # The number of samples over which we take the average and update \mu (for Greedy Attack)
    
    model              = 'Free_Adv_Train'  # Name of the target classifier

    # Load pre-trained flow-based network
    load_file = './flows/{}_{}_blocks_Simple_Gaussian.pt'.format(dataset, n_blocks)

    # Load the target classifier network
    target_arch        = 'wideresnet'
    target_weight_path = './target_models/cifar-10_wide_resnet_free.pth'

##############
# Train mode:
##############

# Pre-training mode hyperparameters
elif mode == 'pre_training':

    train_from_scratch = True
    
    workers         = 1              # Dataloader workers
    batch_size      = 64             # Batch-size
    n_epochs        = 350            # Number of training epochs
    
    lr              = 1e-4           # Initial learning rate of the optimizer
    decay_by        = 0.01           # Learning rate decay
    weight_decay    = 1e-5           # Weight decay of the optimizer
    betas           = (0.9, 0.999)   # Beta parameters of the Adam optimizer
    n_its_per_epoch = 2 ** 10        # Maximum number of training iterations per epoch
    
    do_rev          = False          # Adding the reconstruction error to the objective
    do_fwd          = True           # The usual log-likelihood training
    
    init_scale      = 0.03
    pre_low_lr      = 1
    latent_noise    = 0.1

    # Save parameters under this name
    filename = './flows/{}_{}_blocks_Simple_Gaussian_New.pt'.format(dataset, n_blocks)

    checkpoint_save_interval  = 20
    checkpoint_save_overwrite = True  # Overwrite each checkpoint with the next one
    checkpoint_on_error       = True  # Write out a checkpoint if the training crashes
