## Architechture
lpips_type = 'alex'
first_inv_type = 'w'
optim_type = 'adam'

## Locality regularization
latent_ball_num_of_samples = 1
locality_regularization_interval = 1
use_locality_regularization = False
regulizer_l2_lambda = 0.1
regulizer_lpips_lambda = 0.1
regulizer_alpha = 30

## Loss
pt_l2_lambda = 1
pt_lpips_lambda = 1

## Steps
LPIPS_value_threshold = 0.06
max_pti_steps = 400
first_inv_steps = 400
cam_preheat_steps = 50
max_images_to_invert = 10000

## Optimization
pti_learning_rate = 3e-4
first_inv_lr = 8e-3
e4e_lr = 1e-6
cam_lr_2d = 6e-6
cam_lr_quat = 6e-7
cam_lr_6d = 6e-6
translation_lr = 2e-4
train_batch_size = 1
use_last_w_pivots = False
cam_latent_lr = 7e-3
