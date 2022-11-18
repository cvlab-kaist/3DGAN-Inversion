from random import choice
from string import ascii_uppercase
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
import sys
sys.path.append('/home/cvlab02/project/jaehoon/PTI_repo/PTI_2drot_e4e_warp')
from configs import global_config, paths_config
import wandb

from training.coaches.multi_id_coach import MultiIDCoach
from training.coaches.coach import SingleIDCoach
from utils.ImagesDataset import ImagesDataset
from resnet.resnet2 import resnet50
from resnet.resnet import resnet34
from models.latent_encoder.psp import pSp2
import torch

def run_PTI(run_name='', use_wandb=False, use_multi_id_training=False):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = paths_config.using_GPU

    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name

    if use_wandb:
        run = wandb.init(project=paths_config.pti_results_keyword, reinit=True, name=global_config.run_name)
    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    embedding_dir_path = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}/{paths_config.pti_results_keyword}'
    os.makedirs(embedding_dir_path, exist_ok=True)

    dataset = ImagesDataset(paths_config.input_data_path, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    #initialize camera predictor
    model = resnet34()
    model2 = torch.nn.DataParallel(pSp2())
    init_dir = '/home/cvlab02/project/jaehoon/PTI_repo/PTI_2drot_e4e_warp/initializer'
    e4e_weight_dir = os.path.join(init_dir, 'e4e_24K.pt')
    weight_dir = os.path.join(init_dir, 'pose_estimator.pt')
    #import pdb; pdb.set_trace()
    model.load_state_dict(torch.load(weight_dir))
    model2.load_state_dict(torch.load(e4e_weight_dir, map_location='cpu'))
    encoder = model.cuda()
    w_encoder = model2.cuda()

    if use_multi_id_training:
        coach = MultiIDCoach(dataloader, use_wandb)
    else:
        coach = SingleIDCoach(dataloader, use_wandb)

    coach.train(encoder, w_encoder)

    return global_config.run_name


if __name__ == '__main__':
    run_PTI(run_name='', use_wandb=False, use_multi_id_training=False)
