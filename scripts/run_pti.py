from random import choice
from string import ascii_uppercase
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
import sys
sys.path.append('./')
from configs import global_config, paths_config

from training.coaches.single_id_coach import SingleIDCoach
from utils.ImagesDataset import ImagesDataset
from resnet.resnet import resnet34
from models.e4e.psp import pSp2
import torch

def run_PTI(run_name='', use_wandb=False, use_multi_id_training=False):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = paths_config.using_GPU

    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name

    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    dataset = ImagesDataset(paths_config.input_data_path, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    E = torch.nn.DataParallel(pSp2())
    e4e_weight_dir = os.path.join(paths_config.initializer, 'e4e_ffhq.pt')
    #initialize latent and pose predictor
    if global_config.use_quaternions:
        P = resnet34(4)
        weight_dir = os.path.join(paths_config.initializer, 'pose_estimator_quat.pt')
    elif global_config.use_6d: # for afhq
        P = resnet34(6)
        weight_dir = os.path.join(paths_config.initializer, 'pose_estimator_afhq.pt')
        e4e_weight_dir = os.path.join(paths_config.initializer, 'e4e_afhq.pt')
    else:
        P = resnet34(2)
        weight_dir = os.path.join(paths_config.initializer, 'pose_estimator.pt')
    

    P.load_state_dict(torch.load(weight_dir, map_location='cpu'))
    E.load_state_dict(torch.load(e4e_weight_dir, map_location='cpu'))
    P = P.cuda()
    E = E.cuda()

    coach = SingleIDCoach(dataloader, use_wandb)
    coach.train(P, E)

    return global_config.run_name


if __name__ == '__main__':
    run_PTI(run_name='', use_wandb=False, use_multi_id_training=False)
