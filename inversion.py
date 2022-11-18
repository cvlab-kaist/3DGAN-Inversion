from random import choice
from string import ascii_uppercase
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os

from configs import global_config, paths_config
from training.coaches.coach import Coach
from utils.ImagesDataset import ImagesDataset


def run_PTI(run_name=''):
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = paths_config.using_GPU

    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name

    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    embedding_dir_path = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}/{paths_config.pti_results_keyword}'
    os.makedirs(embedding_dir_path, exist_ok=True)

    dataset = ImagesDataset(paths_config.input_data_path, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


    coach = Coach(dataloader)
    coach.train()

    return global_config.run_name


if __name__ == '__main__':
    run_PTI(run_name='')
