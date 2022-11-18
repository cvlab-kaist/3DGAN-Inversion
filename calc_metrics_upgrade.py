from PIL import Image
import glob
import os
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torchvision.transforms as transforms
from glob import glob
from natsort import natsorted

from criteria import l2_loss
from criteria import id_loss
from lpips import LPIPS
from pytorch_msssim import ms_ssim
lpips_type = 'alex'
lpips_loss = LPIPS(net=lpips_type).cuda().eval()
id_loss = id_loss.IDLoss().cuda().eval()


def parse_args():
	parser = ArgumentParser(add_help=False)
	parser.add_argument('--data_path', type=str, default='pose_1st')
	args = parser.parse_args()
	return args

# 

args = parse_args()
transform = transforms.Compose([transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

print("Loading Dataset")

grids = [os.path.join(args.data_path, file) for file in os.listdir(args.data_path) if file.endswith('.png')]


ls_mse = []
ls_lpips = []
ls_msssim = []
ls_identity = []

for grid in tqdm(grids):
    grid = np.array(Image.open(grid).convert('RGB'))
    input = grid[2:514, 2:514, :] / 255
    input = torch.from_numpy(input).float().permute(2, 0, 1).to('cuda')
    input = torch.unsqueeze(input, 0)
    output = grid[2:514, 516:1028, :] / 255
    output = torch.from_numpy(output).float().permute(2, 0, 1).to('cuda')
    output = torch.unsqueeze(output, 0)

    m_mse = l2_loss.l2_loss(output, input).item()
    m_lpips = lpips_loss(output, input).item()
    m_msssim = ms_ssim(output, input, data_range=1, size_average=False ).item()
    m_identity = 1 - id_loss(output, input).item()
    ls_mse.append(m_mse)
    ls_lpips.append(m_lpips)
    ls_msssim.append(m_msssim)
    ls_identity.append(m_identity)

if len(ls_mse) > 0:
    print("MSE: {}".format(np.mean(ls_mse)))
    print("LPIPS: {}".format(np.mean(ls_lpips)))
    print("MSSSIM: {}".format(np.mean(ls_msssim)))
    print("Identity: {}".format(np.mean(ls_identity)))

else:
    print("No data found")
    exit()