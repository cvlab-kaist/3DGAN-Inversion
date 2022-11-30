import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
sys.path.append('/home/cvlab02/project/jaehoon/PTI_repo/Final')
from glob import glob
from torchvision import transforms
import argparse
from resnet.resnet import resnet34
from tqdm import tqdm
import math
import torch.nn as nn
import json
import dnnlib
import torch.nn.functional as F
from utils.camera_utils import euler2rot, compute_rotation_matrix_from_quaternion, rot6d_to_rotmat
import tensorboard
from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", dest="out_dir", required=True, help='Output directory') 
    parser.add_argument("--img_dir", dest="img_dir", required=True, help='Input image directory') 
    parser.add_argument("--val_dir", dest="val_dir", required=True, help='Validation image directory') 
    parser.add_argument("--lr", dest="lr", required=False, type=float, default=1e-4, help='Learning rate') 
    parser.add_argument("--camera_type", dest="camera_type", choices=['2', '4', '6'], required=True, help='Type of camera dimension')
    args = parser.parse_args()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    output_folder = args.out_dir
    root_dir = os.path.join(args.img_dir, '*.png')
    gt_dir = os.path.join(args.img_dir, 'pseudo_cam_gt.txt')
    val_dir = os.path.join(args.val_dir, '*.png')
    val_gt_dir = os.path.join(args.val_dir, 'val.json')
    lr = args.lr
    camera_type = args.camera_type

    save_dir = os.path.join('./', output_folder)
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().cuda()
    if os.path.isdir(save_dir) == 0:
        os.mkdir(save_dir)

    #open image label
    with open(gt_dir, 'r') as f:
        ext_box = []
        euler_box = []
        name_list = []
        while True:
            ext_list = []
            euler_list = []
            line = f.readline()
            if not line:
                break
            name_list.append(line[:9])
            tmp_list = line[11:-1].split(' ')
            
            for idx, tmp in enumerate(tmp_list):
                if idx<16:
                    ext_list.append(float(tmp))
                else:
                    if tmp != '':
                        euler_list.append(float(tmp))
            ext_box.append(np.array(ext_list))
            euler_box.append(np.array(euler_list))
    
    #open val image label
    with open(val_gt_dir, 'r') as f:
        val_ext_box = []
        val_name_list = []
        json_ = json.load(f)
        for instance in json_['labels']:
            val_ext_box.append(instance[1][:16])
            val_name_list.append(instance[0][6:-4])

    image_dir_list = glob(root_dir)
    val_dir_list = glob(val_dir)
    image_dir_list.sort()
    val_dir_list.sort()
    
    dataset = EG3D_Dataset(ext_box, image_dir_list, name_list, transform=preprocess)
    valset = EG3D_Dataset(val_ext_box, val_dir_list, val_name_list, transform=preprocess)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True) # e4e train batch가 1??
    val_dataset_loader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    model = resnet34(pretrained=True, output_dims=int(camera_type)).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter()

    rot_loss_record = 10000
    trans_loss_record = 10000
    iter=0
    epoch = 0
    radius = 2.7
    for params in vgg16.parameters():
        params.requires_grad=False
    while True:
        for i, batch in tqdm(enumerate(dataset_loader), desc='training resnet...'):
            img_batch = (batch['image'].cuda() + 1) / 2 * 255
            img_batch = img_batch.to(torch.float32)
            img_batch = F.interpolate(img_batch, size=(256, 256), mode='area')
            bs = img_batch.shape[0]
            _ = vgg16(img_batch, resize_images=False, return_lpips=True)
            ext_batch = batch['extrinsic'].float().cuda()
            
            pred = model(img_batch)
            if camera_type == '2':
                theta = math.pi/2 + pred[:, 0]
                phi = math.pi/2 + pred[:, 1]
                roll = torch.zeros(1, 1)
                pred_rotmat = euler2rot(theta, phi, roll, batch_size=1).reshape(-1, 4, 4)[:, :3, :3]
            elif camera_type == '4':
                pred_rotmat = compute_rotation_matrix_from_quaternion(pred) # quaternion
            else:
                pred_rotmat = rot6d_to_rotmat(pred)# 6d
            translation = -radius*pred_rotmat[:, :3, 2]
            pred_ext = torch.eye(4).unsqueeze(0).repeat(pred_rotmat.shape[0], 1, 1).cuda()
            pred_ext[:, :3, :3] = pred_rotmat
            pred_ext[:, :3, 3] = translation

            rot_loss = compute_geodesic_loss(pred_ext[:, :3, :3], ext_batch[:, :3, :3])
            trans_loss = nn.MSELoss()(pred_ext[:, :3, 3], ext_batch[:, :3, 3])/bs * 10
            
            diag = torch.mul(pred_ext[:, :3, :3], torch.eye(3).unsqueeze(0).repeat(pred_ext.shape[0], 1, 1).cuda())
            diag_norm = torch.abs(diag[:, :2, :2]) - torch.eye(2).unsqueeze(0).repeat(diag.shape[0],1,1).cuda()
            diag_norm = torch.pow(diag_norm, 2)
            zerobyzero = diag_norm[:, 0, 0]
            reg_loss = torch.sum(1/zerobyzero)/bs*1e-10 # Restrict rotmat from diagonal

            loss = rot_loss + trans_loss + reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%10==0:
                with torch.no_grad():
                    print()
                    print('rot_loss:', rot_loss.item())
                    print('reg_loss', reg_loss.item())
                    print('trans_loss:', trans_loss.item())
                    print(pred_ext[0])
                    print(ext_batch[0])
                    print('epoch: ',epoch)
            iter+=1

            if i%50==0:
                with torch.no_grad():
                    writer.add_scalar("Loss/rot_loss", rot_loss.item(), iter)
                    writer.add_scalar("Loss/trans_loss", trans_loss.item(), iter)
                    writer.add_scalar("Loss/reg_loss", reg_loss.item(), iter)

            #validation step
            if iter%1000==0:
                print('validating...')
                model.eval()
                with torch.no_grad():
                    val_rot_loss = 0
                    val_trans_loss = 0
                    val_length = len(val_dataset_loader)
                    for val in val_dataset_loader:
                        ext_val_gt = val['extrinsic'].float().cuda()
                        img_val = (val['image'].cuda() + 1) / 2 * 255
                        img_val = img_val.to(torch.float32)
                        img_val = F.interpolate(img_val, size=(256, 256), mode='area')
                        bs = img_val.shape[0]
                        _ = vgg16(img_val, resize_images=False, return_lpips=True)

                        pred = model(img_val) 
                        if camera_type == '2':
                            theta = math.pi/2 + pred[:, 0]
                            phi = math.pi/2 + pred[:, 1]
                            roll = torch.zeros(1, 1)
                            pred_rotmat = euler2rot(theta, phi, roll, batch_size=1).reshape(-1, 4, 4)[:, :3, :3]
                        elif camera_type == '4':
                            pred_rotmat = compute_rotation_matrix_from_quaternion(pred) # quaternion
                        else:
                            pred_rotmat = rot6d_to_rotmat(pred)# 6d
                        translation = -radius*pred_rotmat[:, :3, 2]
                        pred_ext = torch.eye(4).unsqueeze(0).repeat(pred_rotmat.shape[0], 1, 1).cuda()
                        pred_ext[:, :3, :3] = pred_rotmat
                        pred_ext[:, :3, 3] = translation
                        val_rot_loss += torch.norm(torch.bmm(torch.inverse(ext_val_gt[:, :3, :3]), pred_ext[:, :3, :3]) - torch.eye(3).reshape(-1, 3, 3).repeat(1, 1, 1).cuda())
                        val_rot_loss += compute_geodesic_loss(pred_ext[:, :3, :3], ext_val_gt[:, :3, :3])
                        val_trans_loss += nn.L1Loss()(pred_ext[:, :3, 3], ext_val_gt[:, :3, 3])*2
                    print(f'eval scores -> rot_loss : {val_rot_loss/val_length} / trans_loss : {val_trans_loss/val_length}')
                    
                    writer.add_scalar("Eval/val_rot_loss", val_rot_loss.item(), iter)
                    writer.add_scalar("Eval/val_trans_loss", val_trans_loss.item(), iter)

                    if val_rot_loss.item() + val_trans_loss.item() < rot_loss_record + trans_loss_record:
                        rot_loss_record = val_rot_loss.item()
                        trans_loss_record = val_trans_loss.item()
                        torch.save(model.state_dict(), os.path.join(save_dir, 'model_best.pt'))
                model.train()
            if iter%5000==0:
                torch.save(model.state_dict(), os.path.join(save_dir, f'model{iter}.pt'))
        epoch += 1

class EG3D_Dataset(Dataset):
    """EG3D dataset."""

    def __init__(self, txt_file, root_dir, name_list, transform=None):

        self.gt_extrinsic = txt_file
        self.root_dir = root_dir
        self.transform = transform
        self.name_list = name_list

    def __len__(self):
        
        return len(self.root_dir)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.name_list[idx]
        img_dir = self.root_dir[idx]
        image = Image.open(img_dir)
        extrinsic = np.reshape(self.gt_extrinsic[idx], (4,4))

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'extrinsic': extrinsic, 'name': img_name} 

        return sample


def compute_geodesic_loss(gt_r_matrix, out_r_matrix):
    theta = compute_geodesic_distance_from_two_matrices(gt_r_matrix, out_r_matrix)
    error = theta.mean()
    return error

def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )
    theta = torch.acos(cos)
    
    return theta

if __name__ == '__main__':
    main()

