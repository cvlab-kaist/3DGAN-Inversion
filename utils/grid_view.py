import math
import torch


def look_at(grid_num=5, radius=2.7):
    '''
    num_grid should be odd number!
    '''
    at=torch.Tensor([0, 0, 0])
    up=torch.Tensor([0, 0, 1])
    eyes = gen_eyes(grid_num=grid_num)

    mats = []
    for eye in eyes:
        tmp = torch.Tensor([0,0,0,1])
        z_axis = eye - at
        x_axis = torch.cross(up, z_axis)
        x_axis = x_axis/torch.norm(x_axis)
        y_axis = torch.cross(z_axis, x_axis)
        y_axis = y_axis/torch.norm(y_axis)
        mat = torch.stack([x_axis, y_axis, z_axis], dim = -1)
        #x_axis, y_axis, z_axis/torch.norm(z_axis)

        ## For use of EG3D cam convention
        x, y, z = -mat[0], -mat[1], -mat[2]
        mat_revise = torch.stack([x,z,y], dim=0)

        mat_tmp = mat_revise.clone()

        loc = -mat_tmp[:, 2]*radius

        mat_revise = torch.cat([mat_revise, loc.unsqueeze(1)], dim=1)
        mat_revise = mat_revise.view(1,12).squeeze(0)
        mat_16 = torch.cat([mat_revise, tmp], dim=0)
        mats.append(mat_16)
    
    mats = torch.stack(mats, dim=0)
    return mats


def gen_eyes(grid_num=5):
    half = int((grid_num)/2+1) 

    all_xyz = []
    for i in range(half):
        if i == 0:
            num = 1
            x,y,z = 0,1,0
            xyz = torch.Tensor([x,y,z])
            all_xyz.append(xyz)
        else:
            num = int(8*i)
            y = math.cos(math.pi/8/(half-1)*i)
            y_sin = math.sin(math.pi/8/(half-1)*i)
            for i in range(num):
                x = y_sin * math.cos(2*math.pi/num*(i+1))
                z = y_sin * math.sin(2*math.pi/num*(i+1))
                xyz = torch.Tensor([x,y,z])
                all_xyz.append(xyz)

    all_xyz = torch.stack(all_xyz, dim=0)
    return all_xyz