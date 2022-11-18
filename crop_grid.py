from email.mime import image
from PIL import Image
import glob
import os
import numpy as np
import random

input_dir = '/home/cvlab02/project/jaehoon/PTI_repo/PTI_2drot_e4e_warp/w_1st/*.png'
output_dir = '/home/cvlab02/project/jaehoon/PTI_repo/PTI_2drot_e4e_warp/fid'
img_list = glob.glob(input_dir)
crop_every = False
random_sample = True
for img in img_list:
    type_ = img.split('/')[-3]
    method_name = img.split('/')[-2]
    image_name = img[-10:-4]
    grid = np.array(Image.open(img).convert('RGB'))
    if os.path.isdir(output_dir + '/'+ method_name) == 0:
        os.mkdir(output_dir + '/'+ method_name)
    #import pdb; pdb.set_trace()
    # 세로 가로 3
    #2:514      1
    #516:1028   2
    #1030:1542  3
    #1544:2056  4
    #2058:2570  5
    #2572:3084  6
    #3086:3598  7
    if random_sample:
        sample = random.randrange(1,14)
        if sample == 1:
            Image.fromarray(grid[2:514, 1030:1542, :]).save(output_dir + '/'+ method_name + '/' +  image_name   + '.png')
        elif sample == 2:
            Image.fromarray(grid[2:514, 1544:2056, :]).save(output_dir + '/'+ method_name + '/' +  image_name   + '.png')
        elif sample == 3:
            Image.fromarray(grid[2:514, 2058:2570, :]).save(output_dir + '/'+ method_name + '/' +  image_name   + '.png')
        elif sample == 4:
            Image.fromarray(grid[516:1028, 2:514, :]).save(output_dir + '/'+ method_name + '/' +  image_name   + '.png')
        elif sample == 5:
            Image.fromarray(grid[516:1028, 516:1028, :]).save(output_dir + '/'+ method_name + '/' +  image_name   + '.png')
        elif sample == 6:
            Image.fromarray(grid[516:1028, 1030:1542, :]).save(output_dir + '/'+ method_name + '/' +  image_name   + '.png')
        elif sample == 7:
            Image.fromarray(grid[516:1028, 1544:2056, :]).save(output_dir + '/'+ method_name + '/' +  image_name   + '.png')
        elif sample == 8:
            Image.fromarray(grid[516:1028, 2058:2570, :]).save(output_dir + '/'+ method_name + '/' +  image_name   + '.png')
        elif sample == 9:
            Image.fromarray(grid[1544:2056, 2:514, :]).save(output_dir + '/'+ method_name + '/' +  image_name   + '.png')
        elif sample == 10:
            Image.fromarray(grid[1544:2056, 516:1028, :]).save(output_dir + '/'+ method_name + '/' +  image_name   + '.png')
        elif sample == 11:
            Image.fromarray(grid[1544:2056, 1030:1542, :]).save(output_dir + '/'+ method_name + '/' +  image_name   + '.png')
        elif sample == 12:
            Image.fromarray(grid[2058:2570, 1544:2056, :]).save(output_dir + '/'+ method_name + '/' +  image_name   + '.png')
        elif sample == 13:
            Image.fromarray(grid[2058:2570, 2058:2570, :]).save(output_dir + '/'+ method_name + '/' +  image_name   + '.png')
    elif crop_every:
        Image.fromarray(grid[2:514, 2:514, :]).save(output_dir + '/' + image_name + '1' + method_name  + '.png')
        Image.fromarray(grid[2:514, 516:1028, :]).save(output_dir + '/' + image_name + '2' + method_name  + '.png')
        Image.fromarray(grid[2:514, 1030:1542, :]).save(output_dir + '/' + image_name + '3' + method_name  + '.png')
        Image.fromarray(grid[2:514, 1544:2056, :]).save(output_dir + '/' + image_name + '4' + method_name  + '.png')
        Image.fromarray(grid[2:514, 2058:2570, :]).save(output_dir + '/' + image_name + '5' + method_name  + '.png')
        Image.fromarray(grid[516:1028, 2:514, :]).save(output_dir + '/' + image_name + '6' + method_name  + '.png')
        Image.fromarray(grid[516:1028, 516:1028, :]).save(output_dir + '/' + image_name + '7' + method_name  + '.png')
        Image.fromarray(grid[516:1028, 1030:1542, :]).save(output_dir + '/' + image_name + '8' + method_name  + '.png')
        Image.fromarray(grid[516:1028, 1544:2056, :]).save(output_dir + '/' + image_name + '9' + method_name  + '.png')
        Image.fromarray(grid[516:1028, 2058:2570, :]).save(output_dir + '/' + image_name + '10' + method_name  + '.png')
        Image.fromarray(grid[1030:1542, 2:514, :]).save(output_dir + '/' + image_name + '11' + method_name  + '.png')
        Image.fromarray(grid[1030:1542, 516:1028, :]).save(output_dir + '/' + image_name + '12' + method_name  + '.png')
        Image.fromarray(grid[1030:1542, 1030:1542, :]).save(output_dir + '/' + image_name + '13' + method_name  + '.png')
        Image.fromarray(grid[1030:1542, 1544:2056, :]).save(output_dir + '/' + image_name + '14' + method_name  + '.png')
        Image.fromarray(grid[1030:1542, 2058:2570, :]).save(output_dir + '/' + image_name + '15' + method_name  + '.png')
        Image.fromarray(grid[1544:2056, 2:514, :]).save(output_dir + '/' + image_name + '16' + method_name  + '.png')
        Image.fromarray(grid[1544:2056, 516:1028, :]).save(output_dir + '/' + image_name + '17' + method_name  + '.png')
        Image.fromarray(grid[1544:2056, 1030:1542, :]).save(output_dir + '/' + image_name + '18' + method_name  + '.png')
        Image.fromarray(grid[1544:2056, 1544:2056, :]).save(output_dir + '/' + image_name + '19' + method_name  + '.png')
        Image.fromarray(grid[1544:2056, 2058:2570, :]).save(output_dir + '/' + image_name + '20' + method_name  + '.png')
        Image.fromarray(grid[2058:2570, 2:514, :]).save(output_dir + '/' + image_name + '21' + method_name  + '.png')
        Image.fromarray(grid[2058:2570, 516:1028, :]).save(output_dir + '/' + image_name + '22' + method_name  + '.png')
        Image.fromarray(grid[2058:2570, 1030:1542, :]).save(output_dir + '/' + image_name + '23' + method_name  + '.png')
        Image.fromarray(grid[2058:2570, 1544:2056, :]).save(output_dir + '/' + image_name + '24' + method_name  + '.png')
        Image.fromarray(grid[2058:2570, 2058:2570, :]).save(output_dir + '/' + image_name + '25' + method_name  + '.png')
        
    else:
        Image.fromarray(grid[2:514, 516:1028, :]).save(output_dir + image_name + '_recon' + method_name  + '.png') #2 (recon)
        Image.fromarray(grid[2:514, 1030:1542, :]).save(output_dir + image_name + '_canonical' + method_name + '.png') #3 (canonical)
        if type_=='five':
            Image.fromarray(grid[516:1028, 2058:2570, :]).save(output_dir + image_name + '_right' + method_name + '.png') # 10 right
            Image.fromarray(grid[1544:2056, 2:514, :]).save(output_dir + image_name + '_left' + method_name + '.png') #16 left
        else:
            Image.fromarray(grid[516:1028, 1030:1542, :]).save(output_dir + image_name + '_right' + method_name + '.png') # 10 right
            Image.fromarray(grid[1030:1542, 516:1028, :]).save(output_dir + image_name + '_left' + method_name + '.png') #16 left