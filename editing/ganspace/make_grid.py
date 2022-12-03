from torchvision import transforms
from torchvision.transforms import ToPILImage
from PIL import Image
import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt

# Read an image from a file
def image_reader(image_path, resize=None):
    with open(image_path, "rb") as f:
        image = Image.open(f)
        image = image.convert("RGB")
    if resize != None:
        image = image.resize((resize, resize))
    transform = transforms.Compose([
        transforms.ToTensor() # [0, 1]
    ])
    image = transform(image)
    image = image.unsqueeze(0) # (N, C, H, W)
    return image

def make_grid_fn(source_dir, save_dir, file_name):
    inter_imgs_dir = os.listdir(source_dir)
    inter_imgs = torch.tensor([])
    for idx in range(len(inter_imgs_dir)):
        total_path = os.path.join(source_dir,'ganspace'+str(idx)+'.png')
        img = image_reader(total_path, resize=512)
        inter_imgs = torch.cat([inter_imgs, img], dim=0)

    result_image = ToPILImage()(make_grid(inter_imgs))
    result_image.save(os.path.join(save_dir, file_name + ".png"))


