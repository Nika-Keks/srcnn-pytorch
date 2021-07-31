from PIL import Image
from torch.functional import Tensor
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import os

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def image2tensor(image: Image) -> Tensor:
        y, _, _ = image.split()        
        image_tensor = transforms.ToTensor()(y)
        return image_tensor


class CustomImageDataset(data.Dataset):
    '''
        Custom Dataset class for buffered loading images
    '''

    def __init__(self, dir_path: str, file_exten: str = "png", lr_folder: str = "LRx2", gt_folder: str = "GT", num_rotation: int = 1):
        self.image_buffer = []
        self.image_names = []
        self.dir_path = dir_path
        self.file_exten = file_exten
        self.lr_folder = lr_folder
        self.gt_folder = gt_folder
        self.__is_valid = False
        self.n_rot = num_rotation

        if not os.path.exists(dir_path):
            return
        if file_exten not in ["png", "jpg"]:
            return
        
        # Get file anmes and check dataset validity
        lr_names = [name for name in os.listdir(os.path.join(self.dir_path, self.lr_folder)) if name.endswith(self.file_exten)]
        gt_names = [name for name in os.listdir(os.path.join(self.dir_path, self.gt_folder)) if name.endswith(self.file_exten)]

        self.image_names = list(set(lr_names) & set(gt_names))

        if len(self.image_names) != 0:
            self.__is_valid = True

    def __len__(self):
        return len(self.image_names) * self.n_rot
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.dir_path, self.lr_folder, self.image_names[idx // self.n_rot])
        gt_path = os.path.join(self.dir_path, self.gt_folder, self.image_names[idx // self.n_rot])

        low_res = Image.open(file_path).convert("YCbCr")
        hig_res = Image.open(gt_path).convert("YCbCr")
        
        low_res = low_res.rotate(360 / self.n_rot * float(idx % self.n_rot))
        hig_res = hig_res.rotate(360 / self.n_rot * float(idx % self.n_rot))

        low_res = low_res.resize(hig_res.size, Image.BICUBIC)

        lr_y, _, _ = low_res.split()
        lr_tensor = transforms.ToTensor()(lr_y)
        hr_y, _, _ = hig_res.split()
        hr_tensor = transforms.ToTensor()(hr_y)

        return lr_tensor, hr_tensor.view(1, -1, hr_y.size[1], hr_y.size[0])

    def is_valid(self) -> bool:
        return self.__is_valid