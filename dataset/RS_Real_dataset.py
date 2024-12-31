from torch.utils.data import Dataset
from torchvision import transforms as T
from pathlib import Path
from functools import partial
import torch
import numpy as np
from torch import nn
from PIL import Image


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def exists(x):
    return x is not None


    
    
class RS_Real_test_dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['npy', 'npz'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize((image_size,image_size)),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            #T.CenterCrop(image_size),
            T.ToTensor()
        ])
        self.transform_img = T.Compose([
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path_npz = str(self.paths[index])
        #path1 = str(path)
        #extracted_part = str(path).split("/")[-1].rsplit(".", 1)[0] 
        #path1 = "train_flow1/" + extracted_part + ".npy"  
        data_npz = np.load(path_npz)
        rs = data_npz['condition']
        out_flow = data_npz['out_flow']
        gs = data_npz['gs']
        #out_flow = flow.transpose((2,0,1))
        #out_flow = torch.tensor(out_flow, dtype=torch.float32)
        out_flow = torch.from_numpy(out_flow).float()
        
        gs = Image.fromarray(gs)
        rs = Image.fromarray(rs)
        
        #rs = cv2.cvtColor(rs, cv2.COLOR_BGR2RGB)
        rs = self.transform(rs)
        gs = self.transform(gs)

        
        """ extracted_part = str(path).split("/")[-1].rsplit(".", 1)[0] 
        path2 = "test_data_img/" + extracted_part + ".jpg"   """
        path_img = path_npz.replace("npz", "jpg")#.replace(".npz", ".jpg")

        img = Image.open(path_img)
        #condtion = img_array[:, -800:]
        show_rs = img.crop((0, 0, 800, img.height))
        #flow = torch.tensor(homo)/(self.image_size-1)
        show_rs = self.transform_img(show_rs)
    
        show_gs =  img.crop((800, 0, img.width, img.height))
        #show_gs = Image.fromarray(show_gs)
        #flow = torch.tensor(homo)/(self.image_size-1)
        show_gs = self.transform_img(show_gs)
        path_out = Path(path_npz).stem
        return [out_flow,rs,gs,show_rs,show_gs,path_out]


   
    
class RS_Real_Train_dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['npy', 'npz'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize((image_size,image_size)),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            #T.CenterCrop(image_size),
            T.ToTensor()
        ])
        self.transform1 = T.Compose([
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = str(self.paths[index])
        #path1 = str(path)
        
        data_npz = np.load(path)
        rs = data_npz['condition']
        out_flow = data_npz['out_flow']
        gs = data_npz['gs']
        #out_flow = flow.transpose((2,0,1))
        #out_flow = torch.tensor(out_flow, dtype=torch.float32)
        out_flow = torch.from_numpy(out_flow).float()
        
        gs = Image.fromarray(gs)
        rs = Image.fromarray(rs)
        #rs = cv2.cvtColor(rs, cv2.COLOR_BGR2RGB)
        rs = self.transform(rs)
        gs = self.transform(gs)

        return [out_flow,rs,gs]