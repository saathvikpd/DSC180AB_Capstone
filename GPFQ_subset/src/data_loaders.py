import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import random
import os
import glob
import re
import pickle
from utils import parse_imagenet_val_labels

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
# use above function and g to preserve reproducibility.

class Imagenet(Dataset):
    """
    Validation dataset of Imagenet
    """
    def __init__(self, data_dir, transform):
        # we can maybe pput this into diff files.
        self.Y = torch.from_numpy(parse_imagenet_val_labels(data_dir)).long()
        self.X_path = sorted(glob.glob(os.path.join(data_dir, 'ILSVRC2012_img_val/*.JPEG')), 
            key=lambda x: re.search('%s(.*)%s' % ('ILSVRC2012_img_val/', '.JPEG'), x).group(1))
        self.transform = transform

    def __len__(self):
        return len(self.X_path)
    
    def __getitem__(self, idx):
        img = Image.open(self.X_path[idx]).convert('RGB')
        y = self.Y[idx] 
        if self.transform:
            x = self.transform(img)
        return x, y


def data_loader(ds_name, batch_size, num_workers): 
    """
    Prepare data loaders
    """
    if ds_name == 'ILSVRC2012':
        data_dir = '../data/ILSVRC2012'  # customize the data path before run the code 

        if not os.path.isdir(data_dir):
            raise Exception('Please download Imagenet2012 dataset!')

        # see https://pytorch.org/vision/stable/models.html for setting transform
        transform = transforms.Compose([
                            transforms.Resize(256), 
                            transforms.CenterCrop(224),  
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                            ])
        
        train_ds = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'imagenette2/train'),
                                                    transform=transform)
        class_names = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079', 
               'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']
        
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
#         class_name_to_idx = {class_name: idx for class_name, idx in train_ds.class_to_idx.items()}

        if not os.path.isfile('../data/ILSVRC2012/wnid_to_label.pickle'):
            with open('../data/ILSVRC2012/wnid_to_label.pickle', 'wb') as f:
                pickle.dump(train_ds.class_to_idx, f)
        
        with open('../data/ILSVRC2012/wnid_to_label.pickle', 'rb') as f:
            class_name_to_idx = pickle.load(f)

        # Print the mapping
#         for class_name, idx in class_name_to_idx.items():
#             print(f'Class Name: {class_name}, Index: {idx}')
        for class_name in class_names:
            print(f'Class Name: {class_name}, Index: {class_name_to_idx[class_name]}')
#         test_ds = Imagenet(os.path.join(data_dir, 'imagenette2/val'), transform) 
        test_ds = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'imagenette2/val'),
                                                    transform=transform)
        train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers,
                                worker_init_fn=seed_worker, generator=g)
        test_dl = DataLoader(test_ds, min(batch_size, 1024), shuffle=False,
                                num_workers=num_workers) 
        mapping = {i: class_name_to_idx[class_names[i]] for i in range(10)}
        for i, img_and_label in enumerate(test_dl):
            for j in range(10):
                img_and_label[-1][img_and_label[-1]==j] = mapping[j]
            print(f"Batch {i} - Labels: {img_and_label[-1][:10]}")
            if i > 1:  # Only print the first two batches
                break

    elif ds_name == 'CIFAR10':
        data_dir = '../data'

        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, 
            transform=transform_train)
        
        test_ds = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, 
            transform=transform_test)
        
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size,
                              num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
        
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=min(batch_size, 1024),
                             num_workers=num_workers)

    else:
        raise Exception('Unkown dataset!')

    return train_dl, test_dl 
