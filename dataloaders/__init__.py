from .make_dataloader import make_dataloader
import torch.utils.data as data
import torchvision.transforms as transforms
import copy
import numpy as np
from .vehicleid import VehicleID
from .veri import VeRi
from .bases import ImageDataset
import torchvision.transforms as T
from .preprocessing import RandomErasing
from .sampler import Seeds
from .sampler import RandomIdentitySampler
from .bases import IterLoader
from collections import  defaultdict

def get_num_pids(samples):
    pids = []
    for sample in samples:
        pids.append(sample[1])
    return len(set(pids))


class Loaders:

    def __init__(self,config):


        self.source_dataset_name = config.source_dataset_name
        self.target_dataset_name = config.target_dataset_name

        # transforms
        self.train_gen_transforms = T.Compose([
            T.Resize([256,128]),
            T.RandomHorizontalFlip(p = 0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._factory = {
                            'veri': VeRi,
                            'vehicle' : VehicleID
                        }

        self.train_reid_transforms = T.Compose([
            T.Resize([320, 320]),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop([320, 320]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
        ])

        self.val_transforms = T.Compose([
            T.Resize([320, 320]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # dataset configuration
        self.dataset_root = config.dataset_path

        # batch sample configuration

        self.reid_p = config.reid_p
        self.reid_k = config.reid_k
        self.gan_p = config.gan_p
        self.gan_k = config.gan_k
        self.test_batch_size = config.test_batch_size # to complete

        # init　loaders
        self.__init_train_loaders()



    def __init_train_loaders(self):

        # init datasets
        self.source_dataset_samples = self._factory[self.source_dataset_name](root = self.dataset_root)
        self.target_dataset_samples = self._factory[self.target_dataset_name](root = self.dataset_root)

        source_train_dataset_4gen = ImageDataset(self.source_dataset_samples.train,self.train_gen_transforms)
        target_train_dataset_4gen = ImageDataset(self.target_dataset_samples.train,self.train_gen_transforms)

        self.source_train_dataset_4_reid = ImageDataset(self.source_dataset_samples.train,self.train_reid_transforms)
        self.target_train_dataset_4_reid = ImageDataset(self.target_dataset_samples.train,self.train_reid_transforms)

        source_val_dataset = ImageDataset(self.source_dataset_samples.query + self.source_dataset_samples.gallery, self.val_transforms)
        target_val_dataset = ImageDataset(self.target_dataset_samples.query + self.target_dataset_samples.gallery, self.val_transforms)
        self.num_query_source = len(self.source_dataset_samples.query)
        self.num_query_target = len(self.target_dataset_samples.query)
        #init loaders
        seeds = Seeds(np.random.randint(0,1e8,9999))

        self.num_source_pids = get_num_pids(self.source_dataset_samples.train)
        self.num_target_pids = get_num_pids(self.target_dataset_samples.train)
        #train loaders
        self.gen_source_train_loader = data.DataLoader(
            source_train_dataset_4gen, self.gan_p *self.gan_k, shuffle= False,
            sampler= RandomIdentitySampler(self.source_dataset_samples.train,batch_size= self.gan_k*self.gan_p, num_instances= self.gan_p),
            num_workers= 4,drop_last= True
        )
        self.gen_target_train_loader = data.DataLoader(
            target_train_dataset_4gen, self.gan_p*self.gan_k, shuffle= False,
            sampler=RandomIdentitySampler(self.target_dataset_samples.train, batch_size=self.gan_k * self.gan_p,
                                          num_instances=self.gan_p),
            num_workers=4, drop_last=True
        )
        self.reid_source_train_loader = data.DataLoader(
            self.source_train_dataset_4_reid, self.reid_k* self.reid_p, shuffle= False,
            sampler=RandomIdentitySampler(self.source_dataset_samples.train, batch_size=self.reid_k * self.reid_p,
                                          num_instances=self.reid_p),
            num_workers=4, drop_last=False

        )
        self.reid_target_train_loader = data.DataLoader(
            self.target_train_dataset_4_reid, self.reid_k* self.reid_p, shuffle= False,
            sampler=RandomIdentitySampler(self.target_dataset_samples.train, batch_size=self.reid_k * self.reid_p,
                                          num_instances=self.reid_p),
            num_workers=4, drop_last=False
        )



        # val loaders
        self.source_val_loader = data.DataLoader(
            source_val_dataset, batch_size= self.test_batch_size, shuffle= False,
            num_workers=4
        )
        self.target_val_loader = data.DataLoader(
            target_val_dataset, batch_size= self.test_batch_size,shuffle= False,
            num_workers= 4
        )

        # init iters
        self.gen_source_train_iter = IterLoader(self.gen_source_train_loader)
        self.gen_target_train_iter = IterLoader(self.gen_target_train_loader)

        self.reid_source_train_iter = IterLoader(self.reid_source_train_loader)
        self.reid_target_train_iter = IterLoader(self.reid_target_train_loader)

        self.source_val_iter = IterLoader(self.source_val_loader)
        self.target_val_iter = IterLoader(self.target_val_loader)

    def get_self_train_loaders(self):
        veri_dataset_samples = self._factory['veri'](root = self.dataset_root)
        vehicle_dataset_samples  = self._factory['vehicle'](root = self.dataset_root)
        vehicle_pid_dict = defaultdict(list)
        for image_path, pid, _ in vehicle_dataset_samples.train:
            vehicle_pid_dict[pid].append(image_path)
        vehicle_new_dataset_train_samples = []
        for pid in vehicle_pid_dict.keys():
            img_paths = vehicle_pid_dict[pid]
            if len(img_paths) >= 2:
                chosen_imgs = np.random.choice(img_paths,2,replace= False)
                vehicle_new_dataset_train_samples.append([chosen_imgs[0],pid,1])
                vehicle_new_dataset_train_samples.append([chosen_imgs[1],pid,1])
            else:
                chosen_imgs = np.random.choice(img_paths,2,replace= True)
                vehicle_new_dataset_train_samples.append([chosen_imgs[0], pid, 1])
                vehicle_new_dataset_train_samples.append([chosen_imgs[1], pid, 1])

        veri_dataset = ImageDataset(veri_dataset_samples.train,self.train_reid_transforms)
        vehicle_dataset = ImageDataset(vehicle_new_dataset_train_samples, self.train_reid_transforms)

        veri_train_loader_self_training = data.DataLoader(
           veri_dataset , self.test_batch_size, shuffle=False,
            num_workers=4, drop_last=False
        )

        vehicle_train_loader_self_training = data.DataLoader(
            vehicle_dataset, self.test_batch_size, shuffle=False,
            num_workers=4, drop_last=False
        )
        if self.source_dataset_name == 'veri':
            return veri_dataset_samples.train,veri_train_loader_self_training,vehicle_new_dataset_train_samples,vehicle_train_loader_self_training
        else:
            return vehicle_new_dataset_train_samples,vehicle_train_loader_self_training, veri_dataset_samples.train,veri_train_loader_self_training











