import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .veri import VeRi
from .vehicleid import VehicleID
from .bases import ImageDataset
from .preprocessing import RandomErasing
from .sampler import RandomIdentitySampler

__factory = {
    'veri': VeRi,
    'vehicle' : VehicleID
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, _, _,_ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids

def val_collate_fn(batch):##### revised by luo
    imgs, pids, camids, trackids, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, trackids, img_paths


def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize([320, 320]),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop([320, 320]),
            T.ToTensor(),
            T.Normalize(mean= [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406] )
        ])
    val_transforms = T.Compose([
        T.Resize([320,320]),
        T.ToTensor(),
        T.Normalize(mean= [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
    ])

    num_workers = cfg.num_workers

    dataset = __factory[cfg.dataset_name](root= cfg.dataset_root_dir)
    num_classes = dataset.num_train_pids

    train_set = ImageDataset(dataset.train, train_transforms)


    train_loader = DataLoader(
            train_set, batch_size=cfg.batch_size,
            sampler=RandomIdentitySampler(dataset.train, cfg.batch_size, cfg.num_instances),
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    train_gen_transforms = T.Compose([
            T.Resize([256, 128]),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop([256, 128]),
            T.ToTensor(),
            T.Normalize(mean= [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406] )
        ]

    )
    train_set_gen = ImageDataset(dataset.train,train_gen_transforms)
    train_loader_gen = DataLoader(
        train_set_gen, batch_size= cfg.batch_size_gen,
        sampler= RandomIdentitySampler(dataset.train, cfg.batch_size_gen, cfg.num_instances_gen),
        num_workers= num_workers, collate_fn= train_collate_fn
    )


    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.test_batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader,train_loader_gen, val_loader, len(dataset.query), num_classes
