from .extract_feature import extract_features
import torch
from .rerank import re_ranking
import numpy as np
from sklearn.cluster import DBSCAN
from torch.utils.data import DataLoader
from dataloaders.bases import ImageDataset
import torchvision.transforms as T
from dataloaders.preprocessing import RandomErasing
import logging
import time

train_reid_transforms = T.Compose([
            T.Resize([320, 320]),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop([320, 320]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
        ])

def generate_labeled_dataset(model,iter_n,src_dataset,source_data_loader,trg_dataset,target_data_loader):
    """
    :param model: the base network
    :param source_data_loader: the dataloader for the source dataset
    :param target_data_loader:
    :return: a labeled target dataset
    """
    logger =  logging.getLogger('adaptation_reid')
    t1 = time.time()
    source_features,_ = extract_features(model,source_data_loader)
    source_features = torch.cat([source_features[f].unsqueeze(0) for f, _, _ in src_dataset], 0)
    target_features,_ = extract_features(model,target_data_loader)
    target_features = torch.cat([target_features[f].unsqueeze(0) for f, _, _ in trg_dataset], 0)
    t2 = time.time()
    logger.info("Extracting features take time {}".format(t2-t1))
    # calculate distance and rerank result
    print('Calculating feature distances')

    source_features = source_features.numpy()
    target_features = target_features.numpy()
    rerank_dist = re_ranking(source_features,target_features,lambda_value= 0.1)

    if iter_n == 0:
        # DBSCAN cluster
        tri_mat = np.triu(rerank_dist, 1)  # tri_mat.dim=2
        tri_mat = tri_mat[np.nonzero(tri_mat)]  # tri_mat.dim=1
        tri_mat = np.sort(tri_mat, axis=None)
        top_num = np.round(1.6e-3 * tri_mat.size).astype(int)
        eps = tri_mat[:top_num].mean()
        print('eps in cluster: {:.3f}'.format(eps))
        cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=8)

        # select & cluster images as training set of this epochs
        logger.info('Clustering and labeling...')
        labels = cluster.fit_predict(rerank_dist)
        num_ids = len(set(labels)) - 1
        logger.info('Iteration {} have {} training ids'.format(iter_n + 1, num_ids))
        # generate new dataset

        new_dataset = []
        for (fname, _, _), label in zip(trg_dataset, labels):
            if label == -1:
                continue
            # dont need to change codes in trainer.py _parsing_input function and sampler function after add 0
            new_dataset.append((fname, label, 0))
        logger.info('Iteration {} have {} training images'.format(iter_n + 1, len(new_dataset)))

        trg_labeled_dataloader = DataLoader(
               ImageDataset(new_dataset,transform= train_reid_transforms)
        )
        return trg_labeled_dataloader







