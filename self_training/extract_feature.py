from .feature_extraction import extract_cnn_feature
from tools.meter import AverageMeter
from collections import OrderedDict
from collections import  namedtuple
import time
import torch

def extract_features(model, data_loader, print_freq=10, metric=None):
    model.set_eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids,_) in enumerate(data_loader):
            data_time.update(time.time() - end)
            imgs = imgs.to(model.device)
            outputs = extract_cnn_feature(model, imgs)
           # print(fnames)
            for fname, output, pid in zip(fnames, outputs, pids):
               # print(fname)
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

        return features, labels