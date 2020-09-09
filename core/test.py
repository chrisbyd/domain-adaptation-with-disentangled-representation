import numpy as np
import scipy.io as sio
import os
from tools.logger import setup_logger
import torch
from torchvision.utils import save_image
import logging
from tools import *
from tools.metrics import R1_mAP_eval

logger = setup_logger('adapatation_test','./out/base',if_train= False)

def test(config, base, loaders,epoch, brief=False):

	base.set_eval()
	evaluator = R1_mAP_eval(num_query= loaders.num_query_target, max_rank= 50, feat_norm= 'yes', eval_dataset_name = config.target_dataset_name)
	evaluator.reset()
	for n_iter, (img,fname, vid, camid) in enumerate(loaders.target_val_loader):
		#print("first iter test")
		with torch.no_grad():
			img = img.to(base.device)
			feature_maps, feat, _ = base.encoder(img,True,False)
			evaluator.update((feat, vid, camid))
			# print(feat.requires_grad)
			# print("vid requires gradient ",vid.requires_grad)
	cmc, mAP = evaluator.compute()
	logger.info("Validation Results - Epoch: {}".format(epoch))
	logger.info("mAP: {:.1%}".format(mAP))
	for r in [1, 5, 10]:
		logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

	return cmc,mAP



def compute_and_save_features(config, base, loaders):

	class XX:
		def __init__(self):
			self.val = {}
		def update(self, key, value):
			if key not in list(self.val.keys()):
				self.val[key] = value
			else:
				self.val[key] = np.concatenate([self.val[key], value], axis=0)
		def get_val(self, key):
			if key in list(self.val.keys()):
				return self.val[key]
			else:
				return np.array([[]])


	print('Time:{}.  Start to compute features'.format(time_now()))
	# compute features
	features_meter, pids_meter, cids_meter = CatMeter(), CatMeter(), CatMeter()

	with torch.no_grad():
		for i, data in enumerate(loaders.rgb_all_loader):
			# load data
			images, pids, cids, _ = data
			# forward
			images = images.to(base.device)
			_, features, _ = base.encoder(images, True, sl_enc=False)
			# meter
			features_meter.update(features.data)
			pids_meter.update(pids.data)
			cids_meter.update(cids.data)

		for i, data in enumerate(loaders.ir_all_loader):
			# load data
			images, pids, cids, _ = data
			# forward
			images = images.to(base.device)
			_, features, _ = base.encoder(images, True, sl_enc=False)
			# meter
			features_meter.update(features.data)
			pids_meter.update(pids.data)
			cids_meter.update(cids.data)

	features = features_meter.get_val_numpy()
	pids = pids_meter.get_val_numpy()
	cids = cids_meter.get_val_numpy()

	print('Time: {}.  Note: Start to save features as .mat file'.format(time_now()))
	# save features as .mat file
	results = {1: XX(), 2: XX(), 3: XX(), 4: XX(), 5: XX(), 6: XX()}
	for i in range(features.shape[0]):
		feature = features[i, :]
		feature = np.resize(feature, [1, feature.shape[0]])
		cid, pid = cids[i], pids[i]
		results[cid].update(pid, feature)

	pid_num_of_cids = [333, 333, 533, 533, 533, 333]
	cids = [1, 2, 3, 4, 5, 6]
	for cid in cids:
		a_result = results[cid]
		xx = []
		for pid in range(1, 1+ pid_num_of_cids[cid - 1]):
			xx.append([a_result.get_val(pid).astype(np.double)])
		xx = np.array(xx)
		sio.savemat(os.path.join(config.save_features_path, 'feature_cam{}.mat'.format(cid)), {'feature': xx})

