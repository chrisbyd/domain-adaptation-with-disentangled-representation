import sys
sys.path.append('..')
import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .networks import Generator, Discriminator, Encoder, weights_init,DomainDiscriminator
from tools import os_walk, time_now, make_dirs, CrossEntropyLabelSmooth, TripletLoss
from tools import to_gray

class Base:

    def __init__(self, config, loaders):

        self.config=config
        self.loaders=loaders
        self.device=torch.device('cuda')

        self._init_networks()
        self._init_optimizers()
        self._init_creterion()
        self._init_fixed_items()



    def _init_networks(self):


        # init models
        self.encoder = Encoder(self.loaders.num_source_pids) # include set-level(sl) and instance-level(sl) encoders
        self.generator_source = Generator(self.encoder) # include modality-specific encoder and decoder for source images
        self.generator_target = Generator(self.encoder) # include modality-specific encoder and decoder for target images
        '''as shown above, the two generators shares the same sel-level(sl) encoder'''
        self.discriminator_source = Discriminator(n_layer=4, middle_dim=32, num_scales=2)
        self.discriminator_target = Discriminator(n_layer=4, middle_dim=32, num_scales=2)

        self.discriminator_source.apply(weights_init('gaussian'))
        self.discriminator_target.apply(weights_init('gaussian'))

        self.dom_discriminator = DomainDiscriminator('lsgan')
        self.dom_discriminator.apply(weights_init('gaussian'))



        # data parallel
        self.generator_source = nn.DataParallel(self.generator_source).to(self.device)
        self.generator_target = nn.DataParallel(self.generator_target).to(self.device)
        self.discriminator_source = nn.DataParallel(self.discriminator_source).to(self.device)
        self.discriminator_target = nn.DataParallel(self.discriminator_target).to(self.device)
        self.dom_discriminator = nn.DataParallel(self.dom_discriminator).to(self.device)


        # recored all models for saving and loading
        self.model_list = []
        self.model_list.append(self.generator_source)
        self.model_list.append(self.generator_target)
        self.model_list.append(self.discriminator_source)
        self.model_list.append(self.discriminator_target)
        self.model_list.append(self.dom_discriminator)



    def _init_optimizers(self):

        sl_enc_params = list(self.encoder.resnet_conv1.parameters())
        il_enc_params = list(self.encoder.resnet_conv2.parameters()) + list(self.encoder.classifier.parameters())
        gen_params = list(self.generator_source.parameters()) + list(self.generator_target.parameters())
        gen_params = list(set(gen_params).difference(set(sl_enc_params).union(set(il_enc_params))))
        dis_params = list(self.discriminator_source.parameters()) + list(self.discriminator_target.parameters())
        dis_dom_params = list(self.dom_discriminator.parameters())
        gen_dom_params = list(il_enc_params)

        sl_enc_params = [p for p in sl_enc_params if p.requires_grad]
        gen_params = [p for p in gen_params if p.requires_grad]
        dis_params = [p for p in dis_params if p.requires_grad]
        il_enc_params = [p for p in il_enc_params if p.requires_grad]
        dis_dom_params = [p for p in dis_dom_params if p.requires_grad]
        gen_dom_params = [p for p in gen_dom_params if p.requires_grad]



        self.sl_enc_optimizer = optim.Adam(sl_enc_params, lr=self.config.learning_rate_reid, betas=[0.9, 0.999], weight_decay=5e-4)
        self.gen_optimizer = optim.Adam(gen_params, lr=0.0001, betas=[0.5, 0.999], weight_decay=0.0001)
        self.dis_optimizer = optim.Adam(dis_params, lr=0.0001, betas=[0.5, 0.999], weight_decay=0.0001)
        self.il_enc_optimizer = optim.Adam(il_enc_params, lr=self.config.learning_rate_reid, betas=[0.9, 0.999], weight_decay=5e-4)
        self.dom_gen_optimizer = optim.Adam(gen_dom_params,lr=0.0001, betas=[0.5, 0.999], weight_decay=0.0001)
        self.dom_dis_optimizer = optim.Adam(dis_dom_params, lr= 0.0001,betas=[0.5, 0.999], weight_decay=0.0001)


        self.sl_enc_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.sl_enc_optimizer, milestones=self.config.milestones, gamma=0.1)
        self.gen_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.gen_optimizer, milestones=self.config.milestones, gamma=0.1)
        self.dis_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.dis_optimizer, milestones=self.config.milestones, gamma=0.1)
        self.il_enc_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.il_enc_optimizer, milestones=self.config.milestones, gamma=0.1)
        self.dom_gen_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.dom_gen_optimizer, milestones=self.config.milestones, gamma=0.1)
        self.dom_dis_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.dom_dis_optimizer, milestones=self.config.milestones, gamma=0.1)

    def lr_scheduler_step(self, current_epoch):
        self.sl_enc_lr_scheduler.step(current_epoch)
        self.gen_lr_scheduler.step(current_epoch)
        self.dis_lr_scheduler.step(current_epoch)
        self.il_enc_lr_scheduler.step(current_epoch)
        self.dom_gen_lr_scheduler.step(current_epoch)
        self.dom_dis_lr_scheduler.step(current_epoch)


    def _init_creterion(self):
        self.reconst_loss = nn.L1Loss()
        self.id_loss = nn.CrossEntropyLoss()
        self.ide_creiteron = CrossEntropyLabelSmooth()
        self.triplet_creiteron = TripletLoss(0.3)


    def kl_loss(self, score1, score2,  mini=1e-8):
        score2 = score2.detach()
        prob1 = F.softmax(score1, dim=1)
        prob2 = F.softmax(score2, dim=1)
        loss = torch.sum(prob2 * torch.log(mini + prob2 / (prob1 + mini)), 1) + \
                 torch.sum(prob1 * torch.log(mini + prob1 / (prob2 + mini)), 1)
        return loss.mean()


    def _init_fixed_items(self):

        source_images, _, _,_ = self.loaders.gen_source_train_iter.next_one()
        target_images, _, _,_ = self.loaders.gen_target_train_iter.next_one()
        self.source_images = source_images.to(self.device)
        self.target_images = target_images.to(self.device)


    def save_model(self, save_epoch,warmup):
        if warmup:
            save_models_path = self.config.save_wp_models_path
        else:
            save_models_path = self.config.save_st_models_path
        # save model
        for ii, _ in enumerate(self.model_list):
            torch.save(self.model_list[ii].state_dict(), os.path.join(save_models_path, 'model-{}_{}.pkl'.format(ii, save_epoch)))

        # if saved model is more than max num, delete the model with smallest epoch
        if self.config.max_save_model_num > 0:
            root, _, files = os_walk(save_models_path)

            # get indexes of saved models
            indexes = []
            for file in files:
                indexes.append(int(file.replace('.pkl', '').split('_')[-1]))

            # remove the bad-case and get available indexes
            model_num = len(self.model_list)
            available_indexes = copy.deepcopy(indexes)
            for element in indexes:
                if indexes.count(element) < model_num:
                    available_indexes.remove(element)

            available_indexes = sorted(list(set(available_indexes)), reverse=True)
            unavailable_indexes = list(set(indexes).difference(set(available_indexes)))

            # delete all unavailable models
            for unavailable_index in unavailable_indexes:
                try:
                    # os.system('find . -name "{}*_{}.pkl" | xargs rm  -rf'.format(save_models_path, unavailable_index))
                    for ii in range(len(self.model_list)):
                        os.remove(os.path.join(root, 'model-{}_{}.pkl'.format(ii, unavailable_index)))
                except:
                    pass

            # delete extra models
            if len(available_indexes) >= self.config.max_save_model_num:
                for extra_available_index in available_indexes[self.config.max_save_model_num:]:
                    # os.system('find . -name "{}*_{}.pkl" | xargs rm  -rf'.format(self.config.save_models_path, extra_available_index))
                    for ii in range(len(self.model_list)):
                        os.remove(os.path.join(root, 'model-{}_{}.pkl'.format(ii, extra_available_index)))


    def resume_model(self,warmup, resume_epoch):
        if warmup:
            save_models_path = self.config.save_wp_models_path
        else:
            save_models_path = self.config.save_st_models_path
        for ii, _ in enumerate(self.model_list):
            self.model_list[ii].load_state_dict(
                torch.load(os.path.join(save_models_path, 'model-{}_{}.pkl'.format(ii, resume_epoch))))
        print('Time: {}, successfully resume model from {}'.format(time_now(), resume_epoch))


    ## resume model from a path
    def resume_model_from_path(self, path, resume_epoch):
        for ii, _ in enumerate(self.model_list):
            self.model_list[ii].load_state_dict(
                torch.load(os.path.join(path, 'model-{}_{}.pkl'.format(ii, resume_epoch))))
        print('Time: {}, successfully resume model from {}'.format(time_now(), resume_epoch))


    ## set model as train mode
    def set_train(self):
        for ii, _ in enumerate(self.model_list):
            self.model_list[ii] = self.model_list[ii].train()


    ## set model as eval mode
    def set_eval(self):
        for ii, _ in enumerate(self.model_list):
            self.model_list[ii] = self.model_list[ii].eval()
