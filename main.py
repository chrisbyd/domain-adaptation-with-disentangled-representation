import argparse
import os
import ast
import copy
import numpy as np
from self_training.generate_labeled_dataset import generate_labeled_dataset
from core import  Base, train_an_epoch, test, visualize
from dataloaders import Loaders
from tools import make_dirs, time_now, os_walk
from tools.logger import setup_logger

def main(config):

    # loaders and base
    loaders = Loaders(config)
    base = Base(config, loaders)

    # make dirs
    make_dirs(config.save_images_path)
    make_dirs(config.save_wp_models_path)
    make_dirs(config.save_st_models_path)
    make_dirs(config.save_features_path)

    logger = setup_logger('adaptation_reid',config.output_path,if_train= True)

    if config.mode == 'train':


        if config.resume:
        # automatically resume model from the latest one
            if config.resume_epoch_num == 0:
                start_train_epoch = 0
                root, _, files = os_walk(config.save_models_path)
                if len(files) > 0:
                    # get indexes of saved models
                    indexes = []
                    for file in files:
                        indexes.append(int(file.replace('.pkl', '').split('_')[-1]))

                    # remove the bad-case and get available indexes
                    model_num = len(base.model_list)
                    available_indexes = copy.deepcopy(indexes)
                    for element in indexes:
                        if indexes.count(element) < model_num:
                            available_indexes.remove(element)

                    available_indexes = sorted(list(set(available_indexes)), reverse=True)
                    unavailable_indexes = list(set(indexes).difference(set(available_indexes)))

                    if len(available_indexes) > 0: # resume model from the latest model
                        base.resume_model(available_indexes[0])
                        start_train_epoch = available_indexes[0] + 1
                        logger.info('Time: {}, automatically resume training from the latest step (model {})'.format(time_now(), available_indexes[0]))
                    else: #
                        logger.info('Time: {}, there are no available models')
            else:
                start_train_epoch = config.resume_epoch_num
        else:
            start_train_epoch = 0

        # main loop
        for current_epoch in range(start_train_epoch, config.warmup_reid_epoches + config.warmup_gan_epoches + config.warmup_adaptation_epoches):

            # train
            if current_epoch < config.warmup_reid_epoches: # warmup reid model
                results = train_an_epoch(config,0, loaders, base, current_epoch, train_gan=True, train_reid=True, self_training= False, optimize_sl_enc=True,train_adaptation= False)
            elif current_epoch < config.warmup_reid_epoches + config.warmup_gan_epoches: # warmup GAN model
                results = train_an_epoch(config,0, loaders, base, current_epoch, train_gan=True, train_reid=False, self_training= False, optimize_sl_enc=False,train_adaptation= False)# joint train
            elif current_epoch < config.warmup_reid_epoches + config.warmup_gan_epoches + config.warmup_adaptation_epoches: #warmup adaptation
                results = train_an_epoch(config,0, loaders, base, current_epoch, train_gan=True, train_reid=False, self_training= False, optimize_sl_enc=False,train_adaptation= True)

            print("another epoch")
            logger.info('Time: {};  Epoch: {};  {}'.format(time_now(), current_epoch, results))
            # save model
            if current_epoch% config.save_model_interval == 0:
                base.save_model(current_epoch, True)

            if current_epoch %config.test_model_interval ==0:
                visualize(config, loaders, base, current_epoch)
                test(config, base, loaders, epoch=0, brief=False)

        total_wp_epoches = config.warmup_reid_epoches + config.warmup_gan_epoches

        for iter_n in range(config.iteration_number):
            src_dataset, src_dataloader,trg_dataset, trg_dataloader = loaders.get_self_train_loaders()

            trg_labeled_dataloader = generate_labeled_dataset(base, iter_n, src_dataset, src_dataloader, trg_dataset,
                                                              trg_dataloader)
            for epoch in range(total_wp_epoches +1,config.self_train_epoch):
                results =  train_an_epoch(config,iter_n, loaders,base,epoch,train_gan= True,train_reid= False,
                                          self_training=True,optimize_sl_enc=True,trg_labeled_loader=trg_labeled_dataloader)
                logger.info('Time: {};  Epoch: {};  {}'.format(time_now(), current_epoch, results))

                if epoch %config.save_model_interval == 0:
                    base.save_model(iter_n*config.self_train_epoch + epoch, False)

    elif config.mode == 'test':
        # resume from pre-trained model and test
        base.resume_model_from_path(config.pretrained_model_path, config.pretrained_model_epoch)
        cmc,map = test(config, base, loaders, epoch= 100, brief=False)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')

    # dataset configuration
    parser.add_argument('--dataset_path', type=str, default='./datasets')
    parser.add_argument('--image_size_4reid', type=int, nargs='+', default=[256, 128], help='image size when training reid')
    parser.add_argument('--image_size_4gan', type=int, nargs='+', default=[128, 64], help='image size when training gan. for saving memory, we use small size')
    parser.add_argument('--reid_p', type=int, default=6, help='person count in a batch')
    parser.add_argument('--reid_k', type=int, default=4, help='images count of a person in a batch')
    parser.add_argument('--gan_p', type=int, default=2, help='person count in a batch')
    parser.add_argument('--gan_k', type=int, default=3, help='images count of a person in a batch')

    parser.add_argument('--source_dataset_name', type= str, default= 'veri')
    parser.add_argument('--target_dataset_name', type= str, default= 'vehicle')


    # loss configuration
    parser.add_argument('--learning_rate_reid', type=float, default=0.00045)
    parser.add_argument('--weight_pixel_loss', type=float, default=0.01)
    parser.add_argument('--weight_gan_image', type=float, default=10.0)
    parser.add_argument('--weight_gan_feature', type=float, default=1.0)

    # train configuration
    parser.add_argument('--warmup_reid_epoches', type=int, default=0)
    parser.add_argument('--warmup_gan_epoches', type=int, default=0, help='our model is robust to this parameter, works well when larger than 100')
    parser.add_argument('--warmup_adaptation_epoches', type=int, default=0)

    parser.add_argument('--train_epoches', type=int, default=50)
    parser.add_argument('--milestones', type=int, nargs='+', default=[30])

    # logger configuration
    parser.add_argument('--output_path', type=str, default='out/base/')
    parser.add_argument('--max_save_model_num', type=int, default=2, help='0 for max num is infinit')
    parser.add_argument('--save_images_path', type=str, default=parser.parse_args().output_path+'images/')
    parser.add_argument('--save_wp_models_path', type=str, default=parser.parse_args().output_path+'wp_models/')
    parser.add_argument('--save_st_models_path', type=str, default=parser.parse_args().output_path+'st_models/')
    parser.add_argument('--save_features_path', type=str, default=parser.parse_args().output_path+'features/')

    # self training configuration
    parser.add_argument('--iteration_number', type= int, default= 30)
    parser.add_argument('--self_train_epoch', type= int, default= 10)
    # test configuration
    parser.add_argument('--test_batch_size', type= int, default= 128)
    parser.add_argument('--pretrained_model_path', type=str, default='')
    parser.add_argument('--pretrained_model_epoch', type=str, default='')

    # resume configuration
    parser.add_argument('--resume',type=bool, default= False)
    parser.add_argument('--resume_epoch_num', type= int, default= 0)
    parser.add_argument('--save_model_interval', type= int, default= 20)
    parser.add_argument('--test_model_interval', type= int, default= 1)



    # run
    config = parser.parse_args()
    config.milestones = list(np.array(config.milestones) + config.warmup_reid_epoches + config.warmup_gan_epoches)

    main(config)









