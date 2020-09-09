import numpy as np
import torch
from tools import MultiItemAverageMeter, accuracy


def train_an_epoch(config, iter_n,loaders, base, current_epoch, train_gan, train_reid,self_training,train_adaptation, optimize_sl_enc,trg_labeled_loader = None):

    # set train mode
    base.set_train()
    base.lr_scheduler_step(current_epoch)
    meter = MultiItemAverageMeter()


    # train loop
    for _ in range(5):

        # zero grad
        base.sl_enc_optimizer.zero_grad()
        base.gen_optimizer.zero_grad()
        base.dis_optimizer.zero_grad()
        base.il_enc_optimizer.zero_grad()

        #
        results = {}
        # gan
        if train_gan:
            gen_loss_without_feature, gen_loss_gan_feature, dis_loss, image_list = train_gan_an_iter(config, loaders, base)

            gen_loss_gan_feature.backward(retain_graph=True)
            base.sl_enc_optimizer.zero_grad()
            gen_loss_without_feature.backward()

            base.dis_optimizer.zero_grad()
            dis_loss.backward()

            results['gen_loss_gan_feature'] = gen_loss_gan_feature.item()
            results['gen_loss_without_feature'] = gen_loss_without_feature.item()
            results['dis_loss'] = dis_loss.item()

        if train_reid:
            cls_loss, triplet_loss, acc = train_reid_an_iter(config, loaders, base,True)
            reid_loss = cls_loss + triplet_loss
            reid_loss.backward()
            results['cls_loss'] = cls_loss.item()
            results['triplet_loss'] = triplet_loss.item()
            results['acc'] = acc

        if self_training:
            assert trg_labeled_loader is not None, "self training requires labeled target loader"
            cls_loss, triplet_loss, acc = train_reid_an_iter(config, loaders, base, False, trg_labeled_loader)
            reid_loss = cls_loss + triplet_loss
            reid_loss.backward()
            results['cls_loss'] = cls_loss.item()
            results['triplet_loss'] = triplet_loss.item()
            results['acc'] = acc

        if train_adaptation:
            gen_loss_real, gen_loss_fake, gen_loss_cycle, dis_loss_real, dis_loss_fake, dis_loss_cycle = \
                                         train_adaptation_an_iter(config,loaders,base)
            gen_loss = gen_loss_real + gen_loss_fake + gen_loss_cycle
            dis_loss = dis_loss_real + dis_loss_fake + dis_loss_cycle

            base.dom_gen_optimizer.zero_grad()
            gen_loss.backward()

            base.dom_dis_optimizer.zero_grad()
            dis_loss.backward()
            results['dom_gen_loss'] = gen_loss.item()
            results['dom_dis_loss'] = dis_loss.item()

        # optimize
        if optimize_sl_enc:
            base.sl_enc_optimizer.step()
        if train_gan:
            base.gen_optimizer.step()
            base.dis_optimizer.step()
        if train_reid or self_training:
            base.il_enc_optimizer.step()
        if train_adaptation:
            base.dom_gen_optimizer.step()
            base.dom_dis_optimizer.step()
        # record
        print("a new iter")
        meter.update(results)
    return meter.get_str()


def train_gan_an_iter(config, loaders, base):

    ### load data
    source_images,_, source_ids, source_cams= loaders.gen_source_train_iter.next_one()
    target_images, _,target_ids, ir_cams = loaders.gen_target_train_iter.next_one()
    source_images, target_images = source_images.to(base.device), target_images.to(base.device)
    source_ids, target_ids = source_ids.to(base.device), target_ids.to(base.device)


    # encode
    real_source_contents, real_source_styles, real_source_predicts,_ = base.generator_source.module.encode(source_images, True)
    real_target_contents, real_targets_styles, real_target_predicts, _ = base.generator_target.module.encode(target_images, True)

    # decode (within domain)
    reconst_source_images = base.generator_source.module.decode(real_source_contents, real_source_styles)
    reconst_target_images = base.generator_source.module.decode(real_target_contents, real_targets_styles)

    # decode (cross domain)
    fake_source_images = base.generator_source.module.decode(real_target_contents, real_source_styles)
    fake_target_images = base.generator_target.module.decode(real_source_contents, real_targets_styles)

    # encode again
    fake_ir_contents, fake_rgb_styles, _ ,_= base.generator_source.module.encode(fake_source_images, True)
    fake_rgb_contents, fake_ir_styles, _ ,_ = base.generator_target.module.encode(fake_target_images, True)

    # decode again
    cycreconst_source_images = base.generator_source.module.decode(fake_rgb_contents, real_source_styles)
    cycreconst_target_images = base.generator_target.module.decode(fake_ir_contents, real_targets_styles)


    # reconstruction loss
    gen_loss_reconst_images = base.reconst_loss(reconst_source_images, source_images) + base.reconst_loss(reconst_target_images, target_images)
    gen_loss_cyclereconst_images = base.reconst_loss(cycreconst_source_images, source_images) + base.reconst_loss(cycreconst_target_images, target_images)
    gen_loss_reconst_contents = base.reconst_loss(fake_rgb_contents, real_source_contents) + base.reconst_loss(fake_ir_contents, real_target_contents)
    gen_loss_reconst_styles = base.reconst_loss(fake_rgb_styles, real_source_styles) + base.reconst_loss(fake_ir_styles, real_targets_styles)

    # gan loss
    gen_loss_gan = base.discriminator_source.module.calc_gen_loss(fake_source_images) + base.discriminator_target.module.calc_gen_loss(fake_target_images)

    # overall loss
    gen_loss_without_gan_feature = 1.0 * gen_loss_gan + \
                               base.config.weight_gan_image * (gen_loss_reconst_images + gen_loss_cyclereconst_images)
    gen_loss_gan_feature = base.config.weight_gan_feature * (gen_loss_reconst_contents + gen_loss_reconst_styles)

    # images list
    image_list = [source_images, fake_target_images.detach(), target_images, fake_source_images.detach()]

    ### discriminator
    dis_loss = base.discriminator_source.module.calc_dis_loss(fake_source_images.detach(), source_images) + \
               base.discriminator_target.module.calc_dis_loss(fake_target_images.detach(), target_images)

    return gen_loss_without_gan_feature, gen_loss_gan_feature, dis_loss, image_list




def train_pixel_an_iter(config, loaders, base, image_list):


    source_images, fake_target_images, target_images, fake_source_images = image_list

    ### compute feature
    _, _, rgb_cls_score = base.encoder(source_images, True, sl_enc=False)
    _, _, fake_ir_score = base.encoder(fake_target_images, True, sl_enc=False)
    _, _, ir_cls_score = base.encoder(target_images, True, sl_enc=False)
    _, _, fake_rgb_score = base.encoder(fake_source_images, True, sl_enc=False)

    ###
    loss_rgb = base.kl_loss(fake_ir_score, rgb_cls_score)
    loss_ir = base.kl_loss(fake_rgb_score, ir_cls_score)

    ###
    return loss_rgb + loss_ir


def train_reid_an_iter(config, loaders, base,only_one,target_loader_iter = None):

    if only_one:

        source_images,_,source_ids, source_cams= loaders.reid_source_train_iter.next_one()
        source_images, source_ids = source_images.to(base.device), source_ids.to(base.device)
        _, source_feature_vectors, source_cls_score = base.encoder(source_images, True, sl_enc=False)
        # loss classification loss and triplet loss

        source_cls_loss = base.ide_creiteron(source_cls_score, source_ids, loaders.num_source_pids)
        triplet_loss_1 = base.triplet_creiteron(source_feature_vectors,source_ids)
        source_acc = accuracy(source_cls_score,source_ids,[1])[0]

    else:
        assert target_loader_iter is not None, "The labeled target loader should not be none"
        source_images, _,source_ids, source_cams = loaders.reid_source_train_iter.next_one()
        source_images, source_ids = source_images.to(base.device), source_ids.to(base.device)
        _, source_feature_vectors, source_cls_score = base.encoder(source_images, True, sl_enc=False)
        # loss classification loss and triplet loss
        source_cls_loss = base.ide_creiteron(source_cls_score, source_ids, loaders.num_source_pids)
        triplet_loss_1 = base.triplet_creiteron(source_feature_vectors, source_ids)
        source_acc = accuracy(source_cls_score, source_ids, [1])[0]

        target_images, _,target_ids, ir_cams = target_loader_iter.next_one()
        target_images,target_ids = target_images.to(base.device), target_ids.to(base.device)
        # two losses for generated dataset
        _, target_feature_vectors, target_cls_score = base.encoder(target_images, True, sl_enc=False)

        target_cls_loss = base.ide_creiteron(target_cls_score,target_ids)
        triplet_loss_2 = base.triplet_creiteron(target_feature_vectors,target_ids)
        triplet_loss = (triplet_loss_1 + triplet_loss_2) /2.0
        cls_loss = (source_cls_loss + target_cls_loss) /2.0
        target_acc = accuracy(target_cls_score,target_ids,[1])[0]
        acc = torch.Tensor([source_acc,target_acc])
        return  cls_loss , triplet_loss, acc

    return source_cls_loss, triplet_loss_1, source_acc


def train_adaptation_an_iter(config,loaders,base):
    ### load data
    source_images, _, source_ids, source_cams = loaders.gen_source_train_iter.next_one()
    target_images, _, target_ids, ir_cams = loaders.gen_target_train_iter.next_one()
    source_images, target_images = source_images.to(base.device), target_images.to(base.device)
    source_ids, target_ids = source_ids.to(base.device), target_ids.to(base.device)

    # encode
    real_source_contents, real_source_styles, real_source_predicts, real_source_feature_vectors = base.generator_source.module.encode(source_images,
                                                                                                         True)
    real_target_contents, real_targets_styles, real_target_predicts , real_target_feature_vectors = base.generator_target.module.encode(target_images,
                                                                                                          True)

    gen_loss_real = base.dom_discriminator.module.calc_gen_loss(real_target_feature_vectors)
    dis_loss_real = base.dom_discriminator.module.calc_dis_loss(real_target_feature_vectors.detach(),real_source_feature_vectors.detach())

    # decode (within domain)
    reconst_source_images = base.generator_source.module.decode(real_source_contents, real_source_styles)
    reconst_target_images = base.generator_source.module.decode(real_target_contents, real_targets_styles)

    # decode (cross domain)
    fake_source_images = base.generator_source.module.decode(real_target_contents, real_source_styles)
    fake_target_images = base.generator_target.module.decode(real_source_contents, real_targets_styles)

    # encode again
    fake_target_contents, fake_source_styles, _ ,fake_target_features= base.generator_source.module.encode(fake_source_images, True)
    fake_source_contents, fake_target_styles, _ ,fake_source_features= base.generator_target.module.encode(fake_target_images, True)

    # domain loss 2
    gen_loss_fake = base.dom_discriminator.module.calc_gen_loss(fake_target_features)
    dis_loss_fake = base.dom_discriminator.module.calc_dis_loss(fake_target_features.detach(),fake_source_features.detach())

    # decode again
    cycreconst_source_images = base.generator_source.module.decode(fake_source_contents, real_source_styles)
    cycreconst_target_images = base.generator_target.module.decode(fake_target_contents, real_targets_styles)

    cycle_source_contents, cycle_source_style, _, cycle_source_features = base.generator_source.module.encode(cycreconst_source_images, True)
    cycle_target_contents, cycle_target_style, _, cycle_target_features = base.generator_target.module.encode(cycreconst_target_images, True)

    #domain loss cycle
    gen_loss_cycle = base.dom_discriminator.module.calc_gen_loss(cycle_target_features)
    dis_loss_cycle = base.dom_discriminator.module.calc_dis_loss(cycle_target_features.detach(),cycle_source_features.detach())


    return gen_loss_real,gen_loss_fake,gen_loss_cycle, dis_loss_real, dis_loss_fake, dis_loss_cycle
