import os
import torch
from torchvision.utils import save_image
import numpy as np
from tools import make_dirs


def visualize(config, loaders, base, current_epoch):

    # set eval mode
    base.set_eval()

    # generate images
    with torch.no_grad():
        # fixed images
        source_images, target_images = base.source_images, base.target_images
        # encode
        real_source_contents, real_source_styles, _, _ = base.generator_source.module.encode(source_images, True)
        real_target_contents, real_target_styles, _, _ = base.generator_target.module.encode(target_images, True)
        # decode (cross domain)
        fake_source_images = base.generator_source.module.decode(real_target_contents, real_source_styles)
        fake_target_images = base.generator_target.module.decode(real_source_contents, real_target_styles)
        # encode again
        # fake_target_contents, fake_source_styles, _ = base.generator_source.module.encode(fake_source_images, True)
        # fake_source_contents, fake_target_styles, _ = base.generator_target.module.encode(fake_target_images, True)

        # # decode again
        # cycreconst_rgb_images = base.generator_rgb.module.decode(fake_rgb_contents, real_rgb_styles)
        # cycreconst_ir_images = base.generator_ir.module.decode(fake_ir_contents, real_ir_styles)
        #
        # cycreconst2_rgb_images = base.generator_rgb.module.decode(real_rgb_contents, fake_rgb_styles)
        # cycreconst2_ir_images = base.generator_ir.module.decode(real_ir_contents, fake_ir_styles)
        #
        # wrong_fake_rgb_style, shuffled_fake_rgb_style = shuffle_styles(fake_rgb_styles, config.gan_k)
        # wrong_fake_ir_style, shuffled_fake_ir_style = shuffle_styles(fake_ir_styles, config.gan_k)
        #
        # cycreconst3_rgb_images = base.generator_rgb.module.decode(fake_rgb_contents, shuffled_fake_rgb_style)
        # cycreconst3_ir_images = base.generator_ir.module.decode(fake_ir_contents, shuffled_fake_ir_style)
        #
        # cycreconst4_rgb_images = base.generator_rgb.module.decode(fake_rgb_contents, wrong_fake_rgb_style)
        # cycreconst4_ir_images = base.generator_ir.module.decode(fake_ir_contents, wrong_fake_ir_style)

    # save images
    images = (torch.cat([source_images, target_images, fake_source_images, fake_target_images,
                         # cycreconst_rgb_images, cycreconst_ir_images,
                         # cycreconst2_rgb_images, cycreconst2_ir_images,
                         # cycreconst3_rgb_images, cycreconst3_ir_images,
                         # cycreconst4_rgb_images, cycreconst4_ir_images
                         ], dim=0) + 1.0) / 2.0
    save_image(images.data.cpu(), os.path.join(config.save_images_path, '{}.jpg'.format(current_epoch)), config.gan_p*config.gan_k)



def shuffle_styles(styles, k):
    wrong_styles = torch.cat((styles[k:], styles[:k]), dim=0)

    random_index = []
    for i in range(styles.shape[0] // k):
        random_index.extend(list(np.random.permutation(k) + i * k))
    random_styles = styles[random_index]

    return wrong_styles, random_styles