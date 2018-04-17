import argparse
import os
import sys
import pandas as pd
sys.path.append(os.getcwd())
from gan import GAN
from celeba.generator import Generator
from celeba.discriminator import Discriminator
from image_sampler import ImageSampler
from noise_sampler import NoiseSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=str)
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--nb_epoch', '-e', type=int, default=1000)
    parser.add_argument('--noise_dim', '-nd', type=int, default=100)
    parser.add_argument('--height', '-ht', type=int, default=128)
    parser.add_argument('--width', '-wd', type=int, default=128)
    parser.add_argument('--save_steps', '-ss', type=int, default=1)
    parser.add_argument('--visualize_steps', '-vs', type=int, default=1)
    parser.add_argument('--model_dir', '-md', type=str, default="./params")
    parser.add_argument('--result_dir', '-rd', type=str, default="./result")
    parser.add_argument('--noise_mode', '-nm', type=str, default="uniform")
    parser.add_argument('--upsampling', '-up', type=str, default="deconv")
    parser.add_argument('--metrics', '-m', type=str, default="JSD")
    parser.add_argument('--lr_d', type=float, default=1e-4)
    parser.add_argument('--lr_g', type=float, default=1e-4)
    parser.add_argument('--norm_d', type=str, default=None)
    parser.add_argument('--norm_g', type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # output config to csv
    config_path = os.path.join(args.result_dir, "config.csv")
    dict_ = vars(args)
    df = pd.DataFrame(list(dict_.items()), columns=['attr', 'status'])
    df.to_csv(config_path, index=None)

    input_shape = (args.height, args.width, 3)

    image_sampler = ImageSampler(args.image_dir,
                                 target_size=(args.width, args.height))
    noise_sampler = NoiseSampler(args.noise_mode)

    generator = Generator(args.noise_dim,
                          upsampling=args.upsampling,
                          normalization=args.norm_g)
    discriminator = Discriminator(input_shape,
                                  normalization=args.norm_d)
    gan = GAN(generator,
              discriminator,
              metrics=args.metrics,
              lr_d=args.lr_d,
              lr_g=args.lr_g)

    gan.fit(image_sampler.flow_from_directory(args.batch_size),
            noise_sampler,
            nb_epoch=args.nb_epoch,
            result_dir=args.result_dir,
            model_dir=args.model_dir,
            save_steps=args.save_steps,
            visualize_steps=args.visualize_steps)


if __name__ == '__main__':
    main()