import argparse
import math
import os
from librosa.core import audio

import torch
import torchvision
from torch import optim
from tqdm import tqdm
import sys
import librosa
import numpy as np
import random
import torch.nn.functional as F

import cv2

sys.path.append("./")
from criteria.soundclip_loss import SoundCLIPLoss
from criteria.id_loss import IDLoss
from models.stylegan2.model import Generator
from utils.utils import ensure_checkpoint_exists

# check: DiffusionCLIP Import
from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM

from utils.text_dic import SRC_TRG_TXT_DIC
from utils.diffusion_utils import get_beta_schedule, denoising_step

# from diffusionclip import DiffusionClip



def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def main(args):
    # > optimization means Sound-guided Image Manipulation
    # > use a direct latent code optimization method to manipulate the given image similar to StyleCLIP
    # > Minimizes the distance between a given source latent code and an audio driven latent code in the
    # > learned joint embedding space to produce sound-guided manipulated images.
    # check: Need to bring the clip_latent_optim or clip_finetune process (loss is there!)
    # TODO: We need to find what is input to the generator

    # #*--------------------DEVICE--------------------
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model_var_type = config.model.model_var_type
    # betas = get_beta_schedule(
    #     beta_start=config.diffusion.beta_start,
    #     beta_end=config.diffusion.beta_end,
    #     num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
    # )
    # betas = torch.from_numpy(betas).float().to(device)
    # num_timesteps = betas.shape[0]

    # alphas = 1.0 - betas
    # alphas_cumprod = np.cumprod(alphas, axis=0)
    # alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    # posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    # if model_var_type == "fixedlarge":
    #     logvar = np.log(np.append(posterior_variance[1], 1e-20))
    # elif model_var_type == "fixedsmall":
    #     logvar = np.log(np.maximum(posterior_variance, 1e-20))

    # if args.edit_attr is None:
    #     src_txts = args.src_txts
    #     trg_txts = args.trg_txts
    # else:
    #     src_txts = SRC_TRG_TXT_DIC[args.edit_attr][0]
    #     trg_txts = SRC_TRG_TXT_DIC[args.edit_attr][1]

    # #*--------------------MODEL--------------------
    # #*--------------------DIFFUSIONCLIP LOAD--------------------

    # # TODO: Prevent conffusion between the two models by fixing argparse
    # # check: fix model_path
    # # check: fix dataset?
    # # check: fix ckpt_diffusion (ok)

    # print(args.exp)
    # print(f'     {src_txts}')
    # print(f'->   {trg_txts}')

    # if config.data.dataset == "LSUN":
    #     if config.data.category == "bedroom":
    #         url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
    #     elif config.data.category == "church_outdoor":
    #         url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
    # elif config.data.dataset == "CelebA_HQ":
    #     url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
    # elif config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
    #     pass
    # else:
    #     raise ValueError

    # if config.data.dataset in ["CelebA_HQ", "LSUN"]:
    #     model = DDPM(config)
    #     if args.model_path:
    #         ckpt_diffusion = torch.load(args.model_path)
    #     else:
    #         ckpt_diffusion = torch.hub.load_state_dict_from_url(url, map_location=device)
    #     learn_sigma = False
    #     print("Original diffusion Model loaded.")
    # elif config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
    #     model = i_DDPM(config.data.dataset)
    #     if args.model_path:
    #         ckpt_diffusion = torch.load(args.model_path)
    #     else:
    #         ckpt_diffusion = torch.load(MODEL_PATHS[config.data.dataset])
    #     learn_sigma = True
    #     print("Improved diffusion Model loaded.")
    # else:
    #     print('Not implemented dataset')
    #     raise ValueError

    #*--------------------SGSIM LOAD--------------------
    #*--------------------Audio Encoder--------------------
    # ? We can easily change the generator not regarding a audio input?
    ensure_checkpoint_exists(args.ckpt)

    y, sr = librosa.load(args.audio_path, sr=44100)
    n_mels = 128
    time_length = 864
    audio_inputs = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    audio_inputs = librosa.power_to_db(audio_inputs, ref=np.max) / 80.0 + 1

    audio_inputs = audio_inputs

    zero = np.zeros((n_mels, time_length))
    resize_resolution = 512
    h, w = audio_inputs.shape
    if w >= time_length:
        j = 0
        j = random.randint(0, w-time_length)
        audio_inputs = audio_inputs[:,j:j+time_length]
    else:
        zero[:,:w] = audio_inputs[:,:w]
        audio_inputs = zero


    audio_inputs = cv2.resize(audio_inputs, (n_mels, resize_resolution))
    audio_inputs = np.array([audio_inputs])
    audio_inputs = torch.from_numpy(audio_inputs.reshape((1, 1, n_mels, resize_resolution))).float().cuda()

    os.makedirs(args.results_dir, exist_ok=True)

    #* SGSIM - Generator (StyleGAN)
    # > The generator is a StyleGAN2 generator with a mapping network.
    # > The generator's input in this code is a latent code, which is a 512-dimensional vector.
    # > The code is initializes the latent code in several ways depending on the mode
    # check : Eventually, the generator's input is just pretrained model.
    g_ema = Generator(args.stylegan_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()

    # # TODO: Load diffusion model -> What module do i got?
    # #* DIFFUSIONCLIP - Generator (Diffusion)
    # model.load_state_dict(ckpt_diffusion)
    # model.to(device)

    # # DataParallel - splitting and distributing the input tensor across multiple GPUs
    # model = torch.nn.DataParallel(model)

    # print(f"setting optimizer with lr={args.lr_clip_finetune}")
    # optim_ft = torch.optim.Adam(model.parameters(), weight_decay=0, lr=args.lr_clip_finetune)

    # > mean_latent is a tensor containing the mean latent vector for the generator
    # > In StyleGAN, the input to the generator is not a random vector, but a latent vector that is mapped from a random noise vector.
    # > This mapping is learned by the generator during training.

    # > The mean_latent vector is the average of the latent vectors generated by mapping random noise vectors through the generator.
    # > It represents a "typical" latent vector that is likely to produce realistic images.

    # > The code set mean_latent to the mean of 4096 latent vectors generated by generator,
    # > which is a common practice in StyleGAN-based image generation.
    # check: mean_latent = ([1, 512])
    mean_latent = g_ema.mean_latent(4096)
    layer_masking_weight = torch.ones(14)

    if args.latent_path:
        latent_code_init = torch.load(args.latent_path).cuda()
        with torch.no_grad():
            img_orig, _ = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False)

    elif args.mode == "edit":
        latent_code_init_not_trunc = torch.randn(1, 512).cuda()
        with torch.no_grad():
            img_orig, latent_code_init = g_ema([latent_code_init_not_trunc], return_latents=True,
                                        truncation=args.truncation, truncation_latent=mean_latent)

    else:
        latent_code_init = mean_latent.detach().clone().repeat(1, 18, 1)

    latent = latent_code_init.detach().clone()
    latent.requires_grad = True
    soundclip_loss = SoundCLIPLoss(args)
    id_loss = IDLoss(args)
    optimizer = optim.Adam([latent], lr=args.lr)

    pbar = tqdm(range(args.step))

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr

        # check [latent] is a list that contains th latent code tensor, and 'input_is_latent=True' specifies
        # check that the input to the generator is a latent code tensor.
        img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False)
        cosine_distance_loss = soundclip_loss(img_gen, audio_inputs)

        if args.mode == "edit":
            if not args.adaptive_layer_masking:
                similarity_loss = ((latent_code_init - latent) ** 2).sum()
            else:
                similarity_loss = 0
                for idx in range(14):
                    layer_per_loss = F.sigmoid(layer_masking_weight[idx]) * ((latent_code_init[:,idx,:] - latent[:,idx,:]) ** 2).sum()
                    similarity_loss += layer_per_loss
                    layer_masking_weight[idx] = layer_masking_weight[idx] - 0.1 * layer_per_loss.item() * (1 - layer_per_loss.item())

            loss = args.lambda_similarity * similarity_loss + cosine_distance_loss  + args.lambda_identity * id_loss(img_orig, img_gen)[0]

        else:
            loss = cosine_distance_loss

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        pbar.set_description(
            (
                f"loss: {loss.item():.4f};"
            )
        )
        if args.save_intermediate_image_every > 0 and i % args.save_intermediate_image_every == 0:
            with torch.no_grad():
                img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False)
            torchvision.utils.save_image(img_gen, f"results/{str(i).zfill(5)}.png", normalize=True, range=(-1, 1))

    if args.mode == "edit":
        with torch.no_grad():
            img_orig, _ = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False)

        final_result = torch.cat([img_orig, img_gen])
    else:
        final_result = img_gen


    return final_result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", type=str, default="sobbing person", help="the text that guides the editing/generation")
    parser.add_argument("--audio_path", type=str, default="./audiosample/explosion.wav")
    parser.add_argument("--ckpt", type=str, default="./pretrained_models/stylegan2-ffhq-config-f.pt", help="pretrained StyleGAN2 weights")
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--step", type=int, default=300, help="number of optimization steps")
    parser.add_argument("--mode", type=str, default="edit", choices=["edit", "free_generation"], help="choose between edit an image an generate a free one")
    parser.add_argument("--lambda_similarity", type=float, default=0.008, help="weight of the latent distance (used for editing only)")
    parser.add_argument("--lambda_identity", type=float, default=0.005, help="weight of the identity loss")
    parser.add_argument("--latent_path", type=str, default=None, help="starts the optimization from the given latent code if provided. Otherwose, starts from"
                                                                      "the mean latent in a free generation, and from a random one in editing. "
                                                                       "Expects a .pt format")
    parser.add_argument("--truncation", type=float, default=0.7, help="used only for the initial latent vector, and only when a latent code path is"
                                                                      "not provided")
    parser.add_argument("--save_intermediate_image_every", type=int, default=20, help="if > 0 then saves intermidate results during the optimization")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--adaptive_layer_masking", type=bool, default=False)
    parser.add_argument("--save_latent_path", type=str, default=None)
    parser.add_argument("--save_source_image_path", type=str, default=None)
    parser.add_argument("--save_manipulated_image_path", type=str, default=None)
    parser.add_argument("--save_manipulated_latent_code_path", type=str, default=None)

    args = parser.parse_args()

    result_image = main(args)

    torchvision.utils.save_image(result_image.detach().cpu(), os.path.join(args.results_dir, "final_result.jpg"), normalize=True, scale_each=True, range=(-1, 1))


