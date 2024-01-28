import argparse
import math
import os
import random
import sys
import time

import cv2
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as tvu
from librosa.core import audio
from PIL import Image
from torch import optim
from tqdm import tqdm

sys.path.append("./")
import wandb
from configs.paths_config import (
    DATASET_PATHS,
    HYBRID_CONFIG,
    HYBRID_MODEL_PATHS,
    MODEL_PATHS,
)
from criteria.soundclip_loss import SoundCLIPLoss
from criteria.vgg_perceptual_loss import VGGPerceptualLoss
from datasets.data_utils import get_dataloader, get_dataset
from losses import id_loss
from losses.clip_loss import CLIPLoss
from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from utils.align_utils import run_alignment
from utils.diffusion_utils import denoising_step, get_beta_schedule
from utils.text_dic import SRC_TRG_TXT_DIC
from utils.utils import ensure_checkpoint_exists


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


class Optimization(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == "fixedsmall":
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

        if self.args.edit_attr is None:
            self.src_txts = self.args.src_txts
            self.trg_txts = self.args.trg_txts
        else:
            self.src_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][0]
            self.trg_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][1]

    def matching_multimodal_representations(self):
        # *********** Load Diffusion Model ************#
        # DiffusionCLIP authors provided this pretrained model
        #! Not available in 2023.09
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":  # Bedroom
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":  # Church
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":  # Human Face
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset == "AFHQ":
            pass
        elif self.config.data.dataset == "custom":
            if self.config.data.category == "bedroom":  # Bedroom
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":  # Church
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        else:
            raise ValueError

        if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
            model = DDPM(self.config)
            if self.args.model_path:
                diffusion_ckpt = torch.load(self.args.model_path)
            else:
                diffusion_ckpt = torch.hub.load_state_dict_from_url(
                    url, map_location=self.device
                )
            learn_sigma = False
            print("Original diffusion Model loaded.")
        elif self.config.data.dataset in ["FFHQ", "AFHQ"]:
            model = i_DDPM(self.config.data.dataset)
            if self.args.model_path:
                diffusion_ckpt = torch.load(self.args.model_path)
            else:
                diffusion_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            learn_sigma = True
            print("Improved diffusion Model loaded.")
        elif self.config.data.dataset in ["custom"]:
            model = DDPM(self.config)
            if self.args.model_path:
                diffusion_ckpt = torch.load(self.args.model_path)
            else:
                diffusion_ckpt = torch.hub.load_state_dict_from_url(
                    url, map_location=self.device
                )
            learn_sigma = False
            print("Original diffusion Model loaded. Custom!!!!!!!!!!!!!")
        else:
            print("Not implemented dataset")
            raise ValueError

        # load checkpoint to model
        model.load_state_dict(diffusion_ckpt)
        model.to(self.device)
        model = torch.nn.DataParallel(model)

        # *********** Load Audio Model ************#

        y, sr = librosa.load(self.args.audio_path, sr=44100)
        if self.args.audio_intensity == "5":
            y = y * 10
        elif self.args.audio_intensity == "4":
            y = y * 1.5
        elif self.args.audio_intensity == "3":
            y = y * 1
        elif self.args.audio_intensity == "2":
            y = y * 0.5
        elif self.args.audio_intensity == "1":
            y = y * 0.01

        n_mels = 128
        time_length = 864
        audio_inputs = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)

        zero = np.zeros((n_mels, time_length))
        resize_resolution = 512
        h, w = audio_inputs.shape

        if w >= time_length:
            j = 0
            j = random.randint(0, w - time_length)
            audio_inputs = audio_inputs[:, j : j + time_length]
        else:
            zero[:, :w] = audio_inputs[:, :w]
            audio_inputs = zero

        audio_inputs = cv2.resize(audio_inputs, (n_mels, resize_resolution))
        audio_inputs = np.array([audio_inputs])
        audio_inputs = torch.from_numpy(
            audio_inputs.reshape(1, 1, n_mels, resize_resolution)
        )

        os.makedirs(self.args.results_dir, exist_ok=True)

        # *********** Optimizer and Scheduler ************#
        print(f"Setting optimizer with lr={self.args.lr_clip_finetune}")
        optim_ft = torch.optim.Adam(
            model.parameters(), weight_decay=0, lr=self.args.lr_clip_finetune
        )
        init_opt_ckpt = optim_ft.state_dict()
        scheduler_ft = torch.optim.lr_scheduler.StepLR(
            optim_ft, step_size=1, gamma=self.args.sch_gamma
        )
        init_sch_ckpt = scheduler_ft.state_dict()

        # *********** Loss ************#
        print("Loading losses")
        clip_loss_func = CLIPLoss(
            self.device,
            lambda_direction=1,
            lambda_patch=0,
            lambda_global=0,
            lambda_manifold=0,
            lambda_texture=0,
            clip_model=self.args.clip_model_name,
        )
        id_loss_func = id_loss.IDLoss().to(self.device).eval()
        soundclip_loss = SoundCLIPLoss(self.args).to(self.device)
        vgg_loss = VGGPerceptualLoss().to(self.device)

        # *********** Precompute Latents ************#
        print("Prepare identity latents")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = self.args.bs_train
        img_lat_pairs_dic = {}
        for mode in ["train", "test"]:
            img_lat_pairs = []
            pairs_path = os.path.join(
                "precomputed/",
                f"{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth",
            )
            print(pairs_path)
            if os.path.exists(pairs_path):
                print(f"{mode} pairs exists")
                img_lat_pairs_dic[mode] = torch.load(pairs_path)
                for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs_dic[mode]):
                    tvu.save_image(
                        (x0 + 1) * 0.5,
                        os.path.join(
                            self.args.image_folder, f"{mode}_{step}_0_orig.png"
                        ),
                    )
                    tvu.save_image(
                        (x_id + 1) * 0.5,
                        os.path.join(
                            self.args.image_folder,
                            f"{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png",
                        ),
                    )

                    if step == self.args.n_precomp_img - 1:
                        break
                continue
            else:
                train_dataset, test_dataset = get_dataset(
                    self.config.data.dataset, DATASET_PATHS, self.config
                )
                loader_dic = get_dataloader(
                    train_dataset,
                    test_dataset,
                    bs_train=self.args.bs_train,
                    num_workers=self.config.data.num_workers,
                )
                loader = loader_dic[mode]

            for step, img in enumerate(loader):
                x0 = img.to(self.device)
                tvu.save_image(
                    (x0 + 1) * 0.5,
                    os.path.join(self.args.image_folder, f"{mode}_{step}_0_orig.png"),
                )
                x = x0.clone()

                model.eval()
                with torch.no_grad():
                    with tqdm(
                        total=len(seq_inv), desc=f"Inversion process {mode} {step}"
                    ) as progress_bar:
                        for it, (i, j) in enumerate(
                            zip((seq_inv_next[1:]), (seq_inv[1:]))
                        ):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(
                                x,
                                t=t,
                                t_next=t_prev,
                                models=model,
                                logvars=self.logvar,
                                sampling_type="ddim",
                                b=self.betas,
                                eta=0,
                                learn_sigma=learn_sigma,
                            )

                            progress_bar.update(1)
                    x_lat = x.clone()
                    tvu.save_image(
                        (x_lat + 1) * 0.5,
                        os.path.join(
                            self.args.image_folder,
                            f"{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png",
                        ),
                    )

                    with tqdm(
                        total=len(seq_inv), desc=f"Generative process {mode} {step}"
                    ) as progress_bar:
                        for it, (i, j) in enumerate(
                            zip(reversed((seq_inv)), reversed((seq_inv_next)))
                        ):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(
                                x,
                                t=t,
                                t_next=t_next,
                                models=model,
                                logvars=self.logvar,
                                sampling_type=self.args.sample_type,
                                b=self.betas,
                                learn_sigma=learn_sigma,
                            )
                            progress_bar.update(1)

                    img_lat_pairs.append(
                        [x0, x.detach().clone(), x_lat.detach().clone()]
                    )
                tvu.save_image(
                    (x + 1) * 0.5,
                    os.path.join(
                        self.args.image_folder,
                        f"{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png",
                    ),
                )
                if step == self.args.n_precomp_img - 1:
                    break

            img_lat_pairs_dic[mode] = img_lat_pairs
            pairs_path = os.path.join(
                "precomputed/",
                f"{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth",
            )
            torch.save(img_lat_pairs, pairs_path)

        # *********** Finetune Diffusion Models ************#
        print("Start finetuning")
        print(
            f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}"
        )
        if self.args.n_train_step != 0:
            seq_train = np.linspace(0, 1, self.args.n_train_step) * self.args.t_0
            seq_train = [int(s) for s in list(seq_train)]
            print("Uniform skip type")
        else:
            seq_train = list(range(self.args.t_0))
            print("No skip")
        seq_train_next = [-1] + list(seq_train[:-1])

        seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
        seq_test = [int(s) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])

        for src_txt, trg_txt in zip(self.src_txts, self.trg_txts):
            print(f"CHANGE {src_txt} TO {trg_txt}")
            model.module.load_state_dict(diffusion_ckpt)
            optim_ft.load_state_dict(init_opt_ckpt)
            scheduler_ft.load_state_dict(init_sch_ckpt)
            clip_loss_func.target_direction = None

            # *********** Train ************#
            for it_out in range(self.args.n_iter):
                exp_id = os.path.split(self.args.exp)[-1]
                if not os.path.exists(f"checkpoint/{exp_id}_{self.args.edit_attr}"):
                    os.mkdir(f"checkpoint/{exp_id}_{self.args.edit_attr}")
                save_name = f"checkpoint/{exp_id}_{self.args.edit_attr}/{it_out}.pth"
                if self.args.do_train:
                    if os.path.exists(save_name):
                        print(f"{save_name} already exists.")
                        model.module.load_state_dict(torch.load(save_name))
                        continue
                    else:
                        for step, (x0, x_id, x_lat) in enumerate(
                            img_lat_pairs_dic["train"]
                        ):
                            model.train()
                            time_in_start = time.time()

                            optim_ft.zero_grad()
                            x = x_lat.clone()

                            with tqdm(
                                total=len(seq_train), desc=f"CLIP iteration"
                            ) as progress_bar:
                                for t_it, (i, j) in enumerate(
                                    zip(reversed(seq_train), reversed(seq_train_next))
                                ):
                                    # check: t means time step
                                    t = (torch.ones(n) * i).to(self.device)
                                    t_next = (torch.ones(n) * j).to(self.device)

                                    # check: x.shape is torch.Size([1, 3, 256, 256])
                                    x = denoising_step(
                                        x,
                                        t=t,
                                        t_next=t_next,
                                        models=model,
                                        logvars=self.logvar,
                                        sampling_type=self.args.sample_type,
                                        b=self.betas,
                                        eta=self.args.eta,
                                        learn_sigma=learn_sigma,
                                    )

                                    progress_bar.update(1)

                            # TODO: Check the objective loss function and loss function's input (is that audio or image?)

                            # > CLIPLoss is the key component to supervise the optimization
                            # check x0: original image, x: reconstructed image, src_txt: original text, trg_txt: target text
                            loss_clip = (
                                2 - clip_loss_func(x0, src_txt, x, trg_txt)
                            ) / 2
                            loss_clip = -torch.log(loss_clip)

                            # > identity loss composed of the L1 loss and face identity loss
                            # > The identity loss is employed to prevent the unwanted changes and preserve the identity of the object.
                            # > loss_id means face identity loss
                            loss_id = torch.mean(id_loss_func(x0, x))
                            loss_l1 = nn.L1Loss()(x0, x)
                            loss_l2 = nn.MSELoss()(
                                x0, x
                            )  # because diffusion uses L2 loss
                            loss_vgg = vgg_loss(x0, x)

                            loss_soundclip = soundclip_loss(x, audio_inputs)

                            # > Below equation composed of the directional CLIP loss and the identity loss
                            # > Direct Code Optimization Loss
                            # TODO: l1 to l2
                            # loss = self.args.clip_loss_w * loss_clip + self.args.id_loss_w * loss_id + self.args.l1_loss_w * loss_l1 + self.args.soundclip_loss_w * loss_soundclip
                            # loss = self.args.clip_loss_w * loss_clip + self.args.id_loss_w * loss_id + self.args.l2_loss_w * loss_l2 + self.args.soundclip_loss_w * loss_soundclip
                            # loss = self.args.clip_loss_w * loss_clip + self.args.id_loss_w * loss_id + self.args.l2_loss_w * loss_l2 + self.args.soundclip_loss_w * loss_soundclip + self.args.vgg_loss_w * loss_vgg
                            loss = (
                                self.args.clip_loss_w * loss_clip
                                + self.args.id_loss_w * loss_id
                                + self.args.l1_loss_w * loss_l1
                                + self.args.l2_loss_w * loss_l2
                                + self.args.soundclip_loss_w * loss_soundclip
                                + self.args.vgg_loss_w * loss_vgg
                            )

                            loss.backward()

                            optim_ft.step()
                            # TODO : why loss_soundclip is a tensor?
                            # print(f"CLIP {step}-{it_out}: loss_id: {loss_id:.3f}, loss_clip: {loss_clip:.3f}, loss_soundclip: {loss_soundclip:.3f}")
                            print(
                                f"CLIP {step}-{it_out}: loss_id: {loss_id:.3f}, loss_clip: {loss_clip:.3f}"
                            )
                            wandb.log(
                                {
                                    "loss_id": loss_id,
                                    "loss_clip": loss_clip,
                                    "loss_soundclip": loss_soundclip,
                                    "loss_soundclip": loss_soundclip.float(),
                                    "loss_vgg": loss_vgg.float(),
                                    "loss_l2": loss_l2.float(),
                                }
                            )

                            if self.args.save_train_image:
                                tvu.save_image(
                                    (x + 1) * 0.5,
                                    os.path.join(
                                        self.args.image_folder,
                                        f'train_{step}_2_clip_{trg_txt.replace(" ", "_")}_{it_out}_ngen{self.args.n_train_step}.png',
                                    ),
                                )
                            time_in_end = time.time()
                            print(
                                f"Training for 1 image takes {time_in_end - time_in_start:.4f}s"
                            )
                            if step == self.args.n_train_img - 1:
                                break

                        if isinstance(model, nn.DataParallel):
                            torch.save(model.module.state_dict(), save_name)
                        else:
                            torch.save(model.state_dict(), save_name)
                        print(f"Model {save_name} is saved.")
                        scheduler_ft.step()

                # *********** Eval ************#
                if self.args.do_test:
                    if not self.args.do_train:
                        print(save_name)
                        model.module.load_state_dict(torch.load(save_name))

                    wandb_images = []
                    model.eval()
                    img_lat_pairs = img_lat_pairs_dic[mode]
                    for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs):
                        with torch.no_grad():
                            x = x_lat
                            with tqdm(
                                total=len(seq_test), desc=f"Eval iteration"
                            ) as progress_bar:
                                for i, j in zip(
                                    reversed(seq_test), reversed(seq_test_next)
                                ):
                                    t = (torch.ones(n) * i).to(self.device)
                                    t_next = (torch.ones(n) * j).to(self.device)

                                    x = denoising_step(
                                        x,
                                        t=t,
                                        t_next=t_next,
                                        models=model,
                                        logvars=self.logvar,
                                        sampling_type=self.args.sample_type,
                                        b=self.betas,
                                        eta=self.args.eta,
                                        learn_sigma=learn_sigma,
                                    )

                                    progress_bar.update(1)

                            print(f"Eval {step}-{it_out}")
                            tvu.save_image(
                                (x + 1) * 0.5,
                                os.path.join(
                                    self.args.image_folder,
                                    f'{mode}_{step}_2_clip_{trg_txt.replace(" ", "_")}_{it_out}_ngen{self.args.n_test_step}.png',
                                ),
                            )
                            image = wandb.Image(
                                (x + 1) * 0.5,
                                caption=f'{mode}_{step}_2_clip_{trg_txt.replace(" ", "_")}_{it_out}_ngen{self.args.n_test_step}.png',
                            )
                            wandb_images.append(image)
                            if step == self.args.n_test_img - 1:
                                break
                    wandb.log({"eval": wandb_images})

    def edit_one_image(self):
        # *********** Data ************#
        n = self.args.bs_test

        if self.args.align_face and self.config.data.dataset in ["FFHQ", "CelebA_HQ"]:
            try:
                img = run_alignment(
                    self.args.img_path, output_size=self.config.data.image_size
                )
            except:
                img = Image.open(self.args.img_path).convert("RGB")
        else:
            img = Image.open(self.args.img_path).convert("RGB")

        img = img.resize(
            (self.config.data.image_size, self.config.data.image_size), Image.ANTIALIAS
        )
        img = np.array(img) / 255  # for normalization
        img = (
            torch.from_numpy(img)
            .type(torch.FloatTensor)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(n, 1, 1, 1)
        )  # ! Should i do to(device)??
        img = img.to(self.config.device)
        tvu.save_image(img, os.path.join(self.args.image_folder, f"0_orig.png"))
        x0 = (img - 0.5) * 2.0

        # *********** Model ************#
        # *********** Load Diffusion Model ************#
        # DiffusionCLIP authors provided this pretrained model
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":  # Bedroom
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":  # Church
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":  # Human Face
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset == "AFHQ":
            pass
        else:
            raise ValueError

        if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
            model = DDPM(self.config)
            if self.args.model_path:
                diffusion_ckpt = torch.load(self.args.model_path)
            else:
                # URL is forbidden
                # diffusion_ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
                diffusion_ckpt = torch.load(
                    "pretrained/diffusionclip/church_outdoor.ckpt"
                )
            learn_sigma = False
            print("Original diffusion Model loaded.")
        elif self.config.data.dataset in ["FFHQ", "AFHQ"]:
            model = i_DDPM(self.config.data.dataset)
            if self.args.model_path:
                diffusion_ckpt = torch.load(self.args.model_path)
            else:
                diffusion_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            learn_sigma = True
            print("Improved diffusion Model loaded.")
        else:
            print("Not implemented dataset")
            raise ValueError

        models = []

        if self.args.hybrid_noise:
            model_paths = [None] + HYBRID_MODEL_PATHS
        else:
            model_paths = [None, self.args.model_path]

        for model_path in model_paths:
            if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
                model_i = DDPM(self.config)
                if model_path:
                    ckpt = torch.load(model_path)
                else:
                    # ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
                    ckpt = torch.load("pretrained/diffusionclip/church_outdoor.ckpt")
                learn_sigma = False
            elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
                model_i = i_DDPM(self.config.data.dataset)
                if model_path:
                    ckpt = torch.load(model_path)
                else:
                    ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
                learn_sigma = True
            else:
                print("Not implemented dataset")
                raise ValueError
            model_i.load_state_dict(ckpt)
            model_i.to(self.device)
            model_i = torch.nn.DataParallel(model_i)
            model_i.eval()
            print(f"{model_path} is loaded.")
            models.append(model_i)

        # *********** Invert Image to Latent in case of Deterministic Inversion process ************#
        with torch.no_grad():
            if self.args.deterministic_inv:
                x_lat_path = os.path.join(
                    self.args.image_folder,
                    f"x_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}.pth",
                )
                if not os.path.exists(x_lat_path):
                    # Latent Initialize with random noise
                    seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
                    seq_inv = [int(s) for s in list(seq_inv)]
                    seq_inv_next = [-1] + list(seq_inv[:-1])

                    x = x0.clone()
                    with tqdm(
                        total=len(seq_inv), desc=f"Inversion process "
                    ) as progress_bar:
                        for it, (i, j) in enumerate(
                            zip((seq_inv_next[1:]), (seq_inv[1:]))
                        ):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(
                                x,
                                t=t,
                                t_next=t_prev,
                                models=models,
                                logvars=self.logvar,
                                sampling_type="ddim",
                                b=self.betas,
                                eta=0,
                                learn_sigma=learn_sigma,
                                ratio=0,
                            )

                            progress_bar.update(1)
                        x_lat = x.clone()
                        print(
                            "x_lat shape: ",
                            x_lat.shape,
                            "if it is not (1, 256, 256, 3) it have to be modificated",
                        )
                        torch.save(x_lat, x_lat_path)
                else:
                    print("Latent exists.")
                    x_lat = torch.load(x_lat_path)

            # *********** Generative Process ************#
            print(
                f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}, "
                f" Steps: {self.args.n_test_step}/{self.args.t_0}"
            )
            if self.args.n_test_step != 0:
                seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
                seq_test = [int(s) for s in list(seq_test)]
                print("Uniform skip type")
            else:
                seq_test = list(range(self.args.t_0))
                print("No skip")
            seq_test_next = [-1] + list(seq_test[:-1])

            # may be here, authors doing 'Fine-tune Diffusion model' from CLIP latent space
            for it in range(self.args.n_iter):
                if self.args.deterministic_inv:
                    x = x_lat.clone()
                else:
                    # *********** Stochastic Forward Process ************#
                    e = torch.randn_like(x0)
                    a = (1 - self.betas).cumprod(dim=0)
                    x = (
                        x0 * a[self.args.t_0 - 1].sqrt()
                        + e * (1.0 - a[self.args.t_0 - 1]).sqrt()
                    )
                tvu.save_image(
                    (x + 1) * 0.5,
                    os.path.join(
                        self.args.image_folder, f"1_lat_ninv{self.args.n_inv_step}.png"
                    ),
                )

                with tqdm(
                    total=len(seq_test), desc="Generative process {}".format(it)
                ) as progress_bar:
                    for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                        t = (torch.ones(n) * i).to(self.device)
                        t_next = (torch.ones(n) * j).to(self.device)

                        x = denoising_step(
                            x,
                            t=t,
                            t_next=t_next,
                            models=models,
                            logvars=self.logvar,
                            sampling_type=self.args.sample_type,
                            b=self.betas,
                            eta=self.args.eta,
                            learn_sigma=learn_sigma,
                            ratio=self.args.model_ratio,
                            hybrid=self.args.hybrid_noise,
                            hybrid_config=HYBRID_CONFIG,
                        )

                        # added intermediate step vis
                        if (i - 99) % 100 == 0:
                            tvu.save_image(
                                (x + 1) * 0.5,
                                os.path.join(
                                    self.args.image_folder,
                                    f"2_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_{i}_it{it}.png",
                                ),
                            )
                        progress_bar.update(1)

                x0 = x.clone()
                if self.args.model_path:
                    tvu.save_image(
                        (x + 1) * 0.5,
                        os.path.join(
                            self.args.image_folder,
                            f"3_gen_t{self.args.t_0}_it{it}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_mrat{self.args.model_ratio}_{self.args.model_path.split('/')[-1].replace('.pth','')}.png",
                        ),
                    )
                else:
                    tvu.save_image(
                        (x + 1) * 0.5,
                        os.path.join(
                            self.args.image_folder,
                            f"3_gen_t{self.args.t_0}_it{it}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_mrat{self.args.model_ratio}.png",
                        ),
                    )
