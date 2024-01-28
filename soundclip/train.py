import torch
import clip
import random
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from datasets import VggsoundCurationDataset, AudiosetBalancedCurationDataset, AudiosetUnbalancedCurationDataset
from models import AudioEncoder
import torch.nn.functional as F
import math
import time
import wandb
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Audio Text Clip Implementation")

parser.add_argument("--epochs", default=50, type=int,
                help="epochs of training")
parser.add_argument("--batch_size", default=491, type=int,
                help="batch size of training")

# because # of dataset is 150,246
# 491 * 306 = 150,246
# When batch_size is 800, CUDA out of memory error occurs.

parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.8685, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--step_size', default=1, type=float,
                    help='Step size for SGD')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')


args = parser.parse_args()



if __name__ == "__main__":
    wandb.login()

    run = wandb.init(project="soundclip", reinit=True, config={
        "epochs" : args.epochs,
        "batch_size" : args.batch_size,
        "learning_rate" : args.lr,
    })

    random.seed(42)

    vggsound_dataset = VggsoundCurationDataset()
    # audioset_balanced_dataset = AudiosetBalancedCurationDataset()
    # audioset_unbalanced_dataset = AudiosetUnbalancedCurationDataset()

    # dataset = torch.utils.data.ConcatDataset([vggsound_dataset, audioset_balanced_dataset, audioset_unbalanced_dataset])
    # dataset = torch.utils.data.ConcatDataset([vggsound_dataset])

    lengths = len(vggsound_dataset)
    train_dataset, validation_dataset = torch.utils.data.random_split(vggsound_dataset, [int(0.8 * lengths), lengths - int(0.8 * lengths)])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/16", device=device)    # because DiffusionCLIP uses ViT-B/16

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True)

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True)

    audioencoder = AudioEncoder()
    audioencoder = audioencoder.cuda()

    optimizer = optim.SGD(audioencoder.parameters(), lr=args.lr,
               momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=5, mode="triangular")

    ce = torch.nn.CrossEntropyLoss()
    audioencoder.train()

    min_validation_loss_value = 50000

    # check: Author said this is old version.
    for epoch in range(args.epochs):
        start = time.time()
        train_loss_value, validation_loss_value = 0, 0

        for idx, (batch_audio, batch_audio_aug, batch_text) in tqdm(enumerate(train_dataloader), desc="Soundclip Training"):

            audio_embedding = audioencoder(batch_audio.cuda())
            audio_aug_embedding = audioencoder(batch_audio_aug.cuda())

            text_tokens = torch.cat([clip.tokenize(text) for text in batch_text])

            with torch.no_grad():
                text_embedding = clip_model.encode_text(text_tokens.to(device)).float()
                text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

            optimizer.zero_grad()
            audio_embedding = audio_embedding / audio_embedding.norm(dim=-1, keepdim=True)
            audio_aug_embedding = audio_aug_embedding / audio_aug_embedding.norm(dim=-1, keepdim=True)

            loss = 0

            projection_audio_text = (audio_embedding @ text_embedding.T) * math.exp(0.07)
            projection_self_audio = (audio_embedding @ audio_aug_embedding.T) * math.exp(0.07)

            label = torch.arange(args.batch_size, dtype=torch.long).cuda()

            audio_contrastive_loss = ce(projection_audio_text, label) + ce(projection_audio_text.T, label)
            self_contrastive_loss = ce(projection_self_audio, label) + ce(projection_self_audio.T, label)
            loss = (audio_contrastive_loss + self_contrastive_loss) / 4
            loss.backward()

            optimizer.step()

            train_loss_value += loss.item()

            if idx % 100 == 0:
                wandb.log({"Batch" : idx, "Train Total Loss" : loss.item(), "Train Audio Loss" : audio_contrastive_loss.item(), "Train Self Loss" : self_contrastive_loss.item()})
                print("VGG, Batch : {:3d} , total loss : {:.3f}, audio loss : {:.3f}, self loss : {:.3f}".format(idx, loss.item(), audio_contrastive_loss.item(), self_contrastive_loss.item()))

        scheduler.step()

        print("Validation !")
        for idx, (batch_audio, batch_audio_aug, batch_text) in tqdm(enumerate(validation_dataloader), desc="Soundclip Validation"):

            with torch.no_grad():
                audio_embedding = audioencoder(batch_audio.cuda())
                audio_aug_embedding = audioencoder(batch_audio_aug.cuda())

                text_tokens = torch.cat([clip.tokenize(text) for text in batch_text])
                text_embedding = clip_model.encode_text(text_tokens.to(device)).float()
                text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

                audio_embedding = audio_embedding / audio_embedding.norm(dim=-1, keepdim=True)
                audio_aug_embedding = audio_aug_embedding / audio_aug_embedding.norm(dim=-1, keepdim=True)

                loss = 0

                projection_audio_text = (audio_embedding @ text_embedding.T) * math.exp(0.07)
                projection_self_audio = (audio_embedding @ audio_aug_embedding.T) * math.exp(0.07)

                label = torch.arange(args.batch_size, dtype=torch.long).cuda()

                audio_contrastive_loss = ce(projection_audio_text, label) + ce(projection_audio_text.T, label)
                self_contrastive_loss = ce(projection_self_audio, label) + ce(projection_self_audio.T, label)
                loss = (audio_contrastive_loss + self_contrastive_loss) / 4

            validation_loss_value += loss.item()
            if idx % 100 == 0:
                wandb.log({"Validation Batch": idx, "Validation Total Loss": loss.item(), "Validation Audio Loss": audio_contrastive_loss.item(), "Validation Self Loss": self_contrastive_loss.item()})
                print("VGG, Batch : {:3d} , total loss : {:.3f}, audio loss : {:.3f}, self loss : {:.3f}".format(idx, loss.item(), audio_contrastive_loss.item(), self_contrastive_loss.item()))

        # train_dataloader and validation_dataloader are both zero.
        print("Epoch : {:2d} , train loss : {:.5f}, validation loss : {:.5f}, Time : {}".format(epoch, train_loss_value / len(train_dataloader), validation_loss_value / len(validation_dataloader), time.time() - start))
        if min_validation_loss_value > validation_loss_value:
            save_path = "./pretrained/audio_encoder/audio_encoder" + str(epoch) + ".pth"
            torch.save(audioencoder.state_dict(), save_path)
            min_validation_loss_value = validation_loss_value