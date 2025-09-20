import torch
from dataset.dataset import AnimeDataset
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
from model.discriminator import Discriminator
from model.generator import Generator
import albumentations as A
from albumentations.pytorch import ToTensorV2

transforms = A.Compose([
    A.Resize(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
    ToTensorV2(),
],
additional_targets={'image0':'image'})

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location='cuda')
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def train_fn(
    disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (anime, image) in enumerate(loop):
        anime = anime.to('cuda')
        image = image.to('cuda')

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_image = gen_H(anime)
            D_H_real = disc_H(image)
            D_H_fake = disc_H(fake_image.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_amime = gen_Z(anime)
            D_Z_real = disc_Z(image)
            D_Z_fake = disc_Z(fake_amime.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it togethor
            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_image)
            D_Z_fake = disc_Z(fake_amime)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_zebra = gen_Z(fake_image)
            cycle_horse = gen_H(fake_amime)
            cycle_zebra_loss = l1(anime, cycle_zebra)
            cycle_horse_loss = l1(image, cycle_horse)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_zebra = gen_Z(anime)
            identity_horse = gen_H(image)
            identity_zebra_loss = l1(anime, identity_zebra)
            identity_horse_loss = l1(image, identity_horse)

            # add all togethor
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_zebra_loss * 10
                + cycle_horse_loss * 10
                + identity_horse_loss * 0.0
                + identity_zebra_loss * 0.0
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_image * 0.5 + 0.5, f"saved_images/horse_{idx}.png")
            save_image(fake_amime * 0.5 + 0.5, f"saved_images/zebra_{idx}.png")

        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))

def main():
    disc_H = Discriminator().to('cuda')
    disc_Z = Discriminator().to('cuda')
    gen_Z = Generator().to('cuda')
    gen_H = Generator().to('cuda')
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=1e-5,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=1e-5,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    # if config.LOAD_MODEL:
    #     load_checkpoint(
    #         config.CHECKPOINT_GEN_H,
    #         gen_H,
    #         opt_gen,
    #         config.LEARNING_RATE,
    #     )
    #     load_checkpoint(
    #         config.CHECKPOINT_GEN_Z,
    #         gen_Z,
    #         opt_gen,
    #         config.LEARNING_RATE,
    #     )
    #     load_checkpoint(
    #         config.CHECKPOINT_CRITIC_H,
    #         disc_H,
    #         opt_disc,
    #         config.LEARNING_RATE,
    #     )
    #     load_checkpoint(
    #         config.CHECKPOINT_CRITIC_Z,
    #         disc_Z,
    #         opt_disc,
    #         config.LEARNING_RATE,
    #     )

    dataset = AnimeDataset(
        root_horse="config.TRAIN_DIR" + "/horses",
        root_zebra="config.TRAIN_DIR "+ "/zebras",
        transform=transforms,
    )
    val_dataset = AnimeDataset(
        root_horse="cyclegan_test/horse1",
        root_zebra="cyclegan_test/zebra1",
        transform=transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(20):
        train_fn(
            disc_H,
            disc_Z,
            gen_Z,
            gen_H,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        save_checkpoint(gen_H, opt_gen, filename="genh.pth.tar")
        save_checkpoint(gen_Z, opt_gen, filename="genz.pth.tar")
        save_checkpoint(disc_H, opt_disc, filename="critich.pth.tar")
        save_checkpoint(disc_Z, opt_disc, filename="criticz.pth.tar")

if __name__ == "__main__":
    main()