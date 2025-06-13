"""
Training for CycleGAN

Pro"grammed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-05: Initial coding
* 2022-12-21: Small revision of code, checked that it works with latest PyTorch version
"""
import torch
import warnings
warnings.filterwarnings("ignore")
from dataset import HorseZebraDataset
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_SN import Discriminator
from unetsagenerator import Generator
import numpy as np
import matplotlib.pyplot as plt
from pytorch_msssim import ssim, ms_ssim
import lpips  # 导入 LPIPS 库
from torch.optim import lr_scheduler
import torch_fidelity


def calculate_fid_kid(gen_H, gen_Z, val_loader):
    weights_path = 'weights-inception-2015-12-05-6726825d.pth'
    # 计算 Horse ↔ Fake Horse 的 FID 和 KID
    metrics_horse = torch_fidelity.calculate_metrics(
        input1="data/val1/Tomato___Bacterial_spot1",  # 真实患病叶子的图像的文件夹
        input2="saved_image_val/Tomato___Bacterial_spot",  # 生成患病叶子的图像的文件夹
        cuda=True,  # 使用 GPU 计算
        batch_size=32,  # 批量大小，可以调整以适应显存
        fid=True,  # 显式计算 FID
        kid=True,  # 显式计算 KID
        verbose=False,
        inception_weights_path=weights_path
    )
    print(f"Tomato___Bacterial_spot ↔ Fake Tomato___Bacterial_spot FID: {metrics_horse['frechet_inception_distance']:.4f}, KID: {metrics_horse['kernel_inception_distance_mean']:.4f}")

    # 计算 Zebra ↔ Fake Zebra 的 FID 和 KID
    metrics_zebra = torch_fidelity.calculate_metrics(
        input1="data/val1/healthy1",  # 真实健康叶片图像的文件夹
        input2="saved_image_val/healthy",  # 生成健康叶片图像的文件夹
        cuda=True,  # 使用 GPU 计算
        batch_size=32,  # 批量大小
        fid=True,  # 显式计算 FID
        kid=True,  # 显式计算 KID
        verbose=False,
        inception_weights_path=weights_path
    )
    print(f"healthy ↔ Fake healthy FID: {metrics_zebra['frechet_inception_distance']:.4f}, KID: {metrics_zebra['kernel_inception_distance_mean']:.4f}")


class MyLambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


def PSNR(fake, real, pixel_max=2.0):
    # 确保输入是PyTorch张量并转换为float32
    if not isinstance(fake, torch.Tensor):
        fake = torch.tensor(fake, dtype=torch.float32)
    else:
        fake = fake.float()
    if not isinstance(real, torch.Tensor):
        real = torch.tensor(real, dtype=torch.float32)
    else:
        real = real.float()

    # 确保输入在相同设备
    fake = fake.to(real.device)

    # 调整维度顺序至[N, C, H, W]
    def format_tensor(tensor):
        if tensor.dim() == 3:
            # 若通道在最后一维（HWC），调整为CHW
            if tensor.size(-1) in [1, 3]:  # 假设通道数为1或3
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            else:
                tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 4:
            # 若为NHWC，调整为NCHW
            if tensor.size(1) not in [1, 3]:
                tensor = tensor.permute(0, 3, 1, 2)
        return tensor

    fake = format_tensor(fake)
    real = format_tensor(real)

    # 计算MSE
    mse = torch.mean((fake - real) ** 2)

    if mse < 1e-10:
        return float('inf')  # 或返回100.0视需求而定

    psnr = 20 * torch.log10(pixel_max / torch.sqrt(mse))
    return psnr.item()


def validate_model(gen_H, gen_Z, val_loader, lpips_model):
    gen_H.eval()
    gen_Z.eval()
    psnr_horse_scores = []
    psnr_zebra_scores = []
    ssim_horse_scores = []
    ssim_zebra_scores = []
    ms_ssim_horse_scores = []
    ms_ssim_zebra_scores = []
    lpips_horse_scores = []
    lpips_zebra_scores = []

    with torch.no_grad():
        for i, (zebra, horse) in enumerate(val_loader):
            horse = horse.to(config.DEVICE)
            zebra = zebra.to(config.DEVICE)

            fake_horse = gen_H(zebra)
            fake_zebra = gen_Z(horse)

            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)

            # 在这里添加保存图像的代码
            if i % 1 == 0:
                save_image(fake_horse * 0.5 + 0.5, f"saved_image_val/Tomato___Bacterial_spot/fake_horse_{i}.png")
                save_image(fake_zebra * 0.5 + 0.5, f"saved_image_val/healthy/fake_zebra_{i}.png")


            # 计算 PSNR
            psnr_horse = PSNR(cycle_horse, horse)
            psnr_zebra = PSNR(cycle_zebra, zebra)
            psnr_horse_scores.append(psnr_horse)
            psnr_zebra_scores.append(psnr_zebra)

            # 计算 SSIM
            ssim_horse = ssim(cycle_horse, horse, data_range=2.0, size_average=True)
            ssim_zebra = ssim(cycle_zebra, zebra, data_range=2.0, size_average=True)
            ssim_horse_scores.append(ssim_horse.item())
            ssim_zebra_scores.append(ssim_zebra.item())

            # 计算 MS-SSIM
            ms_ssim_horse = ms_ssim(cycle_horse, horse, data_range=2.0, size_average=True)
            ms_ssim_zebra = ms_ssim(cycle_zebra, zebra, data_range=2.0, size_average=True)
            ms_ssim_horse_scores.append(ms_ssim_horse.item())
            ms_ssim_zebra_scores.append(ms_ssim_zebra.item())

            # 计算 LPIPS
            lpips_zebra = lpips_model(fake_zebra, zebra)
            lpips_horse = lpips_model(fake_horse, horse)
            lpips_horse_scores.append(lpips_horse.item())
            lpips_zebra_scores.append(lpips_zebra.item())


    gen_H.train()
    gen_Z.train()

    # 分别计算马和斑马的平均 PSNR、SSIM、MS-SSIM
    avg_psnr_horse = np.mean(psnr_horse_scores)
    avg_psnr_zebra = np.mean(psnr_zebra_scores)

    avg_ssim_horse = np.mean(ssim_horse_scores)
    avg_ssim_zebra = np.mean(ssim_zebra_scores)

    avg_ms_ssim_horse = np.mean(ms_ssim_horse_scores)
    avg_ms_ssim_zebra = np.mean(ms_ssim_zebra_scores)

    avg_lpips_horse = np.mean(lpips_horse_scores)
    avg_lpips_zebra = np.mean(lpips_zebra_scores)

    gen_H.train()
    gen_Z.train()

    return (avg_psnr_horse, avg_psnr_zebra, avg_ssim_horse, avg_ssim_zebra,
            avg_ms_ssim_horse, avg_ms_ssim_zebra, avg_lpips_horse, avg_lpips_zebra)


# 添加噪声扰动函数
def add_noise(images, noise_factor=0.05):
    noise = torch.randn_like(images) * noise_factor
    return images + noise


def train_fn(
        disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)
    total_G_loss = 0

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # 添加噪声到输入的真实和伪造图像
        real_horse_noisy = add_noise(horse)  # 给真实马图像添加噪声
        fake_horse = gen_H(zebra).detach()  # 生成马图像
        fake_horse_noisy = add_noise(fake_horse)  # 给生成马图像添加噪声

        real_zebra_noisy = add_noise(zebra)  # 给真实斑马图像添加噪声
        fake_zebra = gen_Z(horse).detach()  # 生成斑马图像
        fake_zebra_noisy = add_noise(fake_zebra)  # 给生成斑马图像添加噪声

        # 训练判别器 H 和 Z
        with torch.cuda.amp.autocast():
            # 判别器 H 对真实和生成马图像的判断
            D_H_real = disc_H(real_horse_noisy)
            D_H_fake = disc_H(fake_horse_noisy)
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()

            # 判别器 H 的损失，使用标签平滑
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real) * 0.95)  # 标签平滑后的真实图像损失
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))  # 生成图像损失
            D_H_loss = D_H_real_loss + D_H_fake_loss

            # 判别器 Z 对真实和生成斑马图像的判断
            D_Z_real = disc_Z(real_zebra_noisy)
            D_Z_fake = disc_Z(fake_zebra_noisy)

            # 判别器 Z 的损失，使用标签平滑
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real) * 0.95)  # 标签平滑后的真实图像损失
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))  # 生成图像损失
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # 判别器的总损失
            D_loss = (D_H_loss + D_Z_loss) / 2

        # 优化判别器
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # 训练生成器 H 和 Z
        with torch.cuda.amp.autocast():
            # 生成器对抗损失
            fake_horse = gen_H(zebra)
            fake_zebra = gen_Z(horse)
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # 循环一致性损失
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            cycle_zebra_loss = l1(zebra, cycle_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)

            # 身份损失
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)
            identity_zebra_loss = l1(zebra, identity_zebra)
            identity_horse_loss = l1(horse, identity_horse)

            # 修改总损失
            G_loss = (
                    loss_G_Z
                    + loss_G_H
                    + cycle_zebra_loss * config.LAMBDA_CYCLE
                    + cycle_horse_loss * config.LAMBDA_CYCLE
                    + identity_horse_loss * config.LAMBDA_IDENTITY
                    + identity_zebra_loss * config.LAMBDA_IDENTITY
            )

        # 优化生成器
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        total_G_loss += G_loss.item()

        save_image(fake_horse * 0.5 + 0.5, f"saved_image_val/Tomato___Bacterial_spot/fake_Tomato___Bacterial_spot_{idx}.png")
        save_image(fake_zebra * 0.5 + 0.5, f"saved_image_val/healthy/fake_healthy_{idx}.png")

        if idx % 100 == 0:
            save_image(fake_horse * 0.5 + 0.5, f"saved_images/fake_Tomato___Bacterial_spot_{idx}.png")
            save_image(fake_zebra * 0.5 + 0.5, f"saved_images/fake_healthy_{idx}.png")
            save_image(horse * 0.5 + 0.5, f"saved_images/real_Tomato___Bacterial_spot_{idx}.png")
            save_image(zebra * 0.5 + 0.5, f"saved_images/real_healthy_{idx}.png")
            save_image(cycle_horse * 0.5 + 0.5, f"saved_images/cycle_Tomato___Bacterial_spot_{idx}.png")
            save_image(cycle_zebra * 0.5 + 0.5, f"saved_images/cycle_healthy_{idx}.png")

        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))

    return total_G_loss / len(loader)


def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    # 初始化 LPIPS 模型 (使用 VGG 作为 backbone)
    lpips_model = lpips.LPIPS(net='vgg').to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    # 学习率调度器从第50个epoch开始衰减
    lr_lambda = MyLambdaLR(n_epochs=config.NUM_EPOCHS_1, offset=0, decay_start_epoch=50)
    scheduler_gen = lr_scheduler.LambdaLR(opt_gen, lr_lambda.step)
    scheduler_disc = lr_scheduler.LambdaLR(opt_disc, lr_lambda.step)

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H,
            gen_H,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z,
            gen_Z,
            opt_gen,
            config.LEARNING_RATE,
        )

    dataset = HorseZebraDataset(

        root_horse=config.TRAIN_DIR + "/Tomato___Bacterial_spot",
        root_zebra=config.TRAIN_DIR + "/healthy",
        transform=config.transforms,
    )
    val_dataset = HorseZebraDataset(
        root_horse=config.VAL_DIR + "/Tomato___Bacterial_spot",
        root_zebra=config.VAL_DIR + "/healthy",
        transform=config.transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.amp.GradScaler('cuda')
    d_scaler = torch.amp.GradScaler('cuda')

    # 初始化用于存储损失的列表
    G_avg_losses = []

    for epoch in range(config.NUM_EPOCHS_1):
        avg_G_loss = train_fn(
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

        G_avg_losses.append(avg_G_loss)

# 每个epoch都计算的指标
        psnr_horse, psnr_zebra, ssim_horse, ssim_zebra, ms_ssim_horse, ms_ssim_zebra, lpips_horse, lpips_zebra = validate_model(
                gen_H, gen_Z, val_loader, lpips_model)

        # 使用平均LPIPS分数作为早停指标
        validation_metric = (lpips_horse + lpips_zebra) / 2

        print(f"Epoch {epoch}:")
        print(f"Avg G_loss: {avg_G_loss:.4f}")
        print(f"PSNR (Horse): {psnr_horse:.4f}, PSNR (Zebra): {psnr_zebra:.4f}")
        print(f"SSIM (Horse): {ssim_horse:.4f}, SSIM (Zebra): {ssim_zebra:.4f}")
        print(f"MS-SSIM (Horse): {ms_ssim_horse:.4f}, MS-SSIM (Zebra): {ms_ssim_zebra:.4f}")
        print(f"LPIPS (Horse): {lpips_horse:.4f}, LPIPS (Zebra): {lpips_zebra:.4f}")

        # 在每个epoch结束后调用学习率调度器
        scheduler_gen.step()
        scheduler_disc.step()

        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)

            if epoch % 50 == 0:
                print("Calculating FID and KID metrics for both Horse and Zebra domains...")
                calculate_fid_kid(gen_H, gen_Z, val_loader)

    # 训练后绘制损失图
    plt.figure(figsize=(10, 5))
    plt.plot(G_avg_losses, label="Average G_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Average Generator Loss Per Epoch")
    plt.show()


if __name__ == "__main__":
    main()

