"""
revised by Hongling Chen and Jie Chen on 2024.07.1 in xi'an Jiaotong university.
"""
import os
os.environ['SIZE'] = str(64) # it varies with the size of data
import torch
torch.cuda.set_device(1)
import sys
sys.path.append('path/SAII-DDPM_github')
from improved_diffusion.data_tools import *
from improved_diffusion.condition_inversion import *
import argparse
import numpy as np
import torch.distributed as dist
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from pylops.utils.wavelets import ricker
from scipy.signal import filtfilt
from scipy.signal import butter
import matplotlib.pyplot as plt
import time
from torchvision.utils import make_grid
torch.cuda.empty_cache()


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10,
        batch_size=8,
        use_ddim=False,
        model_path="",
        flag=3,
        flagl=1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def add_colornoise_to_data(data, sigman):
    np.random.seed(123)
    noise = filtfilt(np.ones(3) / 3, 1,
                 filtfilt(np.ones(3) / 3, 1, np.random.normal(0, sigman, (data.shape[0], data.shape[1], data.shape[2], data.shape[3])).T, method='gust').T,
                 method='gust') # band-pass noise
    # 将噪音加入到原始数据中
    data_with_noise = data + torch.tensor(noise.copy()).to('cuda')
    return data_with_noise

########################################################################################################################
########################################################################################################################
def main():
    args = create_argparser().parse_args()
    args.image_size = 64
    args.num_channels = 128
    args.num_res_blocks = 3
    args.learn_sigma = True
    args.diffusion_steps = 4000
    args.noise_schedule = 'cosine'

    args.add_condition = True # 当这个为True时，args.img_channels = 2
    args.img_channels = 2 # 输入为2个通道，因为以低频作为条件输入到了网络中
    args.out_channels = 1
    args.batch_size = 1
    args.num_samples = 1  # run num_samples results
    args.timestep_respacing = [500]
    args.model_path = 'path/logger/weights/model_064_00300000.pt' #包含多种低频情况0.012-0.06，低频不平滑
    args.flag = 2  # inverse manner
    args.flagl = 1 # low frequency prior information


    if args.flagl == 1:
        ### smooth background
        B, A = butter(2, 0.012, 'low')
        m_loww = filtfilt(B, A, data.cpu().numpy().squeeze().T).T
        nsmoothz, nsmoothx = 2, 3
        mback = filtfilt(np.ones(nsmoothz) / float(nsmoothz), 1, m_loww,
                         axis=0)
        mback = filtfilt(np.ones(nsmoothx) / float(nsmoothx), 1, mback, axis=1)
        imp_0 = torch.tensor(np.ascontiguousarray(mback[None, None, :, :]), dtype=torch.float32).to('cuda')
        imp_0 = torch.cat([imp_0] * args.batch_size, dim=0)
    elif args.flagl == 2:
        ### constant background
        imp_0 = (torch.ones_like(data) * torch.mean(data)).to('cuda')
        imp_0 = torch.cat([imp_0] * args.batch_size, dim=0)
    else:
        imp_0 = None


    dist_util.setup_dist()
    logger.configure('image_sample')
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to('cuda')
    model.eval()

    logger.log("creating inverse_model")
    DDC_model = None

    start_time = time.time()
    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        print( args.batch_size)
        model_kwargs = {}
        if args.class_cond:
            classes = torch.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        if imp_0 is not None:
            sample, sample_list = sample_fn(
                model,
                DDC_model,
                (args.batch_size, 1, args.image_size, args.image_size),
                flag=args.flag,
                wav=wav,
                condition_x=imp_0,
                measurement=measurement,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
        else:
            sample, sample_list = sample_fn(
                model,
                (args.batch_size, 1, args.image_size, args.image_size),
                flag=args.flag,
                wav=wav,
                condition_x=measurement,
                measurement=measurement,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

        sample = sample.clamp(0, 1)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"The code took {execution_time:.5f} seconds to execute.")

        gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                torch.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")
    # pdb.set_trace()
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        out_path1 = os.path.join(logger.get_dir(), f"samples_list_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
            np.savez(out_path1, sample_list)
    dist.barrier()
    logger.log("sampling complete")

    image_grid = make_grid(torch.tensor(arr).permute(0, 3, 1, 2), nrow=4, padding=4)
    plt.figure()
    plt.imshow(image_grid.permute(1, 2, 0)[...,0],cmap=plt.cm.jet)
    # plt.colorbar()  # 添加色标
    plt.axis('off')  # 关闭坐标轴
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    dt0 = 0.002  # 2ms 采样
    ntwav = 21  # half size
    wav, twav, wavc = ricker(np.arange(ntwav) * dt0, 30)  # 30hz ricker wavelet

    data = np.load('path/datasets/test_data.npy')
    data = torch.from_numpy(data).float()
    plt.figure();plt.imshow(data[0, 0].cpu(), cmap=plt.cm.jet);plt.show()

    operator = ImpedanceOperator(np.float32(wav))
    syn_dataset1 = operator.forward_linear(data.type(torch.cuda.FloatTensor))
    syn_dataset = syn_dataset1
    syn_dataset = add_colornoise_to_data(syn_dataset1, 0.5)
    measurement = syn_dataset
    SNR = 10 * torch.log10(torch.sum(torch.square(syn_dataset1)) / torch.sum(torch.square(syn_dataset1 - syn_dataset)))
    print(SNR)
    plt.figure();
    plt.imshow(measurement.squeeze().cpu(), 'seismic');
    plt.show()

    main()





