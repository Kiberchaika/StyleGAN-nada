'''
Train a zero-shot GAN using CLIP-based supervision.

Example commands:
    CUDA_VISIBLE_DEVICES=1 python train.py --size 1024 
                                           --batch 2 
                                           --n_sample 4 
                                           --output_dir /path/to/output/dir 
                                           --lr 0.002 
                                           --frozen_gen_ckpt /path/to/stylegan2-ffhq-config-f.pt 
                                           --iter 301 
                                           --source_class "photo" 
                                           --target_class "sketch" 
                                           --lambda_direction 1.0 
                                           --lambda_patch 0.0 
                                           --lambda_global 0.0 
                                           --lambda_texture 0.0 
                                           --lambda_manifold 0.0 
                                           --phase None 
                                           --auto_layer_k 0 
                                           --auto_layer_iters 0 
                                           --auto_layer_batch 8 
                                           --output_interval 50 
                                           --clip_models "ViT-B/32" "ViT-B/16" 
                                           --clip_model_weights 1.0 1.0 
                                           --mixing 0.0
                                           --save_interval 50
'''

import argparse
import os
import numpy as np

import os, sys
from subprocess import call, DEVNULL

'''
if sys.platform.startswith('win'):
    #output = os.popen('"{}" && set'.format("%ProgramFiles(x86)%/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/vcvars64.bat")).read()
    output = os.popen('"{}" && set'.format("%ProgramFiles(x86)%/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build/vcvars64.bat")).read()

    for line in output.splitlines():
        pair = line.split("=", 1)
        if(len(pair) >= 2):
            os.environ[pair[0]] = pair[1]
            #print(pair[0] + " = " + pair[1])
    os.system("where cl.exe")
    #print(output)
#'''

import torch

from tqdm import tqdm

from model.ZSSGAN import ZSSGAN

import shutil
import json

from utils.file_utils import copytree, save_images, save_paper_image_grid
from utils.training_utils import mixing_noise

from options.train_options import TrainOptions

from id_loss import IDLoss

#TODO convert these to proper args
SAVE_SRC = False
SAVE_DST = True


def train(args):

    # Set up networks, optimizers.
    print("Initializing networks...")
    net = ZSSGAN(args)

    args.arcface_model_paths = "pretrained/model_ir_se50.pth"
    id_loss = IDLoss(args).to(device).eval()

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1) # using original SG2 params. Not currently using r1 regularization, may need to change.

    g_optim = torch.optim.Adam(
        net.generator_trainable.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    # Set up output directories.
    sample_dir = os.path.join(args.output_dir, "sample")
    ckpt_dir   = os.path.join(args.output_dir, "checkpoint")

    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # set seed after all networks have been initialized. Avoids change of outputs due to model changes.
    torch.manual_seed(2)
    np.random.seed(2)

    # Training loop
    fixed_z = torch.randn(args.n_sample, 512, device=device)

    for i in tqdm(range(args.iter)):

        net.train()

        sample_z = mixing_noise(args.batch, 512, args.mixing, device)

        for clip_iter in range(0, args.clip_iterations):
            [sampled_src, sampled_dst], loss = net(sample_z)
            net.zero_grad()
            loss.backward()

            # tqdm.write(f"Clip loss: {loss}")

            del loss

        # [sampled_src, sampled_dst], loss = net(sample_z)
        # loss.backward()

        # del loss

        sum_dc_loss = 0.0
        if args.dc_loss_iterations > 0:
            for dc_loss_iter in range(0, args.dc_loss_iterations):
                dc_loss = net.compute_dist_consistency_loss(float(args.dc_loss_weight) / float(args.dc_loss_iterations), args.dc_loss_bypass_last_layers)
                dc_loss.backward() # distance consistency loss
                sum_dc_loss += float(dc_loss)
                del dc_loss

        # tqdm.write(f"DC Loss: {sum_dc_loss}")

        # del loss, dc_loss

        args.id_loss_iterations = 10
        if args.id_loss_iterations > 0:
            for id_loss_iter in range(0, args.id_loss_iterations):
                z = torch.randn(1, 512, device=device)

                x = net.generator_frozen([z], return_feats=False)
                y = net.generator_trainable([z], return_feats=False)
               
                x_resized = id_loss.face_pool(x)
                y_resized = id_loss.face_pool(y)

                loss_id = id_loss(y_resized, x_resized) * 0.1 # opts.id_lambda
                loss_id.backward() 

        g_optim.step()



        if i % args.output_interval == 0:
            net.eval()

            with torch.no_grad():
                [sampled_src, sampled_dst], loss = net([fixed_z], truncation=args.sample_truncation)

                if args.crop_for_cars:
                    sampled_dst = sampled_dst[:, :, 64:448, :]

                grid_rows = int(args.n_sample ** 0.5)

                if SAVE_SRC:
                    save_images(sampled_src, sample_dir, "src", grid_rows, i)

                if SAVE_DST:
                    save_images(sampled_dst, sample_dir, "dst", grid_rows, i)

        if (args.save_interval is not None) and (i > 0) and (i % args.save_interval == 0):
            torch.save(
                {
                    "g_ema": net.generator_trainable.generator.state_dict(),
                    "g_optim": g_optim.state_dict(),
                },
                f"{ckpt_dir}/{str(i).zfill(6)}.pt",
            )

    for i in range(args.num_grid_outputs):
        net.eval()

        with torch.no_grad():
            sample_z = mixing_noise(16, 512, 0, device)
            [sampled_src, sampled_dst], _ = net(sample_z, truncation=args.sample_truncation)

            if args.crop_for_cars:
                sampled_dst = sampled_dst[:, :, 64:448, :]

        save_paper_image_grid(sampled_dst, sample_dir, f"sampled_grid_{i}.png")
            

if __name__ == "__main__":
    device = "cuda"

    args = TrainOptions().parse()

    # save snapshot of code / args before training.
    os.makedirs(os.path.join(args.output_dir, "code"), exist_ok=True)
    copytree("criteria/", os.path.join(args.output_dir, "code", "criteria"), )
    shutil.copy2("model/ZSSGAN.py", os.path.join(args.output_dir, "code", "ZSSGAN.py"))
    
    with open(os.path.join(args.output_dir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    train(args)
    