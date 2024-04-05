""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import json
import os
import re

from pathlib import Path

from timm.utils import NativeScaler

from datasets import build_dataset

from models import UltraLight_VM_UNet

from util.samplers import RASampler
from util import utils as utils
from util.lr_scheduler import create_lr_scheduler
from util.optimizer import SophiaG
from util.engine import train_one_epoch, evaluate
from util.losses import BceDiceLoss


def get_args_parser():
    parser = argparse.ArgumentParser(
        'UltraLight-VMUNet training and evaluation script', add_help=False)
    parser.add_argument('--data_root', default='/data/', help='dataset')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--nb_classes', default=1, type=int, help='num_classes')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')

    # TODO Dataloader parameters
    parser.add_argument('--sync_bn', type=bool, default=False, help='whether using SyncBatchNorm')
    parser.add_argument('--pin_mem', type=bool, default=False, help='pin-memory in dataloader')
    parser.add_argument('--num_workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # TODO Optimizer parameters
    parser.add_argument('--lr', default=0.0003, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    # TODO Transfer learning & checkpoint
    parser.add_argument('--output-dir', default='./multi_train', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # TODO Distributed training
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist-eval', action='store_true',
                        default=False, help='Enabling distributed evaluation')
    # parser.add_argument("--amp", default=False, type=bool,
    #                     help="Use torch.cuda.amp for mixed precision training")
    return parser


def main(args):
    print(args)
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    dataset_train, dataset_val = build_dataset(args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    print(f"Creating model")

    model = UltraLight_VM_UNet(num_classes=args.nb_classes)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr

    print('*****************')
    print('Initial LR is ', linear_scaled_lr)
    print('*****************')

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_scaler = NativeScaler()
    lr_scheduler = create_lr_scheduler(optimizer, len(data_loader_train), args.epochs, warmup=True)

    criterion = BceDiceLoss()

    best_score = 0.0

    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "model.txt").open("a") as f:
            f.write(str(model))
    if args.output_dir and utils.is_main_process():
        with (output_dir / "args.txt").open("a") as f:
            f.write(json.dumps(args.__dict__, indent=2) + "\n")
    if args.resume or os.path.exists(f'{args.output_dir}/best_checkpoint.pth'):
        args.resume = f'{args.output_dir}/best_checkpoint.pth'
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            print("Loading local checkpoint at {}".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
        msg = model_without_ddp.load_state_dict(checkpoint['model'])
        print(msg)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:

            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():  # load parameters to cuda
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            best_score = checkpoint['best_score']
            print(f'Now max accuracy is {best_score}')
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:
        # util.replace_batchnorm(model) # Users may choose whether to merge Conv-BN layers during eval
        confmat = evaluate(data_loader_val, model, device, args)
        val_info = str(confmat)
        print(
            f"Accuracy of the network on the {len(dataset_val)} + \n"
        )
        print(val_info)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        mean_loss, lr = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, args.clip_mode,
            # set_training_mode=args.finetune == ''  # keep in eval mode during finetuning
            set_training_mode=True,
            set_bn_eval=args.set_bn_eval,  # set bn to eval if finetune
            lr_scheduler=lr_scheduler,
        )

        confmat = evaluate(data_loader_val, model, device, args)

        val_info = str(confmat)

        # Extract meanIoU info
        pattern = r"mean\s+IoU:\s+(\d+\.\d+)"
        match = re.search(pattern, val_info)
        if match:
            mean_iou = match.group(1)
            print("Mean IoU:", mean_iou)

        # TODO only operate in main process
        if args.rank in [-1, 0]:
            with open(results_file, "a") as f:
                train_info = f"[epoch: {epoch}]\n" \
                             f"train_loss: {mean_loss:.4f}\n" \
                             f"lr: {lr:.6f}\n"
                f.write(train_info + val_info + "\n\n")

        if best_score < mean_iou:
            best_score = mean_iou
            if args.output_dir:
                ckpt_path = os.path.join(output_dir, 'best_checkpoint.pth')
                checkpoint_paths = [ckpt_path]
                print("Saving checkpoint to {}".format(ckpt_path))
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'best_score': best_score,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)

        print(f'Max MeanIoU: {best_score:.3f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'UltraLight-VMUNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)