"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable

import torch

from .losses import BceDiceLoss
from util import utils as utils


def set_bn_state(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

def train_one_epoch(model: torch.nn.Module, criterion: BceDiceLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    set_training_mode=True,
                    set_bn_eval=False,
                    lr_scheduler=None,
                    ):

    model.train(set_training_mode)

    if set_bn_eval:
        set_bn_state(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    for idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        with torch.cuda.amp.autocast():
            loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                        parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        lr_scheduler.step()

        learning_rate = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=learning_rate)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    return metric_logger.meters["loss"].global_avg, learning_rate


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    # criterion = BceDiceLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    confmat = utils.ConfusionMatrix(args.nb_classes)

    header = 'Test:'
    # switch to evaluation mode
    model.eval()

    print_freq = 20
    for images, target in metric_logger.log_every(data_loader, print_freq, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            confmat.update(target.flatten(), output.argmax(1).flatten())


    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    confmat.reduce_from_all_processes()
    return confmat