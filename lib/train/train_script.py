# For import modules
import importlib
import os

from torch import nn
from torch.nn.functional import l1_loss
# Distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP

# Network related
from lib.models.aiatrack import build_aiatrack
# Forward propagation related
from lib.train.actors import AIATRACKActor
# Train pipeline related
from lib.train.trainers import LTRTrainer
# Loss function related
from lib.utils.box_ops import giou_loss
# Some more advanced functions
from .base_functions import *


def run(settings):
    settings.description = 'training script'

    # Update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("ERROR: %s doesn't exist" % settings.cfg_file)
    config_module = importlib.import_module('lib.config.%s.config' % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)

    # Update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, '%s-%s.log' % (settings.script_name, settings.config_name))

    # Create network
    net = build_aiatrack(cfg)

    # Wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank])  # find_unused_parameters=True
        settings.device = torch.device('cuda:%d' % settings.local_rank)
    else:
        settings.device = torch.device('cuda:0')

    # Loss functions and actors
    objective = {'giou': giou_loss, 'l1': l1_loss, 'iou': nn.MSELoss()}
    loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'iou': cfg.TRAIN.IOU_WEIGHT}

    actor = AIATRACKActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

    loader_train = build_dataloaders(cfg, settings)
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler)

    # Train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
