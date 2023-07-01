import yaml
from easydict import EasyDict as edict

# Add default config

cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.HEAD_TYPE = 'CORNER'
cfg.MODEL.HIDDEN_DIM = 256
cfg.MODEL.NUM_OBJECT_QUERIES = 1
cfg.MODEL.POSITION_EMBEDDING = 'sine'  # Sine or learned
cfg.MODEL.PREDICT_MASK = False
# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = 'resnet50'  # ResNet50, ResNeXt101_32x8d
cfg.MODEL.BACKBONE.OUTPUT_LAYERS = ['layer3']
cfg.MODEL.BACKBONE.DILATION = False
# MODEL.TRANSFORMER
cfg.MODEL.TRANSFORMER = edict()
cfg.MODEL.TRANSFORMER.NHEADS = 8
cfg.MODEL.TRANSFORMER.DROPOUT = 0.1
cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD = 2048
cfg.MODEL.TRANSFORMER.ENC_LAYERS = 6
cfg.MODEL.TRANSFORMER.DEC_LAYERS = 6
cfg.MODEL.TRANSFORMER.PRE_NORM = False
cfg.MODEL.TRANSFORMER.DIVIDE_NORM = False
# MODEL.AIA
cfg.MODEL.AIA = edict()
cfg.MODEL.AIA.USE_AIA = True
cfg.MODEL.AIA.MATCH_DIM = 64
cfg.MODEL.AIA.FEAT_SIZE = 400

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = 'ADAMW'
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.IOU_WEIGHT = 2.0
cfg.TRAIN.DEEP_SUPERVISION = False
cfg.TRAIN.FREEZE_BACKBONE_BN = True
cfg.TRAIN.FREEZE_LAYERS = ['conv1', 'layer1']
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = 'step'
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ['LASOT', 'GOT10K_vot_train']
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 500
cfg.TEST.HYPER = edict()
cfg.TEST.HYPER.DEFAULT = [100, 3, 0.7]
cfg.TEST.HYPER.LASOT = [100, 4, 0.8]
cfg.TEST.HYPER.LASOT_EXT = [100, 6, 0.8]
cfg.TEST.HYPER.TRACKINGNET = [100, 6, 0.7]
cfg.TEST.HYPER.GOT10K_TEST = [80, 4, 0.7]
cfg.TEST.HYPER.NFS = [80, 3, 0.6]
cfg.TEST.HYPER.OTB = [100, 3, 0.7]
cfg.TEST.HYPER.UAV = [100, 3, 0.7]


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = dict()
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = dict()
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError('ERROR: {} not exist in config.py'.format(k))
    else:
        return


def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)
