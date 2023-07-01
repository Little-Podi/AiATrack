import os

from lib.config.aiatrack.config import cfg, update_config_from_file
from lib.test.evaluation.environment import env_settings
from lib.test.utils import TrackerParams


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # Update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/aiatrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg

    # Search region
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    params.checkpoint = os.path.join(save_dir, 'checkpoints/train/aiatrack/%s/AIATRACK_ep%04d.pth.tar' %
                                     (yaml_name, cfg.TEST.EPOCH))

    # Whether to save boxes from all queries
    params.save_all_boxes = False
    return params
