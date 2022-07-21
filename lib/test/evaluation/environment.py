import importlib
import os


class EnvSettings:
    def __init__(self):
        test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        self.results_path = '{}/tracking_results/'.format(test_path)
        self.segmentation_path = '{}/segmentation_results/'.format(test_path)
        self.network_path = '{}/networks/'.format(test_path)
        self.result_plot_path = '{}/result_plots/'.format(test_path)
        self.otb_path = ''
        self.nfs_path = ''
        self.uav_path = ''
        self.got10k_path = ''
        self.lasot_path = ''
        self.trackingnet_path = ''

        self.got_packed_results_path = ''
        self.got_reports_path = ''
        self.tn_packed_results_path = ''

        self.show_result = False


def env_settings():
    env_module_name = 'lib.test.evaluation.local'
    env_module = importlib.import_module(env_module_name)
    return env_module.local_env_settings()
