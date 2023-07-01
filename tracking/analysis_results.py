import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [8, 8]

import _init_paths
from lib.test.analysis.plot_results import print_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = list()
trackers.extend(trackerlist(name='aiatrack', parameter_name='baseline', dataset_name='NOTU',
                            run_ids=None, display_name='Ours'))

dataset = get_dataset('otb')
print_results(trackers, dataset, 'OTB100', merge_results=True, plot_types=('success', 'prec'))
dataset = get_dataset('nfs')
print_results(trackers, dataset, 'NFS30', merge_results=True, plot_types=('success', 'prec'))
dataset = get_dataset('uav')
print_results(trackers, dataset, 'UAV123', merge_results=True, plot_types=('success', 'prec'))
