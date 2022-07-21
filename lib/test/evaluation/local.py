from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here

    settings.got10k_path = 'PATH/GOT10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = 'PATH/LaSOT'
    settings.network_path = 'PATH/AiATrack/test/networks'  # Where tracking networks are stored
    settings.nfs_path = 'PATH/NFS30'
    settings.otb_path = 'PATH/OTB100'
    settings.prj_dir = 'PATH/AiATrack'
    settings.result_plot_path = 'PATH/AiATrack/test/result_plots'
    settings.results_path = 'PATH/AiATrack/test/tracking_results'  # Where to store tracking results
    settings.save_dir = 'PATH/AiATrack'
    settings.segmentation_path = 'PATH/AiATrack/test/segmentation_results'
    settings.tn_packed_results_path = ''
    settings.trackingnet_path = 'PATH/TrackingNet'
    settings.uav_path = 'PATH/UAV123'
    settings.show_result = False

    return settings
