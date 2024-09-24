from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.dev_path = '/mnt/data1/sqx/sot/data/dev'
    settings.got10k_lmdb_path = '/mnt/data1/sqx/sot/data/got10k_lmdb'
    settings.got10k_path = '/mnt/data1/sqx/sot/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/mnt/data1/sqx/sot/data/itb'
    settings.lasot_extension_subset_path = '/mnt/data1/sqx/sot/data/lasotextension'
    settings.lasot_lmdb_path = '/mnt/data1/sqx/sot/data/lasot_lmdb'
    settings.lasot_path = '/mnt/data1/sqx/sot/data/lasot'
    settings.network_path = '/mnt/data1/sqx/sot/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/mnt/data1/sqx/sot/data/nfs'
    settings.otb_path = '/mnt/data1/sqx/sot/data/otb'
    settings.prj_dir = '/mnt/data1/sqx/sot'
    settings.result_plot_path = '/mnt/data1/sqx/sot/output/test/result_plots'
    settings.results_path = '/mnt/data1/sqx/sot/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/mnt/data1/sqx/sot/test1'
    settings.segmentation_path = '/mnt/data1/sqx/sot/output/test/segmentation_results'
    settings.tc128_path = '/mnt/data1/sqx/sot/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/mnt/data1/sqx/sot/data/tnl2k/test'
    settings.tpl_path = ''
    settings.trackingnet_path = '/mnt/data1/sqx/sot/data/trackingnet'
    settings.uav_path = '/mnt/data1/sqx/sot/data/uav'
    settings.vot18_path = '/mnt/data1/sqx/sot/data/vot2018'
    settings.vot22_path = '/mnt/data1/sqx/sot/data/vot2022'
    settings.vot_path = '/mnt/data1/sqx/sot/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

