from imports import *

MAIN_DIR = '/work/cpotvin/WOFS_ML/test'
version = 21
direction = 'forward'
perm_method = 'multipass'

num_classes=5
class_balancing = False
train_ratio = [1,1,1]
test_ratio = [1,3,1]
combine_classes = True#False
UQ = False
stratify_verif = True
stratify = False
pers_bl = ratio_bl = False
#ratio_bl = True
lead_time = 180#60#180
learn_algo = 'ORF'#'GBhist'#'OLR'
reduced_cc = 0.70
min_corr = 0.1
#reduced_cc = min_corr = False

imp_vars = 10#False
nvars = 40#28  #60-44, 120-30/41, 180-28/40      #45#37#31
num_folds = 5#6
do_train = True
do_hyper_opt = True
do_train = do_hyper_opt = False
n_iter = 100
train_test_split_param = 'dates'
tuning_CV_split_method = 'dates'
target = 'fcst_eFSS'
persistence_var = 'init_box_eFSS'
breakpoint_feature = persistence_var
baseline_feature = 'fcst_REFLCOMP_stdev_max'

if learn_algo == 'GB' or learn_algo == 'GBhist':
  n_points = 2
  n_jobs = 10
elif learn_algo == 'NN':
  n_jobs = 15
else:
  n_points = 10
  n_jobs = 50

if learn_algo == 'AT':
  do_hyper_opt = False

prelim_features_fname = '%s/features%d_%dmin.feather' % (MAIN_DIR, version, lead_time)
orig_features_fname = '%s/features_version=%d_%dmin_target=%s_num_classes=%d_stratify=%s_orig.feather' % (MAIN_DIR, version, lead_time, target, num_classes, stratify)
reduced_features_fname = orig_features_fname.split('_orig.feather')[0]+'_reduced_cc=%.2f_min_corr=%.2f.feather' % (reduced_cc, min_corr)

if reduced_cc or min_corr:
  fname = reduced_features_fname
else:
  fname = prelim_features_fname

exp_name = '%dmin_target=%s_version=%d_num_classes=%d_stratify=%s_%s_niter=%d_train-test-split_param=%s_tuning-split-by-year=%s' % (lead_time, target, version, num_classes, stratify, learn_algo, n_iter, str(train_test_split_param), tuning_CV_split_method)
if reduced_cc or min_corr:
  exp_name += '_reduced_cc=%.2f_min_corr=%.2f' % (reduced_cc, min_corr)
if pers_bl:
  exp_name += '_PERS'
if ratio_bl:
  exp_name += '_RATIO'

imp_vars_fname = '%s/imp_vars_combined_%s-%s_%s_nvars=%d.pkl' % (MAIN_DIR, direction, perm_method, exp_name, nvars)

if imp_vars and not pers_bl and not ratio_bl and 'stacked' not in learn_algo:
  exp_name += '_impvars=%d_%s' % (imp_vars, direction)
if pers_bl:
  predictions_fname = '%s/pers_preds%d_%s_orig.feather' % (MAIN_DIR, version, exp_name.split('_PERS')[0])
elif ratio_bl:
  predictions_fname = '%s/base_preds%d_%s_orig.feather' % (MAIN_DIR, version, exp_name.split('_RATIO')[0])
else:
  predictions_fname = '%s/predictions%d_%s_orig.feather' % (MAIN_DIR, version, exp_name)

best_params_fname = '%s/best_params_%s.pkl'  % (MAIN_DIR, exp_name)
best_model_fname = '%s/best_model_%s.joblib' % (MAIN_DIR, exp_name)
train_test_fname = '%s/train_test_%s.joblib' % (MAIN_DIR, exp_name)
ROC_fname =        '%s/ROC_CURVES_%s.png'    % (MAIN_DIR, exp_name)
PD_fname =         '%s/PERF_DIAG_%s.png'     % (MAIN_DIR, exp_name)
AD_fname =         '%s/ATTR_DIAG_%s.png'     % (MAIN_DIR, exp_name)
roc_data_fname =   '%s/ROC_DATA_%s.pkl'      % (MAIN_DIR, exp_name)
attr_data_fname =  '%s/ATTR_DATA_%s.pkl'     % (MAIN_DIR, exp_name)

logfile = '%s/log_%s.dat' % (MAIN_DIR, exp_name)
errfile = '%s/err_%s.dat' % (MAIN_DIR, exp_name)

class_colors = {'POOR': 'blue', 'FAIR': 'orange', 'GOOD': 'green'}

ens_stats_functions = {'median': np.median, 'stdev': np.std} #{'median': np.median, 'stdev': np.std, '90pc': np.percentile, '10pc': np.percentile}
NSE_all_stats_functions = {'median': np.nanmedian}
NSE_pos_stats_functions = {'90pc': np.nanpercentile}#, 'max': np.nanmax}
NSE_neg_stats_functions = {'10pc': np.nanpercentile}#, 'min': np.nanmin}
storm_pos_stats_functions = {'max': np.nanmax}
storm_neg_stats_functions = {'min': np.nanmin}

NSE_pos_fields = ['cin_ml', 'srh_0to3', 'cape_ml', 'stp', 'scp', 'bunk_r_tot', 'pw', 'shear_u_0to6', 'shear_v_0to6', 'shear_tot_0to6', 'w_up', 'uh_2to5_instant', 'uh_0to2_instant', 'wz_0to2_instant', 'fed', 'ws_80', 'srh_0to1', 'srh_0to500']
NSE_neg_fields = ['pw', 'shear_u_0to6', 'shear_v_0to6', 'shear_tot_0to6', 'w_down', 'okubo_weiss', 'lcl_ml']
storm_pos_fields = ['w_up', 'uh_2to5_instant', 'uh_0to2_instant', 'wz_0to2_instant', 'fed', 'REFLCOMP', 'rain', 'ws_80']
storm_neg_fields = ['w_down', 'okubo_weiss', 'lcl_ml']

storm_init_predictors = ['valid_time', 'init_time']+['init_' + feature for feature in ['mrms_storm_peak_refl', 'mrms_storm_type', 'mrms_storm_cat', 'mrms_storm_length', 'frac_nearby_storm', 'wofs_storm_peak_refl_ens_mean', 'peak_refl_ratio', 'frac_matching_storm_type', 'frac_matching_storm_cat', 'norm_dist_nearest_storm', 'storm_area_ratio', 'wofs_storm_area_ens_mean', 'mrms_storm_area', 'wofs_refl_ens_mean_99th', 'refl_99th_ratio', 'mrms_refl_99th', 'error_refl_field_local_nbrhd', 'spread_refl_field_local_nbrhd', 'ratio_refl_field_local_nbrhd', 'entropy_refl_field_local_nbrhd', 'error_refl_field_local_40DBZ', 'spread_refl_field_local_40DBZ', 'ratio_refl_field_local_40DBZ', 'entropy_refl_field_local_40DBZ', 'error_refl_field_local_40DBZ_nbrhd', 'spread_refl_field_local_40DBZ_nbrhd', 'ratio_refl_field_local_40DBZ_nbrhd', 'entropy_refl_field_local_40DBZ_nbrhd']]#+['init_subscore%d' % s for s in range(9)]+['init_object_score', 'init_field_score', 'init_comb_score']

mean_storm_init_predictors = ['mean_' + feature for feature in storm_init_predictors if feature not in ['valid_time', 'init_time']]

storm_types = ['SUPERCELL', 'ORDINARY', 'OTHER', 'QLCS', 'SUP_CLUST', 'QLCS_ORD', 'QLCS_MESO']
storm_cats = ['ORG_MULTICELL', 'CELLULAR']

storm_fcst_predictors = ['fcst_' + feature for feature in ['wofs_mean_pred_num_storms', 'wofs_mean_pred_norm_dist_nearest_storm', 'wofs_mean_pred_total_storm_area', 'wofs_stdev_pred_num_storms', 'wofs_stdev_pred_norm_dist_nearest_storm', 'wofs_stdev_pred_total_storm_area'] + ['wofs_mean_num_%s' % item for item in storm_types+storm_cats] + ['wofs_stdev_num_%s' % item for item in storm_types+storm_cats]] + ['fcst_%s_refl_field_pred_%s' % (affixes[0], affixes[1]) for affixes in itertools.product(['spread', 'entropy'], ['30DBZ_nbrhd', '30DBZ', 'nbrhd'])] + ['fcst_%s_refl_field_pred' % affix for affix in ['spread', 'entropy']]

spread_metrics = ['fcst_%s_refl_field_local_%s' % (affixes[0], affixes[1]) for affixes in itertools.product(['spread', 'error', 'ratio', 'entropy'], ['30DBZ_nbrhd', '30DBZ', 'nbrhd'])] + ['fcst_%s_refl_field_local' % affix for affix in ['spread', 'error', 'ratio', 'entropy']] + ['fcst_%s_refl_field_%s' % (affixes[0], affixes[1]) for affixes in itertools.product(['spread', 'error', 'ratio', 'entropy'], ['30DBZ_nbrhd', '30DBZ', 'nbrhd'])] + ['fcst_%s_refl_field' % affix for affix in ['spread', 'error', 'ratio', 'entropy']]


box_predictors = ['init_box_eFSS', 'init_eFSS_fcst_box', 'init_box_FSS', 'init_FSS_fcst_box', 'init_mrms_cvg', 'init_wofs_cvg', 'init_mrms_cvg_fcst_box', 'init_wofs_cvg_fcst_box'] + ['mrms_init_cvg_%ddbz' % thres for thres in [35, 40, 45]] + ['wofs_init_cvg_%ddbz' % thres for thres in [35, 40, 45]] + ['ratio_init_cvg_%ddbz' % thres for thres in [35, 40, 45]] + ['diff_init_cvg_%ddbz' % thres for thres in [35, 40, 45]] #+ ['init_box_eFSS_%d_%.1f' % (thres_scale[0], thres_scale[1]) for thres_scale in itertools.product([30, 35, 40, 45], [1, 1.5, 2])] + ['init_box_FSS_%d_%.1f' % (thres_scale[0], thres_scale[1]) for thres_scale in itertools.product([30, 35, 40, 45], [1, 1.5, 2])]

label_metrics = ['fcst_eFSS'] + ['fcst_eFSS_%d_%.1f' % (thres_scale[0], thres_scale[1]) for thres_scale in itertools.product([30, 35, 40, 45], [1, 1.5, 2] )] + ['mrms_pred_cvg_%ddbz' % thres for thres in [35, 40, 45]] + ['wofs_pred_cvg_%ddbz' % thres for thres in [35, 40, 45]] + ['ratio_pred_cvg_%ddbz' % thres for thres in [35, 40, 45]] + ['diff_pred_cvg_%ddbz' % thres for thres in [35, 40, 45]] + ['fcst_FSS'] + ['fcst_FSS_%d_%.1f' % (thres_scale[0], thres_scale[1]) for thres_scale in itertools.product([30, 35, 40, 45], [1, 1.5, 2])]#['mean_fcst_' + feature for feature in ['frac_nearby_storm', 'frac_matching_storm_type', 'frac_matching_storm_cat', 'peak_refl_ratio', 'norm_dist_nearest_storm', 'storm_area_ratio', 'eFSS']]+['mean_fcst_subscore%d' % s for s in range(9)]+['mean_fcst_object_score', 'mean_fcst_field_score', 'mean_fcst_comb_score', 'fcst_eFSS']

env_init_predictors = []; env_fcst_predictors = []
for field in NSE_pos_fields:
  for ens_func in ens_stats_functions.keys():
    for nse_func in NSE_pos_stats_functions.keys():
      if field not in ['rain', 'hailcast']:
        env_init_predictors.append('init_%s_%s_%s' % (field, ens_func, nse_func))
      env_fcst_predictors.append('fcst_%s_%s_%s' % (field, ens_func, nse_func))
for field in NSE_neg_fields:
  for ens_func in ens_stats_functions.keys():
    for nse_func in NSE_neg_stats_functions.keys():
      if field not in ['rain', 'hailcast']:
        env_init_predictors.append('init_%s_%s_%s' % (field, ens_func, nse_func))
      env_fcst_predictors.append('fcst_%s_%s_%s' % (field, ens_func, nse_func))
for field in list(set(NSE_pos_fields+NSE_neg_fields)):
  for ens_func in ens_stats_functions.keys():
    for nse_func in NSE_all_stats_functions.keys():
      if field not in ['rain', 'hailcast']:
        env_init_predictors.append('init_%s_%s_%s' % (field, ens_func, nse_func))
      env_fcst_predictors.append('fcst_%s_%s_%s' % (field, ens_func, nse_func))
for field in storm_pos_fields:
  for ens_func in ens_stats_functions.keys():
    for nse_func in storm_pos_stats_functions.keys():
      if field not in ['rain', 'hailcast']:
        env_init_predictors.append('init_%s_%s_%s' % (field, ens_func, nse_func))
      env_fcst_predictors.append('fcst_%s_%s_%s' % (field, ens_func, nse_func))
for field in storm_neg_fields:
  for ens_func in ens_stats_functions.keys():
    for nse_func in storm_neg_stats_functions.keys():
      if field not in ['rain', 'hailcast']:
        env_init_predictors.append('init_%s_%s_%s' % (field, ens_func, nse_func))
      env_fcst_predictors.append('fcst_%s_%s_%s' % (field, ens_func, nse_func))

other_features = ['lead_time', 'date', 'init_mrms_storm_type', 'init_mrms_cvg', 'init_time', 'valid_time', 'init_lat_NW', 'init_lon_NW', 'pred_lat_NW', 'pred_lon_NW', 'init_lat_NE', 'init_lon_NE', 'pred_lat_NE', 'pred_lon_NE', 'init_lat_SW', 'init_lon_SW', 'pred_lat_SW', 'pred_lon_SW', 'init_lat_SE', 'init_lon_SE', 'pred_lat_SE', 'pred_lon_SE', 'fcst_box_icen', 'fcst_box_jcen', 'mrms_storm_icen', 'mrms_storm_jcen']

#class Unbuffered:
#  def __init__(self, stream, output):
#    self.stream = stream
#    self.output = output
#  def write(self, data):
#    self.stream.write(data)
#    self.stream.flush()
#    self.output.write(data)    # Write the data of stdout here to a text file as well
#  def flush(self):
#    self.stream.flush()
