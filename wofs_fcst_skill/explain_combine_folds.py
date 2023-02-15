from params import *

def main(argv=None):
  if argv is None:
    argv = sys.argv

  perm_method = 'singlepass'#'multipass'#'singlepass'
  direction = 'backward'#'forward'#'backward' 
  class_index = 2
  nvars=imp_vars#15#45#10#45
  plot_nvars=10
  plot_nvars2=9#16
  fnames= ['%s/perm_imp_%s_%s_%d_nvars=%d.joblib' % (MAIN_DIR, direction, exp_name, n, nvars) for n in range(5)]

  explainer = skexplain.ExplainToolkit()
  all_ranks = []

  if 'OLR' in exp_name:
    estimator = 'LogisticRegression'
  elif 'ORF' in exp_name:
    estimator = 'RandomForestClassifier'

  for fname in fnames:
    print ('Reading %s' % fname)
    perm_imp = explainer.load(fname)
    ranks = list(perm_imp['%s_rankings__%s' % (perm_method, estimator)].values)
    all_ranks.append(ranks)

  unique_features = list(set([item for sublist in all_ranks for item in sublist]))
  print ('Union of all features: %s' % len(unique_features))
  unique_features = [feature for feature in unique_features if all(feature in ranks for ranks in all_ranks)]
  num_common_features = len(unique_features)
  print ('Intersection of all features: %s' % num_common_features)
  imp_vars_fname = '%s/imp_vars_combined_%s-%s_%s_nvars=%d.pkl' % (MAIN_DIR, direction, perm_method, exp_name, num_common_features)

  all_features_rank_nums = []
  all_features_rank_means = []
  all_features_rank_errs = []

  for feature in unique_features:
    all_rank_nums = []
    for ranks in all_ranks:
      rank_num = ranks.index(feature)
      all_rank_nums.append(rank_num)
    all_features_rank_nums.append(all_rank_nums)
    all_features_rank_means.append(np.median(np.asarray(all_rank_nums)))
    all_features_rank_errs.append(np.std(np.asarray(all_rank_nums)))

  all_features_rank_means = np.asarray(all_features_rank_means)
  unique_features = np.asarray(unique_features)
  all_features_rank_errs = np.asarray(all_features_rank_errs)

  sort_indices = np.argsort(all_features_rank_means) 
  all_features_rank_means = all_features_rank_means[sort_indices]
  unique_features = unique_features[sort_indices]
  all_features_rank_errs = all_features_rank_errs[sort_indices]

  print ('Writing to: %s' % imp_vars_fname)
  with open(imp_vars_fname, 'wb') as handle:
    pickle.dump(unique_features, handle)  

  for n in range(len(unique_features)):
    print (unique_features[n], all_features_rank_means[n])

  convert_name = {'init_box_eFSS': 'init_eFSS', 'fcst_wz_0to2_instant_median_max': 'fcst_vort_med_max', 'fcst_REFLCOMP_stdev_max': 'fcst_refl_std_max', 'fcst_wofs_mean_pred_total_storm_area': 'fcst_mean_storm_area', 'fcst_wofs_mean_num_CELLULAR': 'fcst_mean_num_cell', 'fcst_wofs_stdev_num_QLCS_ORD': 'fcst_std_num_qlcs_ord', 'init_pw_median_90pc': 'init_pw_med_90pc', 'fcst_ws_80_median_max': 'fcst_ws_med_max', 'init_lcl_ml_stdev_10pc': 'init_lcl_std_10pc', 'fcst_spread_refl_field_pred_nbrhd': 'fcst_spread_refl_upscaled', 'fcst_uh_2to5_instant_stdev_median': 'fcst_uh_std_med', 'init_fed_median_90pc': 'init_fed_med_90pc', 'fcst_bunk_r_tot_stdev_median': 'fcst_bunk_std_med', 'fcst_w_down_stdev_min': 'fcst_wdn_std_min', 'fcst_rain_stdev_max': 'fcst_rain_std_max', 'fcst_pw_median_90pc': 'fcst_pw_med_90pc', 'mean_init_mrms_refl_99th': 'init_mrms_refl_99pc', 'init_scp_median_90pc': 'init_scp_med_90pc', 'mean_init_error_refl_field_local_40DBZ_nbrhd': 'error_refl_upscaled_thres', 'mean_init_wofs_refl_ens_mean_99th': 'init_wofs_refl_99pc', 'fcst_spread_refl_field_pred_30DBZ_nbrhd': 'fcst_spread_refl_upscaled_thres', 'init_wz_0to2_instant_median_max': 'init_vort_med_max', 'fcst_bunk_r_tot_stdev_90pc': 'fcst_bunkers_std_90pc', 'fcst_wofs_stdev_pred_total_storm_area': 'fcst_std_total_storm_area', 'fcst_entropy_refl_field_pred': 'fcst_entropy_refl', 'init_stp_median_90pc': 'init_stp_med_90pc', 'fcst_wofs_mean_num_ORG_MULTICELL': 'fcst_mean_num_org_multicell'}

  unique_features = unique_features[:plot_nvars]
  orig_feature_names = unique_features.copy()
  all_features_rank_means = all_features_rank_means[:plot_nvars]
  all_features_rank_errs = all_features_rank_errs[:plot_nvars]

  #unique_features = [convert_name[item] for item in unique_features[:nvars]]
  temp = []
  for item in unique_features:
    try:
      temp.append(convert_name[item])
    except:
      temp.append(item)
  unique_features = temp

  fig, ax = P.subplots(figsize = (22, 16))
  ax.barh(unique_features, all_features_rank_means, xerr=all_features_rank_errs, capsize=7)
  P.xlim([0, 10])
  ax.invert_yaxis()
  ax.tick_params(axis='both', which='major', labelsize=20)

  for i in ax.patches:
    P.text(i.get_width()+0.2, i.get_y()+0.5, str(int(round((i.get_width()), 0))), fontsize=30, fontweight='bold', color='grey')

  fname = '%s/combined_perm_imp_%s-%s_%s_nvars=%d_plot_nvars=%d.png' % (MAIN_DIR, direction, perm_method, exp_name, nvars, plot_nvars)
  print ('Saving plot: %s'  % fname)
  P.savefig(fname, bbox_inches='tight')

  ### ALE CURVES ###

  colors = ['red', 'green', 'blue', 'magenta', 'orange']

  fnames= ['%s/ale_%s_%s_%d_nvars=%d.joblib' % (MAIN_DIR, direction, exp_name, n, nvars) for n in range(5)]
  #fnames= ['%s/ale_%s_%s_%d_class=%d_nvars=%d.joblib' % (MAIN_DIR, direction, exp_name, n, class_index, nvars) for n in range(5)]

  all_ale = []

  features = pd.read_feather(reduced_features_fname)
  print (features.columns)
  labels = np.array(features['labels'].values.tolist())
  labels[((labels>0) & (labels<4))] = 1
  labels[labels==4] = 2
  num_bins = 10

  for fname in fnames:
    print ('Reading %s' % fname)
    ale = explainer.load(fname)
    all_ale.append(ale)

  

  for n, feature in enumerate(orig_feature_names[:plot_nvars2]):

    print (feature)
    event_rates = []
    bins2 = []
    brkpts = [np.nanpercentile(features[feature], n/num_bins*100) for n in range(num_bins+1) ]
    brkpts[-1] += .001
    brkpts = np.array(brkpts)
    #for nn in range(len(brkpts)-1):
    #  if brkpts[nn+1]<=brkpts[nn]:
    #    brkpts[nn+1] = brkpts[nn]+1e-9
    bins = (brkpts[:-1]+brkpts[1:])/2
    conditions = [ ((features[feature] >= brkpts[n]) & (features[feature] < brkpts[n+1])) for n in range(len(brkpts)-1) ]
    for c, condition in enumerate(conditions):
      sub_labels = labels[condition]
      num_events = len(sub_labels[sub_labels==class_index])
      num_total  = len(sub_labels)
      if num_total > 0:
        event_rate = num_events/num_total
        event_rates.append(event_rate)
        bins2.append(bins[c])

    event_rates = np.array(event_rates)*100
    event_rates -= event_rates.mean()
    ales = [all_ale[i]['%s__%s__ale' % (feature, estimator)][0]*100 for i in range(len(all_ale))]
    ale_mins = np.array([ale.min() for ale in ales])
    ale_maxs = np.array([ale.max() for ale in ales])
    ymin = min(ale_mins.min(), event_rates.min())
    ymax = max(ale_maxs.max(), event_rates.max())
    print (ymin, ymax)

    ax1=P.subplot(int(sqrt(plot_nvars2)),int(sqrt(plot_nvars2)),n+1)
    #ax2 = ax1.twinx()
    P.title(unique_features[n], fontsize=20, fontweight='bold')

    for i in range(len(all_ale)):
      ale = all_ale[i]
      X = ale['%s__bin_values' % feature]
      Y = ale['%s__%s__ale' % (feature, estimator)][0]*100
      ax1.plot(X,Y,color=colors[i])
      ax1.plot(bins2,event_rates,color='k',ls='--')
    ax1.axhline(y=0, color='k', ls='--')
    ax1.set_ylim([ymin, ymax])
    #ax2.set_ylim([ymin, ymax])
    ax1.xaxis.set_tick_params(labelsize=18)
    #ax2.yaxis.set_tick_params(labelsize=18)
    ax1.yaxis.set_tick_params(labelsize=18)
  #P.supylabel('Centered ALE (%)')

  fname = '%s/combined_ale2_%s-%s_%s_nvars=%d_class=%d_plot_nvars=%d.png' % (MAIN_DIR, direction, perm_method, exp_name, nvars, class_index, plot_nvars2)
  print ('Saving plot: %s'  % fname)
  P.savefig(fname, bbox_inches='tight')
#-------------------------------------------------------------------------------                                                                                                                                              
if __name__ == "__main__":
    sys.exit(main())
