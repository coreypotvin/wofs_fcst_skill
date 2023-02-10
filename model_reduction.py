from params import *

def main(argv=None):
  if argv is None:
    argv = sys.argv


  #label_metrics = ['mean_fcst_' + feature for feature in ['frac_nearby_storm', 'frac_matching_storm_type', 'frac_matching_storm_cat', 'peak_refl_ratio', 'norm_dist_nearest_storm', 'storm_area_ratio', 'eFSS']]+['mean_fcst_subscore%d' % s for s in range(9)]+['mean_fcst_object_score', 'mean_fcst_field_score', 'mean_fcst_comb_score', 'fcst_eFSS', 'fcst_FSS'] + ['mrms_pred_cvg_%ddbz' % thres for thres in [35, 40, 45]] + ['wofs_pred_cvg_%ddbz' % thres for thres in [35, 40, 45]] + ['diff_pred_cvg_%ddbz' % thres for thres in [35, 40, 45]]+[target]

  global other_features

  if target=='fcst_FSS':
    other_features += ['init_box_FSS']
  elif  target=='fcst_eFSS':
    other_features += ['init_box_eFSS']#, 'fcst_wofs_mean_pred_total_storm_area']
  elif 'fcst_eFSS' in target:
    other_features += [target.replace('fcst_eFSS', 'init_box_eFSS'), 'fcst_wofs_mean_pred_total_storm_area']
    print (other_features)
  for thres in [35, 40, 45]:
    if target=='diff_pred_cvg_%ddbz' % thres:
      other_features += ['diff_init_cvg_%ddbz' % thres, 'fcst_wofs_mean_pred_total_storm_area']

  print ('Reading %s:' % orig_features_fname)
  features = pd.read_feather(orig_features_fname)
  print (features.shape)
  clf = CorrelationFilter()
  clf.EXTRA_VARIABLES = other_features; clf.TARGETS_VARIABLES = label_metrics
  result = clf.filter_df_by_correlation(features, 'labels', reduced_cc)

  reduced_features, dropped_cols, corr_features = result[0], result[1], result[2]

  if min_corr:
    dropped = []
    for feature in reduced_features.columns:
      if feature not in other_features+label_metrics and is_numeric_dtype(reduced_features[feature]):
        if abs(np.corrcoef(reduced_features[feature], reduced_features['labels'])[0][1]) < min_corr:
          dropped += [feature]
    print ('Dropping %d features not meeting min_corr with target!' % len(dropped))
    reduced_features.drop(dropped, axis=1, inplace=True)
  print ('Writing %d features to %s' % (reduced_features.shape[1], reduced_features_fname))
  reduced_features.to_feather(reduced_features_fname)

#-------------------------------------------------------------------------------                                                                                                                                              
if __name__ == "__main__":
    sys.exit(main())
