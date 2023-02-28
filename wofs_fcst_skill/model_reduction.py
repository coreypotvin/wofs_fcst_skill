# Correlation-based feature set reduction

from params import *

def main(argv=None):
  if argv is None:
    argv = sys.argv

  print ('\nReading %s' % orig_features_fname)
  features = pd.read_feather(orig_features_fname)
  print ('\nFeatures shape: ', features.shape, '\n')

  clf = CorrelationFilter()

  # Exclude non-predictor features and features required for baseline models
  other_features2 = other_features+[pers_feature, sprd_feature]
  clf.EXTRA_VARIABLES = other_features2; clf.TARGETS_VARIABLES = label_metrics

  # Identify pairs of highly correlated features and remove the one less correlated with target
  result = clf.filter_df_by_correlation(features, 'labels', reduced_cc)
  reduced_features, dropped_cols, corr_features = result[0], result[1], result[2]

  # Remove features poorly correlated with target
  if min_corr:
    dropped = []
    for feature in reduced_features.columns:
      if feature not in other_features2+label_metrics and is_numeric_dtype(reduced_features[feature]):
        if abs(np.corrcoef(reduced_features[feature], reduced_features['labels'])[0][1]) < min_corr:
          dropped += [feature]
    print ('\nDropping %d features not meeting min_corr with target...' % len(dropped))
    reduced_features.drop(dropped, axis=1, inplace=True)

  # Write reduced feature set
  print ('\nWriting %d features to %s\n' % (reduced_features.shape[1], reduced_features_fname))
  reduced_features.to_feather(reduced_features_fname)

#-------------------------------------------------------------------------------                                                                                                                                              
if __name__ == "__main__":
    sys.exit(main())
