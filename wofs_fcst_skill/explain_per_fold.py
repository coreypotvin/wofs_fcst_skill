from params import *

def main(argv=None):
  if argv is None:
    argv = sys.argv

  write = True
  max_nvars = 45
  class_index = 2#4
  grouped = False
  start_time = time.time()

  direction = 'forward'#'backward'

  if direction=='forward':
    scoring_strategy = 'minimize'
  else:
    scoring_strategy = 'maximize'

  if 'OLR' in exp_name:
    estimator = 'LogisticRegression'
  elif 'ORF' in exp_name:
    estimator = 'RandomForestClassifier'

  iter_names = [exp_name+'_%d' % n for n in range(5)]

  for iter_name in iter_names:

    model_fname = '%s/best_model_%s.joblib' % (MAIN_DIR, iter_name) 
    train_test_fname = '%s/train_test_%s.joblib' % (MAIN_DIR, iter_name)

    print ('Reading: %s' %model_fname)

    model = load(model_fname)
    train_test = load(train_test_fname)
    train_features, train_labels, test_features, test_labels = train_test[0], train_test[1], train_test[2], train_test[3]

    train_labels = train_labels.astype(int)
    if combine_classes:
      train_labels[((train_labels > 0) & (train_labels < 4))] = 1
      train_labels[train_labels==4] = 2

    nvars=min(max_nvars, train_features.shape[1])

    if grouped:
      perm_imp_fname = '%s/grouped_perm_imp_%s.joblib' % (MAIN_DIR, iter_name)
    else:
      perm_imp_fname = '%s/perm_imp_%s_%s.joblib' % (MAIN_DIR, direction, iter_name)
    if grouped:
      rankings_fname = '%s/grouped_rankings_%s_class=%d.png' % (MAIN_DIR, iter_name, class_index)
    else:
      rankings_fname = '%s/rankings_%s_%s_class=%d.png' % (MAIN_DIR, direction, iter_name, class_index)
    ale_fname = '%s/ale_%s_%s_class=%d.joblib' % (MAIN_DIR, direction, iter_name, class_index)
    ale_plot_fname = '%s/ale_%s_%s_class=%d.png' % (MAIN_DIR, direction, iter_name, class_index)
    imp_vars_fname = '%s/imp_vars_%s_%s.pkl' % (MAIN_DIR, direction, iter_name)

    if 'NN' in iter_name or 'GB' in iter_name or 'LR' in iter_name or 'AT' in iter_name:
      n_jobs = 1
    else:
      n_jobs = 10

    if 'OLR' in iter_name:
      scaler = StandardScaler().fit(train_features)
      model = (estimator, MyPipeline(model, scaler, combine_classes=True))
    elif 'ORF' in iter_name:
      model = (estimator, MyPipeline(model, scaler=False, combine_classes=True))

    explainer = skexplain.ExplainToolkit(estimators=model, X=train_features, y=train_labels, estimator_output='probability')

    features = copy.deepcopy(explainer.feature_names)
    for feature in explainer.feature_names:
      if len(np.unique(explainer.X[feature])) < 3:
        features.remove(feature)
        print ('Removing: %s' % feature)

    if write:

      if grouped:
        perm_imp_results, groups = explainer.grouped_permutation_importance(perm_method = 'grouped_only', evaluation_fn = metrics.RPS, scoring_strategy = "maximize", sample_size=100, n_permute=1, n_jobs=n_jobs)#, clustering_kwargs = {'n_clusters' : 3})
        print (groups)
      else:
        perm_imp_results = explainer.permutation_importance(n_vars=nvars, direction=direction, evaluation_fn = metrics.RPS, scoring_strategy = scoring_strategy, n_permute=5, n_jobs=n_jobs, verbose=True, random_seed=0)

      print ('Writing to: %s' % perm_imp_fname)

      explainer.save(fname=perm_imp_fname, data=perm_imp_results)

      ale = explainer.ale(features, n_bins=20, subsample=5000, n_jobs=30, n_bootstrap=1, random_seed=0, class_index=class_index)
      explainer.save(fname=ale_fname, data=ale)

    else:

      print ('Loading %s' % perm_imp_fname)
      perm_imp_results = explainer.load(perm_imp_fname)
      print (perm_imp_results)

      print ('Loading %s' % ale_fname)
      ale = explainer.load(ale_fname)

    print (perm_imp_results)

    ale_var_results = explainer.ale_variance(ale)

    if grouped:
      panels = [('grouped_only', estimator)]
      data = [perm_imp_results]
      imp_plot = explainer.plot_importance(data = data, panels = panels)
      P.savefig(rankings_fname)
      sys.exit(1)

    important_vars = list_scores(perm_imp_results, 'multipass', estimator, nvars=nvars)
    with open(imp_vars_fname, 'wb') as handle:
      pickle.dump(important_vars, handle)

    important_vars = list_scores(ale_var_results, 'ale_variance', estimator, nvars=nvars)

    panels = [('multipass', estimator), ('ale_variance', estimator)]
    data = [perm_imp_results,ale_var_results]

    imp_plot = explainer.plot_importance(data = data, panels = panels, plot_correlated_features=True, num_vars_to_plot=nvars)
    P.savefig(rankings_fname)

    explainer.plot_ale(ale, features=important_vars, to_probability=True)
    P.savefig(ale_plot_fname)

    print(("\n TOTAL RUNTIME: %.1f min \n" % ((time.time() - start_time)/60.)))


def list_scores(results, method, estimator_name, nvars=15):

  print ('~~~~~ %s %s ~~~~~' % (method, estimator_name))
  sorted_var_names = list(results[f"{method}_rankings__{estimator_name}"].values)
  scores = results[f"{method}_scores__{estimator_name}"].values

  for i in range(min(nvars, len(scores))):
    print (i, sorted_var_names[i], np.array(scores[i]).mean())

  return sorted_var_names[:min(nvars, len(scores))]

class MyPipeline:

        def __init__(self, clf, scaler=False, combine_classes=False):
                self.scaler = scaler
                self.clf = clf 
                self.combine_classes = combine_classes
        def predict_proba(self, X):
                if self.scaler:
                  X = self.scaler.transform(X)
                pred = self.clf.predict_proba(X)
                if self.combine_classes:
                  temp = np.empty((pred.shape[0], 3))
                  temp[:,0] = pred[:,0]
                  temp[:,2] = pred[:,4]
                  temp[:,1] = np.sum(pred[:,1:4], axis=1)
                  pred = temp
                return pred 
        def predict(self, X):
                if self.scaler:
                  X = self.scaler.transform(X)
                pred = self.clf.predict(X)
                if self.combine_classes:
                  temp = np.empty((pred.shape[0], 3))
                  temp[:,0] = pred[:,0]
                  temp[:,2] = pred[:,4]
                  temp[:,1] = np.sum(pred[:,1:4], axis=1)
                  pred = temp
                return pred
#-------------------------------------------------------------------------------                                                                                                                                              
if __name__ == "__main__":
    sys.exit(main())
