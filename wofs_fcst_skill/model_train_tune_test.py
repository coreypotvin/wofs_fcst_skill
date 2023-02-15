from defs import *

def main(argv=None):
  if argv is None:
    argv = sys.argv

  output1 = open(logfile,'w')
  output2 = open(errfile,'w')
  sys.stdout=Unbuffered(sys.stdout, output1)
  sys.stderr=Unbuffered(sys.stderr, output2)

  global num_classes

  rng = np.random.RandomState(0)

  start_time = time.time()

  if 1==1:

    categorical_metrics = ['mean_init_mrms_storm_type', 'mean_init_mrms_storm_cat', 'init_mrms_storm_type', 'init_mrms_storm_cat']

    if pers_bl:
      training_features = [persistence_var]
    elif ratio_bl:
      training_features = [baseline_feature]
    elif imp_vars:
      print ('Reading important varnames: %s' % imp_vars_fname)
      with open(imp_vars_fname, 'rb') as handle:
        training_features = pickle.load(handle)[:imp_vars].tolist()
    else:
      training_features = storm_init_predictors+mean_storm_init_predictors+storm_fcst_predictors+env_init_predictors+env_fcst_predictors+box_predictors
      training_features.remove('init_time'); training_features.remove('valid_time'); #training_features.remove('lead_time')

    try:
      training_features.remove('init_eFSS_fcst_box')
    except:
      pass

    print ('Reading features from: %s' % fname)
    features = pd.read_feather(fname)

    dummy_cols = [metric for metric in categorical_metrics if (metric in training_features and metric in features.columns)]
    if len(dummy_cols) > 0:
      features = pd.get_dummies(features, columns=dummy_cols)
      for feature in features.columns:
        if any(item+'_' in feature for item in categorical_metrics):
          training_features.append(feature)
    
    print ('Before removing duplicates: ', features.shape[0])
    features.drop_duplicates(inplace=True, ignore_index=True)
    print ('After removing duplicates: ', features.shape[0])

    print ('Before removing target/BL = -999: ', features.shape[0])
    features = features[((features[target] != -999) & (features[persistence_var] != -999))]
    print ('After removing target/BL = -999: ', features.shape[0])

    #print ('Before removing overlapping domains: ', features.shape[0])
    #features = remove_overlapping_domains(features, lead_time)
    #print ('After removing overlapping domains: ', features.shape[0])

    #print ('Before removing ALL = -999: ', features.shape[0])
    #features = features[features['init_eFSS_fcst_box']!=-999]
    #print ('After removing ALL = -999: ', features.shape[0])


    if 'init_time' in features.columns and fname == prelim_features_fname and not pers_bl and not ratio_bl:
      features['init_time2'] = convert_time(features['init_time'])
      training_features += ['init_time2']

    print (len(training_features))#, training_features)
    removed_features = [feature for feature in training_features if feature not in list(features.keys())]
    training_features = [feature for feature in training_features if feature in list(features.keys())]
    print ('%d removed features' % len(removed_features))

    corr_features = training_features

    all_features = copy.deepcopy(features)

    print ('Feature correlations with target:\n')
    try:
      corrs = np.array([np.corrcoef(features[feature], features['labels'])[0][1] for feature in corr_features])
    except:
      corrs = np.array([np.corrcoef(features[feature], features[target])[0][1] for feature in corr_features])
    sorted_inds = np.argsort(-np.fabs(corrs))
    sorted_corrs = corrs[sorted_inds]
    for c, corr in enumerate(sorted_corrs):
      print (c, corr, corr_features[sorted_inds[c]], features[training_features[sorted_inds[c]]].min(), features[training_features[sorted_inds[c]]].max())
    labels = np.zeros(len(features[target]))

    if not stratify:

      if num_classes==5:
        breakpoints = [np.nanpercentile(features[target], n/num_classes*100) for n in range(num_classes+1)]
      elif num_classes==3:
        breakpoints = [np.nanpercentile(features[target], n) for n in [0, 20, 80, 100]]
      breakpoints[-1] += .001
      print ('Breakpoints for %s:' %target, breakpoints)
      for n in range(len(breakpoints)-1):
        labels[( (features[target] >= breakpoints[n]) & (features[target] < breakpoints[n+1]) )] = n

      labels = labels.astype(int)

      if num_classes==5:
        pers_breakpoints = [np.nanpercentile(features[persistence_var], n/num_classes*100) for n in range(num_classes+1)]
      elif num_classes==3:
        pers_breakpoints = [np.nanpercentile(features[persistence_var], n) for n in [0, 20, 80, 100]]
      pers_breakpoints[-1] += .001
      print ('Breakpoints for %s: ' % persistence_var, pers_breakpoints)
      persistence_preds = np.zeros(len(features[persistence_var]))
      for n in range(len(pers_breakpoints)-1):
        persistence_preds[( (features[persistence_var] >= pers_breakpoints[n]) & (features[persistence_var] < pers_breakpoints[n+1]) )] = n

      if num_classes==5:
        ratio_breakpoints = [np.nanpercentile(features[baseline_feature], n/num_classes*100) for n in range(num_classes+1)]
      elif num_classes==3:
        ratio_breakpoints = [np.nanpercentile(features[baseline_feature], n) for n in [0, 20, 80, 100]]
      ratio_breakpoints[-1] += .001
      print ('Breakpoints for %s:' % baseline_feature, ratio_breakpoints)
      if baseline_feature in ['fcst_spread_refl_field_pred_nbrhd']:
        ratio_preds = 2*np.ones(features.shape[0])
        ratio_preds[( (features[baseline_feature] >= ratio_breakpoints[1]) & (features[baseline_feature] < ratio_breakpoints[2]) )] = 1
        ratio_preds[features[baseline_feature] >= ratio_breakpoints[2]] = 0
      else:
        ratio_preds = np.zeros(features.shape[0])
        for n in range(len(ratio_breakpoints)-1):
          ratio_preds[( (features[baseline_feature] >= ratio_breakpoints[n]) & (features[baseline_feature] < ratio_breakpoints[n+1]) )] = n

    else:

      sys.exit(1)

    if 'reduced' not in fname:
      temp = copy.deepcopy(features)
      temp = temp.drop([feature for feature in temp.columns if feature not in training_features+label_metrics+spread_metrics+other_features], axis=1)
      temp['labels'] = labels
      temp.reset_index(drop=True, inplace=True)
      temp.to_feather(orig_features_fname)
      print ('Writing original features to: %s' % orig_features_fname)
      print ('Features dims = ', temp.shape)

    features = features.drop([feature for feature in features.columns if (feature not in training_features and feature[-2] != '_')], axis=1)

  print ('%d samples total for training/testing' % features.shape[0])

  if train_test_split_param in ['2017', '2018', '2019', '2020', '2021']:

    index = np.where((all_features['date'].str.contains(train_test_split_param)))[0].tolist()
    index2 = [x for x in range(features.shape[0]) if x not in index]
    index = np.asarray(index)
    index2 = np.asarray(index2)
    test_features = features.iloc[index]
    train_features = features.iloc[index2]
    test_labels = labels[index]
    train_labels = labels[index2]

    ratio_preds = ratio_preds[index]

    all_features_train = all_features.iloc[index2]

  elif train_test_split_param == 'dates':

    dates = sorted(list(set(all_features['date'])))
    tot=0
    date_groups = []
    date_group = []
    for date in dates:
      date_group.append(date)
      tot += len(all_features[all_features['date']==date])
      if tot >= (len(date_groups)+1)*len(all_features)/num_folds:
        date_groups.append(date_group)
        date_group=[]

    allfolds_test_features = []; allfolds_test_orig_features = []; allfolds_train_features = []; allfolds_test_labels = []; allfolds_train_labels = []; allfolds_ratio_preds = []; allfolds_persistence_preds = []; allfolds_all_features_train = []

    for ind in range(num_folds):
      date_group = date_groups[ind]
      index = np.where((all_features['date'].isin(date_group)))[0].tolist()
      index2 = [x for x in range(features.shape[0]) if x not in index]
      index = np.asarray(index)
      index2 = np.asarray(index2)
      test_features = features.iloc[index]
      train_features = features.iloc[index2]
      test_labels = labels[index]
      train_labels = labels[index2]

      if class_balancing:

        index = subsample(test_labels, index, ratio=test_ratio)
        test_features = features.iloc[index]
        test_labels = labels[index]

      all_features_train = all_features.iloc[index2] 
      allfolds_test_features.append(test_features)
      allfolds_train_features.append(train_features)
      allfolds_test_labels.append(test_labels)
      allfolds_train_labels.append(train_labels)
      allfolds_all_features_train.append(all_features_train)

      if stratify_verif:
        print ('Reading original features: %s' % orig_features_fname)
        orig_features = pd.read_feather(orig_features_fname)
        test_orig_features = orig_features.iloc[index]
        allfolds_test_orig_features.append(test_orig_features)

  elif type(train_test_split_param)==str and 'dates' in train_test_split_param:

    dates = sorted(list(set(all_features['date'])))
    tot=0
    date_groups = []
    date_group = []
    for date in dates:
      date_group.append(date)
      tot += len(all_features[all_features['date']==date])
      if tot >= (len(date_groups)+1)*len(all_features)/num_folds:
        date_groups.append(date_group)
        date_group=[]

    ind = int(train_test_split_param[-1])
    date_group = date_groups[ind]
    index = np.where((all_features['date'].isin(date_group)))[0].tolist()
    index2 = [x for x in range(features.shape[0]) if x not in index]
    index = np.asarray(index)
    index2 = np.asarray(index2)
    test_features = features.iloc[index]
    train_features = features.iloc[index2]
    test_labels = labels[index]
    train_labels = labels[index2]
    ratio_preds = ratio_preds[index]
    persistence_preds = persistence_preds[index]
    all_features_train = all_features.iloc[index2]

  else:

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = train_test_split_param, random_state = rng)
    dummy, dummy, dummy, ratio_preds = train_test_split(features, ratio_preds, test_size = train_test_split_param, random_state = rng)
    dummy, dummy, dummy, persistence_preds = train_test_split(features, persistence_preds, test_size = train_test_split_param, random_state = rng)
    all_features_train, dummy, dummy, dummy = train_test_split(all_features, labels, test_size = train_test_split_param, random_state = rng)

  allfolds_train_features_unscaled = allfolds_train_features.copy()

  if learn_algo in ['LR', 'NN', 'OLR', 'AT', 'stacked_LR', 'stacked_NN']: 

    if train_test_split_param == 'dates':
 
      #allfolds_train_features_unscaled = allfolds_train_features.copy()

      for fold in range(num_folds):

        scaling_model = StandardScaler().fit(allfolds_train_features[fold])
        test_features = scaling_model.transform(allfolds_test_features[fold])
        train_features = scaling_model.transform(allfolds_train_features[fold])
        allfolds_test_features[fold] = pd.DataFrame(test_features, columns = features.columns)
        allfolds_train_features[fold] = pd.DataFrame(train_features, columns = features.columns)
   
    else:

      scaling_model = StandardScaler().fit(train_features)
      train_features = scaling_model.transform(train_features)
      test_features = scaling_model.transform(test_features)
      train_features = pd.DataFrame(train_features, columns = features.columns)
      test_features = pd.DataFrame(test_features, columns = features.columns)
  
  if train_test_split_param == 'dates':
    train_test_fnames = [train_test_fname.split('.joblib')[0] + '_%d.joblib' % fold for fold in range(num_folds)]
    for fold in range(num_folds):
      train_features = allfolds_train_features_unscaled[fold]; train_labels = allfolds_train_labels[fold]; test_features = allfolds_test_features[fold]; test_labels = allfolds_test_labels[fold]
      dump((train_features, train_labels, test_features, test_labels), train_test_fnames[fold])
      print ('%d/%d training/testing samples' % (len(train_labels), len(test_labels)))
      print ('Class balance of training dataset: ', [round(len(train_labels[train_labels==n])/train_features.shape[0],2) for n in range(num_classes)])
      print ('Class balance of testing dataset: ', [round(len(test_labels[test_labels==n])/test_features.shape[0],2) for n in range(num_classes)])
  else:
    dump((train_features, train_labels, test_features, test_labels), train_test_fname)
    print ('%d/%d training/testing samples' % (len(train_labels), len(test_labels)))
    print ('Class balance of training dataset: ', [round(len(train_labels[train_labels==n])/train_features.shape[0],2) for n in range(num_classes)])
    print ('Class balance of testing dataset: ', [round(len(test_labels[test_labels==n])/test_features.shape[0],2) for n in range(num_classes)])

  if do_train:

    if do_hyper_opt:

      if learn_algo in ['RF', 'ORF', 'stacked_RF']:

        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        max_features = ['log2']#, 'sqrt']
        max_depth = [int(x) for x in np.arange(5, 41, 5)]
        min_samples = [x for x in range(3, 25, 3)]
        min_split = [x for x in range(5, 31, 5)]

        if learn_algo in ['RF', 'stacked_RF']:
          param_distributions = {'n_estimators': n_estimators,
             'max_features': max_features,
             'min_samples_leaf': min_samples,#[1, 2, 4, 6, 9],
             'min_samples_split': min_split,#[5, 10, 15, 20, 30, 50],
             'max_depth': max_depth}
          learner = RandomForestClassifier(random_state=rng)

        elif learn_algo == 'ORF':
          param_distributions = {'estimator__n_estimators': n_estimators,
             'estimator__max_features': max_features,
             'estimator__min_samples_leaf': min_samples,
             'estimator__min_samples_split': min_split,
             'estimator__max_depth': max_depth}
          learner = OrdinalClassifier(RandomForestClassifier(random_state=rng), n_jobs=1)

      elif learn_algo in ['LR', 'OLR', 'stacked_LR']:

        penalty = ['elasticnet']
        l1_ratio = np.arange(0, 1.0000001, 0.1)
        C = np.logspace(-3, 2, num=10)

        if learn_algo in ['LR', 'stacked_LR']:
          param_distributions = {'penalty': penalty, 'l1_ratio': l1_ratio, 'C': C}
          learner = LogisticRegression(solver='saga', max_iter=1000, random_state=rng)
        elif learn_algo == 'OLR':
          param_distributions = {'estimator__penalty': penalty, 'estimator__l1_ratio': l1_ratio, 'estimator__C': C}
          learner = OrdinalClassifier(LogisticRegression(solver='saga', max_iter=1000, random_state=rng), n_jobs=1)

      elif learn_algo in ['NN', 'stacked_NN']:

        param_distributions = {
          'hidden_layer_sizes': [(sp_randint.rvs(50,500,1,random_state=rng),sp_randint.rvs(50,500,1,random_state=rng),) for i in range(100)] ,
          #'hidden_layer_sizes': [(sp_randint.rvs(50,500,1,random_state=rng),sp_randint.rvs(50,500,1,random_state=rng),)],#, 
                                          #(sp_randint.rvs(50,500,1,random_state=rng),)],
          'activation': ['tanh', 'relu', 'logistic'],
          'alpha': np.arange(0.1, 0.91, 0.1),
          'learning_rate_init': np.arange(0.0001, 0.0031, .001)
        }

        learner = MLPClassifier(max_iter=1000, random_state=rng)

      elif learn_algo=='GB':

        param_distributions = {
          'n_estimators'     : [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
          'max_depth'        : [int(x) for x in np.arange(2, 15, 3)],
          'grow_policy'      : ['depthwise', 'lossguide'],
          'learning_rate'    : np.arange(0.1, 0.91, 0.2),
          'subsample'        : [0.5, 0.75, 1],
          'reg_lambda'       : np.arange(0, 1.01, 0.2),
          'reg_alpha'        : np.arange(0, 1.01, 0.2),
          "min_child_weight" : [ 1, 3, 5, 7 ],
          "gamma"            : [ 0.0, 0.1, 0.2, 0.3, 0.4 ],
          "colsample_bytree" : [ 0.3, 0.4, 0.5, 0.7 ]
        }

        learner = XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', nthread=1, random_state=rng)

      elif learn_algo=='GBhist':

        param_distributions = {
          'max_iter'         : [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
          'max_depth'        : [int(x) for x in np.arange(2, 15, 3)],
          'learning_rate'    : [0.0001, 0.0003, 0.001, .003, .01, .03, .1, .3],
          'min_samples_leaf' : [x for x in range(3, 25, 3)],
          'l2_regularization': np.arange(0.1, 0.91, 0.2),
          'max_bins'         : [50, 100, 150, 200]
        }

        learner = HistGradientBoostingClassifier(random_state=rng)

      if tuning_CV_split_method == 'year':

        print ('Splitting hyperparameter optimization CV folds by year')

        CV = []

        years = ['2017', '2018', '2019', '2020', '2021']
        if train_test_split_param in years:
          years.remove(train_test_split_param)

        for y, year in enumerate(years):

          test_indices = np.where((all_features_train['date'].str.contains(year)))[0].tolist()
          train_indices = np.where((~(all_features_train['date'].str.contains(year))))[0].tolist()#[x for x in range(len(features['mean_init_score'])) if x not in test_indices]
          #np.random.seed(0)
          #random.shuffle(test_indices)
          #np.random.seed(0)
          #random.shuffle(train_indices)

          print ('CV fold #%d (validation year: %s): %d/%d training/validation samples' % (y, year, len(train_indices), len(test_indices)))

          CV.append((train_indices, test_indices))

      elif train_test_split_param == 'dates':

        allfolds_CV = []
 
        for fold in range(num_folds):

          CV = []
          date_groups2 = date_groups.copy()
          del date_groups2[fold]
          train_labels = allfolds_train_labels[fold]          
          all_features_train = allfolds_all_features_train[fold]

          for d, date_group in enumerate(date_groups2):

            dates = pd.array(all_features_train['date'], dtype='string')
            test_indices = np.where(np.isin(dates, date_group))[0].tolist()
            train_indices = np.where(~np.isin(dates, date_group))[0].tolist()

            if class_balancing:
              print ('CV fold #%d: %d/%d training/validation samples' % (d, len(train_indices), len(test_indices)))
              train_indices = subsample(train_labels[train_indices], train_indices, ratio=train_ratio)
              test_indices = subsample(train_labels[test_indices], test_indices, ratio=test_ratio)

            print ('CV fold #%d: %d/%d training/validation samples' % (d, len(train_indices), len(test_indices)))
            try:
              print ('      Class balance of training samples:  ', [round(len(train_labels.iloc[train_indices][train_labels.iloc[train_indices]==n])/len(train_labels.iloc[train_indices]),2) for n in range(num_classes)])
              print ('      Class balance of validation samples:', [round(len(train_labels.iloc[test_indices][train_labels.iloc[test_indices]==n])/len(train_labels.iloc[test_indices]),2) for n in range(num_classes)])
            except:
              print ('      Class balance of training samples:  ', [round(len(train_labels[train_indices][train_labels[train_indices]==n])/len(train_labels[train_indices]),2) for n in range(num_classes)])
              print ('      Class balance of validation samples:', [round(len(train_labels[test_indices][train_labels[test_indices]==n])/len(train_labels[test_indices]),2) for n in range(num_classes)])

            CV.append((train_indices, test_indices))

          allfolds_CV.append(CV)

      elif tuning_CV_split_method == 'dates':

        print ('Splitting hyperparameter optimization CV folds by date')

        CV = []


        ind = int(train_test_split_param[-1])
        del date_groups[ind]

        for d, date_group in enumerate(date_groups):
          dates = pd.array(all_features_train['date'], dtype='string')
          test_indices = np.where(np.isin(dates, date_group))[0].tolist()
          train_indices = np.where(~np.isin(dates, date_group))[0].tolist()

          print ('CV fold #%d: %d/%d training/validation samples' % (d, len(train_indices), len(test_indices)))
          print ('      Class balance of training samples:  ', [round(len(train_labels[train_indices][train_labels[train_indices]==n])/len(train_labels[train_indices]),2) for n in range(num_classes)])
          print ('      Class balance of validation samples:', [round(len(train_labels[test_indices][train_labels[test_indices]==n])/len(train_labels[test_indices]),2) for n in range(num_classes)])

          CV.append((train_indices, test_indices))
 
      elif tuning_CV_split_method == 'random':

        CV = 5

      if train_test_split_param == 'dates':

        allfolds_best_params = []

        for fold in range(num_folds):
       
          CV = allfolds_CV[fold]
          #make_scorer(my_custom_score_hinge, labels=np.array(range(num_classes)))  'neg_log_loss' make_scorer(RPS, greater_is_better=False, needs_proba=True) make_scorer(RPSS, needs_proba=True) scoring=make_scorer(custom_CSI, num_classes=num_classes) 'balanced_accuracy'
          #clf = RandomizedSearchCV(estimator = learner, scoring=make_scorer(RPSS, needs_proba=True), param_distributions = param_distributions, n_iter = n_iter, cv=CV, verbose=1, random_state=rng, n_jobs = 20)
          #clf = BayesSearchCV(estimator = learner, scoring = make_scorer(RPSS, needs_proba=True), search_spaces = param_distributions, n_iter = n_iter, cv=CV, verbose=1, random_state=rng, n_points = 10, n_jobs = 40)
          clf = BayesSearchCV(estimator = learner, scoring = make_scorer(RPS, greater_is_better=False, needs_proba=True), search_spaces = param_distributions, n_iter = n_iter, cv=CV, verbose=1, random_state=rng, n_points = 5, n_jobs = 5)#n_points = 10, n_jobs = 40)
          #print (learner.get_params().keys())
          best_model = clf.fit(allfolds_train_features[fold], allfolds_train_labels[fold])

          best_params = best_model.best_params_
          print (best_params)
          allfolds_best_params.append(best_params)

          best_params_fname2 = best_params_fname.split('.pkl')[0]+'_%d.pkl' % fold
          with open(best_params_fname2, 'wb') as handle:
            pickle.dump(best_params, handle)

      else:

        clf = RandomizedSearchCV(estimator = learner, param_distributions = param_distributions, n_iter = n_iter, cv=CV, verbose=1, random_state=rng, n_jobs = 10)
        best_model = clf.fit(train_features, train_labels)
        best_params = best_model.best_params_
        print (best_params)

        with open(best_params_fname, 'wb') as handle:
          pickle.dump(best_params, handle)

    elif learn_algo != 'AT':

      if train_test_split_param == 'dates':

        allfolds_best_params = []
        for fold in range(num_folds):
          best_params_fname2 = best_params_fname.split('.pkl')[0]+'_%d.pkl' % fold
          with open(best_params_fname2, 'rb') as handle:
            allfolds_best_params.append(pickle.load(handle))

      else:

        with open(best_params_fname, 'rb') as handle:
          best_params = pickle.load(handle)

    if train_test_split_param == 'dates':

      allfolds_model = []

      for fold in range(num_folds):

        if learn_algo != 'AT':
          best_params = allfolds_best_params[fold]

        train_features = allfolds_train_features[fold]
        train_labels = allfolds_train_labels[fold]

        #if class_balancing:
        #  index2 = subsample(train_labels, np.arange(len(train_labels)).astype(int), ratio=train_ratio)
        #  train_features = train_features.iloc[index2]
        #  train_labels = train_labels[index2] 

        if learn_algo=='ORF':

          model = OrdinalClassifier(estimator = RandomForestClassifier(n_estimators = best_params['estimator__n_estimators'], max_features=best_params['estimator__max_features'], max_depth=best_params['estimator__max_depth'], min_samples_leaf = best_params['estimator__min_samples_leaf'], min_samples_split=best_params['estimator__min_samples_split'], random_state=rng), n_jobs=1)

        elif learn_algo in ['RF', 'stacked_RF']:
 
          model = RandomForestClassifier(n_estimators = best_params['n_estimators'], max_features=best_params['max_features'], max_depth=best_params['max_depth'], min_samples_leaf = best_params['min_samples_leaf'], min_samples_split=best_params['min_samples_split'], random_state=rng)

        elif learn_algo=='AT':

          model = mord.LogisticAT(alpha=1)
          #model = mord.LogisticIT(alpha=0)
          #model = mord.LogisticSE(alpha=0)

        elif learn_algo in ['LR', 'stacked_LR']:

          model = LogisticRegression(penalty = best_params['penalty'], l1_ratio = best_params['l1_ratio'], C = best_params['C'], solver='saga', max_iter=1000, random_state=rng)
         
        elif learn_algo=='OLR':

          model = OrdinalClassifier(estimator = LogisticRegression(penalty = best_params['estimator__penalty'], l1_ratio = best_params['estimator__l1_ratio'], C = best_params['estimator__C'], solver='saga', max_iter=1000, random_state=rng), n_jobs=1)

        elif learn_algo in ['NN', 'stacked_NN']:
  
          model = MLPClassifier(hidden_layer_sizes = best_params['hidden_layer_sizes'], activation = best_params['activation'], learning_rate_init = best_params['learning_rate_init'], alpha = best_params['alpha'], max_iter=1000, random_state=rng)

        elif learn_algo=='GB':

          model = XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', n_estimators = best_params['n_estimators'], max_depth=best_params['max_depth'], grow_policy=best_params['grow_policy'], learning_rate=best_params['learning_rate'], subsample=best_params['subsample'], reg_lambda=best_params['reg_lambda'], reg_alpha=best_params['reg_alpha'], min_child_weight = best_params['min_child_weight'], gamma = best_params['gamma'], colsample_bytree = best_params['colsample_bytree'], nthread=10, random_state=rng)

        elif learn_algo=='GBhist':

          model = HistGradientBoostingClassifier(max_iter = best_params['max_iter'], max_depth = best_params['max_depth'], learning_rate = best_params['learning_rate'], min_samples_leaf = best_params['min_samples_leaf'], l2_regularization =  best_params['l2_regularization'], max_bins =  best_params['max_bins'], early_stopping=True, random_state=rng)

        model.fit(train_features, train_labels.astype(int))
        allfolds_model.append(model)
        best_model_fname2 = best_model_fname.split('.joblib')[0]+'_%d.joblib' % fold
        dump(model, best_model_fname2)

    else:

      if learn_algo=='RF':

        model = OrdinalClassifier(RandomForestClassifier(n_estimators = best_params['n_estimators'], max_features=best_params['max_features'], max_depth=best_params['max_depth'], min_samples_leaf = best_params['min_samples_leaf'], min_samples_split=best_params['min_samples_split'], random_state=rng), n_jobs=1)

      elif learn_algo=='LR':

        model = OrdinalClassifier(LogisticRegression(penalty = best_params['penalty'], l1_ratio = best_params['l1_ratio'], C = best_params['C'], solver='saga', max_iter=1000, random_state=rng), n_jobs=1)

      elif learn_algo=='NN':

        model = MLPClassifier(hidden_layer_sizes = best_params['hidden_layer_sizes'], activation = best_params['activation'], learning_rate_init = best_params['learning_rate_init'], alpha = best_params['alpha'], max_iter=1000, random_state=rng) 

      elif learn_algo=='GB':

        model = XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', n_estimators = best_params['n_estimators'], max_depth=best_params['max_depth'], grow_policy=best_params['grow_policy'], learning_rate=best_params['learning_rate'], subsample=best_params['subsample'], reg_lambda=best_params['reg_lambda'], reg_alpha=best_params['reg_alpha'], min_child_weight = best_params['min_child_weight'], gamma = best_params['gamma'], colsample_bytree = best_params['colsample_bytree'], nthread=10, random_state=rng)

      elif learn_algo=='GBhist':

        model = HistGradientBoostingClassifier(max_iter = best_params['max_iter'], max_depth = best_params['max_depth'], learning_rate = best_params['learning_rate'], min_samples_leaf = best_params['min_samples_leaf'], l2_regularization =  best_params['l2_regularization'], max_bins =  best_params['max_bins'], early_stopping=True, random_state=rng)

      model.fit(train_features, train_labels)
      dump(model, best_model_fname) 

  else:

    if train_test_split_param == 'dates':

      allfolds_model = []

      for fold in range(num_folds):

        best_model_fname2 = best_model_fname.split('.joblib')[0]+'_%d.joblib' % fold
        model = load(best_model_fname2)
        allfolds_model.append(model)

    else:

      model = load(best_model_fname)

  if train_test_split_param == 'dates':

    baseline_preds = np.empty(0); prob_predictions = []; persistence_preds = np.empty(0); ratio_preds = np.empty(0); test_labels = np.empty(0); test_features = np.empty(0)
    train_predictions = np.empty(0); predictions = np.empty(0); all_train_errors = np.empty(0)
    allfolds_errors = []; class_acc = []; allfolds_auc = []

    for fold in range(num_folds):

      model = allfolds_model[fold]

      new_test_features = allfolds_test_features[fold]
      new_train_features = allfolds_train_features[fold]

      if stratify_verif:
        new_test_orig_features = allfolds_test_orig_features[fold]
      new_prob_predictions = model.predict_proba(new_test_features)
      new_train_prob_predictions = model.predict_proba(new_train_features)
      new_predictions = model.predict(new_test_features)
      new_train_predictions = model.predict(new_train_features)
      new_test_labels = allfolds_test_labels[fold]
      new_train_labels = allfolds_train_labels[fold]
      train_predictions = model.predict(allfolds_train_features[fold])
      train_labels = allfolds_train_labels[fold]
    
      if 'RF' in learn_algo and UQ:
        tree_preds = np.empty((len(model.estimators_), new_test_features.shape[0], num_classes))
        tree_train_preds = np.empty((len(model.estimators_), new_train_features.shape[0], num_classes))
        for t, tree in enumerate(model.estimators_):
          tree_preds[t] = tree.predict_proba(new_test_features.values)
          tree_train_preds[t] = tree.predict_proba(new_train_features.values)
        new_pred_uncertainty = np.nanpercentile(tree_preds, 90, axis=0) - np.nanpercentile(tree_preds, 10, axis=0) 
        new_pred_stds = tree_preds.var(axis=0)#tree_preds.std(axis=0)
        new_train_pred_uncertainty = np.nanpercentile(tree_train_preds, 90, axis=0) - np.nanpercentile(tree_train_preds, 10, axis=0)
        new_train_pred_stds = tree_train_preds.var(axis=0)

      if combine_classes and num_classes==5:

        temp = np.ones(new_predictions.shape[0])
        temp[new_predictions==0] = 0
        temp[new_predictions==4] = 2
        new_predictions = temp.copy().astype(int)

        temp = np.ones(train_predictions.shape[0])

        temp[train_predictions==0] = 0                   
        temp[train_predictions==4] = 2
        train_predictions = temp.copy().astype(int)

        temp = np.ones(new_test_labels.shape[0])
        temp[new_test_labels==0] = 0                   
        temp[new_test_labels==4] = 2
        new_test_labels = temp.copy().astype(int) 

        temp = np.ones(train_labels.shape[0])
        temp[train_labels==0] = 0           
        temp[train_labels==4] = 2
        train_labels = temp.copy().astype(int)

        temp = np.empty((new_prob_predictions.shape[0], 3))
        temp[:,0] = new_prob_predictions[:,0]
        temp[:,2] = new_prob_predictions[:,4]
        temp[:,1] = np.sum(new_prob_predictions[:,1:4], axis=1)
        new_prob_predictions = temp.copy()

      new_errors = abs(new_predictions - new_test_labels)
      allfolds_errors.append(np.mean(new_errors))
      new_class_acc = balanced_accuracy_score(new_test_labels, new_predictions)#np.count_nonzero(new_predictions == new_test_labels)/len(new_test_labels)
      class_acc.append(new_class_acc)
      train_errors = abs(train_predictions - train_labels)
      all_train_errors = np.concatenate((train_errors, all_train_errors))

      if fold==0:
        test_features = copy.deepcopy(new_test_features)
        if stratify_verif:
          test_orig_features = copy.deepcopy(new_test_orig_features)
        predictions = copy.deepcopy(new_predictions)
        train_predictions2 = copy.deepcopy(new_train_predictions)
        prob_predictions = copy.deepcopy(new_prob_predictions)
        train_prob_predictions = copy.deepcopy(new_train_prob_predictions)
        test_labels = copy.deepcopy(new_test_labels)
        train_labels2 = copy.deepcopy(new_train_labels)
        if 'RF' in learn_algo and UQ:
          pred_uncertainty = copy.deepcopy(new_pred_uncertainty)
          pred_stds = copy.deepcopy(new_pred_stds)
          train_pred_uncertainty = copy.deepcopy(new_train_pred_uncertainty)
          train_pred_stds = copy.deepcopy(new_train_pred_stds)
      else:
        test_features = pd.concat((test_features, new_test_features))
        if stratify_verif:
          test_orig_features = pd.concat((test_orig_features, new_test_orig_features))
        predictions = np.concatenate((predictions, new_predictions))
        train_predictions2 = np.concatenate((train_predictions2, new_train_predictions))
        prob_predictions = np.concatenate((prob_predictions, new_prob_predictions))
        train_prob_predictions = np.concatenate((train_prob_predictions, new_train_prob_predictions))
        test_labels = np.concatenate((test_labels, new_test_labels))
        train_labels2 = np.concatenate((train_labels2, new_train_labels))
        if 'RF' in learn_algo and UQ:
          pred_uncertainty = np.concatenate((pred_uncertainty, new_pred_uncertainty))
          pred_stds = np.concatenate((pred_stds, new_pred_stds))
          train_pred_uncertainty = np.concatenate((train_pred_uncertainty, new_train_pred_uncertainty))
          train_pred_stds = np.concatenate((train_pred_stds, new_train_pred_stds))

      #baseline_preds = np.concatenate((baseline_preds, np.repeat((num_classes+1)/2-1, len(allfolds_test_labels[fold]))))
      baseline_preds = np.concatenate((baseline_preds, np.random.choice([0,1,2], size=len(allfolds_test_labels[fold]), p=[0.2, 0.6, 0.2])))

      print ('Mean training prediction error (fold %s): %.2f' % (fold, round(np.mean(train_errors), 2))) 
      print ('Mean testing  prediction error (fold %s): %.2f' % (fold, round(np.mean(new_errors), 2)))
      print ('Classification Accuracy (fold %s): %.3f' % (fold, new_class_acc))

      if num_classes==3 or combine_classes:
        classes = ['Below Avg', 'Near Avg', 'Above Avg']
      elif num_classes==5:
        classes = ['Poor', 'Below Avg', 'Near Avg', 'Above Avg', 'Great']
      fig, SRs, tpr, all_auc, all_max_csi, all_max_csi_thres = plot_ROC(new_prob_predictions, new_test_labels, classes)
      print ('Macro-average AUC (fold %s): %.3f ' % (fold, mean(all_auc)))
      allfolds_auc.append(mean(all_auc))

    if combine_classes:
      num_classes = 3

    allfolds_errors = np.asarray(allfolds_errors)
    class_acc = np.asarray(class_acc)
    allfolds_auc = np.asarray(allfolds_auc)

    print ('Testing error across folds:            %.3f +/- %.3f' % (np.mean(allfolds_errors), np.std(allfolds_errors))) 
    print ('Classification Accuracy across folds:  %.3f +/- %.3f' % (np.mean(class_acc), np.std(class_acc)))
    print ('Macro-Average AUC across folds:        %.3f +/- %.3f' % (np.mean(allfolds_auc), np.std(allfolds_auc)))

    if stratify_verif:
      test_orig_features['prob_predictions'] = prob_predictions.tolist()
      test_orig_features['test_labels'] = test_labels.tolist()
      test_orig_features['predictions'] = predictions.tolist()
      test_orig_features.reset_index(drop=True, inplace=True) 
      test_orig_features.to_feather(predictions_fname) 

  else:

    baseline_preds = np.repeat(1.0, len(test_labels))
    prob_predictions = model.predict_proba(test_features)

    train_predictions = model.predict(train_features)
    predictions = model.predict(test_features)

  if train_test_split_param != 'dates':
    all_train_errors = abs(train_predictions - train_labels)
  errors = abs(predictions - test_labels)

  if 'RF' in learn_algo and UQ:

   all_train_RMSEs = []; all_train_SDs = []; all_test_RMSEs = []; all_test_SDs = []
   for n in range(num_classes):
     condition = (train_predictions2 == n)
     bins, train_RMSEs, train_SDs = spread_error(train_prob_predictions[condition][:,n], np.where(train_labels2[condition]==n, 1, 0), train_pred_stds[condition][:,n])
     all_train_RMSEs.append(train_RMSEs); all_train_SDs.append(train_SDs)
     for nn in range(len(bins)):
       print (n, train_SDs[nn], train_RMSEs[nn])
   for n in range(num_classes):
     condition = (predictions == n)
     bins, test_RMSEs, test_SDs = spread_error(prob_predictions[condition][:,n], np.where(test_labels[condition]==n, 1, 0), pred_stds[condition][:,n])
     all_test_RMSEs.append(test_RMSEs); all_test_SDs.append(test_SDs)
     for nn in range(len(bins)):
       print (n, test_SDs[nn], test_RMSEs[nn])

   for n in range(len(all_train_RMSEs)):
     train_SDs = all_train_SDs[n]; train_RMSEs = all_train_RMSEs[n]; test_SDs = all_test_SDs[n]; test_RMSEs = all_test_RMSEs[n]
     P.figure(figsize=(20, 12))
     ax = P.subplot(121)
     P.title('Training')
     P.scatter(train_SDs, train_RMSEs)
     P.xlabel('spread')
     P.ylabel('error')
     P.axis('equal') 
     P.plot([0, 1], [0, 1], transform=ax.transAxes)
     P.xlim([0, 0.6]); P.ylim([0, 0.6])
     ax = P.subplot(122)
     P.title('Testing')
     P.scatter(test_SDs, test_RMSEs)
     P.xlabel('spread')
     P.ylabel('error')
     P.axis('equal')
     P.plot([0, 1], [0, 1], transform=ax.transAxes)
     P.xlim([0, 0.6]); P.ylim([0, 0.6])
     P.savefig('%s/spread_error_%s_class=%d.png' % (MAIN_DIR, exp_name, n), bbox_inches='tight') 

   prob_bins = [ [0.2*n, 0.2*(n+1)] for n in range(5) ] 
   prob_bins[-1][1] += .001

   for prob_bin in prob_bins:

     print('Probabilities: %.1f-%.1f\n' % (prob_bin[0], prob_bin[1]))

     all_train_RMSEs = []; all_train_SDs = []; all_test_RMSEs = []; all_test_SDs = []; N_train = []; N_test = []

     for n in range(num_classes):

       condition = ( (train_prob_predictions[:,n] >= prob_bin[0]) & (train_prob_predictions[:,n] < prob_bin[1]) )
       train_prob_predictions_sub = train_prob_predictions[condition]; train_labels2_sub = train_labels2[condition];  train_pred_stds_sub = train_pred_stds[condition]; train_predictions2_sub = train_predictions2[condition]
       condition = ( (prob_predictions[:,n] >= prob_bin[0]) & (prob_predictions[:,n] < prob_bin[1]) )
       prob_predictions_sub = prob_predictions[condition]; test_labels_sub = test_labels[condition]; pred_stds_sub = pred_stds[condition]; predictions_sub = predictions[condition]
 
       condition = (train_predictions2_sub == n)
       bins, train_RMSEs, train_SDs = spread_error(train_prob_predictions_sub[condition][:,n], np.where(train_labels2_sub[condition]==n, 1, 0), train_pred_stds_sub[condition][:,n])
       all_train_RMSEs.append(train_RMSEs); all_train_SDs.append(train_SDs)
 
       N = len(train_prob_predictions_sub[condition])
       N_train.append(N)
       if N >= 0:
         print ('N (training) = %d' % N)
         for nn in range(len(bins)):
           print (n, train_SDs[nn], train_RMSEs[nn])
        
     for n in range(num_classes):

       condition = (predictions_sub == n)
       bins, test_RMSEs, test_SDs = spread_error(prob_predictions_sub[condition][:,n], np.where(test_labels_sub[condition]==n, 1, 0), pred_stds_sub[condition][:,n])
       all_test_RMSEs.append(test_RMSEs); all_test_SDs.append(test_SDs)

       N = len(prob_predictions_sub[condition])
       N_test.append(N)
       if N >= 0:
         print ('N (testing) = %d' % N)
         for nn in range(len(bins)):
           print (n, test_SDs[nn], test_RMSEs[nn])

     for n in range(len(all_train_RMSEs)):

       train_SDs = all_train_SDs[n]; train_RMSEs = all_train_RMSEs[n]; test_SDs = all_test_SDs[n]; test_RMSEs = all_test_RMSEs[n]
       P.figure(figsize=(20, 12))

       ax = P.subplot(121)
       P.title('Training (P=%.1f-%.1f, N=%d)' % (prob_bin[0], prob_bin[1], N_train[n]))
       P.scatter(train_SDs, train_RMSEs)
       P.xlabel('spread')
       P.ylabel('error')
       P.axis('equal')
       P.plot([0, 1], [0, 1], transform=ax.transAxes)
       P.xlim([0, 0.6]); P.ylim([0, 0.6])

       ax = P.subplot(122)
       P.title('Testing (P=%.1f-%.1f, N=%d)' % (prob_bin[0], prob_bin[1], N_test[n]))
       P.scatter(test_SDs, test_RMSEs)
       P.xlabel('spread')
       P.ylabel('error')
       P.axis('equal')
       P.plot([0, 1], [0, 1], transform=ax.transAxes)
       P.xlim([0, 0.6]); P.ylim([0, 0.6])
       P.savefig('%s/spread_error_%s_class=%d_P=%.1f-%.1f.png' % (MAIN_DIR, exp_name, n, prob_bin[0], prob_bin[1]), bbox_inches='tight')

  baseline_errors = abs(baseline_preds - test_labels)
  print ('Mean random error: ', round(np.mean(baseline_errors), 2))
  print ('Random Classification Accuracy: ', round(balanced_accuracy_score(test_labels, baseline_preds),2))#round(np.count_nonzero(baseline_preds == test_labels)/len(test_labels), 2))

  print ('Mean training prediction error: ', round(np.mean(all_train_errors), 2))
  print ('Mean prediction error: ', round(np.mean(errors), 2))
  print ('Classification Accuracy: ', round(balanced_accuracy_score(test_labels, predictions),2))#round(np.count_nonzero(predictions == test_labels)/len(test_labels), 2))
  for i in range(num_classes):
    hits   = len( predictions[( (test_labels==i) & (predictions==i) )] )
    misses = len( predictions[( (test_labels==i) & (predictions!=i) )] )
    false_alarms = len( predictions[( (test_labels!=i) & (predictions==i) )] ) 
    corr_negs = len( predictions[( (test_labels!=i) & (predictions!=i) )] )
    POD = hits/(hits+misses)
    FAR = false_alarms/(hits+false_alarms)
    CSI = hits/(hits+misses+false_alarms)
    #print (POD,FAR,CSI,1/(1/(1-FAR)+1/POD-1))
    #POD = np.count_nonzero(predictions[test_labels==i] == i)/len(test_labels[test_labels==i]) 
    #FAR = np.count_nonzero(predictions[test_labels!=i] == i)/len(test_labels[test_labels!=i])
    print ('POD (CLASS %d): %.2f' % (i, round(POD, 2)))
    print ('FAR (CLASS %d): %.2f' % (i, round(FAR, 2)))
    print ('CSI (CLASS %d): %.2f' % (i, round(CSI, 2)))#(i, round(np.count_nonzero(predictions[test_labels==i] == i)/(len(test_labels[test_labels==i])+np.count_nonzero(predictions[test_labels!=i] == i)), 2)))

  matt_corr = matthews_corrcoef(test_labels, predictions)
  print ('Matthews correlation coef = ', round(matt_corr,2)) 

  score = log_loss(test_labels, prob_predictions)

  print ('Log loss = ', round(score,2))

  if num_classes==3:
    classes = ['POOR', 'FAIR', 'GOOD']
  elif num_classes==5:
    classes = ['Poor', 'Below Avg', 'Near Avg', 'Above Avg', 'Great']

  fig, SRs, tpr, all_auc, all_max_csi, all_max_csi_thres = plot_ROC(prob_predictions, test_labels, classes)
  P.savefig(ROC_fname, bbox_inches='tight')

  with open(roc_data_fname, 'wb') as handle:
    pickle.dump((prob_predictions, test_labels), handle)

  mean_auc = mean(all_auc)
  print ('Macro-average AUC = ', round(mean_auc,2))

  mean_probs = []; cond_freqs = []; bss_rels = []
  for i in range(num_classes):
    y_true = np.where((test_labels==i), 1, 0)
    y_pred = prob_predictions[:,i]
    #mean_prob, cond_freq, ef_low, ef_up = reliability_uncertainty(y_true, y_pred, n_iter = 1000, n_bins=5)
    mean_prob, cond_freq = reliability_curve(y_true, y_pred, bin_edges = [0, .10, .20, .30, .40, .50, .60, .70])
    base_rate = sum(y_true) / len(test_labels)
    mean_probs.append(mean_prob)
    cond_freqs.append(cond_freq)
    #print (mean_prob, cond_freq, base_rate)
    bss_rel = bss_reliability(y_true, y_pred, bin_edges = [0, .10, .20, .30, .40, .50, .60, .70])
    #bss_rel = brier_skill_score(y_true, y_pred)
    print (i, bss_rel)
    bss_rels.append(bss_rel)
  fig = attr_diag(mean_probs, cond_freqs, base_rate, bss_rels, classes)
  P.savefig(AD_fname, bbox_inches='tight')

  with open(attr_data_fname, 'wb') as handle:
    pickle.dump((mean_probs, cond_freqs, base_rate, bss_rels), handle)

  fig1 = perf_diag_empty()
  class_proportions = {'POOR': 0.2, 'FAIR': 0.6, 'GOOD': 0.2}
  AUPDCs = compute_aupdc(prob_predictions, test_labels, classes)
  for label in range(len(classes)):
    indices = np.argsort(SRs[label])
    sorted_SRs = np.array(SRs[label])[indices] 
    sorted_tpr = tpr[label][indices]
    c = class_proportions[classes[label]]
    aupdc2 = auc(sorted_SRs, sorted_tpr)+c
    aupdc = AUPDCs[classes[label]]
    SRmins = [c*POD/(1-c+c*POD) for POD in tpr[label]] 
    pos = sum(np.where((test_labels==label), 1, 0))
    neg = sum(np.where((test_labels!=label), 1, 0))
    AUPDCmin = 1/pos*np.sum([i/(i+neg) for i in range(pos)])
    NAUPDC = (aupdc - AUPDCmin) / (1 - AUPDCmin)
    #print (NAUPDC, aupdc, aupdc2, AUPDCmin)
    NCSI = (all_max_csi[label]-c) / (1-c)
    P.plot(SRs[label], tpr[label], label='%s: NAUPDC = %.2f, MAX NCSI = %.2f' % (classes[label], NAUPDC, NCSI), lw=3, color=class_colors[classes[label]])
    #P.plot(SRs[label], tpr[label], label='%s: NAUPDC = %.2f, MAX NCSI = %.2f, (%02d%%)' % (classes[label], NAUPDC, NCSI, 100*all_max_csi_thres[label]), lw=3, color=class_colors[classes[label]])
    #P.plot(SRmins, tpr[label], ls='--', lw=3, color=class_colors[classes[label]])
    print ('Class %d: NAUPDC = %.2f, NCSI = %.2f' % (label, NAUPDC, NCSI))
  P.legend(loc='best', fontsize=11)
  P.savefig(PD_fname, bbox_inches='tight')

  # Compute confusion matrix
  cnf_matrix = confusion_matrix(test_labels, predictions)
  np.set_printoptions(precision=2)

  if num_classes==5:
    bad_preds = cnf_matrix[0, 4] + cnf_matrix[0, 3] + cnf_matrix[1, 4] + cnf_matrix[3, 0] + cnf_matrix[4, 0] + cnf_matrix[4, 1]
  elif num_classes==3:
    bad_preds = cnf_matrix[0, 2] + cnf_matrix[2, 0]
  print ('BAD PREDICTIONS = %.2f' % (bad_preds/len(test_labels)))

  RPS_val = RPS(test_labels, prob_predictions)
  print ('RPS = %.2f' % RPS_val)
  RPSS_val = RPSS(test_labels, prob_predictions)
  print ('RPSS = %.2f' % RPSS_val)

  # Plot non-normalized confusion matrix
  P.figure()
  plot_confusion_matrix(cnf_matrix, classes=classes,
                    title='Confusion matrix')
  P.savefig('%s/conf_matrix_%s.png' % (MAIN_DIR, exp_name))

  # Plot normalized confusion matrix
  P.figure()
  plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                    title='Normalized confusion matrix')
  P.savefig('%s/conf_matrix_norm_%s.png' % (MAIN_DIR, exp_name))

  indices = np.where( (np.amax(prob_predictions, axis=1) >= 0.6) )[0]
  print (len(indices), len(predictions), round(len(indices)/len(predictions),2))
  predictions2 = predictions[indices]
  test_labels2 = test_labels[indices]

  cnf_matrix = confusion_matrix(test_labels2, predictions2)

  # Plot non-normalized confusion matrix
  P.figure()
  plot_confusion_matrix(cnf_matrix, classes=classes,
                    title='Confusion matrix')
  P.savefig('%s/conf_matrix_highconfidence_%s.png' % (MAIN_DIR, exp_name))

  # Plot normalized confusion matrix
  P.figure()
  cnf_matrix = confusion_matrix(test_labels2, predictions2)
  plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                    title='Normalized confusion matrix')
  P.savefig('%s/conf_matrix_norm_highconfidence_%s.png' % (MAIN_DIR, exp_name))

  print(("\n TOTAL RUNTIME: %.1f \n" % ((time.time() - start_time)/60.)))


#-------------------------------------------------------------------------------                                                                                                                                              
if __name__ == "__main__":
    sys.exit(main())
