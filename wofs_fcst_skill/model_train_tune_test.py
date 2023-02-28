# Model training, hyperparameter optimization, testing, evaluation

from defs import *

# Write stdout/stderr to log files (in addition to screen)
output1 = open(logfile,'w')
output2 = open(errfile,'w')
sys.stdout=Unbuffered(sys.stdout, output1)
sys.stderr=Unbuffered(sys.stderr, output2)

rng = np.random.RandomState(0)

start_time = time.time()

# Read feature dataset
print ('Reading features from: %s' % fname)
features = pd.read_feather(fname)

# Create list of allowable ML features; intersection of this set and the feature set read above will be final feature set

# If using one of the baseline models, the training set can only be the persistence or spread feature
if pers_bl:
  training_features = [pers_feature]
elif sprd_bl:
  training_features = [sprd_feature]
# Else if using top-ranked features based on importance (imp_vars=True), only allow the top-N features where N = imp_vars
elif imp_vars:
  print ('Reading important varnames: %s' % imp_vars_fname)
  with open(imp_vars_fname, 'rb') as handle:
    training_features = pickle.load(handle)[:imp_vars].tolist()
# Else if using a full model with a feature set not restricted by feature importance, allow all features 
else:
  training_features = storm_init_predictors+mean_storm_init_predictors+storm_fcst_predictors+env_init_predictors+env_fcst_predictors+box_predictors
  # Time features are replaced by converted versions
  training_features.remove('init_time'); training_features.remove('valid_time'); #training_features.remove('lead_time')

# Convert times from 'hhmm' to # hours after 20 UTC (only done when a new WoFS stats dataset is used)
if fname == prelim_features_fname and not pers_bl and not sprd_bl:
  for var in ['init_time', 'valid_time']:
    if var in features.columns:
      features[var+'_2'] = convert_time(features[var])
      training_features += [var+'_2']

# Special treatment for categorical features
dummy_cols = [metric for metric in categorical_metrics if (metric in training_features and metric in features.columns)]
if len(dummy_cols) > 0:
  features = pd.get_dummies(features, columns=dummy_cols)
  for feature in features.columns:
    if any(item+'_' in feature for item in categorical_metrics):
      training_features.append(feature)

training_features = [feature for feature in training_features if feature in list(features.keys())]
print ('%d training features' % len(training_features))

# Remove duplicate examples
print ('Before removing duplicate examples: ', features.shape[0])
features.drop_duplicates(inplace=True, ignore_index=True)
print ('After removing duplicate examples: ', features.shape[0])

# Remove examples with undefined label, and to keep example sets the same between full and baseline models, also remove examples where a baseline variable is undefined
print ('Before removing examples with undefined target/BL feature: ', features.shape[0])
features = features[((features[target] != -999) & (features[pers_feature] != -999) & (features[sprd_feature] != -999))]
print ('After removing examples with undefined target/BL feature: ', features.shape[0])

# retain a copy of all features for use in verification; only training features will be retained in 'features' dataset for use in ML 
all_features = copy.deepcopy(features)

# Read labels or generate labels by converting eFSS to percentiles and classifying
if 'labels' in features.columns:
  labels = features['labels'].values
else:
  if num_classes==5:
    breakpoints = [np.nanpercentile(features[target], n/num_classes*100) for n in range(num_classes+1)]
  elif num_classes==3:
    breakpoints = [np.nanpercentile(features[target], n) for n in [0, 20, 80, 100]]
  breakpoints[-1] += .001
  print ('Breakpoints for %s for label generation:' % target, breakpoints)
  labels = np.zeros(len(features[target]))
  for n in range(len(breakpoints)-1):
    labels[( (features[target] >= breakpoints[n]) & (features[target] < breakpoints[n+1]) )] = n
  labels = labels.astype(int)

print ('Feature correlations with target:\n')
corrs = np.array([np.corrcoef(features[feature], labels)[0][1] for feature in training_features])
sorted_inds = np.argsort(-np.fabs(corrs))
sorted_corrs = corrs[sorted_inds]
for c, corr in enumerate(sorted_corrs):
  print (c, corr, training_features[sorted_inds[c]], features[training_features[sorted_inds[c]]].min(), features[training_features[sorted_inds[c]]].max())

# Retain set of original (i.e., prior to model reduction) training features plus labels and auxiliary features
if 'reduced' not in fname:
  temp = copy.deepcopy(features)
  temp = temp.drop([feature for feature in temp.columns if feature not in training_features+label_metrics+spread_metrics+other_features], axis=1)
  temp['labels'] = labels
  temp.reset_index(drop=True, inplace=True)
  temp.to_feather(orig_features_fname)
  print ('\nWriting original features to: %s' % orig_features_fname)
  print ('Features dims = ', temp.shape)

# Finally, drop all features that won't be used for ML
features = features.drop([feature for feature in features.columns if (feature not in training_features and feature[-2] != '_')], axis=1)

print ('\n%d samples total for training/testing' % features.shape[0])

# Split examples into training/testing sets

if train_test_split_param == 'dates':
  allfolds_train_features, allfolds_train_labels, allfolds_test_features, allfolds_test_labels, allfolds_all_features_train, allfolds_test_orig_features, date_groups \
        = dataset_split(features, all_features, labels, train_test_split_param, class_balancing, num_folds)
else:
  train_features, train_labels, test_features, test_labels, all_features_train \
        = dataset_split(features, all_features, labels, train_test_split_param, class_balancing, None, rng)

allfolds_train_features_unscaled = allfolds_train_features.copy()

# Scale the features if required by the learning algorithm

if learn_algo in ['LR', 'NN', 'OLR']: 

  if train_test_split_param == 'dates':

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

# Store features & labels for each train and test fold

if train_test_split_param == 'dates':
  train_test_fnames = [train_test_fname.split('.joblib')[0] + '_%d.joblib' % fold for fold in range(num_folds)]
  for fold in range(num_folds):
    train_features = allfolds_train_features_unscaled[fold]; train_labels = allfolds_train_labels[fold]; test_features = allfolds_test_features[fold]; test_labels = allfolds_test_labels[fold]
    dump((train_features, train_labels, test_features, test_labels), train_test_fnames[fold])
    print ('\n%d/%d training/testing samples' % (len(train_labels), len(test_labels)))
    print ('Class balance of training dataset: ', [round(len(train_labels[train_labels==n])/train_features.shape[0],2) for n in range(num_classes)])
    print ('Class balance of testing dataset: ', [round(len(test_labels[test_labels==n])/test_features.shape[0],2) for n in range(num_classes)])
else:
  dump((train_features, train_labels, test_features, test_labels), train_test_fname)
  print ('%d/%d training/testing samples' % (len(train_labels), len(test_labels)))
  print ('Class balance of training dataset: ', [round(len(train_labels[train_labels==n])/train_features.shape[0],2) for n in range(num_classes)])
  print ('Class balance of testing dataset: ', [round(len(test_labels[test_labels==n])/test_features.shape[0],2) for n in range(num_classes)])

# Train the model (i.e., don't read previously trained model)

if do_train:

  # Perform hyperparameter optimization (i.e., don't read previously optimized hyperparameters)

  if do_hyper_opt:

    # Get hyperparameter search space and model training parameters; user prescribes these in hyper_opt_learn_prep 
    learner, param_distributions = hyper_opt_learn_prep(learn_algo, rng)

    # Generate train-test splits for CV 
    if tuning_CV_split_method != 'dates':
      CV_param = hyper_opt_unnestedCV_prep(train_test_split_param, tuning_CV_split_method, num_classes, all_features['date'].values, date_groups)
    else:
      CV_param = hyper_opt_nestedCV_prep(allfolds_all_features_train, allfolds_train_labels, train_test_split_param, tuning_CV_split_method, num_classes, date_groups, class_balancing)

    # Perform the hyperparameter tuning and return the optimal values
    if tuning_CV_split_method == 'random':
      best_params = hyper_opt_perform_unnestedCV(train_features, train_labels, best_params_fname, learner, param_distributions, CV_param, n_iter, rng)
    else:
      allfolds_best_params = hyper_opt_perform_nestedCV(allfolds_train_features, allfolds_train_labels, best_params_fname, learner, param_distributions, CV_param, n_iter, rng)

    print ('\nTuned hyperparameters!')

  # Otherwise read previously optimized hyperparameters

  else:

    if train_test_split_param == 'dates':

      allfolds_best_params = []
      for fold in range(num_folds):
        best_params_fname2 = best_params_fname.split('.pkl')[0]+'_%d.pkl' % fold
        with open(best_params_fname2, 'rb') as handle:
          allfolds_best_params.append(pickle.load(handle))

    else:

      with open(best_params_fname, 'rb') as handle:
        best_params = pickle.load(handle)

    print ('\nRead in hyperparameters!')

  # Hyperparameters are now set; can proceed with training the model(s)
  if train_test_split_param == 'dates':
    allfolds_model = train_model_nestedCV(allfolds_best_params, allfolds_train_features, allfolds_train_labels, learn_algo, best_model_fname, rng)
  else:
    model = train_model_unnestedCV(best_params, train_features, train_labels, learn_algo, best_model_fname, rng)

  print ('\nTrained model(s)!')

# If not training model(s), read them in

else:

  if train_test_split_param == 'dates':

    allfolds_model = []

    for fold in range(num_folds):

      best_model_fname2 = best_model_fname.split('.joblib')[0]+'_%d.joblib' % fold
      model = load(best_model_fname2)
      allfolds_model.append(model)

  else:

    model = load(best_model_fname)

  print ('\nRead in model(s)!')

############################################################
# Verify deterministic and probabilistic model performance #
############################################################

# Get/collate model predictions, features, labels

if train_test_split_param != 'dates':   

  baseline_preds = np.repeat(1.0, len(test_labels))
  prob_predictions = model.predict_proba(test_features)

  train_predictions = model.predict(train_features)
  predictions = model.predict(test_features)

else:

  baseline_preds, predictions, train_predictions, prob_predictions, test_labels, num_classes2 \
     = verify(allfolds_model, allfolds_train_labels, allfolds_test_labels, allfolds_train_features, allfolds_test_features, allfolds_test_orig_features, combine_classes, num_classes, stratify_verif=True) 

# Compute deterministic model errors and classification accuracy
 
all_train_errors = abs(train_predictions - train_labels)
errors = abs(predictions - test_labels)
baseline_errors = abs(baseline_preds - test_labels)

print ('\nMean random error: ', round(np.mean(baseline_errors), 2))
print ('Random Balanced Classification Accuracy: ', round(balanced_accuracy_score(test_labels, baseline_preds),2))#round(np.count_nonzero(baseline_preds == test_labels)/len(test_labels), 2))

print ('Mean training error: ', round(np.mean(all_train_errors), 2))
print ('Mean prediction error: ', round(np.mean(errors), 2))
print ('Balanced Classification Accuracy: ', round(balanced_accuracy_score(test_labels, predictions),2))#round(np.count_nonzero(predictions == test_labels)/len(test_labels), 2))

# Compute contingency table statistics

for i in range(num_classes2):

  hits   = len( predictions[( (test_labels==i) & (predictions==i) )] )
  misses = len( predictions[( (test_labels==i) & (predictions!=i) )] )
  false_alarms = len( predictions[( (test_labels!=i) & (predictions==i) )] ) 
  corr_negs = len( predictions[( (test_labels!=i) & (predictions!=i) )] )

  POD = hits/(hits+misses)
  FAR = false_alarms/(hits+false_alarms)
  CSI = hits/(hits+misses+false_alarms)

  print ('\nPOD (CLASS %d): %.2f' % (i, round(POD, 2)))
  print ('FAR (CLASS %d): %.2f' % (i, round(FAR, 2)))
  print ('CSI (CLASS %d): %.2f' % (i, round(CSI, 2)))#(i, round(np.count_nonzero(predictions[test_labels==i] == i)/(len(test_labels[test_labels==i])+np.count_nonzero(predictions[test_labels!=i] == i)), 2)))

print ()
matt_corr = matthews_corrcoef(test_labels, predictions)
print ('Matthews correlation coef = ', round(matt_corr,2)) 

score = log_loss(test_labels, prob_predictions)
print ('Log loss = ', round(score,2))

RPS_val = RPS(test_labels, prob_predictions)
print ('RPS = %.2f' % RPS_val)
RPSS_val = RPSS(test_labels, prob_predictions)
print ('RPSS = %.2f' % RPSS_val)

if num_classes2==3:
  classes = ['POOR', 'FAIR', 'GOOD']
elif num_classes2==5:
  classes = ['Poor', 'Below Avg', 'Near Avg', 'Above Avg', 'Great']

# ROC Diagram and AURC

fig, SRs, tpr, all_auc, all_max_csi, all_max_csi_thres = plot_ROC(prob_predictions, test_labels, classes)
P.savefig(ROC_fname, bbox_inches='tight')
print ('Saved %s' % ROC_fname)

with open(roc_data_fname, 'wb') as handle:
  pickle.dump((prob_predictions, test_labels), handle)

mean_auc = mean(all_auc)
print ('\nMacro-average AUC = %.2f\n' % round(mean_auc,2))

# Reliability Diagram and BSSrel

mean_probs = []; cond_freqs = []; bss_rels = []
for i in range(num_classes2):
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
  print ('Class %d: BSSrel = %.3f' % (i, bss_rel))
  bss_rels.append(bss_rel)
fig = attr_diag(mean_probs, cond_freqs, base_rate, bss_rels, classes)
P.savefig(AD_fname, bbox_inches='tight')
print ('Saved %s' % AD_fname)

with open(attr_data_fname, 'wb') as handle:
  pickle.dump((mean_probs, cond_freqs, base_rate, bss_rels), handle)

# Performance Diagram and AUPDC

print ()
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
  NCSI = (all_max_csi[label]-c) / (1-c)
  P.plot(SRs[label], tpr[label], label='%s: NAUPDC = %.2f, MAX NCSI = %.2f' % (classes[label], NAUPDC, NCSI), lw=3, color=class_colors[classes[label]])
  print ('Class %d: NAUPDC = %.2f, NCSI = %.2f' % (label, NAUPDC, NCSI))
P.legend(loc='best', fontsize=11)
P.savefig(PD_fname, bbox_inches='tight')
print ('Saved %s\n' % PD_fname)

# Confusion Matrices

cnf_matrix = confusion_matrix(test_labels, predictions)
np.set_printoptions(precision=2)

# Rate of bad predictions (GOOD as POOR or vice versa)
if num_classes2==5:
  bad_preds = cnf_matrix[0, 4] + cnf_matrix[0, 3] + cnf_matrix[1, 4] + cnf_matrix[3, 0] + cnf_matrix[4, 0] + cnf_matrix[4, 1]
elif num_classes2==3:
  bad_preds = cnf_matrix[0, 2] + cnf_matrix[2, 0]
print ('BAD PREDICTIONS = %.2f\n' % (bad_preds/len(test_labels)))

# Plot non-normalized confusion matrix
P.figure()
plot_confusion_matrix(cnf_matrix, classes=classes,
                  title='Confusion matrix')
P.savefig('%s/conf_matrix_%s.png' % (MAIN_DIR, exp_name))
print ('Saved %s/conf_matrix_%s.png' % (MAIN_DIR, exp_name))

# Plot normalized confusion matrix
P.figure()
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                  title='Normalized confusion matrix')
P.savefig('%s/conf_matrix_norm_%s.png' % (MAIN_DIR, exp_name))
print ('Saved %s/conf_matrix_norm_%s.png' % (MAIN_DIR, exp_name))

# Focus on high-confidence predictions

prob_thres = 0.6
indices = np.where( (np.amax(prob_predictions, axis=1) >= prob_thres) )[0]
print ('\n%d high-confidence (>%.2f) predictions out of %d (%2d %%)\n' % (len(indices), prob_thres, len(predictions), 100*round(len(indices)/len(predictions),2)))
predictions2 = predictions[indices]
test_labels2 = test_labels[indices]

cnf_matrix = confusion_matrix(test_labels2, predictions2)

# Plot non-normalized confusion matrix
P.figure()
plot_confusion_matrix(cnf_matrix, classes=classes,
                  title='Confusion matrix')
P.savefig('%s/conf_matrix_highconfidence_%s.png' % (MAIN_DIR, exp_name))
print ('Saved %s/conf_matrix_highconfidence_%s.png' % (MAIN_DIR, exp_name))

# Plot normalized confusion matrix
P.figure()
cnf_matrix = confusion_matrix(test_labels2, predictions2)
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                  title='Normalized confusion matrix')
P.savefig('%s/conf_matrix_norm_highconfidence_%s.png' % (MAIN_DIR, exp_name))
print ('Saved %s/conf_matrix_norm_highconfidence_%s.png' % (MAIN_DIR, exp_name))

print(("\n TOTAL RUNTIME: %.1f \n" % ((time.time() - start_time)/60.)))

