from params import *

class Unbuffered:
  def __init__(self, stream, output):
    self.stream = stream
    self.output = output
  def write(self, data):
    self.stream.write(data)
    self.stream.flush()
    self.output.write(data)    # Write the data of stdout here to a text file as well
  def flush(self):
    self.stream.flush()

def compute_aupdc(prob_predictions, test_labels, classes):

  AUPDCs = {}

  for label in range(len(classes)):

    SRs = []; PODs = []
    
    y_true = np.zeros(len(test_labels))
    y_true[test_labels==label] = 1

    y_score = prob_predictions[:,label]

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)

    print ('AAAAAAAAAAAAAA', len(thresholds))

    for threshold in thresholds:

      indices = np.where( (y_true == 1) & (y_score >= threshold) )[0]
      hits = len(indices)
      indices = np.where( (y_true == 1) & (y_score < threshold) )[0]
      misses = len(indices)
      indices = np.where( (y_true == 0) & (y_score >= threshold) )[0]
      false_alarms = len(indices)
      indices = np.where( (y_true == 0) & (y_score < threshold) )[0]
      corr_negs = len(indices)

      if false_alarms == 0:
        SR = 1
      else:
        SR = 1 - false_alarms/float(false_alarms+hits)
      SRs.append(SR)

      POD = hits/float(hits+misses)
      PODs.append(POD)

    aupdc = sum([(PODs[k]-PODs[k-1])*SRs[k] for k in range(1,len(thresholds))])

    AUPDCs[classes[label]] = aupdc
   
  return AUPDCs

def plot_ROC(prob_predictions, test_labels, classes):

  fig = P.figure()#figsize=(16,12))
  P.plot([0, 1], [0, 1], 'k--', linewidth=3)
  P.xlabel('False Positive Rate', fontweight='bold')
  P.ylabel('Probability of Detection', fontweight='bold')
  #P.title('ROC curve')

  fpr = {}; tpr = {}; biases = {label: [] for label in range(len(classes))}; csis = {label: [] for label in range(len(classes))}; SRs = {label: [] for label in range(len(classes))} 
  all_auc = []; all_max_csi = []; all_max_csi_thres = []

  for label in range(len(classes)):

    y_true = np.zeros(len(test_labels))
    y_true[test_labels==label] = 1

    y_score = prob_predictions[:,label]

    fpr[label], tpr[label], thresholds = roc_curve(y_true, y_score, pos_label=1)

    for threshold in thresholds:

      indices = np.where( (y_true == 1) & (y_score >= threshold) )[0]
      hits = len(indices)
      indices = np.where( (y_true == 1) & (y_score < threshold) )[0]
      misses = len(indices)
      indices = np.where( (y_true == 0) & (y_score >= threshold) )[0]
      false_alarms = len(indices)
      indices = np.where( (y_true == 0) & (y_score < threshold) )[0]
      corr_negs = len(indices)

      bias = (hits+false_alarms)/float(hits+misses)
      biases[label].append(bias)

      csi = hits/float(hits+false_alarms+misses)
      csis[label].append(csi)

      if false_alarms == 0:
        SR = 1
      else:
        SR = 1 - false_alarms/float(false_alarms+hits)
      SRs[label].append(SR)

    index = np.argmax(np.asarray(csis[label]))#np.argmin(abs(1-np.asarray(biases[label])))
    max_csi = csis[label][index]
    max_csi_thres = thresholds[index]

    roc_auc = auc(fpr[label], tpr[label])

    all_auc.append(roc_auc)
    all_max_csi.append(max_csi)
    all_max_csi_thres.append(max_csi_thres)

    P.xticks(fontsize=14)
    P.yticks(fontsize=14)

    try:
      P.plot(fpr[label], tpr[label], label='%s: AUC=%.2f' % (classes[label], roc_auc), linewidth=3, color = class_colors[classes[label]])
      #P.plot(fpr[label], tpr[label], label='%s: AUC=%.2f, MAX CSI=%.2f' % (classes[label], roc_auc, max_csi), linewidth=3, color = class_colors[classes[label]])
    except:
      P.plot(fpr[label], tpr[label], label='%s: AUC=%.2f' % (classes[label], roc_auc), linewidth=3)
      #P.plot(fpr[label], tpr[label], label='%s: AUC=%.2f, MAX CSI=%.2f' % (classes[label], roc_auc, max_csi), linewidth=3)

  P.legend(loc='lower right', fontsize=11)#'best')

  return fig, SRs, tpr, all_auc, all_max_csi, all_max_csi_thres



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=P.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    P.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=0.8)
    #P.title(title)
    cbar = P.colorbar()
    cbar.ax.tick_params(labelsize=14)
    tick_marks = np.arange(len(classes))
    P.xticks(tick_marks, classes, rotation=45, fontsize=14)
    P.yticks(tick_marks, classes, fontsize=14)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        P.text(j, i, round(cm[i, j],2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #P.tight_layout()
    P.ylabel('True label', fontsize=18, fontweight='bold')
    P.xlabel('Predicted label', fontsize=18, fontweight='bold')
    P.grid(b=None)
    P.tight_layout()

def perf_diag_empty():

   gray5 = (150/255., 150/255., 150/255.)
   gray7 = (82/255., 82/255., 82/255.)

   purple5 = (158/255., 154/255., 200/255.)
   purple6 = (128/255., 125/255., 186/255.)
   purple7 = (106/255., 81/255., 163/255.)

   grid = np.arange(0.,1.005,0.005)

   sr_grid, pod_grid = np.meshgrid(grid,grid)

   bias_grid = pod_grid / sr_grid
   csi_grid = 1. / (1. / sr_grid + 1. / pod_grid - 1)

   csi_levels = np.arange(0.1,1.1,0.1)
   csi_colors = [purple5, purple5, purple5, purple5, \
                 purple5, purple5, purple5, purple5, \
                 purple5, purple5]

   bias_levels = [0.25, 0.5, 1., 1.5, 2., 3., 5.]
   bias_colors = [gray5, gray5, gray7, gray5, gray5, \
                  gray5, gray5]

   fig1 = P.figure(figsize=(7.,7.))
   ax1 = fig1.add_axes([0.13, 0.13, 0.85, 0.85])
   ax1.spines["top"].set_alpha(0.7)
   ax1.spines["top"].set_linewidth(0.5)
   ax1.spines["bottom"].set_alpha(0.7)
   ax1.spines["left"].set_alpha(0.7)
   ax1.spines["bottom"].set_linewidth(0.5)
   ax1.spines["left"].set_linewidth(0.5)

   ax1.spines["right"].set_linewidth(0.5)
   ax1.spines["right"].set_alpha(0.7)

   x_ticks = np.arange(0.,1.1,0.1)
   y_ticks = np.arange(0.,1.1,0.1)

   x_labels = ['', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '']
   y_labels = ['', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '']

   P.xlim(0.,1.)
   P.ylim(0.,1.)

   P.xticks(x_ticks, x_labels, fontsize=16, alpha=0.7)
   P.yticks(y_ticks, y_labels, fontsize=16, alpha=0.7)

   P.tick_params(axis="both", which="both", bottom="on", top="off", labelbottom="on", left="on", right="off", labelleft="on")

   ax1.set_xlabel('Success Ratio (1 - FAR)', fontsize=18, alpha=0.7, fontweight='bold')
   ax1.set_ylabel('Probability of Detection', fontsize=18, alpha=0.7, fontweight='bold')

   csi_con = P.contour(sr_grid, pod_grid, csi_grid, levels=csi_levels, colors=csi_colors, linewidths=1.25, alpha=0.8)
   bias_con = P.contour(sr_grid, pod_grid, bias_grid, levels=bias_levels, colors=bias_colors, linewidths=1., alpha=0.4)
   P.clabel(bias_con, fmt='%1.1f', manual=[(.2, 0.95), (.3, .95), (.47, .95), (.57, .95), (.8, .8), (.8, .42), (.8, .2)])
   P.clabel(csi_con, fmt='%1.1f', manual=[(.92, 0.1), (.92, .2), (.92, 0.3), (.92, .4), (.92, 0.5), (.92, .6), \
                                          (.92, 0.7), (.95, .85), (.95, 0.95)])
   return fig1

def perf_diag_data(fig, PODs, FARs):

  ax = fig.axes[0]

  for n in range(len(PODs)):
    if (n+1) % 2 == 0:
      marker = 'o'
    else:
      marker = 'o'#'+'
    if (n+1) % 6 in [1,2]:
      color = 'b'
    elif (n+1) % 6 in [3,4]:
      color = 'g'
    elif (n+1) % 6 in [5,0]:
      color = 'r'
    ax.plot(1-FARs[n], PODs[n], marker=marker, color=color, label=models[n], alpha=0.5)

  #P.legend(loc='lower left')#, shadow=True)

  return fig

def attr_diag(mean_probs, cond_freqs, base_rate, bss_rels, classes, ef_low=None, ef_up=None):

  fig, ax = P.subplots(dpi=300,figsize=(5,5))
  climo = 1/len(classes)#base_rate

  for i in range(len(mean_probs)):
    ax.plot(mean_probs[i], cond_freqs[i], label='%s : %.3f' % (classes[i], bss_rels[i]), lw=3)#, color='xkcd:darkish blue')
  ax.set_xlim([0,1])
  ax.set_ylim([0,1])
  ax.set_ylabel('Conditional Event Frequency', fontweight='bold')
  ax.set_xlabel('Mean Forecast Probability', fontweight='bold')
  x = np.linspace(0,1,100)
  P.xticks(fontsize=14)
  P.yticks(fontsize=14)


  ax.plot(x,x, linestyle='dashed', color='k', alpha=0.6)
  '''y = np.linspace( 0.5 * climo, 0.5 * (1.0 + climo), 100)
  ax.plot(
            [0, 1],
            [0.5 * climo, 0.5 * (1.0 + climo)],
            color="k",
            alpha=0.65,
            linewidth=0.5,
        )
  bottom = np.zeros((100,))
  ax.fill_between(x, bottom, y, where=y >= bottom, facecolor='grey', interpolate=True, alpha=0.5)

  props =  {'ha': 'center', 'va': 'bottom'}
  ang = to_angle(x, y)
  ax.text(0.7, 0.15, 'No Skill\nNegative BSS', props, color='grey', alpha=0.5,
                rotation=0, transform = ax.transAxes, fontsize=15)
  ax.grid(linestyle='dashed', linewidth=0.2)
  # Resolution Line
  ax.axhline(climo, linestyle="dashed", color="gray")
  ax.text(0.15, climo + 0.015, "No Resolution", fontsize=8, color="gray")

  # Uncertainty Line
  ax.axvline(climo, linestyle="dashed", color="gray")
  ax.text(
        climo + 0.01, 0.35, "Uncertainty", rotation=90, fontsize=8, color="gray"
        )
  '''
  if ef_low != None:
    for i in range(len(mean_prob)):
      ax.axvline(
            mean_prob[i], ymin=ef_low[i], ymax=ef_up[i], color='xkcd:darkish blue', alpha=0.5
        )
    #print (mean_prob[i], ef_low[i], ef_up[i])
  P.legend(loc='lower right', fontsize=11)#'best')

  return fig

def to_angle(xdata, ydata):
    import math
    dx = xdata[-1] - xdata[0]
    dy = ydata[-1] - ydata[0]
    ang = math.degrees(math.atan2(dy,dx))

    return ang

def brier_score(y, predictions):
    return np.mean((predictions - y) ** 2)

def brier_skill_score(y, predictions):
    return 1.0 - brier_score(y, predictions) / brier_score(y, y.mean())

def bss_reliability(y, predictions, n_bins=10, bin_edges=None):
    """
    Reliability component of BSS. Weighted MSE of the mean forecast probabilities
    and the conditional event frequencies. 
    """
    mean_fcst_probs, event_frequency, indices = reliability_curve(y, predictions, n_bins, bin_edges, return_indices=True)
    # Add a zero for the origin (0,0) added to the mean_fcst_probs and event_frequency
    counts = [1e-5]
    for i in indices:
        if i is np.nan:
            counts.append(1e-5)
        else:
            counts.append(len(i[0]))

    mean_fcst_probs[np.isnan(mean_fcst_probs)] = 1e-5
    event_frequency[np.isnan(event_frequency)] = 1e-5
    diff = (mean_fcst_probs-event_frequency)**2
    #for i in range(len(mean_fcst_probs)):
      #print (i, bin_edges[i], mean_fcst_probs[i], event_frequency[i])
    #print (event_frequency - mean_fcst_probs)
    return np.average(diff, weights=counts)

def reliability_curve(y, predictions, n_bins=10, bin_edges=None, return_indices=False):
    """
    Generate a reliability (calibration) curve. 
    Bins can be empty for both the mean forecast probabilities 
    and event frequencies and will be replaced with nan values. 
    Unlike the scikit-learn method, this will make sure the output
    shape is consistent with the requested bin count. The output shape
    is (n_bins+1,) as I artifically insert the origin (0,0) so the plot
    looks correct. 
    """

    if bin_edges == None:
      bin_edges = np.linspace(0,1, n_bins+1)
    else:
      n_bins = len(bin_edges) - 1
    bin_indices = np.clip(
                np.digitize(predictions, bin_edges, right=True) - 1, 0, None
                )

    indices = [np.where(bin_indices==i+1)
               if len(np.where(bin_indices==i+1)[0]) > 0 else np.nan for i in range(n_bins) ]
    #print ([len(np.where(bin_indices==i+1)[0]) for i in range(n_bins)])
    mean_fcst_probs = [np.nan if i is np.nan else np.mean(predictions[i]) for i in indices]
    event_frequency = [np.nan if i is np.nan else np.sum(y[i]) / len(i[0]) for i in indices]

    # Adding the origin to the data
    mean_fcst_probs.insert(0,0)
    event_frequency.insert(0,0)
   
    if return_indices:
        return np.array(mean_fcst_probs), np.array(event_frequency), indices 
    else:
        return np.array(mean_fcst_probs), np.array(event_frequency) 

    
def reliability_uncertainty(y_true, y_pred, n_iter = 1000, n_bins=10 ):
    '''
    Calculates the uncertainty of the event frequency based on Brocker and Smith (WAF, 2007)
    '''
    mean_fcst_probs, event_frequency = reliability_curve(y_true, y_pred, n_bins=n_bins)

    event_freq_err = [ ]
    for i in range( n_iter ):
        Z     = uniform( size = len(y_pred) )
        X_hat = resample( y_pred )
        Y_hat = np.where( Z < X_hat, 1, 0 )
        _, event_freq = reliability_curve(X_hat, Y_hat, n_bins=n_bins)
        event_freq_err.append(event_freq)

    ef_low = np.nanpercentile(event_freq_err, 2.5, axis=0)
    ef_up  = np.nanpercentile(event_freq_err, 97.5, axis=0)

    return mean_fcst_probs, event_frequency, ef_low, ef_up

def my_custom_score(y_true, y_pred):

  y_pred = y_pred[y_true != 1]
  y_true = y_true[y_true != 1]

  return accuracy_score(y_true, y_pred)

def custom_score_ROC(test_labels, prob_predictions, num_classes):

  all_auc = []

  for label in range(num_classes):

    y_true = np.zeros(len(test_labels))
    y_true[test_labels==label] = 1

    y_score = prob_predictions[:,label]

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)

    roc_auc = auc(fpr, tpr)

    all_auc.append(roc_auc)

  return mean(all_auc)

def my_custom_score_hinge(y_true, y_pred, labels):

  return hinge_loss(y_true, y_pred, labels=labels)

  '''num_classes = y_pred.shape[1]

  out = 0.
  h = h = Hinge(.8)

  for j in range(num_classes):
    if j <= y_true:
      out += h.loss(y_pred[:,j] + 1 - j, 1)
    else:
      out += h.loss(-(y_pred[:,j] + 1 - j) + 1, 1)

  return out 
'''

def RPS(y_true, y_pred):

  num_samples = y_pred.shape[0]
  num_class = y_pred.shape[1]

  y_true2 = np.zeros(y_pred.shape)
  for n in range(num_class):
    y_true2[:,n] = np.where((y_true==n), 1, 0)
    

  RPS = np.sum( np.sum( (np.cumsum(y_pred, axis=1) - np.cumsum(y_true2, axis=1))**2, axis=1) / (num_class-1) ) / num_samples

  return RPS

def RPSS(y_true, y_pred):

  num_samples = y_pred.shape[0]
  num_class = y_pred.shape[1]

  y_true2 = np.zeros(y_pred.shape); y_clim = np.zeros(y_pred.shape)
  for n in range(num_class):
    y_true2[:,n] = np.where((y_true==n), 1, 0)
    y_clim[:,n] = len(y_true[y_true==n])/len(y_true)

  RPS = np.sum( np.sum( (np.cumsum(y_pred, axis=1) - np.cumsum(y_true2, axis=1))**2, axis=1) / (num_class-1) ) / num_samples

  RPS_climo = np.sum( np.sum( (np.cumsum(y_clim, axis=1) - np.cumsum(y_true2, axis=1))**2, axis=1) / (num_class-1) ) / num_samples

  #print (RPS, RPS_climo)

  RPSS = 1 - RPS / RPS_climo

  return RPSS

def custom_CSI(true, pred, num_classes):

  tot = 0
  for i in [0, num_classes-1]:
    tot += np.count_nonzero(pred[true==i] == i)/(len(true[true==i])+np.count_nonzero(pred[true!=i] == i))
  tot /= 2.

  return tot

def convert_time(df_row):

  df_row2 = copy.deepcopy(df_row)

  for v, val in enumerate(df_row):
    val = int(val[:2])
    if val >= 20:
      val2 = val-20
    else:
      val2 = val+4
    df_row2.iloc[v] = val2

  return pd.to_numeric(df_row2)

def remove_overlapping_domains2(features, lead_time):

  features_thinned = copy.deepcopy(features)
  drop_indices = []

  for index, row in features.iterrows():
    print (index)
    for index2, row2 in features.iloc[index+1:].iterrows():
      if row['date']==row2['date'] and row['init_time']==row2['init_time']:
        if sqrt((row['mrms_storm_icen']-row2['mrms_storm_icen'])**2+(row['mrms_storm_jcen']-row2['mrms_storm_jcen'])**2) < 20 and sqrt((row['fcst_box_icen']-row2['fcst_box_icen'])**2+(row['fcst_box_jcen']-row2['fcst_box_jcen'])**2) < 20+5*int(lead_time/60):
          drop_indices += [index2]
      else:
        break

  features_thinned.drop(index=drop_indices, inplace=True) 
  features_thinned.reset_index(drop=True, inplace=True)

  return features_thinned

def remove_overlapping_domains(features, lead_time):

  features.reset_index(drop=True, inplace=True)
  features_thinned = copy.deepcopy(features)
  drop_indices = []

  for i in range(features.shape[0]):
    #print (i)
    for ii in range(i+1, features.shape[0]):
      #print ('  ', ii)
      if features['date'][i]==features['date'][ii] and features['init_time'][i]==features['init_time'][ii]:
        init_dist = sqrt((features['mrms_storm_icen'][i]-features['mrms_storm_icen'][ii])**2+(features['mrms_storm_jcen'][i]-features['mrms_storm_jcen'][ii])**2)
        fcst_dist = sqrt((features['fcst_box_icen'][i]-features['fcst_box_icen'][ii])**2+(features['fcst_box_jcen'][i]-features['fcst_box_jcen'][ii])**2)  
        if init_dist < 10 and fcst_dist < 0.5*(20+5*int(lead_time/60)):
          drop_indices += [ii]
          #print (init_dist, fcst_dist)
      else:
        break

  features_thinned.drop(index=drop_indices, inplace=True) 
  features_thinned.reset_index(drop=True, inplace=True)

  return features_thinned

def spread_error(prob_predictions, test_labels, pred_stds):

  bins = []; RMSEs = []; SDs = []

  hist, bin_edges = np.histogram(pred_stds, bins=10)

  for i in range(len(bin_edges)-1):
    condition = ( (pred_stds >= bin_edges[i]) & (pred_stds < bin_edges[i+1]) )
    pred_std_mean = sqrt(pred_stds[condition].mean())#pred_stds[condition].mean()
    RMSE = sqrt(np.power(prob_predictions[condition] - test_labels[condition], 2).mean())
    SDs.append(pred_std_mean)
    RMSEs.append(RMSE)
    bins.append([bin_edges[i], bin_edges[i+1]])

  return bins, RMSEs, SDs

def subsample5(labels, indices):

  indices = np.array(indices)
  unique_labels = sorted(list(set(labels)))
  label_counts = [np.count_nonzero(labels==label) for label in unique_labels]

  if max(label_counts) > 2*min(label_counts):
    label = list(x == max(label_counts) for x in label_counts).index(True)
    label_counts.remove(max(label_counts))
    reduced_count = int(np.array(label_counts).mean())
    new_indices = np.random.choice(indices[labels==label], size=reduced_count, replace=False)
    new_indices = np.concatenate((new_indices, indices[labels!=label]))    
    return new_indices
  else:
    return indices

def subsample5(labels, indices):

  indices = np.array(indices)
  unique_labels = sorted(list(set(labels)))
  label_counts = [np.count_nonzero(labels==label) for label in unique_labels]

  for n in range(len(label_counts)):

    if max(label_counts) > 1.5*min(label_counts):

      label = list(x == max(label_counts) for x in label_counts).index(True)
      label_counts.remove(max(label_counts))
      reduced_count = int(np.array(label_counts).mean())

      indices2 = np.random.choice(np.arange(len(indices[labels==label])), size=reduced_count, replace=False).astype(int)
      new_indices = indices[labels==label][indices2]
      new_labels = labels[labels==label][indices2]

      indices = np.concatenate((new_indices, indices[labels!=label]))
      labels = np.concatenate((new_labels, labels[labels!=label]))

      label_counts = [np.count_nonzero(labels==label) for label in unique_labels]

  return indices.astype(int)

def subsample(labels, indices, ratio=None):

  if ratio == None:
    ratio = [1,1,1]

  indices = np.array(indices)
  unique_labels = sorted(list(set(labels)))
  label_counts = [np.count_nonzero(labels==label) for label in unique_labels]
  print ('START:', label_counts)
  unbal_ratio = [label_count/sum(label_counts) for label_count in label_counts]
  #print (unbal_ratio)
  unbal_ratio = [rat / min(unbal_ratio) for rat in unbal_ratio]
  #print (unbal_ratio)
  reduce_factors = [unbal_ratio[i]/ratio[i] for i in range(len(ratio))] 
  reduce_factors = [reduce_factor/min(reduce_factors) for reduce_factor in reduce_factors] 
  #print (reduce_factors) 
  #reduce_factors = [max(reduce_factor, 1) for reduce_factor in reduce_factors]

  for n in range(len(label_counts)):

    label = unique_labels[n]
    #print (label_counts[n], int(1/reduce_factors[n]*label_counts[n]))
    indices2 = np.random.choice(np.arange(label_counts[n]), size=int(1/reduce_factors[n]*label_counts[n]), replace=False).astype(int)
    new_indices = indices[labels==label][indices2]
    new_labels = labels[labels==label][indices2]

    indices = np.concatenate((new_indices, indices[labels!=label]))
    labels = np.concatenate((new_labels, labels[labels!=label]))

    label_counts = [np.count_nonzero(labels==label) for label in unique_labels]

    #print (label_counts)

  print ('END:', label_counts)

  return indices.astype(int)


def subsample2(labels, indices):

  indices = np.array(indices)
  unique_labels = sorted(list(set(labels)))
  label_counts = [np.count_nonzero(labels==label) for label in unique_labels]

  reduced_count = min(label_counts)
  all_new_indices = np.empty(0)
  for label in unique_labels:
    new_indices = np.random.choice(indices[labels==label], size=reduced_count, replace=False)
    all_new_indices = np.concatenate((new_indices, all_new_indices))
  return all_new_indices.astype(int)

def subsample3(labels, indices):

  indices = np.array(indices)
  unique_labels = sorted(list(set(labels)))
  label_counts = [np.count_nonzero(labels==label) for label in unique_labels]

  reduced_count = int(gmean(label_counts))
  if reduced_count > 1.5*min(label_counts):
    reduced_count = int(gmean([reduced_count, min(label_counts)]))

  all_new_indices = np.empty(0)
  for label in unique_labels:
    if len(indices[labels==label]) > reduced_count:
      new_indices = np.random.choice(indices[labels==label], size=reduced_count, replace=False)
    else:
      new_indices = indices[labels==label]
    all_new_indices = np.concatenate((new_indices, all_new_indices))
  return all_new_indices.astype(int)

