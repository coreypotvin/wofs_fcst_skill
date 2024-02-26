# Contains most of the classes and definitions used by package

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

def dataset_split(features, all_features, labels, train_test_split_param, class_balancing=False, num_folds=None, rng=None):

 all_dates = all_features['date'].values
 num_examples = features.shape[0]

 # Un-nested CV options

 if train_test_split_param in ['2017', '2018', '2019', '2020', '2021']:

  index = np.where((all_dates.contains(train_test_split_param)))[0].tolist()
  index2 = [x for x in range(num_examples) if x not in index]
  index = np.asarray(index)
  index2 = np.asarray(index2)
  test_features = features.iloc[index]
  train_features = features.iloc[index2]
  test_labels = labels[index]
  train_labels = labels[index2]
  all_features_train = all_features.iloc[index2]

  return train_features, train_labels, test_features, test_labels, all_features_train

 elif type(train_test_split_param)==str and 'dates' in train_test_split_param and train_test_split_param!='dates':

  dates = sorted(list(set(all_dates)))
  random.seed(42)
  random.shuffle(dates)
  tot=0
  date_groups = []
  date_group = []
  for date in dates:
    date_group.append(date)
    tot += len(all_dates[all_dates==date])
    if tot >= (len(date_groups)+1)*len(all_dates)/num_folds:
      date_groups.append(date_group)
      date_group=[]

  ind = int(train_test_split_param[-1])
  date_group = date_groups[ind]
  index = np.where((all_dates.isin(date_group)))[0].tolist()
  index2 = [x for x in range(num_examples) if x not in index]
  index = np.asarray(index)
  index2 = np.asarray(index2)
  test_features = features.iloc[index]
  train_features = features.iloc[index2]
  test_labels = labels[index]
  train_labels = labels[index2]
  all_features_train = all_features.iloc[index2]

  return train_features, train_labels, test_features, test_labels, all_features_train

 elif type(train_test_split_param) == 'float':

  train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = train_test_split_param, random_state = rng)
  all_features_train, dummy, dummy, dummy = train_test_split(all_features, labels, test_size = train_test_split_param, random_state = rng)

  return train_features, train_labels, test_features, test_labels, all_features_train

 # Nested CV

 elif train_test_split_param == 'dates':

  dates = sorted(list(set(all_dates)))
  random.seed(42)
  random.shuffle(dates)
  #print (dates)
  tot=0
  date_groups = []
  date_group = []
  for date in dates:
    date_group.append(date)
    tot += len(all_dates[all_dates==date])
    if tot >= (len(date_groups)+1)*len(all_dates)/num_folds:
      date_groups.append(date_group)
      date_group=[]

  ## Measure effect of potential auto-correlations between consecutive dates split between training & testing folds
  '''
  for d, date_group in enumerate(date_groups):

    test_dates = date_group
    train_dates = [date for date in dates if date not in test_dates]

    new_test_dates = remove_consecutive_dates(train_dates, test_dates)

    print (len(train_dates), len(test_dates), len(new_test_dates))
    print (train_dates)
    print (test_dates)
    print (new_test_dates)

    date_groups[d] = new_test_dates
  '''
  allfolds_test_features = []; allfolds_test_orig_features = []; allfolds_train_features = []; allfolds_test_labels = []; allfolds_train_labels = []; allfolds_all_features_train = []

  for ind in range(num_folds):
    date_group = date_groups[ind]
    index = np.where(np.isin(all_dates, date_group))[0].tolist()
    index2 = [x for x in range(num_examples) if x not in index]
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
      
  return allfolds_train_features, allfolds_train_labels, allfolds_test_features, allfolds_test_labels, allfolds_all_features_train, allfolds_test_orig_features, date_groups

 else:

  print ('Invalid train_test_split_param!')

def hyper_opt_learn_prep(learn_algo, rng):

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

 return learner, param_distributions

def hyper_opt_unnestedCV_prep(train_test_split_param, tuning_CV_split_method, num_classes=None, all_dates=None, date_groups=None):

 if tuning_CV_split_method == 'year':

   print ('Splitting hyperparameter optimization CV folds by year')

   CV = []

   years = ['2017', '2018', '2019', '2020', '2021']
   if train_test_split_param in years:
     years.remove(train_test_split_param)

   for y, year in enumerate(years):

     test_indices = np.where((all_dates.str.contains(year)))[0].tolist()
     train_indices = np.where((~(all_dates.str.contains(year))))[0].tolist()#[x for x in range(len(features['mean_init_score'])) if x not in test_indices]

     print ('CV fold #%d (validation year: %s): %d/%d training/validation samples' % (y, year, len(train_indices), len(test_indices)))

     CV.append((train_indices, test_indices))

 elif tuning_CV_split_method == 'dates':

   print ('Splitting hyperparameter optimization CV folds by date')

   CV = []

   ind = int(train_test_split_param[-1])
   del date_groups[ind]

   for d, date_group in enumerate(date_groups):
     dates = pd.array(all_dates, dtype='string')
     test_indices = np.where(np.isin(dates, date_group))[0].tolist()
     train_indices = np.where(~np.isin(dates, date_group))[0].tolist()

     print ('\nCV fold #%d: %d/%d training/validation samples' % (d, len(train_indices), len(test_indices)))
     print ('      Class balance of training samples:  ', [round(len(train_labels[train_indices][train_labels[train_indices]==n])/len(train_labels[train_indices]),2) for n in range(num_classes)])
     print ('      Class balance of validation samples:', [round(len(train_labels[test_indices][train_labels[test_indices]==n])/len(train_labels[test_indices]),2) for n in range(num_classes)])

     CV.append((train_indices, test_indices))

 elif tuning_CV_split_method == 'random':

   CV = 5

 return CV

def hyper_opt_nestedCV_prep(allfolds_all_features_train, allfolds_train_labels, train_test_split_param, tuning_CV_split_method, num_classes, date_groups, class_balancing=False, train_ratio=None, test_ratio=None):

 allfolds_CV = []

 for fold in range(num_folds):

   CV = []
   date_groups2 = date_groups.copy()
   del date_groups2[fold]
   train_labels = allfolds_train_labels[fold]          
   all_features_train = allfolds_all_features_train[fold]

   print ('     ~~~~~~~~~~~~~~~~~~~~~~~\n')

   for d, date_group in enumerate(date_groups2):

     dates = pd.array(all_features_train['date'], dtype='string')
     test_indices = np.where(np.isin(dates, date_group))[0].tolist()
     train_indices = np.where(~np.isin(dates, date_group))[0].tolist()

     if class_balancing:
       print ('CV fold #%d: %d/%d training/validation samples' % (d, len(train_indices), len(test_indices)))
       train_indices = subsample(train_labels[train_indices], train_indices, ratio=train_ratio)
       test_indices = subsample(train_labels[test_indices], test_indices, ratio=test_ratio)

     print ('\nCV fold #%d: %d/%d training/validation samples' % (d, len(train_indices), len(test_indices)))
     try:
       print ('      Class balance of training samples:  ', [round(len(train_labels.iloc[train_indices][train_labels.iloc[train_indices]==n])/len(train_labels.iloc[train_indices]),2) for n in range(num_classes)])
       print ('      Class balance of validation samples:', [round(len(train_labels.iloc[test_indices][train_labels.iloc[test_indices]==n])/len(train_labels.iloc[test_indices]),2) for n in range(num_classes)])
     except:
       print ('      Class balance of training samples:  ', [round(len(train_labels[train_indices][train_labels[train_indices]==n])/len(train_labels[train_indices]),2) for n in range(num_classes)])
       print ('      Class balance of validation samples:', [round(len(train_labels[test_indices][train_labels[test_indices]==n])/len(train_labels[test_indices]),2) for n in range(num_classes)])

     CV.append((train_indices, test_indices))

   allfolds_CV.append(CV)

 return allfolds_CV

def hyper_opt_perform_unnestedCV(train_features, train_labels, best_params_fname, learner, param_distributions, num_folds, n_iter, rng):

 clf = RandomizedSearchCV(estimator = learner, param_distributions = param_distributions, n_iter = n_iter, cv=num_folds, verbose=1, random_state=rng, n_jobs = 10)
 best_model = clf.fit(train_features, train_labels)
 best_params = best_model.best_params_
 print (best_params)

 with open(best_params_fname, 'wb') as handle:
   pickle.dump(best_params, handle)

 return best_params

def hyper_opt_perform_nestedCV(allfolds_train_features, allfolds_train_labels, best_params_fname, learner, param_distributions, allfolds_CV, n_iter, rng):


 allfolds_best_params = []

 for fold in range(len(allfolds_CV)):

   CV = allfolds_CV[fold]
   clf = BayesSearchCV(estimator = learner, scoring = make_scorer(RPS, greater_is_better=False, needs_proba=True), search_spaces = param_distributions, n_iter = n_iter, cv=CV, verbose=1, random_state=rng, n_points = 10, n_jobs = 40)
   best_model = clf.fit(allfolds_train_features[fold], allfolds_train_labels[fold])

   best_params = best_model.best_params_
   print (best_params)
   allfolds_best_params.append(best_params)

   best_params_fname2 = best_params_fname.split('.pkl')[0]+'_%d.pkl' % fold
   with open(best_params_fname2, 'wb') as handle:
     pickle.dump(best_params, handle)

 return allfolds_best_params

def train_model_nestedCV(allfolds_best_params, allfolds_train_features, allfolds_train_labels, learn_algo, best_model_fname, rng):

  allfolds_model = []

  for fold in range(len(allfolds_best_params)):

    best_params = allfolds_best_params[fold]

    train_features = allfolds_train_features[fold]
    train_labels = allfolds_train_labels[fold]

    if learn_algo=='ORF':

      model = OrdinalClassifier(estimator = RandomForestClassifier(n_estimators = best_params['estimator__n_estimators'], max_features=best_params['estimator__max_features'], max_depth=best_params['estimator__max_depth'], min_samples_leaf = best_params['estimator__min_samples_leaf'], min_samples_split=best_params['estimator__min_samples_split'], random_state=rng), n_jobs=1)

    elif learn_algo in ['RF', 'stacked_RF']:

      model = RandomForestClassifier(n_estimators = best_params['n_estimators'], max_features=best_params['max_features'], max_depth=best_params['max_depth'], min_samples_leaf = best_params['min_samples_leaf'], min_samples_split=best_params['min_samples_split'], random_state=rng)

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

  return allfolds_model

def train_model_unnestedCV(best_params, train_features, train_labels, learn_algo, best_model_fname, rng):

    if learn_algo=='RF':

      model = RandomForestClassifier(n_estimators = best_params['n_estimators'], max_features=best_params['max_features'], max_depth=best_params['max_depth'], min_samples_leaf = best_params['min_samples_leaf'], min_samples_split=best_params['min_samples_split'], random_state=rng)

    elif learn_algo=='LR':

      model = LogisticRegression(penalty = best_params['penalty'], l1_ratio = best_params['l1_ratio'], C = best_params['C'], solver='saga', max_iter=1000, random_state=rng)

    elif learn_algo=='ORF':

      model = OrdinalClassifier(RandomForestClassifier(n_estimators = best_params['n_estimators'], max_features=best_params['max_features'], max_depth=best_params['max_depth'], min_samples_leaf = best_params['min_samples_leaf'], min_samples_split=best_params['min_samples_split'], random_state=rng), n_jobs=1)

    elif learn_algo=='OLR':

      model = OrdinalClassifier(LogisticRegression(penalty = best_params['penalty'], l1_ratio = best_params['l1_ratio'], C = best_params['C'], solver='saga', max_iter=1000, random_state=rng), n_jobs=1)

    elif learn_algo=='NN':

      model = MLPClassifier(hidden_layer_sizes = best_params['hidden_layer_sizes'], activation = best_params['activation'], learning_rate_init = best_params['learning_rate_init'], alpha = best_params['alpha'], max_iter=1000, random_state=rng) 

    elif learn_algo=='GB':

      model = XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', n_estimators = best_params['n_estimators'], max_depth=best_params['max_depth'], grow_policy=best_params['grow_policy'], learning_rate=best_params['learning_rate'], subsample=best_params['subsample'], reg_lambda=best_params['reg_lambda'], reg_alpha=best_params['reg_alpha'], min_child_weight = best_params['min_child_weight'], gamma = best_params['gamma'], colsample_bytree = best_params['colsample_bytree'], nthread=10, random_state=rng)

    elif learn_algo=='GBhist':

      model = HistGradientBoostingClassifier(max_iter = best_params['max_iter'], max_depth = best_params['max_depth'], learning_rate = best_params['learning_rate'], min_samples_leaf = best_params['min_samples_leaf'], l2_regularization =  best_params['l2_regularization'], max_bins =  best_params['max_bins'], early_stopping=True, random_state=rng)

    model.fit(train_features, train_labels)
    dump(model, best_model_fname) 

def verify(allfolds_model, allfolds_train_labels, allfolds_test_labels, allfolds_train_features, allfolds_test_features, allfolds_test_orig_features, combine_classes, num_classes, stratify_verif=False):

  baseline_preds = np.empty(0); prob_predictions = []; test_labels = np.empty(0); test_features = np.empty(0)
  train_predictions = np.empty(0); predictions = np.empty(0); all_train_errors = np.empty(0)
  allfolds_errors = []; class_acc = []; allfolds_auc = []

  for fold in range(len(allfolds_model)):

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
      prob_predictions = copy.deepcopy(new_prob_predictions)
      train_prob_predictions = copy.deepcopy(new_train_prob_predictions)
      test_labels = copy.deepcopy(new_test_labels)
    else:
      test_features = pd.concat((test_features, new_test_features))
      if stratify_verif:
        test_orig_features = pd.concat((test_orig_features, new_test_orig_features))
      predictions = np.concatenate((predictions, new_predictions))
      prob_predictions = np.concatenate((prob_predictions, new_prob_predictions))
      train_prob_predictions = np.concatenate((train_prob_predictions, new_train_prob_predictions))
      test_labels = np.concatenate((test_labels, new_test_labels))

    #baseline_preds = np.concatenate((baseline_preds, np.repeat((num_classes+1)/2-1, len(allfolds_test_labels[fold]))))
    baseline_preds = np.concatenate((baseline_preds, np.random.choice([0,1,2], size=len(allfolds_test_labels[fold]), p=[0.2, 0.6, 0.2])))

    print ('\nMean training prediction error (fold %s): %.2f' % (fold, round(np.mean(train_errors), 2)))
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
    num_classes2 = 3
  else:
    num_classes2 = num_classes

  allfolds_errors = np.asarray(allfolds_errors)
  class_acc = np.asarray(class_acc)
  allfolds_auc = np.asarray(allfolds_auc)

  print ('Testing error across folds:            %.3f +/- %.3f' % (np.mean(allfolds_errors), np.std(allfolds_errors)))
  print ('Classification Accuracy across folds:  %.3f +/- %.3f' % (np.mean(class_acc), np.std(class_acc)))
  print ('Macro-Average AUC across folds:        %.3f +/- %.3f' % (np.mean(allfolds_auc), np.std(allfolds_auc)))

  #print ('ACC per class: %.2f / %.2f / %.2f' % (np.mean(per_class_acc, axis=1)[0], np.mean(per_class_acc, axis=1)[1], np.mean(per_class_acc, axis=1)[2]))
  #print ('AUC per class: %.2f / %.2f / %.2f' % (np.mean(per_class_auc, axis=1)[0], np.mean(per_class_auc, axis=1)[1], np.mean(per_class_auc, axis=1)[2]))

  cm = confusion_matrix(test_labels, predictions)
  per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
  print ('ACC per class: %.2f / %.2f / %.2f' % (per_class_accuracy[0], per_class_accuracy[1], per_class_accuracy[2]))

  y_true_binarized = label_binarize(test_labels, classes=[0, 1, 2])
  auc_class_1 = roc_auc_score(y_true_binarized[:, 0], prob_predictions[:, 0])
  auc_class_2 = roc_auc_score(y_true_binarized[:, 1], prob_predictions[:, 1])
  auc_class_3 = roc_auc_score(y_true_binarized[:, 2], prob_predictions[:, 2])
  print ('AUC per class: %.2f / %.2f / %.2f' % (auc_class_1, auc_class_2, auc_class_3))

  if stratify_verif:
    test_orig_features['prob_predictions'] = prob_predictions.tolist()
    test_orig_features['test_labels'] = test_labels.tolist()
    test_orig_features['predictions'] = predictions.tolist()
    test_orig_features.reset_index(drop=True, inplace=True)
    #test_orig_features.to_feather(predictions_fname) 

  return baseline_preds, predictions, train_predictions, prob_predictions, test_labels, num_classes2

def remove_consecutive_dates(list_a, list_b):
    """Removes dates from list B that are consecutive with dates in list A."""
    # Ensure both lists are sorted
    list_a = [datetime.strptime(date_str, "%Y%m%d") for date_str in list_a]
    list_b = [datetime.strptime(date_str, "%Y%m%d") for date_str in list_b]
  
    sorted_a = sorted(list_a)
    sorted_b = sorted(list_b)

    # Set for faster removal and search operations
    to_remove = set()

    # Check for each date in list A
    for date in sorted_a:
        prev_date = date - timedelta(days=1)
        next_date = date + timedelta(days=1)
        
        # If previous or next date is in list B, mark it for removal
        if prev_date in sorted_b:
            to_remove.add(prev_date)
        if next_date in sorted_b:
            to_remove.add(next_date)

    # Remove marked dates from list B
    filtered_b = [date.strftime("%Y%m%d") for date in sorted_b if date not in to_remove]
    
    return filtered_b
