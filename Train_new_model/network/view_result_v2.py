
import numpy as np
import matplotlib.pyplot as plt

# Select data
#epoch = 154
res_dir = 'plots'

## Plot and save diagnostics

# Loss value
filename = 'loss_tr_te.npz'
data = np.load(filename)
tr_loss = data['arr_0']
te_loss = data['arr_1']
plt.plot(tr_loss)
plt.plot(te_loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('cross entropy loss')
plt.legend(('train_loss','test_loss'),loc='best')
plt.savefig('%s/loss.png'%(res_dir), bbox_inches='tight')
plt.clf()

# Loss value zoomed
filename = 'loss_tr_te.npz'
data = np.load(filename)
tr_loss = data['arr_0']
te_loss = data['arr_1']
plt.plot(tr_loss)
plt.plot(te_loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim((0,0.8))
plt.title('cross entropy loss zoomed')
plt.legend(('train_loss','test_loss'),loc='best')
plt.savefig('%s/loss_zoomed.png'%(res_dir), bbox_inches='tight')
plt.clf()


# Metrics from | tp tn fp fn
filename = 'metrics_train_test_tp_tn_fp_fn.npz'
data = np.load(filename)
metrics_train = data['arr_0'] # 0-tp 1-tn 2-fp 3-fn
metrics_test = data['arr_1']
print(metrics_train[0].shape)
print(metrics_test.shape)

# precision = tp/(tp+fp) | 0-tp 1-tn 2-fp 3-fn
tr_pr = np.divide(metrics_train[0],(metrics_train[0]+metrics_train[2]))
te_pr = np.divide(metrics_test[0],(metrics_test[0]+metrics_test[2]))
plt.plot(tr_pr)
plt.plot(te_pr)
plt.xlabel('epochs')
plt.ylabel('precision')
plt.title('Metrics - precision')
plt.legend(('train','test'),loc='best')
plt.savefig('%s/metrics_precision.png'%(res_dir), bbox_inches='tight')
plt.clf()

# recall = tp/(tp+fn) | 0-tp 1-tn 2-fp 3-fn
tr_re = np.divide(metrics_train[0],(metrics_train[0]+metrics_train[3]))
te_re = np.divide(metrics_test[0],(metrics_test[0]+metrics_test[3]))
plt.plot(tr_re)
plt.plot(te_re)
plt.xlabel('epochs')
plt.ylabel('recall')
plt.title('Metrics - recall')
plt.legend(('train','test'),loc='best')
plt.savefig('%s/metrics_recall.png'%(res_dir), bbox_inches='tight')
plt.clf()

# training precision-recall plot
plt.plot(tr_pr)
plt.plot(tr_re)
plt.xlabel('epochs')
plt.ylabel('tr precision-recall')
plt.title('Training - precision/recall')
plt.legend(('precision','recall'),loc='best')
plt.savefig('%s/metrics_tr_precision_recall.png'%(res_dir), bbox_inches='tight')
plt.clf()

# testing precision-recall plot
plt.plot(te_pr)
plt.plot(te_re)
plt.xlabel('epochs')
plt.ylabel('te precision-recall')
plt.title('Testing - precision/recall')
plt.legend(('precision','recall'),loc='best')
plt.savefig('%s/metrics_te_precision_recall.png'%(res_dir), bbox_inches='tight')
plt.clf()

# specificity = tn/(tn+fp) | 0-tp 1-tn 2-fp 3-fn
tr_pr = np.divide(metrics_train[1],(metrics_train[1]+metrics_train[2]))
te_pr = np.divide(metrics_test[1],(metrics_test[1]+metrics_test[2]))
plt.plot(tr_pr)
plt.plot(te_pr)
plt.xlabel('epochs')
plt.ylabel('specificity')
plt.title('Metrics - specificity')
plt.legend(('train','test'),loc='best')
plt.savefig('%s/metrics_specificity.png'%(res_dir), bbox_inches='tight')
plt.clf()

# sensitivity = tn/(tn+fn) | 0-tp 1-tn 2-fp 3-fn
tr_pr = np.divide(metrics_train[1],(metrics_train[1]+metrics_train[3]))
te_pr = np.divide(metrics_test[1],(metrics_test[1]+metrics_test[3]))
plt.plot(tr_pr)
plt.plot(te_pr)
plt.xlabel('epochs')
plt.ylabel('sensitivity')
plt.title('Metrics - sensitivity')
plt.legend(('train','test'),loc='best')
plt.savefig('%s/metrics_sensitivity.png'%(res_dir), bbox_inches='tight')
plt.clf()

# accuracy = (tn+tp)/(tn+tp+fn+fp) | 0-tp 1-tn 2-fp 3-fn
tr_pr = np.divide(metrics_train[0]+metrics_train[1],(metrics_train[0]+metrics_train[1]+metrics_train[2]+metrics_train[3]))
te_pr = np.divide(metrics_test[0]+metrics_test[1],(metrics_test[0]+metrics_test[1]+metrics_test[2]+metrics_test[3]))
plt.plot(tr_pr)
plt.plot(te_pr)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Metrics - accuracy')
plt.legend(('train','test'),loc='best')
plt.savefig('%s/metrics_accuracy.png'%(res_dir), bbox_inches='tight')
plt.clf()

# jaccard index = tp/(tp+fp+fn) | 0-tp 1-tn 2-fp 3-fn
tr_pr = np.divide(metrics_train[0],(metrics_train[0]+metrics_train[2]+metrics_train[3]))
te_pr = np.divide(metrics_test[0],(metrics_test[0]+metrics_test[2]+metrics_test[3]))
plt.plot(tr_pr)
plt.plot(te_pr)
plt.xlabel('epochs')
plt.ylabel('jaccardIdx')
plt.title('Metrics - jaccardIdx')
plt.legend(('train','test'),loc='best')
plt.savefig('%s/metrics_jaccardIdx.png'%(res_dir), bbox_inches='tight')
plt.clf()

# dice coefficient = 2tp/(2tp+fp+fn) | 0-tp 1-tn 2-fp 3-fn
tr_pr = np.divide(2*metrics_train[0],(2*metrics_train[0]+metrics_train[2]+metrics_train[3]))
te_pr = np.divide(metrics_test[0],(2*metrics_test[0]+metrics_test[2]+metrics_test[3]))
plt.plot(tr_pr)
plt.plot(te_pr)
plt.xlabel('epochs')
plt.ylabel('diceCoeff')
plt.title('Metrics - diceCoeff')
plt.legend(('train','test'),loc='best')
plt.savefig('%s/metrics_diceCoeff.png'%(res_dir), bbox_inches='tight')
plt.clf()

# Load data
# filename = 'epoch%d/test.npz'%(epoch)
# data = np.load(filename)
# print(data.files)
# view1 = data['arr_0'] 
# view2 = data['arr_1']
# view3 = data['arr_2']
# print(data['arr_0'].shape)
# print(data['arr_1'].shape)
# print(data['arr_2'].shape)
# # View data
# mv = mvc.MultisliceViewer3()
# mv.run_3D_viewer(view1,view2,view3)

# Load train & test accuracy
# filename = 'epoch%d/test_train_accuracy_p.npz'%(epoch)

# data = np.load(filename)
# test_acc = data['arr_0'] 
# train_acc = data['arr_1']


# plt.plot(test_acc)
# plt.xlabel('half epochs')
# plt.ylabel('precision')
# plt.title('Precision')
# plt.legend(('test'))
#plt.show()


# # Load train & test accuracy
# filename = 'epoch%d/loss_tr_te.npz'%(epoch)

# data = np.load(filename)
# tr_loss = data['arr_0'] 
# te_loss = data['arr_1']


# plt.plot(tr_loss)
# plt.plot(te_loss)
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.title('cross entropy loss fcn')
# plt.legend(('train','test'))
# plt.show()
