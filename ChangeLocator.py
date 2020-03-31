import pandas as pd
import numpy as np
import processSeq
import sys

from sklearn.model_selection import train_test_split

from sklearn.metrics import average_precision_score,precision_score,recall_score,f1_score
from sklearn.metrics import roc_auc_score,accuracy_score,matthews_corrcoef

import xgboost as xgb

from optparse import OptionParser

from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
from inspect import signature
plt.switch_backend('Agg')
from matplotlib import rcParams

from sklearn.model_selection import StratifiedKFold

PATH1='.' # directory of dataset

def filter(t_train,t_test):
	bar = np.copy(t_train[-1])
	all_correct = bar.reshape((-1)).shape[0]
	keep = []
	for i in range(len(t_test)):
		
		if np.sum(t_test[i] == bar) == all_correct:
			continue
		else:
			keep.append(i)
	keep = np.array(keep)

	return keep

def filter_train(t_train,t_test):
	bar = np.copy(t_test[0])
	bar1 = np.copy(t_test[-1])
	all_correct = bar.reshape((-1)).shape[0]
	keep = []
	for i in range(len(t_train)):
		
		if (np.sum(t_train[i] == bar) == all_correct) or (np.sum(t_train[i] == bar1) == all_correct):
			continue
		else:
			keep.append(i)
	keep = np.array(keep)

	return keep
			
def build_vec():
	table = pd.read_table(PATH1+"prep_data.txt", sep="\t")
	final_result = np.zeros((len(table),23))
	for i in range(len(table)):
		seq1 = table['target'][i].upper()
		seq2 = table['sequence'][i].upper()

		if i % 10000 == 0:
			print ("%d of %d\r" %(i,len(table)),end = "")
			sys.stdout.flush()

		for j in range(23):
			if seq1[j] != seq2[j]:
				final_result[i,j] = 1

	np.save("feature_matrix.npy",final_result)

def one_hot_encoding(seq, seq_len):
	# seq_len = len(seq)
	vec1 = np.zeros((seq_len,4))
	cnt = 0

	for i in range(0,seq_len):
		if seq[i]=='A':
			vec1[i,0] = 1
		elif seq[i]=='G':
			vec1[i,1] = 1
		elif seq[i]=='C':
			vec1[i,2] = 1
		elif seq[i]=='T':
			vec1[i,3] = 1
		else:
			pass

	return np.int64(vec1)

def index_encoding(seq, seq_len, seq_dict):
	# seq_len = len(seq)
	vec1 = np.zeros(seq_len)

	for i in range(0,seq_len):
		vec1[i] = seq_dict[seq[i]]

	return np.int64(vec1)

def build_vec_matrix():
	table = pd.read_table(PATH1+"prep_data.txt", sep="\t")
	seq_len = 23
	final_result = np.zeros((len(table),seq_len*4))
	print("build_vec_matrix")
	for i in range(len(table)):
		seq1 = table['target'][i].upper()
		seq2 = table['sequence'][i].upper()

		vec1 = one_hot_encoding(seq1, seq_len)
		vec2 = one_hot_encoding(seq2, seq_len)

		temp1 = vec1-vec2
		temp2 = (vec1|vec2)
		# b = np.where(np.sum(temp2,1)==1)[0]
		b = np.where(np.sum(np.abs(temp1),1)==0)[0]
		temp1[b] = temp2[b]
		final_result[i] = np.ravel(temp1)

		if i % 10000 == 0:
			print ("%d of %d\r" %(i,len(table)),end = "")
			sys.stdout.flush()

	np.save("baseline_vec_matrix.npy",final_result)

def build_vec_matrix_mutation(n_base):
	table = pd.read_table(PATH1+"prep_data.txt", sep="\t")
	seq_len = 23
	n_base = 5
	final_result = np.zeros((len(table),seq_len*n_base*n_base))

	seq_dict = dict()
	seq_dict['A'] = 0
	seq_dict['G'] = 1
	seq_dict['C'] = 2
	seq_dict['T'] = 3
	seq_dict['N'] = 4
	
	print("build_vec_matrix_mutation")
	for i in range(len(table)):
		seq1 = table['target'][i].upper()
		seq2 = table['sequence'][i].upper()

		vec1 = index_encoding(seq1, seq_len, seq_dict)
		vec2 = index_encoding(seq2, seq_len, seq_dict)

		mtx1 = np.zeros((seq_len,n_base,n_base))
		for j in range(0,seq_len):
			mtx1[j,vec1[j],vec2[j]] = 1

		final_result[i] = np.int64(np.ravel(mtx1))

		if i % 10000 == 0:
			print ("%d of %d\r" %(i,len(table)),end = "")
			sys.stdout.flush()

	np.save("feature_matrix_mutation.base%d.npy"%(n_base),final_result)

def balance_data(X,y):
	pos_index = np.where(y == 1)[0]
	neg_index = np.where(y == 0)[0]
	np.random.shuffle(neg_index)
	neg_index = neg_index[:len(pos_index)]

	X = np.concatenate((X[pos_index], X[neg_index]), axis=0)
	y = np.concatenate((y[pos_index], y[neg_index]), axis=0)

	return X,y

def balance_data_1(X,t,y):
	pos_index = np.where(y == 1)[0]
	neg_index = np.where(y == 0)[0]
	np.random.shuffle(neg_index)
	neg_index = neg_index[:len(pos_index)]

	X = np.concatenate((X[pos_index], X[neg_index]), axis=0)
	y = np.concatenate((y[pos_index], y[neg_index]), axis=0)
	t = np.concatenate((t[pos_index], t[neg_index]), axis=0)

	return X,t,y

def build_sampleweight(y):

	vec = np.zeros((len(y)))
	for l in np.unique(y):
		vec[y == l] = np.sum(y != l) / len(y)
	return vec

def featureImp_convert(featureImp_vec, character_dict, type_id=1, n_base=5):

	# character_dict = np.asarray(['A','G','C','T'])
	if type_id==1:
		idx = np.int64(featureImp_vec[:,0])
		pos_idx = np.int64(idx/4)+1
		character_idx = idx%4
		b1 = np.where(idx==featureImp_vec.shape[0]-1)[0]
		pos_idx[b1[0]] = -1
		str_vec1 = character_dict[character_idx]
		# featureImp_vec1 = np.hstack((featureImp_vec,pos_idx.reshape((-1,1)),character_idx.reshape((-1,1)),str_vec1.reshape(-1,1)))
		fields = ['index','importance','position','char_idx','char']
		data1 = pd.DataFrame(columns=fields)
		data1[fields[0]], data1[fields[1]], data1[fields[2]] = np.int64(featureImp_vec[:,0]), featureImp_vec[:,1], pos_idx
		data1[fields[3]], data1[fields[4]] = character_idx, str_vec1
	else:
		idx = np.int64(featureImp_vec[:,0])
		num1 = n_base*n_base
		pos_idx = np.int64(idx/num1)+1
		temp1 = idx%num1
		id1 = np.int64(temp1/n_base)
		id2 = temp1%n_base
		
		b1 = np.where(idx==featureImp_vec.shape[0]-1)[0]
		pos_idx[b1[0]] = -1
		str_vec1 = character_dict[id1]
		str_vec2 = character_dict[id2]

		fields = ['index','importance','position','char_idx1','char_idx2','char1','char2']
		data1 = pd.DataFrame(columns=fields)
		data1[fields[0]], data1[fields[1]], data1[fields[2]] = np.int64(featureImp_vec[:,0]), featureImp_vec[:,1], pos_idx
		data1[fields[3]], data1[fields[4]], data1[fields[5]], data1[fields[6]] = id1, id2, str_vec1, str_vec2

	return data1

def plot_curve(run_id):

	filename1 = 'file%d.npy'%(run_id)
	y_test, y_proba = np.load(filename1)
	average_precision = average_precision_score(y_test, y_proba)
	precision, recall, _ = precision_recall_curve(y_test, y_proba)
	
	auc_score = roc_auc_score(y_test,y_proba)
	fpr, tpr, _ = roc_curve(y_test, y_proba)

	plt.rc('font', size=12)
	plt.subplot(1, 2, 2)
	plt.axis('scaled')
	plt.plot(recall, precision, color='navy', alpha=0.8, linewidth=2)

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.xticks(np.arange(0,1.01,0.2))
	
	plt.title('PR Curve: Imbalanced')
	plt.legend(labels=['AUPR={0:0.4f}'.format(average_precision)],loc='lower left')

	plt.rc('font', size=12)
	plt.subplot(1, 2, 1)
	plt.axis('scaled')
	plt.plot(fpr, tpr, color='navy', alpha=0.8, linewidth=2)

	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.xticks(np.arange(0,1.01,0.2))
	
	plt.title('ROC Curve: Imbalanced')
	plt.legend(labels=['AUROC={0:0.4f}'.format(auc_score)],loc='lower left')

	plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95, hspace=0.25,
                    wspace=0.3)

	plt.savefig('test%d.roc.pdf'%(run_id))
	plt.savefig('test%d.roc.png'%(run_id))

def score_function(y_test, y_pred, y_proba):

	auc = roc_auc_score(y_test,y_proba)
	aupr = average_precision_score(y_test,y_proba)
	precision = precision_score(y_test,y_pred)
	recall = recall_score(y_test,y_pred)
	accuracy = (np.sum(y_test == y_pred)*1.0 / len(y_test))

	# print(auc,aupr,precision,recall)
	
	return accuracy, auc, aupr, precision, recall

def KFold(x,y,target,k_fold,run_id,config,good_index):

	skf = StratifiedKFold(n_splits=k_fold, shuffle=False, random_state = 0)
	kf = skf.split(x,y)

	metrics = np.zeros((k_fold,5),dtype="float32")
	featureImp_vec = np.zeros((k_fold,x.shape[1]),dtype="float32")
	labels, predicted, proba = [], [], []
	thresh = 0.5
	cnt = 0
	print("%d %d" % (sum(y==1), sum(y==0)))

	max_depth, learning_rate, num_tree, balanced_mode, feature_mode, flag = config[0], config[1], config[2], config[3], config[4], config[5]

	for train_index, test_index in kf:
		
		print(train_index.shape,test_index.shape)
		if flag>=2:
			train_index = np.intersect1d(train_index,good_index)

		X_train, y_train, t_train = x[train_index], y[train_index], target[train_index]
		X_test, y_test, t_test = x[test_index], y[test_index], target[test_index]
	
		print(X_train.shape,y_train.shape,y_test.shape)
		
		if balanced_mode==1:
			X_train,y_train = balance_data(X_train,y_train)
			X_test,y_test = balance_data(X_test,y_test)
		elif balanced_mode==2:
			X_train,y_train = balance_data(X_train,y_train)
			# X_test,y_test = balance_data(X_test,y_test)
		else:
			pass

		forest = xgb.XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=num_tree, nthread=30)

		forest.fit(X_train,y_train,sample_weight=build_sampleweight(y_train))
		# print(forest.feature_importances_)
		t_feature_importance = forest.feature_importances_

		featureImp_vec[cnt] = t_feature_importance
		y_pred = forest.predict(X_test)
		y_proba = forest.predict_proba(X_test)[:,1]

		accuracy, auc, aupr, precision, recall = score_function(y_test, y_pred, y_proba)
		metrics[cnt] = [accuracy, auc, aupr, precision, recall]
		print(cnt,accuracy,auc,aupr,precision,recall)
		cnt += 1

		labels.extend(y_test)
		predicted.extend(y_pred)
		proba.extend(y_proba)
	
	labels, predicted, proba = np.asarray(labels), np.asarray(predicted), np.asarray(proba)

	print(labels.shape,predicted.shape,proba.shape)
	accuracy, auc, aupr, precision, recall = score_function(labels, predicted, proba)
	print(accuracy, auc, aupr, precision, recall)
	filename1a = 'file_kfold_%d'%(run_id)
	np.save(filename1a,(labels,proba))
	
	character_dict = np.asarray(['A','G','C','T','N'])
	featureImp_mean = np.mean(featureImp_vec,axis=0)
	vec1 = np.vstack((np.asarray(range(0,len(featureImp_mean))), featureImp_mean)).T
	temp1 = (-vec1[:,1]).argsort()
	vec2 = vec1[temp1]
	n_base = 4

	if feature_mode==1:
		filename1 = 'feature_importance_imbalanced_mutation.m%d.lr%.2f.n%d.base%d.%d'%(max_depth, learning_rate, num_tree, n_base, run_id)
		type_id = 2
	else:
		filename1 = 'feature_importance_imbalanced.m%d.lr%.2f.n%d.%d'%(max_depth, learning_rate, num_tree, run_id)
		type_id = 1

	featureImp_vec1 = np.vstack((featureImp_vec,featureImp_mean)).T
	np.savetxt(filename1+'.txt',featureImp_vec1,fmt='%.4f')

	featureImp_vec1 = featureImp_convert(vec2, character_dict, type_id, n_base)
	filename_2 = '%s.table.txt'%(filename1)
	featureImp_vec1.to_csv(filename_2,index=False,sep='\t')

	return True

def parse_args():
	parser = OptionParser(usage="Genome editting", add_help_option=False)
	parser.add_option("-r","--run_id", default="0", help="experiment id")
	parser.add_option("-l","--lr", default="0.1", help="learning rate")
	parser.add_option("-m","--max_depth",default="10",help="max depth of tree")
	parser.add_option("-n","--num_tree",default="1000",help="number of trees")
	parser.add_option("--read_thresh",default="100",help="read number threshold")
	parser.add_option("-b","--balanced_mode",default="0",help="balanced mode: 0: imbalanced data; 1: balanced data; 2: balanced training, imbalanced test")
	parser.add_option("-f","--feature_mode",default="1",help="feature mode: 0: 1D; 1: 2D; 2: position")
	parser.add_option("-d","--distance_mode",default="1",help="use distance feature or not: 0: not use; 1: use")

	(opts, args) = parser.parse_args()
	return opts

def run(run_id, read_thresh, max_depth, num_tree, learning_rate, balanced_mode, distance_mode, feature_mode=1, n_base=4):
	
	run_id = int(run_id)
	max_depth = int(max_depth)
	num_tree = int(num_tree)
	learning_rate = float(learning_rate)
	balanced_mode = int(balanced_mode)
	feature_mode = int(feature_mode)
	read_thresh = int(read_thresh)
	distance_mode = int(distance_mode)
	
	table = pd.read_table(PATH1+"prep_data.txt",sep = "\t")

	extra_feature = np.asarray(table['distance']).reshape((-1,1))

	if feature_mode==1:
		build_vec_matrix_mutation(n_base)
		if n_base==5:
			X = np.load("feature_matrix_mutation.base5.npy")
		else:
			X = np.load("feature_matrix_mutation.base4.npy")
	else:
		build_vec()
		X = np.load("feature_matrix.npy")

	if distance_mode==1:
		X = np.concatenate((X,extra_feature),axis = -1)

	y = np.asarray(table['label'])
	print ((np.sum(y == 1)/ np.sum(y == 0)))
	print (X[:3])

	read = np.asarray(table['reads'])
	print (y)

	good_index = np.where(np.isnan(read)|(read >= read_thresh))[0]

	k_fold = 10
	max_depth = 10
	learning_rate = 0.1
	n_estimators = 1000
	flag = 2
	config = [max_depth,learning_rate,n_estimators,balanced_mode,feature_mode,flag]
	KFold(X,y,target,k_fold,run_id,config,good_index)

if __name__ == '__main__':

	opts = parse_args()
	run(opts.run_id, opts.read_thresh, opts.max_depth, opts.num_tree, opts.lr, opts.balanced_mode, opts.feature_mode, opts.distance_mode)

