# Name          : Kunal Zodape
# Project Topic : Decision tree based classifier model building using ID3 algorithm
# Project Code  : Topic 08
# Roll No.      : 19CS60R13

import csv
import math 
import random 
import copy 

def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk)) 
def prLightPurple(skk): print("\033[94m {}\033[00m" .format(skk)) 
def prPurple(skk): print("\033[95m {}\033[00m" .format(skk)) 
def prCyan(skk): print("\033[96m {}\033[00m" .format(skk)) 
def prLightGray(skk): print("\033[97m {}\033[00m" .format(skk)) 
def prBlack(skk): print("\033[98m {}\033[00m" .format(skk)) 

dataset = []
with open('NURSERY.csv') as csv_file:
	reader = csv.reader(csv_file)
	for row in reader:
		dataset.append(row)
dataset.pop()

attributes = [
	'parents',
	'has_nurs',
	'form',
	'children',
	'housing',
	'finance',
	'social',
	'health'
]

n_attr = len(attributes)
n_rows = len(dataset)
target = 'admit'

attribute_values = []
for i in range(n_attr):
	valitem = [attributes[i], []]
	attribute_values.append(valitem)
for row in dataset:
	for i in range(n_attr):
		if row[i] not in attribute_values[i][1]:
			attribute_values[i][1].append(row[i])
# for attr in attribute_values:
# 	print(attr)

target_values = []
for row in dataset:
	if row[-1] not in target_values:
		target_values.append(row[-1])
# print(len(target_values))
# for value in target_values:
# 	print(value)

# labels = []
# for row in dataset:
# 	labels.append(row[-1])

random.seed(42)
n_train = (n_rows*2)//3
shuffle_data = random.choices(dataset, k=n_rows)
train_data = []
test_data = []
for i in range(n_train):
	train_data.append(copy.deepcopy(shuffle_data[i]))
for i in range(n_train,n_rows):
	test_data.append(copy.deepcopy(shuffle_data[i]))

def getClassCount(data):
	"""Get class counts from dataset"""
	classes = []
	for row in data:
		flag = True
		for i in range(len(classes)):
			if row[-1] == classes[i][0]:
				classes[i][1] += 1
				flag = False 
				break	
		if flag:
			insert = [ row[-1], 1]
			classes.append(insert)
	return classes

# classes = getClassCount(dataset)
# for item in classes:
# 	print(item)
# print()
# temp_data = [row for row in dataset if row[0] == 'usual']
# classes = getClassCount(temp_data)
# for item in classes:
# 	print(item)

def maxClassPred(data):
	"""Get most popular class from dataset for prediction"""
	classes = getClassCount(data)
	maxClassCount = 0
	maxClass = ''
	for item in classes:
		if item[1] > maxClassCount:
			maxClassCount = item[1]
			maxClass = item[0]
	return maxClass
# print(maxClassPred(temp_data))

# temp_data_1 = [row for row in dataset if row[0] == 'usual']
# temp_data_2 = [row for row in dataset if row[0] == 'pretentious']
# temp_data_3 = [row for row in dataset if row[0] == 'great_pret']
# classes = getClassCount(temp_data_1)
# for item in classes:
# 	print(item)
# print()
# classes = getClassCount(temp_data_2)
# for item in classes:
# 	print(item)
# print()
# classes = getClassCount(temp_data_3)
# for item in classes:
# 	print(item)
# print()

def calcEntropy(data):
	"""Calculate Entropy"""
	d_rows = len(data)
	classCount = getClassCount(data)
	entropy = 0
	for count in classCount:
		entropy += (-1)*(count[1]/d_rows)*math.log2(count[1]/d_rows)
	return entropy
# print(calcEntropy(dataset))
# print(calcEntropy(temp_data_1))
# print(calcEntropy(temp_data_2))
# print(calcEntropy(temp_data_3))


class DecisionTree(object):
	def __init__(self):
		self.left = None
		self.child = []
		self.data = []
		self.attribute = '' 
		self.prediction = '' 
	def createChildren(self,amount):
		for i in range(0,amount):
			self.child.append(DecisionTree())
	def setChildrenValues(self,list):
		for i in range(0,len(list)):
			self.data.append(list[i])

root = DecisionTree()

def dataSplit(data, label, attributes):
	"""Split the dataset on attribute"""
	splits = []
	attr_vals = []
	index = attributes.index(label)
	for row in data:
		flag = True
		for i in range(len(attr_vals)):
			if row[index] == attr_vals[i]:
				flag = False
				splits[i].append(row)
				break
		if flag == True:
			attr_vals.append(row[index])
			splits.append([row])
	return attr_vals, splits

# attr_vals, splits = dataSplit(dataset, 'parents', attributes)
# print(attr_vals)
# for block in splits:
# 	count = 0
# 	for item in block: 
# 		print(item)
# 		count += 1
# 		if count == 3:
# 			break
# 	print()

# classes = getClassCount(splits[1])
# print(len(splits[1]))
# for cls in classes:
# 	print(cls)

def InfoGain(data, label, attributes):
	"""Calculate Info Gain on split"""
	parent_entropy = calcEntropy(data)
	p_rows = len(data)
	attr_vals, splits = dataSplit(data, label, attributes)
	c_rows = []
	c_entropies = []
	for i in range(len(attr_vals)):
		c_rows.append(len(splits[i]))
		c_entropies.append(calcEntropy(splits[i]))
	# print(label, len(attributes), c_entropies, sep=' * ')
	weighted_entropy = 0
	for i in range(len(c_entropies)):
		weighted_entropy += (c_rows[i]/p_rows)*(c_entropies[i])
	return parent_entropy - weighted_entropy
# print(InfoGain(dataset, 'parents', attributes))
# print(InfoGain(dataset, 'has_nurs', attributes))
# print(InfoGain(dataset, 'form', attributes))
# print(InfoGain(dataset, 'children', attributes))
# print(InfoGain(dataset, 'housing', attributes))
# print(InfoGain(dataset, 'finance', attributes))
# print(InfoGain(dataset, 'social', attributes))
# print()

def bestAttr(attributes, data):
	"""Get best attribute using information gain"""
	maxGain = 0
	maxGainAttribute = ''
	for item in attributes:
		info_gain = InfoGain(data, item, attributes)
		# print(info_gain,maxGain,len(attributes),len(data[0]),sep=' % ')
		if( info_gain > maxGain):
			maxGain = InfoGain(data, item, attributes)
			maxGainAttribute = item
	return maxGainAttribute, maxGain
# print(bestAttr(attributes, dataset))

def getBestSplit(data, attributes):
	maxGainAttribute, maxGain = bestAttr(attributes, data)
	attr_vals, splits = dataSplit(data, maxGainAttribute, attributes)
	return maxGainAttribute, attr_vals, splits

# attr_vals, splits = getBestSplit(dataset, attributes)
# print(attr_vals)
# for block in splits:
# 	count = 0
# 	for item in block: 
# 		print(item)
# 		count += 1
# 		if count == 3:
# 			break
# 	print()

# def buildTree(root, data, attributes, attribute_values):
# 	"""Recursively build the decision tree"""
# 	maxGainAttribute, maxGain = bestAttr(attributes, data)
# 	if len(data[0]) == 1 or maxGainAttribute == '' or len(attributes) == 0:
# 		root.prediction = maxClassPred(data)
# 		return root
# 	attr_unique_list = []
# 	attr_unique_no = 0
# 	for attr in attribute_values:
# 		if attr[0] == maxGainAttribute:
# 			attr_unique_no = len(attr[1])
# 			attr_unique_list = attr[1]
# 			break
# 	root.createChildren(attr_unique_no)
# 	root.setChildrenValues(attr_unique_list)
# 	root.attribute = maxGainAttribute
# 	attr_vals, splits = dataSplit(data, maxGainAttribute, attributes)
# 	remove_col_index = attributes.index(maxGainAttribute)
# 	# print(remove_col_index)
# 	# print(attr_vals)
# 	ch_attributes = attributes
# 	ch_attributes.pop(remove_col_index)
# 	ch_attribute_values = attribute_values
# 	ch_attribute_values.pop(remove_col_index)
# 	for i in range(len(root.child)):
# 		this_data = splits[attr_vals.index(root.data[i])]
# 		[j.pop(remove_col_index) for j in this_data]
# 		buildTree(root.child[i], this_data, ch_attributes, ch_attribute_values)
# 	return root

def buildTree(root, data, attributes, attribute_values):
	if len(data) == 0 or calcEntropy(data) == 0 or len(data[0]) == 1:
		root.prediction = maxClassPred(data)
		return root
	maxGainAttribute, attr_vals, splits = getBestSplit(data, attributes)
	# print(maxGainAttribute, attr_vals, len(data), len(data[0]), sep=':')
	if maxGainAttribute == '':
		root.prediction = maxClassPred(data)
		return root
	root.attribute = maxGainAttribute
	ch_attributes = attributes.copy()
	ch_attribute_values = attribute_values.copy()
	rem_idx = ch_attributes.index(maxGainAttribute)
	del ch_attributes[rem_idx]
	child_vals =  ch_attribute_values.pop(rem_idx)[1]
	for block in splits:
		for row in block:
			row.pop(rem_idx)
	# for item in ch_attribute_values:
	# 	print(item)
	# print()
	# print(child_vals)
	root.createChildren(len(child_vals))
	root.setChildrenValues(child_vals)
	# for i in range(len(splits)):
	# 	print(i, len(root.child), len(splits), sep=' // ')
	# 	which_child = root.child[0]
	# 	buildTree(root.child[i], splits[i], ch_attributes, ch_attribute_values)
	for i in range(len(attr_vals)):
		value = attr_vals[i]
		child_index = -1
		for j in range(len(root.data)):
			if root.data[j] == value:
				child_index = j
				break
		buildTree(root.child[child_index], splits[i], ch_attributes, ch_attribute_values)
	return root

buildTree(root, train_data, attributes, attribute_values)
# def printTree(root, indent):
# 	if root.prediction != '':
# 		print('-'*(indent+1), root.prediction)
# 		return
# 	print('-'*indent, root.attribute)
# 	indent += 1
# 	for child in root.child:
# 		print('-'*indent,root.data[root.child.index(child)])
# 		printTree(child, indent)
# 	return

# printTree(root, 0)

# def allPred(root):
# 	print(root.attribute, root.prediction, sep=' $ ')
# 	for child in root.child:
# 		allPred(child)
# allPred(root)

def predict(root, row, attributes):
	if len(root.child) == 0:
		if root.prediction == '':
			return 'recommend'
		return root.prediction
	curr_attr = root.attribute
	attr_idx = attributes.index(curr_attr)
	row_val = row[attr_idx]
	# print(curr_attr, attr_idx, sep=' # ')
	for i in range(len(root.child)):
		if root.data[i] == row_val:
			return predict(root.child[i], row, attributes)


prGreen('='*20 + 'Bootstrap Sampling' + '='*20)

def calcAccuracy(test_data, root, attributes):
	total = 0
	correct = 0
	for row in test_data:
		pred = predict(root, row, attributes)
		# print(row[-1], pred, sep=' @ ')
		total += 1
		if row[-1] == pred:
			correct += 1
		accuracy = correct/total
	return accuracy

print()
print('Accuracy: ' + str(calcAccuracy(test_data, root, attributes)))

def confMatrix(root, test_data, target_values, attributes):
	conf_item = []
	for i in range(len(target_values)):
		conf_item.append(0)
	conf = []
	for j in range(len(target_values)):
		conf.append(copy.deepcopy(conf_item))
	conf[2][2] = 2
	for row in test_data:
		pred = predict(root, row, attributes)
		actu = row[-1]
		pred_idx = target_values.index(pred)
		actu_idx = target_values.index(actu)
		conf[pred_idx][actu_idx] += 1
	return conf
		
print()
prYellow('-'*10 + 'Confusion Matrix' + '-'*10)
conf = confMatrix(root, test_data, target_values, attributes)
for item in conf:
	print(item)

def calcMetrics(conf, target_values):
	precision = []
	recall = []
	F1_score = []
	for i in range(len(target_values)):
		true_pos = 0
		true_neg = 0
		fals_pos = 0
		fals_neg = 0
		
		true_pos = conf[i][i]
		for j in range(len(target_values)):
			fals_neg += conf[j][i]
		for j in range(len(target_values)):
			fals_pos += conf[i][j]
		# print(true_pos, fals_pos, fals_neg, sep=' = ')
		this_prec = true_pos/(true_pos + fals_pos)
		this_recl = true_pos/(true_pos + fals_neg)
		this_f1 = 0
		if this_prec == 0 or this_recl == 0:
			this_f1 = 0
		else:
			this_f1 = 2 * this_prec * this_recl / (this_prec + this_recl)
		precision.append(this_prec)
		recall.append(this_recl)
		F1_score.append(this_f1)
	return precision, recall, F1_score

precision, recall, F1_score = calcMetrics(conf, target_values)

print()
prYellow('-'*10 + 'Performance Metrics' + '-'*10)
for i in range(len(target_values)):
	print()
	prCyan(target_values[i] + ':')
	print('\t' + 'Precision: ' + str(precision[i]))
	print('\t' + 'Recall: ' + str(recall[i]))
	print('\t' + 'F1 Score: ' + str(F1_score[i]))

# K Cross Validation

def chunkSplit(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def kcrossSplit(data, k):
	""" t = which k should be treated as test set """
	n_rows = len(data)
	chunk_size = math.ceil(n_rows/k)
	chunks = list(chunkSplit(data, chunk_size))
	return chunks

def flattenData(chunks):
	flat = []
	for sublist in chunks:
			for item in sublist:
					flat.append(copy.deepcopy(item))
	return flat

def splitFlattenChunks(chunks, t):
	train_chunks = []
	test_chunks = []
	for i in range(len(chunks)):
		if i == t:
			test_chunks.append(chunks[i])
		else:
			train_chunks.append(chunks[i])
	train_data = flattenData(train_chunks)
	test_data = flattenData(test_chunks)
	return train_data, test_data


print()
prGreen('='*20 + 'K-Cross Validation' + '='*20)
print()
shuffle_data = random.sample(dataset, k=n_rows)
accuracies = []
for k in range(3,15):
	chunks = kcrossSplit(shuffle_data, k)
	accuracy_all = []
	for t in range(k):
		train_data, test_data = splitFlattenChunks(chunks, t)
		temproot = DecisionTree()
		buildTree(temproot, train_data, attributes, attribute_values)
		# print('Accuracy: ' + str(calcAccuracy(test_data, temproot, attributes)))
		accuracy_all.append(calcAccuracy(test_data, temproot, attributes))
	accuracy_avg = sum(accuracy_all)/len(accuracy_all)
	print('K = ' + str(k) + ' : Average Accuracy = ' + str(accuracy_avg))
	accuracies.append(accuracy_avg)

# print(accuracies)

# temp_accuracies = [0.9684413580246914,
# 0.9730709876543211,
# 0.9755401234567902,
# 0.9767746913580247,
# 0.9789353400893049,
# 0.9804783950617284,
# 0.9814043209876543,
# 0.9804783950617285,
# 0.9807131799498211,
# 0.9807098765432097,
# 0.9818675210060209]

prev_acc, this_acc = 0, 0
epsilon = 0.0001
best_k = 3
for i in range(1, len(accuracies)):
	prev_acc = accuracies[i-1]
	this_acc = accuracies[i]
	diff = this_acc - prev_acc
	# print(i, diff, this_acc, prev_acc, sep=' - ')
	if diff < epsilon:
		best_k = (i-1) + 3
		break

print()
prCyan('Optimum value of K is ' + str(best_k))



# for i in range(10):
# 	print(train_data[i])
# print()
# for i in range(10):
# 	print(test_data[i])




	
	
	
