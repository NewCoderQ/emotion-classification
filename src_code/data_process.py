# -*- coding: utf-8 -*-
# @Author: Zhiqiang
# @Date:   2017-09-25 18:28:16
# @Last Modified by:   funny_QZQ
# @Last Modified time: 2017-09-26 16:58:12

import jieba 	# 结巴分词
import xlrd
import numpy as np
import collections

from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model

MAX_FEATURES = 34000	# the frequence of words
MAX_SENTENCE_LENGTH = 40 	# the max length of sentence

def get_maxlen_wordfreqs():
	'''
		获取样本中句子的最大长度 以及 词频数

		Returns:
			word_freqs: the Counter of sentences
			workbook: the instance of excel
	'''
	maxlen = 0		# the max length of the line
	word_freqs = collections.Counter()		# word counter
	num_rows_counter = 0	# 句子的行数

	workbook = xlrd.open_workbook('../1.xlsx')	# open the xlsx file
	sheet1 = workbook.sheet_by_index(0)	# 获取第一张表对象
	num_rows = sheet1.nrows 			# get the line number
	for i in range(1, num_rows):		# 遍历每一行,除去第一行的标题
		line = sheet1.cell(i, 1).value
		print(line)
		line = line.replace(' ', '')	# 去除句中的空格
		words = list(jieba.cut(line))
		if len(words) > maxlen:
			maxlen = len(words)			# 获得最长句子的长度
		for word in words:
			word_freqs[word] += 1		# num++
		# num_rows_counter += 1
	print(maxlen)						# 104
	print(len(word_freqs))				# 34827
	return word_freqs, sheet1 		# return

def data_prepare(word_freqs, sheet):
	'''
		Data preparation

		Parameters:
			word_freqs: the Counter of sentences
	'''
	print('preparing data...')
	vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2 	# the length of the word dict
	# generate the word to index dict length is 34002
	word2index = {x[0]: i + 2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
	# print(word_freqs)
	# print(word2index['华为'])
	word2index['PAD'] = 0 		# padding 
	word2index['UNK'] = 1
	# index to word
	index2word = {v:k for k, v in word2index.items()}
	num_rows = sheet.nrows
	x = np.empty(num_rows - 1, dtype = list)	# train data
	y = np.zeros(num_rows - 1)					# train label
	# 生成训练数据以及标签
	i = 0
	for index in range(1, num_rows):		# 遍历每一行数据
		data_line = sheet.cell(index, 1).value.strip().replace(' ', '')	# 除去句中以及两端的空行
		# label: 中立：1, 消极：0
		label = 1 if sheet.cell(index, 2).value == '中立' else 0
		words = jieba.cut(data_line)		# words list
		seqs = []	# 创建sequence列表用来存放每句话的内容
		for word in words:
			if word in word2index:
				seqs.append(word2index[word])
			else:
				seqs.append(word2index['UNK'])	# 1
		x[i] = seqs
		y[i] = label
		i += 1
	
	# sentence padding
	x = sequence.pad_sequences(x, maxlen = MAX_SENTENCE_LENGTH)
	xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 42)
	return xtrain, xtest, ytrain, ytest, vocab_size, index2word, word2index
	
def built_net(xtrain, xtest, ytrain, ytest, vocab_size, index2word):
	'''
		build network

		Parameters:
			x_train, x_test: feature data
			y_train, y_test: test label
	'''
	# 网络构建
	EMBEDDING_SIZE = 128
	HIDDEN_LAYER_SIZE = 64
	BATCH_SIZE = 32
	NUM_EPOCHS = 10

	# *******************build network**************************
	'''
		Linear stack of layers
		Sequential中第一层必须指定input shape
	'''
	model = Sequential() 

	'''	
		The first layer: 2D -> 3D
		Parameters：
			input_dim: vocab_size, Size of the vocabulary, maximum integer index + 1
			output_dim: EMBEDDING_SIZE, Dimension of the dense embedding
			input_length: Length of input sequences, when it is constant
		
	'''
	model.add(Embedding(vocab_size, EMBEDDING_SIZE,input_length=MAX_SENTENCE_LENGTH))

	'''
		The second layer: Long-Short Term Memory layer
		Parameters:
			units: dimensionality of the output space, HIDDEN_LAYER_SIZE 64
			dropout: Float between 0 and 1.
	            Fraction of the units to drop for
	            the linear transformation of the inputs.
			recurrent_dropout: Float between 0 and 1.
	            Fraction of the units to drop for
	            the linear transformation of the recurrent state.
	'''
	model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))

	'''
		The third layer:
		Parameters:
			units: Positive integer, dimensionality of the output space, 1
		
		Input shape:
	        nD tensor with shape: `(batch_size, ..., input_dim)`.
	        The most common situation would be
	        a 2D input with shape `(batch_size, input_dim)`.

		Output shape
	        nD tensor with shape: `(batch_size, ..., units)`.
	        For instance, for a 2D input with shape `(batch_size, input_dim)`,
	        the output would have shape `(batch_size, units)`.
	'''
	model.add(Dense(1))

	'''
		The fourth layer: Applies an activation function to an output.
			添加一个sigmoid层，将输出归一化到0-1之间
	'''
	model.add(Activation("sigmoid"))	# 1 / (1 + exp(-x))

	'''
		Configures the learning process
		loss: loss function
			  binary_crossentropy: Binary crossentropy between an output tensor and a target tensor
	'''
	model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
	# 网络训练
	print('training...')
	model.fit(xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(xtest, ytest))
	model.save('../model/model.h5')
	# 预测
	score, acc = model.evaluate(xtest, ytest, batch_size=BATCH_SIZE)
	print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
	# print('{}   {}      {}'.format('预测','真实','句子'))
	# for i in range(5):
	#     idx = np.random.randint(len(xtest))
	#     xtest = xtest[idx].reshape(1, MAX_SENTENCE_LENGTH)
	#     ylabel = ytest[idx]
	#     ypred = model.predict(xtest)[0][0]
	#     sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
	#     print(' {}      {}     {}'.format(int(round(ypred)), int(ylabel), sent))


def test(file_obj, word2index):
	model = load_model('../model/model.h5')
	words = jieba.cut(str)
	seqs = []
	for word in words:
		if word in word2index:
			seqs.append(word2index[word])
		else:
			seqs.append(1)
	x = np.empty(1, dtype = list)
	x[0] = seqs
	# model.add(Embedding(vocab_size, 128, input_length = MAX_SENTENCE_LENGTH))
	x = sequence.pad_sequences(x, maxlen = MAX_SENTENCE_LENGTH)
	print(model.predict(x))

if __name__ == '__main__':
	word_freqs, sheet = get_maxlen_wordfreqs()
	xtrain, xtest, ytrain, ytest, vocab_size, index2word, word2index = data_prepare(word_freqs, sheet)
	# built_net(xtrain, xtest, ytrain, ytest, vocab_size, index2word)
	test(sheet, word2index)