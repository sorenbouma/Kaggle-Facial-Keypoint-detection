from __future__ import print_function

import sys
import os
import time
import string
import random
import pickle

import numpy as np
import theano
import theano.tensor as T
import lasagne

import dill as pickle
from datetime import datetime
import os
import sys
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import batch_norm

from matplotlib import pyplot
import numpy as np
from lasagne import layers
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
from pandas import DataFrame
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import theano

try:
	from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
	from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
	from lasagne.nonlinearities import elu
except ImportError:
	Conv2DLayer = layers.Conv2DLayer
	MaxPool2DLayer = layers.MaxPool2DLayer
	 

sys.setrecursionlimit(99000)  # for pickle...
np.random.seed(42)

FTRAIN = '/home/soren/Desktop/KFKD/kfkd-tutorial-master/training.csv'
FTEST = '/home/soren/Desktop/KFKD/test.csv'
FLOOKUP = '/home/soren/Desktop/KFKD/kfkd-tutorial-master/IdLookupTable.csv'



def float32(k):
	return np.cast['float32'](k)


def load(test=False, cols=None):
	"""Loads data from FTEST if *test* is True, otherwise from FTRAIN.
	Pass a list of *cols* if you're only interested in a subset of the
	target columns.
	"""
	fname = FTEST if test else FTRAIN
	df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

	# The Image column has pixel values separated by space; convert
	# the values to numpy arrays:
	df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

	if cols:  # get a subset of columns
		df = df[list(cols) + ['Image']]

	print(df.count())  # prints the number of values for each column
	df = df.dropna()  # drop all rows that have missing values in them

	X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
	X = X.astype(np.float32)

	if not test:  # only FTRAIN has any target columns
		y = df[df.columns[:-1]].values
		y = (y - 48) / 48  # scale target coordinates to [-1, 1]
		X, y = shuffle(X, y, random_state=42)  # shuffle train data
		y = y.astype(np.float32)
	else:
		y = None
	
	return X, y


def load2d(test=False, cols=None):
	print('loading data')
	X, y = load(test=test, cols=cols)
	X = X.reshape(-1, 1, 96, 96)
	print('finished loading data')
	return X, y


def plot_sample(x, y, axis):
	img = x.reshape(96, 96)
	axis.imshow(img, cmap='gray')
	if y is not None:
		axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)


def plot_weights(weights):
	fig = pyplot.figure(figsize=(6, 6))
	fig.subplots_adjust(
		left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

	for i in range(16):
		ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
		ax.imshow(weights[:, i].reshape(96, 96), cmap='gray')
	pyplot.show()


class FlipBatchIterator(BatchIterator):
	flip_indices = [
		(0, 2), (1, 3),
		(4, 8), (5, 9), (6, 10), (7, 11),
		(12, 16), (13, 17), (14, 18), (15, 19),
		(22, 24), (23, 25),
		]

	def transform(self, Xb, yb):
		Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

		# Flip half of the images in this batch at random:
		bs = Xb.shape[0]
		indices = np.random.choice(bs, bs / 2, replace=False)
		Xb[indices] = Xb[indices, :, :, ::-1]

		if yb is not None:
			# Horizontal flip of all x coordinates:
			yb[indices, ::2] = yb[indices, ::2] * -1

			# Swap places, e.g. left_eye_center_x -> right_eye_center_x
			for a, b in self.flip_indices:
				yb[indices, a], yb[indices, b] = (
					yb[indices, b], yb[indices, a])

		return Xb, yb


class AdjustVariable(object):
	def __init__(self, name, start=0.03, stop=0.001):
		self.name = name
		self.start, self.stop = start, stop
		self.ls = None

	def __call__(self, nn, train_history):
		if self.ls is None:
			self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
		#print('current'+self.name+str(getattr(nn,self.name)))
		epoch = train_history[-1]['epoch']
		new_value = np.cast['float32'](self.ls[epoch - 1])
		getattr(nn, self.name).set_value(new_value)


class EarlyStopping(object):
	def __init__(self, patience=100):
		self.patience = patience
		self.best_valid = np.inf
		self.best_valid_epoch = 0
		self.best_weights = None

	def __call__(self, nn, train_history):
		current_valid = train_history[-1]['valid_loss']
		current_epoch = train_history[-1]['epoch']
		if current_valid < self.best_valid:
			self.best_valid = current_valid
			self.best_valid_epoch = current_epoch
			self.best_weights = nn.get_all_params_values()
		elif self.best_valid_epoch + self.patience < current_epoch:
			print("Early stopping.")
			print("Best valid loss was {:.6f} at epoch {}.".format(
				self.best_valid, self.best_valid_epoch))
			nn.load_params_from(self.best_weights)
			raise StopIteration()
from lasagne import nonlinearities
#from lasagne.non
#custom_rectify = LeakyRectify(0.1)

from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import very_leaky_rectify


dropoutList=[]
global dropoutList
def build_cnn(input_var=None, n=5,output_n=30):
	print('building cnn')
	#counter for number of dropout layers
	# create a residual learning building block with two stacked 3x3 convlayers as in paper
	def residual_block(l, increase_dim=False, projection=False,dropout=0):
		kwargz={};dn=0
		a=layers.DropoutLayer(l,p=dropout)
		input_num_filters = l.output_shape[1]
		if increase_dim:
			first_stride = (2,2)
			out_num_filters = input_num_filters*2
		else:
			first_stride = (1,1)
			out_num_filters = input_num_filters

		stack_1 = batch_norm(ConvLayer(
			a, 
			num_filters=out_num_filters, 
			filter_size=(3,3), 
			stride=first_stride, 
			nonlinearity=nonlinearities.very_leaky_rectify, 
			pad='same', 
			W=lasagne.init.HeNormal(gain='relu')
			))
		if dropout > 0:
			dn+=1
			dropOut1=layers.DropoutLayer(
				incoming=stack_1,
				p=dropout,
				)
			kwargz[str(dn)]=dropout
			stack_2 = batch_norm(ConvLayer(
				dropOut1, 
				num_filters=out_num_filters,
				filter_size=(3,3),
				stride=(1,1),
				nonlinearity=very_leaky_rectify,
				pad='same',
				W=lasagne.init.HeNormal(gain='relu'),
				))
		else:
			stack_2 = batch_norm(ConvLayer(
				stack_1,
				num_filters=out_num_filters,
				filter_size=(3,3),
				stride=(1,1),
				nonlinearity=very_leaky_rectify,
				pad='same',
				W=lasagne.init.HeNormal(gain='relu')
				))

		# add shortcut connections
		if increase_dim:
			if projection:
				# projection shortcut, as option B in paper
				projection = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None))
				block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]),nonlinearity=very_leaky_rectify)
			else:
				# identity shortcut, as option A in paper
				identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
				padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
				block = NonlinearityLayer(
					ElemwiseSumLayer([stack_2, padding]),
					nonlinearity=very_leaky_rectify
					)
		else:
			block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]),nonlinearity=very_leaky_rectify)
		
		return block

	# Building the network
	l_in = InputLayer(shape=(None,1, 96, 96), input_var=input_var)

	# first layer, output is 16 x 32 x 32
	l = batch_norm(ConvLayer(l_in, num_filters=4, filter_size=(3,3), stride=(1,1), nonlinearity=very_leaky_rectify, pad='same', W=lasagne.init.HeNormal(gain='relu')))
	# first stack of residual blocks, output is 16 x 32 x 32
	for _ in range(n):
		l = residual_block(l,dropout=0.01)
	print ('first residual stack built')
	# second stack of residual blocks, output is 32 x 16 x 16
	l = residual_block(l, increase_dim=True,dropout=0.01)
	for _ in range(1,n):
		l = residual_block(l,dropout=0.01)
	print('second residual stack built')

	# third stack of residual blocks, output is 64 x 8 x 8
	l = residual_block(l, increase_dim=True,dropout=0.01)
	for _ in range(1,n):
		l = residual_block(l,dropout=0.01)
	print('third residual stack built')


	l = residual_block(l, increase_dim=True,dropout=0.01)
	for _ in range(1,n):
		l = residual_block(l,dropout=0.01)
	print('fourth residual stack built')

	l = residual_block(l, increase_dim=True,dropout=0.01)
	for _ in range(1,n):
		l = residual_block(l,dropout=0.1)
	print('fifth residual stack built')

	l = residual_block(l, increase_dim=True,dropout=0.1)
	for _ in range(1,n):
		l = residual_block(l,dropout=0.01)
	print('sixth residual stack built')
	

	
	# average pooling
	l = GlobalPoolLayer(l)
	l=DropoutLayer(l,p=0.03)
	l=DenseLayer(l,num_units=256,nonlinearity=very_leaky_rectify)
	
	# fully connected layer
	network = DenseLayer(
			l, num_units=output_n,
			W=lasagne.init.HeNormal(),
			nonlinearity=None)

	return network
print('ready to build cnn')

network=build_cnn(
	input_var=None,
	n=1
	)
#for lyr in  lasagne.layers.get_all_layers(network):

  #  if 'Dropout' in str(type(lyr)):
   #     lyr.p=0
from nolearn.lasagne import visualize as v


def example_occ(n,X,target,sqrL=7,figsize=(9,None)):
	x1=X[n,:,:,:];x1=x1.reshape(1,1,96,96)*-1
	t1=target[n,:]
	v.plot_occlusion(net,x1,t1,sqrL,figsize)
	pyplot.show()

def ex_sal(a,b,X,size=(9,None)):
	plot_saliency(net,X[a:b,:,:,:],size)





	#print(str(lyr) + str(dir(lyr)))
class ActivateDropout(object):
	def __init__(self,doList,verbose=False,threshold=0.004,adjust_lr=False):
		self.doList=doList
		self.verbose=verbose
		self.threshold=threshold
		self.adjust_lr=adjust_lr
	def __call__(self,nn,train_history):
		current_valid = train_history[-1]['valid_loss']
		if np.mean(current_valid) <= self.threshold:
			i=0;
			for lyr in lasagne.layers.get_all_layers(nn.layers[0]):
				if 'Dropout' in str(type(lyr)):
					if lyr.p!=0:
						break
					lyr.p=self.doList[i]
					if self.verbose:
						print('p adjusted to ' +str(lyr.p))
					i+=1
			if self.adjust_lr:
				print('lr adjustment not ready yet')
			#    lr=getattr(nn,'update_learning_rate')

def l2Nesterov(loss_or_grads,params,learning_rate,momentum=0.9,reg=0.001):
	updates=nesterov_momentum(loss_or_grads,params,learning_rate,momentum)
	 
print(dropoutList)

net = NeuralNet(
	layers=[network],
	#input_shape=(None, 1, 96, 96),
	update_learning_rate=theano.shared(float32(0.04)),
	update_momentum=theano.shared(float32(0.96)),
	regression=True,
	batch_iterator_train=FlipBatchIterator(batch_size=32),
	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=0.004, stop=0.0003),
		AdjustVariable('update_momentum', start=0.96, stop=0.9999),
	   #print('?')THESE DONT WORK. YOU HAVE TO DEFINE A CLASS FOR THESE
		#on_epoch_finished_test(),
		#ActivateDropout(doList=dropoutList,verbose=True,threshold=0.0045,adjust_lr=False),
		#SaveWeights(path='PATH_HERE',every_n_epochs=10,only_best=True)
		
		],
	max_epochs=10000,
	verbose=2,
	#objective_loss_function=squaredErrorL2Loss,
	#custom_scores=[('no L2 Val Loss',squared_error)],
	)

X,y=load2d()
print ('ready to fit data')
net.fit(X,y)

with open('NET7.pickle', 'wb') as f:
	pickle.dump(net, f, -1)
