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

import cPickle as pickle
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
# VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# License: non-commercial use only

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax


def build_model():
    net = {}
    net['input'] = InputLayer((None, 1, 96, 96))
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 1, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 1, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 1, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 1, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 1, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 1, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 1, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 1, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 1, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 1, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 1, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 1, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 1, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
       net['fc7_dropout'], num_units=30, nonlinearity=None)

    return net
network=build_model()





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
	 

net = NeuralNet(
	layers=network,
	#input_shape=(None, 1, 96, 96),
	update_learning_rate=theano.shared(float32(0.3)),
	update_momentum=theano.shared(float32(0.96)),
	regression=True,
	batch_iterator_train=FlipBatchIterator(batch_size=32),
	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=0.11, stop=0.003),
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