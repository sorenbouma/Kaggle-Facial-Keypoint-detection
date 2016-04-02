import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from matplotlib import pyplot
FTRAIN = '/home/soren/Desktop/kaggle/facialkeypoints/training.csv'
FTEST = '/home/soren/Desktop/kaggle/facialkeypoints/test.csv'
FLOOKUP = '/home/soren/Desktop/kaggle/facialkeypoints/IdLookupTable.csv'



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


def plot_sample(x,  axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    #axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

X, _ = load(test=False)

fig = pyplot.figure(figsize=(6, 6))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)



def random_grid(X,n):
    for i in range(n**2):
        ax = fig.add_subplot(n, n, i + 1, xticks=[], yticks=[])
        a=np.random.randint(low=0,high=2000)
        plot_sample(X[a],  ax)
    pyplot.show()
random_grid(X,2)