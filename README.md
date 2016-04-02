# Kaggle-Facial-Keypoint-detection
Code for my entry in Kaggle's facial keypoint detection competition. As of the time of writing, it gets 6th
on the Kaggle's leaderboards.

Based on Daniel Nouri's awesome KFKD tutorial, but uses a unique neural net architecture. 
>dnouri's model had 3 conv layers with 2x2 max pooling between them.
>This net has 13 3x3 conv layers, no max pooling and batch normalization with skip layer connections.

It's based on MSRA's deep residual architecture, but with some tweaks to make it
better suited to this problem:
>Unlike the original MSRA paper uses dropout between conv layers and residual connection layers.
>Has a fully connected layer after the global pooling layer.
>Uses very leaky rectifiers instead of relu.

This architecture can get a slight validation loss improvement on dnouri's original kfkd net(0.00069 vs 0.000767)
and gets good results with much less epochs.

Sorry if the code is horrible, I'm very new to programming.

Thank You Daniel Nouri for your amazing tutorial.

Requires theano, lasagne, nolearn,cudnn, cuda.
