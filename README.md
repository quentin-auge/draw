# Teaching a neural network to draw

An implementation of unconditional drawings generation using recurrent
neural networks.

![](images/penguin.gif)

For further insights, see the accompanying blog post: [Teaching a Neural Network to Draw](http://quentin-auge.github.io/2019/04/21/teaching-a-neural-network-to-draw.html).

The model follows the Google Brain paper
[A Neural Representation of Sketch Drawings](https://arxiv.org/abs/1704.03477).

The main entry point is the Jupyter notebook [unconditional.ipynb](unconditional.ipynb).

# Data and models

The dataset comes from [this repository](https://github.com/googlecreativelab/quickdraw-dataset).
The simplified files in binary format are used. They are too big to be
commited, but you can copy them to the `data/` subdirectory.

Serialized models are available in the `models/` subdirectory. They
are single-layer LSTMs outputting parameters for a 20-normals GMM,
with a hidden state size of either 128 or 512.

# Setup

The code runs with Python >= 3.6 and Pytorch >= 1.0.  
The other requirements are listed in `requirements.txt`.
