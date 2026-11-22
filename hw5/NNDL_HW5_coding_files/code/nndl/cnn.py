import numpy as np

from cs231n.fast_layers import (
    conv_forward_fast,
    conv_backward_fast,
    max_pool_forward_fast,
    max_pool_backward_fast,
)

from nndl.layers import (
    affine_forward,
    affine_backward,
    relu_forward,
    relu_backward,
    softmax_loss,
    batchnorm_forward,
    batchnorm_backward,
)
from nndl.conv_layers import (
    conv_forward_naive,
    conv_backward_naive,
    max_pool_forward_naive,
    max_pool_backward_naive,
    spatial_batchnorm_forward,
    spatial_batchnorm_backward,
)
from nndl.layer_utils import (
    affine_relu_forward,
    affine_relu_backward,
)
from nndl.conv_layer_utils import (
    conv_relu_forward,
    conv_relu_backward,
    conv_relu_pool_forward,
    conv_relu_pool_backward,
)


""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""


class ThreeLayerConvNet:
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim: tuple[int, int, int] = (3, 32, 32),
        num_filters: int = 32,
        filter_size: int = 7,
        hidden_dim: int = 100,
        num_classes: int = 10,
        weight_scale: float = 1e-3,
        reg: float = 0.0,
        dtype: np.dtype = np.float32,
        use_batchnorm: bool = False,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.use_batchnorm = use_batchnorm
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        if self.use_batchnorm:
            self.params["gamma1"] = np.ones(num_filters, dtype=dtype)
            self.params["beta1"] = np.zeros(num_filters, dtype=dtype)
            self.params["gamma2"] = np.ones(hidden_dim, dtype=dtype)
            self.params["beta2"] = np.zeros(hidden_dim, dtype=dtype)
            self.bn_params = [{"mode": "train"}, {"mode": "train"}]

        # ================================================================ #
        # YOUR CODE HERE:
        #   Initialize the weights and biases of a three layer CNN. To initialize:
        #     - the biases should be initialized to zeros.
        #     - the weights should be initialized to a matrix with entries
        #         drawn from a Gaussian distribution with zero mean and
        #         standard deviation given by weight_scale.
        # ================================================================ #
        C, H, W = input_dim
        # basically W1 is just the number of (filter, filter dim)
        self.params["W1"] = weight_scale * np.random.randn(
            num_filters, C, filter_size, filter_size
        )
        self.params["b1"] = np.zeros(num_filters)

        # Convolution:  32 feature maps, each 32 × 32
        # ReLU:         32 feature maps, each 32 × 32
        # Max pool:     32 feature maps, each 16 × 16
        # Flatten:      32 × 16 × 16 = 8192 numbers
        # Assumes the convolution preserves H and W, and pooling has stride 2.
        pooled_dim = num_filters * (H // 2) * (W // 2)
        self.params["W2"] = weight_scale * np.random.randn(pooled_dim, hidden_dim)
        self.params["b2"] = np.zeros(hidden_dim)
        self.params["W3"] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params["b3"] = np.zeros(num_classes)


        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}
        scores = None
        cache1 = None
        cache2 = None
        cache3 = None

        # ================================================================ #
        # YOUR CODE HERE:
        #   Implement the forward pass of the three layer CNN.  Store the output
        #   scores as the variable "scores".
        # ================================================================ #
        if self.use_batchnorm == False:
            # conv - relu - 2x2 max pool - affine - relu - affine - softmax
            pool_out, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
            hidden_out, cache2 = affine_relu_forward(pool_out, W2, b2)
            scores, cache3 = affine_forward(hidden_out, W3, b3)
        else:
            conv_out, conv_cache = conv_forward_fast(X, W1, b1, conv_param)
            norm1, bn_cache1 = spatial_batchnorm_forward(
                conv_out, self.params["gamma1"], self.params["beta1"], self.bn_params[0]
            )
            relu1, relu_cache1 = relu_forward(norm1)
            pool_out, pool_cache = max_pool_forward_fast(relu1, pool_param)

            hidden, affine_cache = affine_forward(pool_out, W2, b2)
            norm2, bn_cache2 = batchnorm_forward(
                hidden, self.params["gamma2"], self.params["beta2"], self.bn_params[1]
            )
            hidden_out, relu_cache2 = relu_forward(norm2)
            scores, cache3 = affine_forward(hidden_out, W3, b3)

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        if y is None:
            return scores

        loss, grads = 0, {}
        # ================================================================ #
        # YOUR CODE HERE:
        #   Implement the backward pass of the three layer CNN.  Store the grads
        #   in the grads dictionary, exactly as before (i.e., the gradient of
        #   self.params[k] will be grads[k]).  Store the loss as "loss", and
        #   don't forget to add regularization on ALL weight matrices.
        # ================================================================ #
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (
            np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2)
        )
        dhidden, dW3, db3 = affine_backward(dscores, cache3)
        if self.use_batchnorm:
            # Hidden layer: ReLU → batch norm → affine, in backward order
            dnorm2 = relu_backward(dhidden, relu_cache2)
            dhidden_affine, dgamma2, dbeta2 = batchnorm_backward(dnorm2, bn_cache2)
            dpool, dW2, db2 = affine_backward(dhidden_affine, affine_cache)

            # Conv layer: pool → ReLU → spatial batch norm → conv
            drelu1 = max_pool_backward_fast(dpool, pool_cache)
            dnorm1 = relu_backward(drelu1, relu_cache1)
            dconv, dgamma1, dbeta1 = spatial_batchnorm_backward(dnorm1, bn_cache1)
            dx, dW1, db1 = conv_backward_fast(dconv, conv_cache)
        else:
            dpool, dW2, db2 = affine_relu_backward(dhidden, cache2)
            dx, dW1, db1 = conv_relu_pool_backward(dpool, cache1)

        grads = {
            "W1": dW1 + self.reg * W1,
            "b1": db1,
            "W2": dW2 + self.reg * W2,
            "b2": db2,
            "W3": dW3 + self.reg * W3,
            "b3": db3
        }
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        return loss, grads
