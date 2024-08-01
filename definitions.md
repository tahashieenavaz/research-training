# Definitions

## Vanishing Gradient Problem

The vanishing gradient problem occurs during the training of deep neural networks when the gradients of the loss function with respect to the network parameters become very small. This causes the updates to the weights during backpropagation to be insignificant, effectively preventing the network from learning and improving.

However, in very deep networks, repeated multiplication of gradients (which are often less than one) through many layers can result in extremely small values. When this happens, the gradients in the earlier layers become so small that the weights are updated minimally, if at all, effectively stalling the learning process.

Consider a neural network with \(L\) layers. The gradient of the loss \(L\) with respect to the weights \(w_l\) at layer \(l\) is calculated as:

\[ 
\frac{\partial L}{\partial w_l} = \frac{\partial L}{\partial a_L} \cdot \frac{\partial a_L}{\partial z_L} \cdot \frac{\partial z_L}{\partial a_{L-1}} \cdot \ldots \cdot \frac{\partial a_{l+1}}{\partial z_{l+1}} \cdot \frac{\partial z_{l+1}}{\partial a_l} \cdot \frac{\partial a_l}{\partial z_l} \cdot \frac{\partial z_l}{\partial w_l} 
\]

Here, \(a_l\) represents the activations and \(z_l\) represents the weighted sum of inputs at layer \(l\). Each term \(\frac{\partial z}{\partial a}\) and \(\frac{\partial a}{\partial z}\) in the chain is typically less than 1. For deep networks (large \(L\)), this product can become exceedingly small, leading to vanishing gradients.


## Network Weight Perturbation

This method involves altering the weights of a neural network slightly to create different models, which helps avoid local minima during training and enhances model robustness.

## Bagging

Bagging (Bootstrap Aggregating) is a technique where multiple models are trained on different subsets of the training data created by sampling with replacement. This approach helps reduce variance and improves model stability.

## VGG19

VGG19 is a convolutional neural network model proposed by the Visual Geometry Group (VGG) from Oxford. It consists of 19 layers, including 16 convolutional layers, 3 fully connected layers, 5 max-pooling layers, and a softmax layer. It is known for its simplicity and use of small receptive fields (3x3 convolutional filters).

- Deep architecture with 19 layers.
- Uses small (3x3) convolution filters throughout.

## MobileNet

Definition: MobileNet is a family of convolutional neural networks designed for efficient use in mobile and embedded vision applications. It uses depthwise separable convolutions to reduce the number of parameters and computational load.

- Efficient architecture with depthwise separable convolutions.
- Suitable for mobile and embedded devices.

## DenseNet

DenseNet (Densely Connected Convolutional Networks) is a type of convolutional neural network where each layer receives input from all preceding layers. This direct connection alleviates the vanishing gradient problem and improves feature propagation.

- Dense connections between layers.
- Improved gradient flow and feature reuse

## ResNet50

ResNet50 is a convolutional neural network with 50 layers, part of the Residual Networks (ResNet) family. It uses skip connections, or shortcuts, to jump over some layers, addressing the vanishing gradient problem and enabling the training of very deep networks.

- 50-layer deep architecture.
- Skip connections (residual learning) to improve training.

## Inception V3

Inception v3 is a convolutional neural network that improves upon the original Inception architecture by factorizing convolutions and adding auxiliary classifiers. It is designed to enhance the network’s efficiency and accuracy.

- Factorized convolutions.
- Auxiliary classifiers to improve gradient signal.

## Xception

Xception (Extreme Inception) is a convolutional neural network that replaces the standard Inception modules with depthwise separable convolutions, inspired by the efficiency improvements seen in MobileNets.

- Depthwise separable convolutions.
- Improved performance with a **simpler architecture**.

## AlexNet

AlexNet is a pioneering convolutional neural network that significantly contributed to the success of deep learning by winning the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012. It has 8 layers: 5 convolutional layers and 3 fully connected layers.

- First deep network to achieve breakthrough performance on ImageNet.
- Uses ReLU activation and dropout to improve training.
