In this file, I am going to document all the concepts and architecture I faced during my research training on **deep learning ensembles**.

- [Concepts](#concepts)
- [Perturbation](#perturbation)
  - [Vanishing Gradient Problem](#vanishing-gradient-problem)
  - [Network Weight Perturbation](#network-weight-perturbation)
  - [Bagging](#bagging)
  - [Batch Normalization](#batch-normalization)
  - [Monte Carlo Dropout](#monte-carlo-dropout)
- [Model Architectures](#model-architectures)
  - [VGG19](#vgg19)
  - [MobileNet](#mobilenet)
  - [DenseNet](#densenet)
  - [ResNet50](#resnet50)
  - [Inception V3](#inception-v3)
  - [Xception](#xception)
  - [AlexNet](#alexnet)

## Concepts

## Perturbation

In mathematics, perturbation refers to a small change made to the input of a function or system. The concept is used to study how slight modifications can affect the behavior and properties of mathematical models and solutions. Here are some common contexts where perturbation is used:

- **Perturbation Theory**: This is a set of techniques for finding an approximate solution to a problem by starting from the exact solution of a related, simpler problem and adding corrections to it. It is widely used in various fields such as physics, engineering, and applied mathematics.
- **Differential Equations**: Small changes in the coefficients or the terms of differential equations to study the stability and sensitivity of their solutions.
- **Optimization**: Perturbations are used to understand the sensitivity of the optimal solution with respect to small changes in the parameters of the objective function or constraints.

In deep learning, perturbations are often used in the following contexts:

- **Adversarial Attacks**: Small, intentional changes made to the input data to deceive a machine learning model into making a wrong prediction. These are designed to be imperceptible to humans but can significantly affect the performance of the model.
- **Data Augmentation**: Introducing slight variations to the training data (such as rotations, translations, and noise) to improve the robustness and generalization ability of the model. This technique helps the model to better handle variations and noise in real-world data.
- **Regularization**: Perturbing the model parameters or inputs during training to prevent overfitting. Techniques such as dropout introduce randomness to the training process to improve the model's generalization capabilities.
- **Sensitivity Analysis**: Studying how changes in the input or the model parameters affect the output to understand the robustness and stability of the neural network.

### Vanishing Gradient Problem

The vanishing gradient problem occurs during the training of deep neural networks when the gradients of the loss function with respect to the network parameters become very small. This causes the updates to the weights during backpropagation to be insignificant, effectively preventing the network from learning and improving.

However, in very deep networks, repeated multiplication of gradients (which are often less than one) through many layers can result in extremely small values. When this happens, the gradients in the earlier layers become so small that the weights are updated minimally, if at all, effectively stalling the learning process.

### Network Weight Perturbation

This method involves altering the weights of a neural network slightly to create different models, which helps avoid local minima during training and enhances model robustness.

### Bagging

Bagging (Bootstrap Aggregating) is a technique where multiple models are trained on different subsets of the training data created by sampling with replacement. This approach helps reduce variance and improves model stability.

### Batch Normalization

Batch normalization is a technique used to improve the training of deep neural networks by reducing the internal covariate shift. It normalizes the inputs of each layer so that they have a mean of zero and a variance of one, and it does this for each mini-batch of data.

- Internal Covariate Shift: As training progresses, the distribution of each layer’s inputs changes, which can slow down training. This phenomenon is called internal covariate shift. By normalizing the inputs, batch normalization reduces this shift, making the training faster and more stable.
- Gradient Flow: Normalizing inputs helps maintain gradients in a reasonable range, preventing issues like vanishing and exploding gradients, which can hinder learning.
- Higher Learning Rates: Batch normalization allows for higher learning rates, which can accelerate the training process.

### Monte Carlo Dropout

Monte Carlo Dropout (MC Dropout) is a technique used in deep learning to estimate the uncertainty in model predictions. It leverages dropout, a regularization method typically used during training, at inference time to perform multiple stochastic forward passes. This allows the model to generate a distribution of predictions, from which uncertainty can be quantified.

- Uncertainty Estimation: In many real-world applications, knowing how confident a model is in its predictions is crucial. MC Dropout provides a way to estimate this uncertainty.
- Robustness: Models that can quantify their uncertainty are better at identifying when they are likely to make incorrect predictions, which can improve decision-making processes.
- Regularization: Dropout helps prevent overfitting during training by randomly setting some neurons' outputs to zero. By applying it during inference as well, MC Dropout leverages this randomness to create a distribution of predictions.

## Model Architectures

### VGG19

VGG19 is a convolutional neural network model proposed by the Visual Geometry Group (VGG) from Oxford. It consists of 19 layers, including 16 convolutional layers, 3 fully connected layers, 5 max-pooling layers, and a softmax layer. It is known for its simplicity and use of small receptive fields (3x3 convolutional filters).

- Deep architecture with 19 layers.
- Uses small (3x3) convolution filters throughout.

### MobileNet

Definition: MobileNet is a family of convolutional neural networks designed for efficient use in mobile and embedded vision applications. It uses depthwise separable convolutions to reduce the number of parameters and computational load.

- Efficient architecture with depthwise separable convolutions.
- Suitable for mobile and embedded devices.

### DenseNet

DenseNet (Densely Connected Convolutional Networks) is a type of convolutional neural network where each layer receives input from all preceding layers. This direct connection alleviates the vanishing gradient problem and improves feature propagation.

- Dense connections between layers.
- Improved gradient flow and feature reuse

### ResNet50

ResNet50 is a convolutional neural network with 50 layers, part of the Residual Networks (ResNet) family. It uses skip connections, or shortcuts, to jump over some layers, addressing the vanishing gradient problem and enabling the training of very deep networks.

- 50-layer deep architecture.
- Skip connections (residual learning) to improve training.

### Inception V3

Inception v3 is a convolutional neural network that improves upon the original Inception architecture by factorizing convolutions and adding auxiliary classifiers. It is designed to enhance the network’s efficiency and accuracy.

- Factorized convolutions.
- Auxiliary classifiers to improve gradient signal.

### Xception

Xception (Extreme Inception) is a convolutional neural network that replaces the standard Inception modules with depthwise separable convolutions, inspired by the efficiency improvements seen in MobileNets.

- Depthwise separable convolutions.
- Improved performance with a **simpler architecture**.

### AlexNet

AlexNet is a pioneering convolutional neural network that significantly contributed to the success of deep learning by winning the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012. It has 8 layers: 5 convolutional layers and 3 fully connected layers.

- First deep network to achieve breakthrough performance on ImageNet.
- Uses ReLU activation and dropout to improve training.
