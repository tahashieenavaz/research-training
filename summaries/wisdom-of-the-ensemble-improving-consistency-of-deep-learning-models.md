# Wisdom Of Ensemble: Improving Consistency Of Deep Learning Models

Deep learning models are assisting humans in their decision making process, therefore user trust is of paramount importance. User trust is sometimes a function of constant behavior.

As this sounds like a straightforward requirement as we retrain models over time there are no guarantees that they produce the same **correct** output for the same input. In this paper consistency has been defined as the ability of a model to generate correct samples over the course of its maturity during different cycles of training. By way of an illustration, imagine that a model classifies a picture correctly and we will run an update over night, say, for car drivers, and in our latest version the model is NOT able to classify something that it could detect correctly yesterday. This would greatly affect drivers' trust.

The writers have used ensembles of ResNet50 and MobileNetV2. ResNet50, with 50 convolutional layers and residual blocks, mitigates the vanishing gradient problem through skip connections, enabling effective deep neural network training. Pretrained on datasets like ImageNet, it’s popular for tasks such as object detection and image recognition. Since winning ILSVRC 2015, ResNet50 has gained popularity for its effective gradient propagation and flow of lower-level information to higher layers. Figure 4 visualizes its architecture.

The stochastic approach alters activation functions in ResNet50 or MobileNetV2 by randomly replacing them with functions from a pool of candidates. This creates varied networks for the ensemble, with performance differing by CNN architecture. Referred to as “SE” in the experiments, this method avoids overfitting by randomly replacing activation functions without using specific datasets.

During training, various data-augmentation techniques are used:

- APP1: Generates three new images by reflecting the original image vertically and horizontally, and scaling it along both axes with factors between 1 and 2.
- APP2: Builds on APP1 by adding rotation (−10 to 10 degrees), translation (0 to 5 pixels), and shear transformations (0 to 30 degrees), resulting in six new images.
- APP3: Similar to APP2 but excludes shear and scale transformations, producing four new images.
- APP4: Uses PCA-based transformations to create three new images through random zeroing, noise addition, and component swapping with a 5% probability from images in the same class.
- APP5: Like APP4 but applies Discrete Cosine Transform (DCT) instead of PCA, keeping the DC coefficient unchanged.
- APP6: Applies to color images, creating three new images by color shifting, contrast alteration, and sharpness modification using Gaussian blur.

The PEP method creates ensembles by adding Gaussian random noise to the weights of a single trained network. This perturbation can enhance performance and help the model escape local minima, avoiding the cost of retraining. Variants tested include:

- Dout: Similar to dropout, zeroes out 2% of the weights.
- DCTa: Projects weights onto DCT space, zeroes 3.33% of DCT coefficients (excluding the DC component), then applies inverse DCT.
- DCTb: Projects weights onto DCT space with small random noise (excluding the DC component), then applies inverse DCT.
- PEPa: Similar to the original PEP but with a small amount of random noise.
- PEPb: Similar to PEPa.
