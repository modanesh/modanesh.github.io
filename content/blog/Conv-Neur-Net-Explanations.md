---
title:  "Convolutional Neural Network Explanation Methods"
description: A brief description on explanations methods in the computer vision literature.
date: 2019-11-19
author: ["Mohamad H. Danesh"]
showToc: true
disableAnchoredHeadings: false
---


## Methods:

-   Saliency Maps:


	-   Intuitively, the absolute value of the gradient indicates those input features (pixels, for image classification) that can be perturbed the least in order for the target output to change the most, with no regards for the direction of this change.


-   Gradient Input:


	-   The attribution is computed taking the (signed) partial derivatives of the output with respect to the input and multiplying them feature-wise with the input itself.


-   Integrated Gradient:


	-   Similarly to Gradient Input, computes the partial derivatives of the output with respect to each input feature. However, instead of evaluating the partial derivative at the provided input $x$ only, it computes its average value while the input varies along a linear path from a baseline $x'$ to $x$. The baseline is defined by the user and often chosen to be zero.


-   Layer-wise Relevance Propagation (LRP):


	-   Considers a quantity $r^l_i$ , called "relevance" of unit $i$ of layer $l$. The algorithm starts at the output layer $L$, assigning the relevance of the target neuron $c$ equal to the activation of the neuron itself, and the relevance of all other neurons to zero. Then it proceeds layer by layer, redistributing the prediction score $S_i$ until the input layer is reached.


-   DeepLIFT:


	-   Similar to LRP, but the difference is that there is a baseline. The relevance of unit $i$ of layer $l$ for input $x$ is subtracted by the relevance of unit $i$ of layer $l$ for input $x'$ (baseline). Also, for redistributing the prediction score $S_i$, the baseline plays a role.


## Resources:

-   [Visualizing Expected Gradients](https://psturmfels.github.io/VisualizingExpectedGradients/)

-   [A unified view of gradient-based attribution methods for Deep Neural Networks](https://pdfs.semanticscholar.org/7a56/72796aeca8605b2e370d8a756a7a311fd171.pdf)