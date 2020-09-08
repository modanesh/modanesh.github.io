---
layout: post
title: "Convolutional Neural Network Explanation Methods"
date: 2019-05-16
---

### Methods:

-   Saliency Maps:
    

	-   Intuitively, the absolute value of the gradient indicates those input features (pixels, for image classification) that can be perturbed the least in order for the target output to change the most, with no regards for the direction of this change.
    

-   Gradient Input:
    

	-   The attribution is computed taking the (signed) partial derivatives of the output with respect to the input and multiplying them feature-wise with the input itself.
    

-   Integrated Gradient:
    

	-   Similarly to Gradient Input, computes the partial derivatives of the output with respect to each input feature. However, instead of evaluating the partial derivative at the provided input <b><i>x</i></b> only, it computes its average value while the input varies along a linear path from a baseline <b><i>x'</i></b> to <b><i>x</i></b>. The baseline is defined by the user and often chosen to be zero.
    

-   Layer-wise Relevance Propagation (LRP):
    

	-   Considers a quantity <b><i>r<sup>l</sup><sub>i</sub></i></b> , called "relevance" of unit <b><i>i</i></b> of layer <b><i>l</i></b>. The algorithm starts at the output layer <b><i>L</i></b>, assigning the relevance of the target neuron <b><i>c</i></b> equal to the activation of the neuron itself, and the relevance of all other neurons to zero. Then it proceeds layer by layer, redistributing the prediction score <b><i>S<sub>i</sub></i></b> until the input layer is reached.
    

-   DeepLIFT:
    

	-   Similar to LRP, but the difference is that there is a baseline. The relevance of unit <b><i>i</i></b> of layer <b><i>l</i></b> for input <b><i>x</i></b> is subtracted by the relevance of unit <b><i>i</i></b> of layer <b><i>l</i></b> for input <b><i>x'</i></b> (baseline). Also, for redistributing the prediction score <b><i>S<sub>i</sub></i></b>, the baseline plays a role.
    

### Resources:

-   [Visualizing Expected Gradients](https://psturmfels.github.io/VisualizingExpectedGradients/)
    
-   [A unified view of gradient-based attribution methods for Deep Neural Networks](https://pdfs.semanticscholar.org/7a56/72796aeca8605b2e370d8a756a7a311fd171.pdf)
