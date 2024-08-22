# Artifical Nueron and their types:
Act similarly to an actual neuron, hence the name.

They recieve inputs (dendrites) to the body (soma) and send an output (axon) if an activation threshold is reached.

Inputs can be weighted differently depending on the scenario. Activation thresholds can be of the following each with their respective equations and use cases:

- perceptron
- sigmoid
- Tanh
- Relu -> most commonly used

## How to choose Neuron type:
Need to select a nonlinear function if you want to approx a continous function.

- perceptron -> accepts only binary inputs and therefore not too useful
- sigmoid -> Use when it would be helpful to have neuron output in the range of [0,1]. Otherwise it is slow.
- Tanh -> Solid choice due to its rapid learning.
- Relu -> Preffered neuron choice b/c how efficiently these neurons enable learning networks to perform calcs.
    
# Artificial Neural Networks
Contain an input layer, hidden layers, and output layer.

## Input Layer
Dont preform calcs and are just placeholders for input data.

## Hidden Layers
### Dense Layers (Fully Connected Layers)
Most general form of hidden layer. Each neuron in a given dense layer receive info from every one of the neuorns in the preceding layer of the network. A dense layer is fully connected to the one before it.


## Output Layers
Sigmoid neurons can be used for an output that is deciphering between 2 classes (car or not a car). For models with more than one class, for example the MNIST data with 10 classes, a softmax layer can be used.
### Softmax Layer
A dense layer that allows for the deciphering between 2+ classes. Outputs a network estimate for each class that sum to 1.

> If they all sum to 1 then how can we have a model for multiple different classes?

# Cost Functions (Loss Functions)
Quantify how incorrect the models estimate of the ideal output is.
## Gradient Descent
Minimizes cost with given paramters.
## Stochastic Gradient Descent
Gradient descent performed in multiple smaller batches to allow for use in large datasets.
## Backpropogation
Essentially the chain rule. Uses cost to calc the relative contribution by every single parameter to the total cost and then updates the parameter. Adding hidden layers reduces effectiveness of backprop.

## Tuning Hidden Layers
The more abstract the ground truth is that we want to estimate the more helpful additonal layers are. Recommended starting point is 2-4 layers.

## Initalizing weights
Most common method is glorot initialization.

# Overfitting
Cost continues to goes down while validation goes up. (model is too accurate for application outside of training set)

## Dropout
Prevents any one specific neuron from becoming excessivley influential within the network by pretending that randomly selected neurons dont exist.
Avoids overfitting. In mdoels wth many hidden layers add dropout to the later layers first. Drop between 20-50% of the hidden layer neurons.

## Data Augmentation
- Skew image
- Blur Image
- Shift the image a few pixels
- Apply random noise
- Rotate image slightly

# Fancy Optimizers
## Momentum
Considers revious steps to help bypass local minima (ski example)
## AdaGrad, AdaDelta, RMSProp, and Adam
Have individual learning rates for each parameter enabling those that have reached their mimimum to halt.
Adam is the most used in the book.

# Convolutional Neural Networks
Artifical Network featuring one or more convolutional layers.
## Convolutional Layers
Conv Layers consist of sets of kernels, also known as filters. Each kernel is a samll window (patch) that scans an image top left to bottom right. Kernels are made of weights like dense layers and range in size. (commonly 3x3 or 5x5) The amount of weights is size x # of dims (3x3x1) plus bias term (10 in that ex). There are multiple filters in these layers which output activation maps. To determine the number of filters consider the complexity of the problem (more complex = more features). Early requires less filters as they identify simple features so it is common to have more kernels later in the network.

## Conv Filter Hyperparamters
### Kernel Size
Typically 3x3, 5x5, or at largest 7x7.
### Stride Length 
Refers to the size of step that the kernerl takes over the image. Typically 1,2, or 3 pixels but never more than 3.
### Padding
To produce an activation map that is the same size as the input pad the image with zeroes.


## Pooling Layers
Work in tandem w/ conv layers to reduce activation maps spatially while maintaining the depth. Most often use the max operation (max-pooling layer) to retain the largest activations and disregard other values.




# Applications of Machine Vision
## Object Detection
Typical pipline:
-   Define ROI
-   Extract features in this region
-   Classify region
## YOLO
You only look once. Looks at the whole image instead of just ROIs like the R-CNNs do. Created using a pretrained CNN model.
## Image Segmentation
### U-Net
Best available method for biomedical images - should look into.

## Transfer Learning 
<https://github.com/christianversloot/machine-learning-articles/blob/main/tutorial-building-a-hot-dog-not-hot-dog-classifier-with-tensorflow-and-keras.md>

<https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog>


# Natural Language Processing
## Preprocessing Natural Language Data
-   _Tokenization_: splitting the document into elements of the language; tokens (words/terms)
-   _Convert to lowercase_ - How does it know how to start a sentence then?
-   _Removing stop words_ : Removing frequently occuring words that dont add meaning (the, at, of, etc)
-   _Removing punctuation_
-   _Stemming_ - reduce words to the stem (housing -> hous)
-   _N-grams_ - Some combination of words are better suited as one token instead of multiple (ex: New York City has an n-gram length of 3 )

# Generative Adversarial Networks (GAN)
Contain two components that are working against each oher (adversaries). Components are a generator and a discrimator. They are trained one at a time to perform better than thier counter part until they reach the limits of their architectural design. Then the discrimator is disregarded and the generator has been trained properly.

# Types of Image Annotation:
-   Image classifcation - detects the presence of the object ex: Does a bananna exist in the photo? Y/N
-   Object Detection - Presence, location, and count of object. Ex: There are 4 banannas
-   Semeantic Segmentation - EX: There is bannana in these pixels.
-   Instance Segmentation- Ex: There are 4 bannansa of this shape size and grade





- Define model (model =)
- Configure Model (Model.compile)
- Train Model (Model.fit)