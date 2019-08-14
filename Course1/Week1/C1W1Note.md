Neural Network (NN)?
--------------------

### Single neuron, e.g. ReLu (rectified linear unit)

![](https://github.com/Veronica0206/Coursera_deep-learning/blob/master/Course1/Week1/screenshot/1.PNG)
The figure shows the simplest possible neural networks is to define
*f*(*x*) as a single "neuron" in the network where
*f*(*x*)=max(*a**x* + *b*, 0), for some coefficients *a*, *b*. What
*f*(*x*) does is return a single value: (*a**x* + *b*) or zero, which
even is greater.

### A more complex neural network ("stack" the single neuron output as input into the next neuron)

![](https://github.com/Veronica0206/Coursera_deep-learning/blob/master/Course1/Week1/screenshot/2.PNG)

Here, the input to a neural network is a set of input features (**input
layers**) *x*<sub>1</sub> (size), *x*<sub>2</sub> (\# bedrooms),
*x*<sub>3</sub> (zip code), *x*<sub>4</sub> (wealth). We connect these
four features to three neurons (**hidden layers**). These three
"internal" neurons are called *hidden units*. The goal for the neural
network is to automatically determine three relevant features such that
three features predict the price of a house (**output layers**).

**The only thing we must provide to the neural network is a sufficient
number of training examples (*x*<sup>(*i*)</sup>, *y*<sup>(*i*)</sup>),
and the neural networks performed as a black box.**

### Activation fuction (which is in general non-linear function)

-   Logistic/sigmoid function: $g(z)=\\frac{1}{1+e^{-z}}$

-   ReLU: *g*(*z*)=max(*z*, 0)

-   tanh: $g(z)=\\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$

Supervised learning with neural networks
----------------------------------------

-   Different types of neural networks for supervised learning which
    includes:

1.  Convolutional neural networks (CNN) (useful in computer vision)

2.  Recurrent neural networks (RNN) (useful in speech recognition of
    NLP)

3.  Standard NN (useful for structured data)

4.  Hybrid/custom NN or a collection of NNs types

-   Structured data is like the databases and tables

-   Unstructured data is like images, video, audio, and text

-   Structured data gives more money because companies rely on
    predictions on its big data

Why is deep learning taking off (3 reasons)?
--------------------------------------------

-   Data
![](https://github.com/Veronica0206/Coursera_deep-learning/blob/master/Course1/Week1/screenshot/3.PNG)

-   Computation: GPUs, powerful CPUs, distributed computing, ASICs

-   Algorithm
