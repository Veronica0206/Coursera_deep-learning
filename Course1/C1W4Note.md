Deep L-layer neural network
---------------------------

![](https://github.com/Veronica0206/Coursera_deep-learning/blob/master/Course1/screenshot/11.PNG)

### Deep neural network notation

![](https://github.com/Veronica0206/Coursera_deep-learning/blob/master/Course1/screenshot/12.png)

-   *l* = 4 (\# of layers);

-   *n*<sup>\[*l*\]</sup> (\# of units in layer *l*):
    *n*<sup>\[0\]</sup> = 3, *n*<sup>\[1\]</sup> = 5,
    *n*<sup>\[2\]</sup> = 5, *n*<sup>\[3\]</sup> = 3,
    *n*<sup>\[4\]</sup> = 1

-   *a*<sup>\[*l*\]</sup> (activations in layer *l*):
    *a*<sup>\[*l*\]</sup> = *g*<sup>\[*l*\]</sup>(*z*<sup>\[*l*\]</sup>)

-   *w*<sup>\[*l*\]</sup>, *b*<sup>\[*l*\]</sup>: weights for
    *z*<sup>\[*l*\]</sup>

### Forward propagation in a deef network

$$
\\begin{aligned}
&for\\ l\\ in\\ 1:L:\\\\
&\\quad z^{\[l\]}=w^{\[l\]}a^{\[l-1\]}+b^{\[l\]}\\\\
&\\quad a^{\[l-1\]}=g^{\[l-1\]}(z^{\[l-1\]})
\\end{aligned}
$$

### Getting your matrix dimensions right

-   Dimension of *w*<sup>\[*l*\]</sup> is
    (*n*<sup>\[*l*\]</sup>, *n*<sup>\[*l* − 1\]</sup>), can be thought
    from right to left

-   Dimension of *b*<sup>\[*l*\]</sup> is (*n*<sup>\[*l*\]</sup>, 1)

-   *d**w*<sup>\[*l*\]</sup> has the same shape as
    *w*<sup>\[*l*\]</sup>, while *d**b*<sup>\[*l*\]</sup> has the same
    shape as *b*<sup>\[*l*\]</sup>

-   Dimension of *z*<sup>\[*l*\]</sup>, *A*<sup>\[*l*\]</sup>,
    *d**z*<sup>\[*l*\]</sup>, *d**A*<sup>\[*l*\]</sup> is
    (*n*<sup>\[*l*\]</sup>, *m*)

### Why deep representation?

-   Deep NN makes relation with data from simpler to complex. In each
    layer, it tries to make a relation with the previous layer.

![](https://github.com/Veronica0206/Coursera_deep-learning/blob/master/Course1/screenshot/13.png)

1.  Face recognition application: image→edges→face parts→faces→desired
    face

2.  Audio recognition application: audio→low level sound features like
    "sss" or "bb"→phonemes→words→sentences

**Neural researchers think that deep neural networks "thinks" like
brains**

-   Circuit theory and deep learning 

![](https://github.com/Veronica0206/Coursera_deep-learning/blob/master/Course1/screenshot/14.png)

-   When starting on an application, don't start directly by dozens of
    hidden layers. Try the simplest solutions (e.g., logistic
    regression), then try the shallow neural network and so on.

Building blocks of deep neural networks
---------------------------------------

### Forward and backward functions

![](https://github.com/Veronica0206/Coursera_deep-learning/blob/master/Course1/screenshot/15.png)

![](https://github.com/Veronica0206/Coursera_deep-learning/blob/master/Course1/screenshot/16.png)

### Forward and backward propagation

-   forward propagation
    $$
    \\begin{aligned}
    &Input\\ a^{\[l-1\]}\\\\
    &\\quad z^{\[l\]}=w^{\[l\]}a^{\[l-1\]}+b^{\[l\]}\\\\
    &\\quad a^{\[l\]}=g^{\[l\]}(z^{\[l\]})\\\\
    &Output\\ a^{\[l\]}\\\\
    &Cache\\ z^{\[l\]}
    \\end{aligned}
    $$

-   backward propagation
    $$
    \\begin{aligned}
    &Input\\ da^{\[l\]}, Caches\\\\
    &\\quad dz^{\[l\]}=da^{\[l\]}\*g^{'\[l\]}(z^{\[l\]})\\\\
    &\\quad dw^{\[l\]}=np.dot(dz^{\[l\]}, a^{\[l-1\]})/m\\\\
    &\\quad db^{\[l\]}=np.sum(z^{\[l\]})/m\\\\
    &\\quad da^{\[l-1\]}=np.dot(w^{\[l\]},dz^{\[l\]})\\\\
    &Output\\ da^{\[l-1\]}, dw^{\[l\]}, db^{\[l\]}
    \\end{aligned}
    $$

Parameters vs. hyperparameters
------------------------------

-   Main parameters of the NN is `w` and `b`

-   Hyperparameters (parameters that control the algorithm) are like:
    learning rate, number of iteration, number of hidden layers `L`,
    number of hidden units `n`, choice of activation functions

-   Have to try values of hyperparameters

-   In the earlier days of ML, the learning rate was often called a
    parameter, but it is a hyperparameter

-   On the next course, we will see how to optimize hyperparameters

What does this have to do with the brain
----------------------------------------

![](https://github.com/Veronica0206/Coursera_deep-learning/blob/master/Course1/screenshot/17.png)
