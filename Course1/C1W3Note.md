Neural networks overview
------------------------

-   In logistic regression: *x* → *z* → *a* → *L*(*a*, *y*)

-   In neural networks with one hidden layer:
    *x* → *z*<sub>1</sub> → *a*<sub>1</sub> → *z*<sub>2</sub> → *a*<sub>2</sub> → *L*(*a*<sub>2</sub>, *y*)

 ![](https://github.com/Veronica0206/Coursera_deep-learning/blob/master/Course1/screenshot/7.PNG)

Computing a neural network's output
-----------------------------------

### Neural network representation

-   A neural network contains input layer (layer 0), hidden layer(s) and
    output layer. Here we will define the neural networks which have one
    hidden layer (which means we caNNot see that layers in the training
    set)

-   Using **<sup>\[0\]</sup>, **<sup>\[1\]</sup>, **<sup>\[2\]</sup> to
    represent the 0<sup>*t**h*</sup> layer, the 1<sup>*t**h*</sup>
    layer, and the 2<sup>*t**h*</sup> layer.

-   Using **<sub>1</sub><sup>\[1\]</sup>, **<sub>2</sub><sup>\[1\]</sup>
    to represent the first node (neuron) and the second node (neuron) of
    the first layer.

 ![](https://github.com/Veronica0206/Coursera_deep-learning/blob/master/Course1/screenshot/8.PNG)

 ![](https://github.com/Veronica0206/Coursera_deep-learning/blob/master/Course1/screenshot/9.PNG)


-   Shapes of the variables and parameters

1.  *w*<sup>\[1\]</sup> is the matrix of the first hidden layer, and it
    has a shape (\# of hidden neurons, *n*<sub>*x*</sub>)

2.  *b*<sup>\[1\]</sup> is the vector of the first hidden layer, and it
    has a shape (\# of hidden neurons, 1)

3.  *z*<sup>\[1\]</sup> is the result of the equation
    *z*<sup>\[1\]</sup> = *w*<sup>\[1\]</sup> \* *x* + *b*, it has a
    shape (\# of hidden neurons, 1)

4.  *a*<sup>\[1\]</sup> is the result of the equation
    *a*<sup>\[1\]</sup> = *s**i**g**m**o**i**d*(*z*<sup>\[1\]</sup>), it
    has a shape (\# of hidden neurons, 1)

5.  *w*<sup>\[2\]</sup> is the matrix of the second hidden layer, and it
    has a shape (1, \# of hidden neurons)

6.  *b*<sup>\[2\]</sup> is the matrix of the second hidden layer, and it
    has a shape (1, 1)

7.  *z*<sup>\[2\]</sup> is the result of the equation
    *z*<sup>\[2\]</sup> = *w*<sup>\[2\]</sup> \* *a*<sup>\[1\]</sup> + *b*<sup>\[2\]</sup>,
    it has a shapre (1, 1)

8.  *a*<sup>\[2\]</sup> is the result of the equation
    *a*<sup>\[2\]</sup> = *s**i**g**m**o**i**d*(*z*<sup>\[2\]</sup>), it
    has a shapre (1, 1)

### Vectorizing across multiple examples

-   Pseudo code of the for loop for the forward propagation for the 2
    layers NN:

$$
\\begin{aligned}
&for\\ i=1\\ to\\ m:\\\\
&\\quad z^{\[1\](i)}=w^{\[1\]}x^{(i)}+b^{\[1\]}\\\\
&\\quad a^{\[1\](i)}=sigmoid(z^{\[1\](i)})\\\\
&\\quad z^{\[2\](i)}=w^{\[2\]}a^{\[1\](i)}+b^{\[2\]}\\\\
&\\quad a^{\[2\](i)}=sigmoid(z^{\[2\](i)})\\\\
\\end{aligned}
$$

-   Pseudo code of the vectorization for the forward propagation for the
    2 layers NN (**the number of columns is always *m***):

$$
\\begin{aligned}
&z^{\[1\]}=w^{\[1\]}x+b^{\[1\]}\\\\
&a^{\[1\]}=sigmoid(z^{\[1\]})\\\\
&z^{\[2\]}=w^{\[2\]}a^{\[1\]}+b^{\[2\]}\\\\
&a^{\[2\]}=sigmoid(z^{\[2\]})\\\\
\\end{aligned}
$$

### Activation functions

#### sigmoid() (use for output layer in the case with a binary outcome):

-   Sigmoid() function: $A=g(z)=\\frac{1}{1+\\exp(-z)}$

-   Derivation:
    $g^{'}(z)=\\frac{\\exp(-z)}{(1+\\exp(-z))^{2}}=g(z)\*(1-g(z))$

-   sigmoid() can lead us to gradient problem where the updates are low

-   sigmoid() activation function range is \[0, 1\]

#### tanh()

-   tanh() function:
    $A=g(z)=\\frac{\\exp(z)-\\exp(-z)}{\\exp(z)+\\exp(-z)}$

-   Derivation: *g*<sup>′</sup>(*z*)=1 − *g*(*z*)<sup>2</sup>

-   tanh() activation function range is \[ − 1, 1\]

-   It turns out that the tanh() activation usually works better than
    sigmoid() activation function for hidden units because the mean of
    its output is closer to zero, and so it centers the data better for
    the next layer.

-   The disadvantage of sigmoid/tanh: if the input is too small or too
    high, the slope will be near zero, which will cause us the gradient
    decent problem.

#### ReLU

-   ReLU function: *A* = max(0, *z*), meaning that if *z* is negative
    the slope is 0 and if *z* is positive the slope remains linear.

-   Derivation:
    $g^{'}(z)=\\{\\begin{aligned}0\\quad if\\ z&lt;0\\\\1\\quad if\\ z\\ge0\\end{aligned}$

-   For the case with a binary output, use the sigmoid() as the output
    activation and ReLU as activation functions for other layers.

#### Leaky ReLU

-   Leaky ReLU function: *A* = max(0.01*z*, *z*)

-   Derivation:
    $g^{'}(z)=\\{\\begin{aligned}0.01\\quad if\\ z&lt;0\\\\1\\quad if\\ z\\ge0\\end{aligned}$

-   Leaky ReLU activation function is a modified version of ReLU.

**There are no guidelines for making a choice when conducting NN (\# of
hidden layers, \# of neurons in each hidden layer, learning rate,
activation functions, etc.).**

#### Why do we need nonlinear activation functions:

-   If we removed the activation function from our algorithm, we have
    sorts of the linear activation function.

-   A linear activation function will output a linear combination of
    input

-   We may use a linear activation function in one place--in the output
    layer if the output is real numbers (regression problem). But even
    in this case if the output value is non-negative, we could use ReLU
    instead.

Gradient descent for the neural networks (backward propagation of the neural networks)
--------------------------------------------------------------------------------------

 ![](https://github.com/Veronica0206/Coursera_deep-learning/blob/master/Course1/screenshot/10.PNG)


### Algorithm

$\\begin{aligned}
&Repeat:\\\\
&\\quad compute\\ predictions\\ (\\hat{y}^{(i)},i=1,2,\\dots,m)\\\\
&\\quad get\\ derivatives: dw\_{1},db\_{1},dw\_{2},db\_{2}\\\\
&\\quad update:\\\\
&\\quad\\quad w^{\[1\]}=w^{\[1\]}-\\alpha\*dw^{\[1\]}\\\\
&\\quad\\quad b^{\[1\]}=b^{\[1\]}-\\alpha\*db^{\[1\]}\\\\
&\\quad\\quad w^{\[2\]}=w^{\[2\]}-\\alpha\*dw^{\[2\]}\\\\
&\\quad\\quad b^{\[2\]}=b^{\[2\]}-\\alpha\*db^{\[2\]}
\\end{aligned}$

where
$\\begin{aligned}
&dz^{\[2\]}=a^{\[2\]}-y\\\\
&dw^{\[2\]}=dz^{\[2\]}\*a^{\[1\]T}\\\\
&db^{\[2\]}=dz^{\[2\]}\\\\
&dz^{\[1\]}=w^{\[2\]T}dz^{\[2\]}\*a^{\[1\]'}z^{\[1\]}\\\\
&dw^{\[1\]}=dz^{\[1\]}\*a^{\[0\]T}=dz^{\[1\]}\*x^{T}\\\\
&db^{\[1\]}=dz^{\[1\]}
\\end{aligned}$

### Random initialization

-   In logistic regression, it is not important to initialize the
    weights randomly, while in NN we have to initialize them randomly.

-   If we initialize all the weights with zeros in NN, it won't work
    (though it's fine to initializing bias with zero)

1.  all hidden units will be completely identical (symmetric)--compute
    precisely the same function

2.  on each gradient descent iteration, all the hidden units will always
    update the same

-   We need small values because in sigmoid(or tanh), for example, if
    the weight is too large you are more likely to end up even at the
    very start of training with very large values of *z*, which causes
    the tanh() or sigmoid() activation function to be saturated, thus
    slowing down the learning process. If you don't have any sigmoid()
    or tanh() activation functions throughout your NN, this is less of
    an issue.
