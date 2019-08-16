Binary classification
---------------------

-   How to do a logistic regression to nake a binary classifier. For
    example, to know if the current image contains a cat or not.

-   Notation:

1.  (*x*, *y*), *x* ∈ *R*<sup>*n*<sub>*x*</sub></sup>, *y* ∈ {0, 1}

2.  m training samples:
    {(*x*<sup>(1)</sup>, *y*<sup>(1)</sup>),(*x*<sup>(2)</sup>, *y*<sup>(2)</sup>),…,(*x*<sup>(*m*)</sup>, *y*<sup>(*m*)</sup>)}

3.  *x* ∈ *R*<sup>*m* \* *n*<sub>*x*</sub></sup>,
    *y* ∈ *R*<sup>*m* \* 1</sup>

### Logistic regression

-   An algorithm is used for the classification algorithm of 2 classes

-   Equation: $\\hat{y}=\\sigma(w^{T}x+b)$, where
    $\\sigma(z)=\\frac{1}{1+e^{-z}}$

1.  If *z* is large, $\\sigma(z)\\approx\\frac{1}{1+0}=1$

2.  If *z* is large, $\\sigma(z)\\approx\\frac{1}{1+big\\ \\\#}=0$

-   Parameters: *w* (*n*<sub>*x*</sub> dimension vector) and *b*

### Logistic regression cost function

-   Equation: $\\hat{y}=\\sigma(w^{T}x+b)$, where
    $\\sigma(z)=\\frac{1}{1+e^{-z}}$, given
    {(*x*<sup>(1)</sup>, *y*<sup>(1)</sup>),(*x*<sup>(2)</sup>, *y*<sup>(2)</sup>),…,(*x*<sup>(*m*)</sup>, *y*<sup>(*m*)</sup>)},
    want $\\hat{y}^{(i)}\\approx y^{(i)}$

-   Loss (error) function: the error for a single training example

1.  Square loss: $L(\\hat{y},y)=\\frac{1}{2}(\\hat{y}-y)^{2}$

2.  Logistic loss:
    $L(\\hat{y},y)=-(y\\log\\hat{y}+(1-y)\\log(1-\\hat{y}))$

-   If *y* = 1, $L(\\hat{y},y)=-\\log\\hat{y}$, want $\\log\\hat{y}$
    large, want $\\hat{y}$ large (approach to 1).

-   If *y* = 0, $L(\\hat{y},y)=-\\log(1-\\hat{y})$, want
    $\\log(1-\\hat{y})$ large, want $\\hat{y}$ small (approach to 0).

-   Cost function: the average of the loss function of the entire
    training set.
    $$
    J(w,b)=\\frac{1}{m}\\sum\_{i=1}^{m}\\bigg(-(y\\log\\hat{y}+(1-y)\\log(1-\\hat{y}))\\bigg)
    $$

### Gradient descent

Want to find (*w*, *b*) that minimize *J*(*w*, *b*) (convex function
(**a single big bowl**))

-   Repeat
    $$
    \\begin{aligned}
    &w:w-\\alpha\\frac{\\partial J(w,b)}{\\partial w}\\\\
    &b:b-\\alpha\\frac{\\partial J(w,b)}{\\partial b}\\\\
    \\end{aligned}
    $$
     where *α* is learning rate. When the slope is negative, then *w*
    increases and when the slope is positive, then *w* decreases.

### Derivatives

The derivative is the slope and slope is the difference in different
points in the function, that's why the derivative is a function.

Computation graph
-----------------

-   A computation graph that organizes the computation **(forward) from
    left to right**.

![](screenshot/4.png)

-   A computation graph that organizes derivative (including chain rule)
    \*\* backward (from right to left)\*\*.

### Logistic regression gradient descent (1 training point)

-   Logistic regression recap
    $$
    \\begin{aligned}
    &z=w^{T}x+b\\\\
    &\\hat{y}=a=\\sigma(z)\\\\
    &L(a,y)=-(y\\log(a)+(1-y)\\log(1-a))
    \\end{aligned}
    $$
     ![](screenshot/6.png)

### Gradient descent on *m* examples

*J* = 0, *d**w*<sub>1</sub> = 0, *d**w*<sub>2</sub> = 0, *d**b* = 0

$$
\\begin{aligned}
&For\\ i=1\\ to\\ m:\\\\
&\\quad z^{(i)}=w^{T}x^{(i)}+b\\\\
&\\quad a^{(i)}=\\sigma(z^{(i)})\\\\
&\\quad J+=-(y^{(i)}\\log(a^{(i)})+(1-y^{(i)})\\log(1-a^{(i)}))\\\\
&\\quad dz^{(i)}=a^{(i)}-y^{(i)}\\\\
&\\quad dw\_{1}+=x\_{1}^{(i)}dz^{(i)}\\\\
&\\quad dw\_{2}+=x\_{2}^{(i)}dz^{(i)}\\\\
&\\quad db+=dz^{(i)}\\\\
&J/=m; dw\_{1}/=m; dw\_{2}/=m; db/=m;
\\end{aligned}
$$

Vectorization
-------------

-   Deep learning shines when the dataset is big. However, for loops
    will make you wait a lot for a result. That's why we need
    vectorization to get rid of some of our for loops.

-   NumPy library `dot` function is using vectorization by default.

-   The vectorization can be done on CPU or GPU thought the SIMD
    operation. But it's faster on GPU.

-   Whenever possible, avoid for loops.

-   Most of the NumPy library methods are vectorized version.

### Vectorizing logistic regression

-   Input: *x* : (*n*<sub>*x*</sub>, *m*),
    *y* : (*n*<sub>*y*</sub>, *m*)
    $$
    \\begin{aligned}
    &z=np.dot(w.T, x) +b\\\\
    &a=1/(1+np.exp(-z))\\\\
    &dz=a-y\\\\
    &dw=np.dot(x,dz.T)/m\\\\
    &db=dz.sum()/m\\\\
    \\end{aligned}
    $$

### Notes on Python and NumPy

-   In NumPy, *o**b**j*.*s**u**m*(*a**x**i**s* = 0) sums the columns
    while *o**b**j*.*s**u**m*(*a**x**i**s* = 1) sums the rows.

-   In NumPy, *o**b**j*.*s**h**a**p**e*(1, 4) changes the shape of the
    matrix by broadcasting the values

-   Reshape is cheap in calculations so put it everywhere you're not
    sure about the calculation

-   Broadcasting works when you do a matrix operation with matrices that
    don't match for the operation; in this case, NumPy automatically
    makes the shapes ready for the operation by broadcasting the values.

-   A general principle of broadcasting. If you have an (*m*, *n*)
    matrix and you add ( + ) or subtract ( − ) or multiply ( \* ) or
    divide (/) with a (1, *n*) matrix, then this will copy it *m* times
    into an (*m*, *n*) matrix. The same with if you use those operations
    with a (*m*, 1) matrix, then this will copy it *n* times into
    (*m*, *n*) matrix. And then apply the addition, subtraction, and
    multiplication of division element-wise.

-   Some tricks to eliminate all the strange bugs in the code:

1.  If you didn't specify the shape of a vector, it would take shape of
    (*m*, ) and the transpose operation won't work. You have to reshape
    it to (*m*, 1)

2.  Try not to use the rank one matrix a NN.

3.  Don't hesitate to use `assert(a.shape==(5,1))` to check if your
    matrix shape is the required one.

4.  If you've found a rank one matrix try to run reshape on it.
