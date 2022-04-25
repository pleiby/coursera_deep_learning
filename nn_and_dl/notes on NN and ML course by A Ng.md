## notes on neural net and machine learning course by Andrew Ng

### Week 2
- ML for Binary Outcomes
   - m features
   - n (labeled) observations in training set

#### Logistic Regression
- logistic (sigmoid) fun to represent probability of binary outcome being one value or another.
- attributes x, 
- computer linearly weighed attributes z = wx + b,
- apply sigmoid activation function
 y-hat  =
$$a(z) = sigma(z) = 1/(1+ e^{-x})$$
- for logistic regression (and single layer), estimated prob
- loss fun for each obs
   $$L(y^{(i)}, \hat y^{(i)} = -y ln(\hat y) - (1-y) ln(1- \hat y)$$
    -Note: minimizing the loss corresponds to maximizing $log(p(y|x))$
$$p(y|x) = \hat y^y (1-\hat y)^{(1-y)}$$

- cost fn for all obs C(X.Y, w, b), to be minimized
$$C = 1/m \sum_{i=1}^m L(y^{(i)})$$
   - note: this gives a maximum likelihood estimation of logistic parameters, assuming obs y_i are i.i.d.


#### Gradient descent
- for cost J minimization, a simple gradient decent step is:
- $$ w = w - \alpha dJ(w)/dw$$
  - Note: technically this should be written as partial derivative for the case of multiple input features, high e the case of minimizing a (one-dimensional) function of a multi-dimensional variable
- learning rate \alpha (to be chosen), degree of updated parameters or size of each step.
- Note
  - The slope indicates the direction to take a step in, for each iteration of gradient descent
  - If the slope is negative then a step is taken in the direction to increase W, and thereby decreased the cost function J
  - Conversely the slope is positive and the step is taken in the negative direction, decreasing W and thereby decreasing the cost function J.
- common machine learning notational convention:
  - dw_i = partial of cost function with respect to weight perameter i
  - db = partial derivative of cost function with respect to offset perameter b.

#### Derivatives Within a Computational Graph
- ForwardProp(-agation): step through the network/computation graph to compute objective fn value
- BackProp(-agation): step backward through network to compute derivatives w.r.t. hidden layer and initial layer weights, by numerical application of the chain rule

#### Activation Functions
- Each node of Neural Net is a linear weighting of input features plus offset ($z = w x + b$) and an activation function ($a = g(x)$, e.g.  $a = \sigma(z)$)
- Non-linear activations functions (on at least some layers) necessary to keep multi-layer/deep neural net from collapsing into single-layer linear system
	>"turns out that if you use a linear activation function or alternatively, if you don't have an activation function, then no matter how many layers your neural
network has, all it's doing is just computing a linear activation function."
- **Alternative Activation Functions**
	- sigmoid: $\sigma(z) = 1/(1+e^{-z})$
		- a classic, ranges betweem 0 and 1.0, monotonically increasing. 1/2 at 0.0
      - Suitable for output layer given binary classification problem.
      - Note that assumption of crossover/midpoint at $z=0$ is not constraining since position/shift parameter is included in offset $b$ used in the construction of $z$ from linearly weighted features: $z = w \cdot x + b$
	- tanh: $tanh(z) = \frac{e^{-z}}{(1+e^{-z})}$
      - ranges betweem -1.0 and +1.0, monotonically increasing; zero at 0.0. 
  - consider scaled and (vertically shifted) sigmoid:
    - $2 \sigma(z) - 1$
    - produces activation variables that are "centered" (around zero), so typically better than sigmoid
  - RELU: Rectified Linear Unit
    - $RELU(z) = \left \{  \substack{0 ~if ~z < 0\\ z ~~if ~z \ge 0} \right .$
    - numerically, introduces nonlinearity but converges more rapidly than sigmoid or tanh
    - however, has problem of zero derivative for $z < 0$, (derivative technically undefined for $z=0$)
  - Leaky RELU: Leaky Rectified Linear Unit
    - $LRELU(z) = \left \{  \substack{\epsilon z ~if ~z < 0\\ z ~~if ~z \ge 0} \right .$ for small $\epsilon > 0$
    - feature of always having positive derivative.

#### Forward-propagation (of variable values to cost) for Deep NN
- Recall basic NN equations for forward-propagation of features leading to Loss $L(a, y)$ for each output node activation $a^{(k)}$ and observations ("labels") $y^{(k)}$, $k \in [0, ... m-1]$
  - $A^{[0]} \equiv X$ is matrix input features, $X_{n_x \times m}$ for $n_x$ input features and $m$ observations in training set.
    - $x^{(k)]$ is $[n_0 \times 1]$ column vector of $n_0$ input features for observation $k$
  - $a_{i}^{[j](k)}$ = activation level variable for $i$ th node/neuron, at layer number $j$, $k$ th observation
  - $z_{i}^{[j](k)}$ = weighted average of previous layer features for $i$ th node/neuron, at layer number $j$, $k$ th observation
    - $z_{i}^{[j](k)} = w_{i}^{[j]} \cdot a_{i}^{[j-1](k)} + b_{i}^{[j]}$
    - $W^{[j]} ~is~ [n_j \times n_{j-1}]$, with a row of weights for each node/neuron at layer $j$, and the number of those weights (columns of $W^{[j]}$) equal to the number of inputs from the previous layer, i.e. number of neurons at layer $j-1$
  - ??? getting a little confused re indices $i,j,k$ and superscripts vs. subscripts here.

#### Back-propagation (of gradient) for Deep NN
  - define shorthand notation for derivatives: $dx \equiv dJ/dx$ for any variable or parameter in the NN, where $J$ is the final cost function
    - $J = \frac{1}{m} \sum_{k=0}^m L(a^{[J](m)}, y^{(m)})$ for $a^{[J]}$ being the final/output layer activations (notation confusing $J$ the number of layers with J=C the cost function)
      - i.e. $a^{[J]} \equiv \hat y$
  - omitting some of the details/indices for layers and nodes:
    - $z = w^T x + b$
    - $a = g(z)$
      - for final output layer in binary classification, $a = \sigma(z) = 1/(1+e^{-z})$
    - $L(a,y) = -y ln(a) - (1-y) ln(1-a)$
      - this is *cross-entropy loss*


### Backup Info:

#### Other Loss Functions
- [Cross-Entropy](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy)
  - Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label. 

    ```python
    def CrossEntropy(yHat, y):
    if y == 1:
      return -log(yHat)
    else:
      return -log(1 - yHat)
    ```

  - for multi-class classification, with $M$ classes, Cross-Entropy Loss is, for each observation $o$
  - $L = −\sum_{c=1}^M y_{o,c} ln(p_{o,c})$
    - (i.e. calculate a separate loss for each possible class label and sum the result.
    - $y_{o,c}$ - binary encoding of whether training observation $o$ is of class $c$ or not (1, 0)
    - $p_{o,c}$ - predicted probability that observation $o$ is of class $c$, $p_{o,c} \equiv a_{0,c}$ in prior notation
- [Hinge](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#hinge)
  - Used for classification.
  ```python
  def Hinge(yHat, y):
    return np.max(0, y - (1-2*y)*yHat)
  ```

- [Huber](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#huber)
  - Typically used for regression. It’s less sensitive to outliers than the MSE as it treats error as square only inside an interval.
  - $L_δ= \left \{ \substack {\frac{1}{2}(y− \hat y)^2 ~if |(y− \hat y)| < δ \\
        δ((y− \hat y ) − \frac{1}{2}δ) ~otherwise }
  \right .$

  - Code
    ```python
    def Huber(yHat, y, delta=1.):
        return np.where(np.abs(y-yHat) < delta,.5*(y-yHat)**2 , delta*(np.abs(y-yHat)-0.5*delta))
    ```

- [Kullback-Leibler](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#kullback-leibler)
- [MAE (L1)](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#mae-l1)
  - Mean Absolute Error, or L1 loss. Excellent overview below [6] and [10].
  - Code
    ```python
    def L1(yHat, y):
        return np.sum(np.absolute(yHat - y)) / y.size
    ```



- [MSE (L2)](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#mse-l2)
  - Mean Squared Error, or L2 loss. 
    ```python
    def MSE(yHat, y):
      return np.sum((yHat - y)**2) / y.size

    def MSE_prime(yHat, y): # derivative w.r.t. yHat
        return yHat - y
    ```

