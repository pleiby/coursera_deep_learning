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
		- a classic, ranges betweem 0 and 1.0. Suitable for output layer given binary classification problem
	- tanh: $tanh(z) = \frac{1}{(1+e^{-z}}$$
