#%%
# ### expt ml basic.py - Experiments on basic concepts

# #%%
import numpy as np

# %%
def sigmoid(z):
    return 1 / (1 + exp(z))


#%%
z = np.dot(w, x) + b
a = sigmoid(z)  # y_hat for logistic regression

#%%
def L(y, y_hat):
    # this Loss function definition only defined for
    #  y in {0,1}, 0 < y_hat < 1.0
    return -np.dot(y, np.log(y_hat)) - np.dot((1 - y), np.log(1 - y_hat))


def C(Y, Y_hat):
    return 1 / len(Y) * L(Y, Y_hat)


#%%
dZ = Y - A

#%%
Y = np.array([1, 0, 1, 1, 0])
Y_hat = np.array([0.999, 0.001, 0.999, 0.999, 0.001])

# %%
np.dot(Y, Y_hat)

#%%
L(Y, Y_hat)
# %%
np.log(x)
# %%
