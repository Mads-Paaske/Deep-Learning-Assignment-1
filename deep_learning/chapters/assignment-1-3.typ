#import "../lib.typ": *

= Neural network <nn>

== What it does

The linear classifier in @linear-classifiers can only separate classes with a
straight boundary. A neural network stacks several linear layers and puts a
non-linear function between them, which lets it bend the boundary.

Our network has three layers: an input layer of 2,352 nodes (one per pixel
value), one hidden layer, and an output layer of 8 nodes (one per class). Data
goes in at the input, gets multiplied by a weight matrix at each step, and comes
out as eight class scores.

The non-linear function is ReLU, which is just `max(0, x)`. It passes positive
values through and flattens everything negative to zero. Without it, stacking
layers would be pointless: multiplying several matrices together gives another
matrix, so the whole network would collapse back into one linear layer. ReLU
between the layers is what stops that collapse, and it is the only reason this
model can do anything the linear classifier cannot.

The output layer has no ReLU. It gives raw scores, which the loss function
handles.

== What gets optimized

The same softmax loss as in @linear-classifiers, plus L2 regularization on every
weight matrix.

The difference is how the gradients are computed. With one weight matrix there is
one gradient. With several layers the error has to be passed backwards through
them using the chain rule, which is backpropagation. The network computes the
loss at the output, works out how much each layer contributed, and adjusts every
weight accordingly.

Hyperparameters: number of layers, nodes per layer, learning rate,
regularization strength, batch size, epochs, and choice of optimizer.

== Implementation

The model is one class, `FullyConnectedNN`, built so the layer sizes are just a
list. Passing `[2352, 500, 8]` gives the network above, and adding another number
would add another hidden layer without changing any other code.

#codeblock(caption: [The forward pass])[
```python
for i in range(1, len(self.layers)):
    W, b = self.params['W' + str(i)], self.params['b' + str(i)]
    Z = np.dot(A, W) + b
    if i != len(self.layers) - 1:
        A = self.relu(Z)
    else:
        A = Z  # No activation in the output layer (raw scores for loss)
    cache['Z' + str(i)] = Z
    cache['A' + str(i)] = A
```
]

Each layer multiplies by its weights and adds the bias. The `if` applies ReLU to
every layer except the last, which is left as raw scores. The `cache` stores the
values at each step, because backpropagation needs them again on the way back.

#codeblock(caption: [The backward pass])[
```python
for i in reversed(range(1, len(self.layers))):
    dZ = dA
    A_prev = cache['A' + str(i - 1)]
    grads['W' + str(i)] = np.dot(A_prev.T, dZ) + self.reg_strength * self.params['W' + str(i)]
    grads['b' + str(i)] = np.sum(dZ, axis=0, keepdims=True)
    if i > 1:
        dA = np.dot(dZ, self.params['W' + str(i)].T) * self.relu_derivative(cache['Z' + str(i - 1)])
```
]

The loop runs in reverse, from the output layer back to the first. Each step
computes the gradient for that layer's weights, adding the regularization term.
The last line passes the error down to the previous layer, multiplying by the
ReLU derivative so that nodes switched off by ReLU get no gradient.

The class also offers momentum and Adam as alternatives to plain SGD. We used
SGD for every result below.

#trap(title: [Two paths in the class that are not correct])[
  The hinge branch of `backward` sets the gradient of the correct class to zero
  instead of minus the number of violated margins, so hinge training would not
  descend properly. The Adam branch of `update_params` updates the weights but
  never the biases.

  Neither affects the results here, because everything below uses softmax loss
  with plain SGD. Both would need fixing before using those options.
]

== Training and results

We searched over hidden layer size and learning rate, 60 epochs each, batch size
64, regularization 0.001.

#fig(caption: [Validation accuracy across the grid.])[
  #image("../figures/nn-tuning.png", width: 60%)
]

Accuracy climbs steeply from learning rate 0.01 to 0.05 and then flattens. The
wider hidden layer is slightly better everywhere, but the learning rate matters
far more than the width.

#restable(
  columns: (auto, auto, auto, auto),
  caption: [The full grid. Training accuracy is included because the gap against
    validation is the interesting part.],
  [*Hidden nodes*], [*Learning rate*], [*Train*], [*Validation*],
  [200], [0.01], [78.2%], [78.0%],
  [200], [0.05], [95.0%], [86.6%],
  [200], [0.1], [99.2%], [86.4%],
  [500], [0.01], [79.6%], [78.2%],
  [500], [0.05], [96.3%], [87.0%],
  [500], [0.1], [99.7%], [*88.0%*],
)

The best setting was 500 hidden nodes at learning rate 0.1, giving 88.0% on
validation, 83.9% on the test set and 81.6% balanced accuracy.

#fig(caption: [Training loss for the best setting.])[
  #image("../figures/nn-loss.png", width: 59%)
]

The loss falls quickly over the first epochs and then flattens out close to
zero, which matches the 99.7% training accuracy.

#fig(caption: [Confusion matrix for the best network on the test set.])[
  #image("../figures/nn-confusion.png", width: 48%)
]

The same shape of errors as the other two methods, but fewer of them. Basophils
and monocytes are still the weakest classes and still lose images to immature
granulocytes.

#fig(caption: [First-layer weights for 16 of the 500 hidden nodes.])[
  #image("../figures/nn-weights.png", width: 78%)
]

Unlike the linear classifier, these are not one template per class. Each hidden
node learns a small piece of structure, and the output layer combines 500 of
them into a class score.

== Discussion

The network is the best of the three methods, at 83.9% against 75.2% for kNN and
70.6% for the linear SVM. The non-linearity bought roughly 13 points over the
linear model that is otherwise closest to it.

That fits the data. Cell types differ in shape and texture rather than in overall
brightness, and a single straight boundary in pixel space cannot capture that. A
hidden layer of 500 ReLU nodes can build many small detectors and combine them.

#limitation(title: [It is overfitting, clearly])[
  The best setting reaches 99.7% on the training data and 88.0% on validation, a
  gap of nearly 12 points. The network has essentially memorized the 5,000
  training images.

  This is not surprising given the size. The first weight matrix alone is
  2,352 × 500, which is over a million weights fitted to 5,000 examples. The
  regularization strength of 0.001 is clearly too weak to hold that back.

  The two settings with learning rate 0.01 show the other side: 78% train and 78%
  validation, no gap at all, but worse on both. Those are underfitting.
]

The grid is also too narrow, in the same way as in @linear-classifiers. The best
setting is the largest hidden layer and the highest learning rate we tried, both
at the edge, so the real optimum is probably outside the range.

Given the overfitting, more training data is the obvious lever. We used 5,000 of
11,959 available images, and a model with this much capacity is exactly the case
where the extra data should help most.