#import "../lib.typ": *

== Neural network

// The 1.3 handout has not been downloaded yet. The assignment description
// confirms the third method is a neural network ("loss function (linear models +
// Neural network)"), so the headings below mirror 1.1 and 1.2 to keep the report
// consistent. Adjust once the actual task is known.

=== Theoretical description

#todo[Architecture: layers, widths, activation functions. Why a non-linear model
can represent decision boundaries the linear classifiers cannot.]

=== Optimization objective

#todo[Loss function used, and how backpropagation differs from the single-layer
gradient computation in 1.2.]

=== Hyperparameters

#todo[Hidden layer size, learning rate, regularization, epochs, batch size.]

=== Implementation

#codeblock(caption: [#todo[?]], explain: [#todo[?]])[
```python
# code excerpt
```
]

=== Training process

#todo[Tuning grid and selection procedure. Keep it comparable to 1.1 and 1.2 so
the three methods can be compared fairly.]

=== Results

#todo[Validation and test accuracy, plus a confusion matrix.]

=== Discussion

#todo[Compare against the linear classifiers. If the network beats them, say what
non-linearity bought. If it does not, say why, since with 5,000 training images
overfitting is a plausible explanation.]
