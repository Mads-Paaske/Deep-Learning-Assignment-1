#import "../lib.typ": *

== Linear classifiers

=== Theoretical description

A linear classifier scores each class directly from the image vector and picks
the highest-scoring class. For an image $x$ and a weight matrix $W$ with one row
per class:

#eq[$ f(x, W) = W x $]

The bias trick is used here: a constant 1 is appended to every image vector, so
the bias term folds into $W$ as an extra column and does not need to be carried
separately.

Unlike kNN, this model actually learns. $W$ starts out random and is updated by
gradient descent to minimize a loss function on the training data. Each row of
the trained $W$ works as a template for its class, and the weights say which
pixel regions and color patterns separate that cell type from the others.

=== Optimization objective

Two loss functions were implemented.

==== SVM (hinge) loss

Hinge loss penalises a class score for failing to fall at least a margin of 1
below the correct class's score. For image $i$ with correct label $y_i$ and
scores $s$:

#eq[$ L_i = sum_(j != y_i) max(0, s_j - s_(y_i) + 1) $]

Once a wrong class sits more than a margin below the correct one, it stops
contributing to the loss entirely. The loss only cares about getting the
ordering right by a margin, not about the exact values of the scores.

==== Softmax (cross-entropy) loss

Softmax loss turns the scores into a probability distribution by exponentiating
and normalizing, then penalises the negative log-probability assigned to the
correct class:

#eq[$ L_i = -log (e^(s_(y_i)) / (sum_j e^(s_j))) $]

Unlike hinge loss, this is never fully satisfied. The probability of the correct
class can always get closer to 1, so there is always gradient pressure to
improve, however well the model is already doing.

At random initialization every class receives roughly equal probability $1/k$,
so the loss starts near $-ln(1/8) approx 2.08$. This is a useful check that the
implementation is correct before training begins.

==== Regularization

Both losses add an L2 regularization term over the weights:

#eq[$ L = 1/N sum_i L_i + lambda sum_(k,l) W_(k,l)^2 $]

This discourages large weights and reduces overfitting, and $lambda$ is one of
the tuned hyperparameters.

=== Hyperparameters

Learning rate, regularization strength, batch size, and number of training
epochs.

=== Implementation

#codeblock(
  caption: [Hinge loss],
  explain: [`scores - correct_class_scores + 1` computes the margin violation
    for every class at once, and `np.maximum(0, ...)` is the hinge itself,
    zeroing out any class already beyond the margin. The second line sets the
    correct class's own entry to zero, since the sum in the formula runs over
    $j != y_i$ and would otherwise contribute a constant 1 per image. The final
    line averages over the batch and adds the regularization term.],
)[
```python
margins = np.maximum(0, scores - correct_class_scores + 1)
margins[np.arange(num_train), y] = 0
loss = np.sum(margins) / num_train + reg * np.sum(W * W)
```
]

#codeblock(
  caption: [Softmax loss],
  explain: [The first line is the normalization step, dividing each exponentiated
    score by the row sum so each row becomes a probability distribution.
    `keepdims=True` preserves the column shape so the division broadcasts
    correctly. The second line picks out the probability assigned to the correct
    class for each image, takes the negative log, and averages.],
)[
```python
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
loss = np.sum(-np.log(probs[np.arange(num_train), y])) / num_train + reg * np.sum(W * W)
```
]

Both loss functions return a gradient alongside the loss value. A single
`LinearClassifier` class handles training for either, dispatching to one loss or
the other.

#choice(title: [Design choice: mini-batch gradient descent])[
  Training runs in batches, using `np.array_split`, rather than computing the
  gradient over all 5,000 images at once.

  This is partly a memory consideration, but the more important reason is that it
  makes several small weight updates per epoch instead of one large one. More
  frequent updates mean the model makes progress within a single pass over the
  data rather than only at the end of it.
]

=== Training process

Grid search over 3 learning rates × 3 regularization strengths, for each of the
two loss functions, 30 epochs each. The combination scoring highest on the
validation set was selected.

#fig(
  caption: [#todo[Validation accuracy across the learning rate and
    regularization grid.]],
  reading: [#todo[Say which region of the grid works and how performance falls
    off at the extremes.]],
)[#todo[figures/linear-tuning.png]]

#fig(
  caption: [#todo[Training loss over iterations for both loss functions.]],
  reading: [#todo[Comment on convergence speed and any instability.]],
)[#todo[figures/loss-curves.png]]

=== Results

#restable(
  columns: (auto, auto, auto, auto),
  caption: [Best configuration for each loss function, selected on validation and
    evaluated once on the test set.],
)[*Model*][*Best configuration*][*Validation accuracy*][*Test accuracy*]
[SVM][lr = 1e-6, reg = 1e3][75.8%][70.6%]
[Softmax][lr = 1e-6, reg = 1e2][71.8%][68.8%]

Both are a large step up from kNN's 17.6% test accuracy.

=== Visualizing the learned weights

#fig(
  caption: [#todo[Learned weights for each class, reshaped back into 28×28×3
    images, next to a randomly initialized weight matrix for comparison.]],
  reading: [The trained weights show faint class-shaped blobs, where the randomly
    initialized matrix shows only visual noise. This is direct evidence that the
    model has learned per-class templates rather than arbitrary values.],
)[#todo[figures/weights.png]]

=== Discussion

The jump from kNN to linear classifiers follows from what each model does. kNN
compares raw pixels directly and treats all 2,352 values as equally informative.
A linear classifier learns a weighted template per class during training, so it
can find which pixel regions and color patterns actually separate the cell types
and give less weight to the rest. The weight images above show this directly.

SVM slightly outperformed softmax here, 70.6% against 68.8% on the test set.

#limitation(title: [How much to read into the SVM and softmax gap])[
  The difference is under two percentage points on a 1,000-image test set, which
  is within the range that could be explained by sampling noise alone. It is
  worth being careful about building an argument on top of it.

  #todo[If you want to claim the difference is real rather than noise, the
  honest way is to re-run both with several random seeds and see whether the
  ordering holds. Otherwise report it as a small difference and leave it there.]
]

Training accuracy was close to validation accuracy for both models, which
suggests they are not overfitting much at this data size and regularization
strength. That points towards there being room to improve by training longer or
searching a wider hyperparameter grid, rather than by adding more
regularization.
