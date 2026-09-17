#import "../lib.typ": *

= Linear classifiers <linear-classifiers>

== What it does

A linear classifier works the opposite way from kNN. Instead of keeping the
training images, it learns one template per class and then discards the data.

The templates live in a weight matrix $W$, one row per class, each row the same
length as an image vector. Classifying means multiplying:

#eq[$ f(x, W) = W x $]

This gives eight numbers, one score per class, saying how well the image matches
each template. The highest wins. We used the bias trick, adding a constant 1 to
each image vector so the bias sits inside $W$ as an extra column and the formula
stays a single multiplication.

This model learns. $W$ starts random and gradient descent adjusts it to reduce a
loss function, which is just a number saying how wrong the model currently is.
Training means pushing that number down. Afterwards each row of $W$ can be folded
back into an image and looked at.

== The two loss functions

*Hinge loss (SVM)* wants the correct class to score higher than every wrong class
by a margin of at least 1:

#eq[$ L_i = sum_(j != y_i) max(0, s_j - s_(y_i) + 1) $]

$s_j$ is the score for class $j$ and $y_i$ the correct class. For each wrong
class, work out how far it is from being a margin below the correct one. If it is
already far enough below, `max(0, ...)` makes the term zero. That is the key
behavior: once a wrong class is beaten by enough, hinge loss ignores it and only
pushes on cases still too close.

*Softmax loss* instead turns the scores into probabilities summing to 1, then
penalizes the probability given to the correct class:

#eq[$ L_i = -log (e^(s_(y_i)) / (sum_j e^(s_j))) $]

Exponentiating makes every score positive and dividing by the total makes them
sum to 1. The negative log means high probability gives small loss. Unlike hinge,
softmax is never finished: the correct class can always get nearer to 1, so there
is always gradient left. This gives a check before training, since random weights
give all classes about $1/8$, so the loss should start near
$-ln(1/8) approx 2.08$.

Both add a regularization term:

#eq[$ L = 1/N sum_i L_i + lambda sum_(k,l) W_(k,l)^2 $]

The second part sums every weight squared, so it grows when weights get large.
Since training makes the total small, this discourages large weights and stops the
model latching onto a few pixels. $lambda$ controls how hard it pushes.

The hyperparameters are learning rate, regularization strength, batch size and
epochs. The learning rate sets how big a step each update takes: too small and
training crawls, too large and it overshoots.

== Implementation

#codeblock(
  caption: [Hinge loss],
  explain: [Line 1 does the margin for every class at once, with
    `np.maximum(0, ...)` as the hinge. Line 2 zeroes the correct class's own entry,
    because the formula skips $j = y_i$ and leaving it in would add a constant 1
    per image. Line 3 averages over the batch and adds regularization.],
)[
```python
margins = np.maximum(0, scores - correct_class_scores + 1)
margins[np.arange(num_train), y] = 0
loss = np.sum(margins) / num_train + reg * np.sum(W * W)
```
]

#codeblock(
  caption: [Softmax loss],
  explain: [Line 1 divides each exponentiated score by the row total so the row
    becomes a probability distribution; `keepdims=True` keeps the sum as a column
    so the division lines up. Line 2 picks the probability of the correct class,
    takes the negative log, and averages.],
)[
```python
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
loss = np.sum(-np.log(probs[np.arange(num_train), y])) / num_train + reg * np.sum(W * W)
```
]

Both return the gradient with the loss, since gradient descent needs to know
which way to move. One `LinearClassifier` class handles either.

#choice(title: [Design choice: training in mini-batches])[
  Rather than computing the gradient over all 5,000 images and making one update,
  we split into batches with `np.array_split` and update after each. This uses
  less memory, but the real gain is that one pass gives many small updates instead
  of one large one, so the model improves during the pass rather than only at the
  end.
]

== Tuning and results

We ran a grid of 3 learning rates × 3 regularization strengths for each loss, 30
epochs each. An epoch is one full pass over the training data.

#fig(
  caption: [Validation accuracy across the grid, one line per regularization
    strength.],
  reading: [Accuracy rises with learning rate for both losses, and the weakest
    regularization (1e3) wins everywhere. Both best settings sit at the edge of
    the grid rather than inside it, which is a sign the grid was too narrow.],
)[#image("../figures/linear-tuning.png", width: 71%)]

#fig(
  caption: [Training loss per epoch for the best setting of each loss function.],
  reading: [Both fall steeply over the first few epochs and then flatten. Softmax
    starts near 2.08, which is $-ln(1/8)$ and the expected value for random
    weights, so the implementation checks out.],
)[#image("../figures/loss-curves.png", width: 62%)]

#restable(
  columns: (auto, auto, auto, auto, auto),
  caption: [Best setting per loss, chosen on validation and run once on test.],
  [*Model*],
  [*Best setting*],
  [*Validation*],
  [*Test*],
  [SVM],
  [lr = 1e-6, reg = 1e3],
  [75.8%],
  [70.6%],
  [Softmax],
  [lr = 1e-6, reg = 1e2],
  [71.8%],
  [68.8%],
)

Both land below kNN's 75.2%, which is not what we expected going in.

#fig(
  caption: [Confusion matrix for the best SVM on the test set.],
  reading: [The same pattern as kNN: platelets are nearly perfect and immature
    granulocytes absorb errors from several other classes.],
)[#image("../figures/linear-confusion-svm.png", width: 48%)]

#fig(
  caption: [Learned SVM weights, one per class, reshaped to 28×28×3.],
  reading: [Faint blobs roughly the shape and color of each cell type are
    visible, which is the per-class template the model learned. A randomly
    initialized matrix shows only noise by comparison.],
)[#image("../figures/weights-trained.png", width: 78%)]



== Discussion

SVM came out ahead of softmax, 71.4% against 66.8% on the test set, and ahead on
balanced accuracy by a similar margin.

The more interesting result is that both lose to kNN. Learning a template per
class did not beat simply comparing images to stored examples. One linear
template per class is a strong constraint: every image of a class has to match
one average picture of it. kNN has no such constraint, since it can match any
individual training image, so classes that vary in appearance are easier for it
to handle.

#limitation(title: [The grid was too narrow])[
  Both winning settings sit on the edge of the grid: the weakest regularization
  we tried, and for softmax the highest learning rate we tried. When the best
  value is at the boundary, the real optimum is usually outside it. These numbers
  should be read as a lower bound on what these classifiers can do.
]

Training accuracy came out close to validation accuracy for both, so neither is
overfitting much at this data size and regularization strength. That rules out
too little regularization as the problem, and points at the grid and the amount
of training data instead.
