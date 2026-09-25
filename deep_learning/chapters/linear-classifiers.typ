#import "../lib.typ": *

= Linear classifiers <linear-classifiers>

== What it does

A linear classifier works the opposite way from kNN. Instead of keeping the
training images, it learns one template per class and then deletes the images
afterwards.

The templates live in a weight matrix $W$, one row per class, each row the same
length as an image vector. To classify an image, we multiply it by $W$:

#eq[$ f(x, W) = W x $]

Here $x$ is the image as a column of 2,353 numbers (the 2,352 pixel values plus
the 1 from the bias trick), and $W$ has 8 rows and 2,353 columns, one row per
class. Each score is computed by multiplying the image by one row of $W$ pixel
by pixel and adding the results. An image gets a high score for a class when it
has large values where that class's template has large weights.

This gives eight numbers, one score per class, saying how well the image matches
each template. The highest wins. We used the bias trick, adding a constant 1 to
each image vector so the bias sits inside $W$ as an extra column and the formula
stays a single multiplication.

$W$ starts out random. During training, gradient descent adjusts it step by step
to make the loss smaller. The loss is a single number that says how wrong the
model is. Once trained, each row of $W$ can be turned back into an image, so we
can see what the model learned for each class.

== The two loss functions

*Hinge loss (SVM)* wants the correct class to score higher than every wrong class
by a margin of at least 1:

#eq[$ L_i = sum_(j != y_i) max(0, s_j - s_(y_i) + 1) $]

Here $s_j$ is the score of a wrong class and $s_(y_i)$ is the score of the correct class.
For each wrong class we check whether it is at least 1 point below the correct class. If it is
it adds nothing to the loss, as it doesn't need optimization. If it is too close, it adds the difference.
So hinge loss only changes the wrong classes that are too close, therefore
the coreret class wins by a clear margin.

*Softmax loss* instead turns the scores into probabilities summing to 1, then
penalizes the probability given to the correct class:

#eq[$ L_i = -log (e^(s_(y_i)) / (sum_j e^(s_j))) $]

Exponentiating makes every score positive and dividing by the total makes them
sum to 1. The negative log means high probability gives small loss. Unlike hinge,
softmax is never finished: the correct class can always get nearer to 1, so there
is always gradient left. This gives a check before training, since random weights
give all classes about $1/8$, so the loss should start near
$-ln(1/8) approx 2.08$.

*Regularization.* Both losses get an extra term:

#eq[$ L = 1/N sum_i L_i + lambda sum_(k,l) W_(k,l)^2 $]

The first part is the average loss over all $N$ images. The second part adds up
every weight in $W$ squared, so it gets bigger when the weights are large.
Training tries to make the total small, so the model has to both classify well
and keep its weights small. This stops it from relying on a few pixels and
makes it use more of the image. $lambda$ decides how strong this punishment is.

Hyperparameters are the settings we choose ourselves are the learning rate, the
regularization strength $lambda$, the batch size and the number of epochs. The
learning rate decides how big each update step is: too small and training is
slow, too large and it jumps past the best values.

== Implementation

#codeblock(caption: [Hinge loss])[
```python
margins = np.maximum(0, scores - correct_class_scores + 1)
margins[np.arange(num_train), y] = 0
loss = np.sum(margins) / num_train + reg * np.sum(W * W)
```
]

The first line computes the hinge loss for every class at once. For each class
it subtracts the correct class's score and adds 1, and `np.maximum(0, ...)` turns
any negative result into 0. The second line sets the correct class's own value
to 0, since the formula only looks at the wrong classes. Without it, every image
would get an extra 1 added to its loss. The third line takes the average over
all images and adds the regularization term described above.

#codeblock(caption: [Softmax loss])[
```python
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
loss = np.sum(-np.log(probs[np.arange(num_train), y])) / num_train + reg * np.sum(W * W)
```
]

The first line turns the scores into probabilities by dividing each one by the total for that image, so they add up to 1. 
The second line picks out the probability of the correct class and takes the negative log, so a high probability gives a small loss and a low probability gives a bigger one. 
It then averages over all images and adds the regularization term.

Both functions also return the gradient, which tells gradient descent which way
to adjust $W$. A single `LinearClassifier` class can use either loss.

#choice(title: [Design choice: training in mini-batches])[
  Instead of going through all 5,000 images before updating $W$ once, we split
  them into smaller batches with `np.array_split` and update $W$ after each
  batch. This means the model improves many times during each pass through the
  data instead of only once at the end.
]

== Tuning and results

We ran a grid of 3 learning rates × 3 regularization strengths for each loss, 30
epochs each. An epoch is one full pass over the training data.

#fig(
  caption: [Validation accuracy across the grid, one line per regularization
    strength.],
)[#image("../figures/linear-tuning.png", width: 71%)]

Accuracy rises with learning rate for both losses, and the weakest
regularization wins everywhere. Both best settings sit at the edge of the grid
rather than inside it, which is a sign the grid was too narrow.

#fig(
  caption: [Training loss per epoch for the best setting of each loss function.],
)[#image("../figures/loss-curves.png", width: 62%)]

Both fall steeply over the first few epochs and then flatten. Softmax starts
near 2.08, which is $-ln(1/8)$ and the expected value for random weights, so the
implementation checks out.

#restable(
  columns: (auto, auto, auto, auto),
  caption: [Best setting per loss, chosen on validation and run once on test.],
  [*Model*], [*Best setting*], [*Validation*], [*Test*],
  [SVM], [lr = 5e-7, reg = 1e3], [75.0%], [71.4%],
  [Softmax], [lr = 1e-6, reg = 1e3], [70.2%], [66.8%],
)

Both land below kNN's 75.2%, which is not what we expected going in, since the
linear classifiers actually learn from the data while kNN only compares pixels.

#fig(
  caption: [Confusion matrix for the best SVM on the test set.],
)[#image("../figures/linear-confusion-svm.png", width: 48%)]

The same pattern as kNN: platelets are nearly perfect and immature granulocytes
absorb errors from several other classes.

#fig(
  caption: [Learned SVM weights, one per class, reshaped to 28×28×3.],
)[#image("../figures/weights-trained.png", width: 78%)]

Faint blobs roughly the shape and color of each cell type are visible, which is
the per-class template the model learned. A randomly initialized matrix shows
only noise by comparison.



== Discussion

SVM came out ahead of softmax, 71.4% against 66.8% on the test set, and ahead on
balanced accuracy by a similar margin.

The more interesting result is that both lose to kNN. Learning a template per
class did not beat simply comparing images to stored examples. One linear
template per class is a strong constraint: every image of a class has to match
one average picture of it. kNN has no such constraint, since it can match any
individual training image, so classes that vary in appearance are easier for it
to handle.

Both winning settings used the highest learning rate and the weakest
regularization we tried. When the best value is at the edge of the grid, the
real optimum is probably outside it, so these results are likely a lower bound.
Training and validation accuracy were also close for both models, so they are
not overfitting. This points at the grid and the amount of training data as the
things to improve, rather than the regularization.
