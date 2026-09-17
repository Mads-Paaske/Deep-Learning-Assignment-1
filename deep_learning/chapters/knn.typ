#import "../lib.typ": *

== k-Nearest Neighbors

=== Theoretical description

k-Nearest Neighbors is one of the simplest and oldest algorithms in machine
learning. It takes an instance-based approach: it memorises the training
examples and makes predictions by comparing a new image against the stored ones
using a similarity measure.

Because the training phase consists only of storing the data, all computation is
deferred to prediction time. Training is effectively instantaneous, while
prediction is comparatively slow. This is why kNN is described as a lazy
learner.

To classify a new example, the algorithm computes the distance from it to every
training instance, finds the k nearest of them, and predicts the class by
majority vote among those k labels.

==== Distance metrics

Two distance metrics were implemented. For two flattened image vectors $a$ and
$b$ of length 2,352, the L1 (Manhattan) distance sums the absolute differences
per pixel:

#eq[$ d_1(a, b) = sum_(p=1)^(2352) abs(a_p - b_p) $]

and the L2 (Euclidean) distance takes the square root of the summed squared
differences:

#eq[$ d_2(a, b) = sqrt(sum_(p=1)^(2352) (a_p - b_p)^2) $]

The choice between them affects which training images count as nearest, and is
therefore treated as a hyperparameter rather than fixed in advance.

=== Optimization objective and hyperparameters

kNN has no trainable parameters and no loss function, so there is no optimization
in the usual sense. Nothing is minimized by gradient descent. The only thing
being chosen is the pair of hyperparameters, selected by brute-force search for
whichever combination maximises validation accuracy.

The two hyperparameters are:

- *k*, the number of neighbors that vote. A small k risks overfitting to noise,
  since a single unusual training image can decide the prediction. A large k
  risks underfitting, since the vote is drawn from a neighborhood wide enough to
  include genuinely different cells.
- *The distance metric*, either L1 or L2.

=== Implementation

The distance calculation is the part worth explaining, because it is the
computational bottleneck of the whole method.

#codeblock(
  caption: [Vectorized L2 distance matrix],
  explain: [This uses the identity
    $norm(a - b)^2 = norm(a)^2 - 2 a dot b + norm(b)^2$. Expanding the square
    this way turns the distance computation into one matrix multiplication
    (`cross_term`) plus two row-wise sums, instead of looping over every
    test-train pair. Broadcasting the column vector `test_sq` against the row
    vector `train_sq` produces the full matrix in a single expression. The
    `np.maximum(..., 0)` guards against small negative values from floating
    point rounding before the square root.],
)[
```python
test_sq = np.sum(X_test ** 2, axis=1).reshape(num_test, 1)
train_sq = np.sum(self.X_train ** 2, axis=1).reshape(1, num_train)
cross_term = X_test @ self.X_train.T
dists = np.sqrt(np.maximum(test_sq - 2 * cross_term + train_sq, 0))
```
]

L1 has no equivalent algebraic shortcut, because the absolute value does not
expand into dot products. That implementation loops over the test images only,
not over every test-train pair, which avoids building an array of shape
(num_test × num_train × 2352) that would exhaust memory.

#codeblock(
  caption: [Majority vote over the k nearest labels],
  explain: [`argsort` orders the training images by distance and the slice takes
    the k closest. `np.unique` with `return_counts` tallies the labels among
    them, and `argmax` picks the most frequent. Ties are broken by whichever
    label sorts lowest, since `argmax` returns the first maximum.],
)[
```python
nearest_idx = np.argsort(dists[i])[:k]
nearest_labels = self.y_train[nearest_idx]
values, counts = np.unique(nearest_labels, return_counts=True)
y_pred[i] = values[np.argmax(counts)]
```
]

=== Sanity check

#fig(
  caption: [#todo[Validation images shown alongside their nearest neighbors in
    the training set.]],
  reading: [#todo[Comment on whether the retrieved neighbors look like the same
    cell type, or whether they merely share brightness and background.]],
)[#todo[figures/knn-nearest-neighbors.png]]

=== Training process

There is no training process in the usual sense. The search covered 18 values of
k, namely 1, 3, 5, 7, 13, 21, 33, 45, 55, 67, 79, 91, 111, 131, 149, 167, 193
and 201, evaluated for both L1 and L2 distance. That is 36 configurations, each
evaluated once against the 500-image validation set.

#fig(
  caption: [#todo[Validation accuracy as a function of k, one curve per distance
    metric.]],
  reading: [#todo[Point out where the curve peaks and how it behaves as k grows.]],
)[#todo[figures/knn-tuning.png]]

=== Results

The best configuration on the validation set was *k = 5 with L1 distance*, at
*23.6%* validation accuracy. Run once on the held-out test set, that same
configuration scored *17.6%*.

#restable(
  columns: (auto, auto, auto),
  caption: [Best kNN configuration and its performance.],
)[*Configuration*][*Validation accuracy*][*Test accuracy*]
[k = 5, L1 distance][23.6%][17.6%]

Both numbers are low. Random guessing across eight classes gives roughly 12.5%,
so the model is doing better than chance, but not by a wide margin.

#todo[Add a confusion matrix here. With eight imbalanced classes it will show
which cell types the model actually manages and which it never gets right,
which a single accuracy figure cannot.]

=== Discussion

The low accuracy is roughly what raw-pixel distance on medical images should be
expected to give. L1 and L2 distance between flattened pixel vectors respond to
brightness, staining variation, and small shifts in where the cell sits in the
frame, and none of those relate to cell type. Two images of the same cell type
stained slightly differently can end up further apart in pixel space than two
images of different types that happen to share a background.

The number of dimensions makes this worse. Each image is a point in
2,352-dimensional space and there are only 5,000 training points to fill it, so
the space is covered very sparsely. The nearest neighbor of a given image is then
not necessarily similar to it, it is just the closest one available. This is the
curse of dimensionality, and kNN is especially exposed to it because distance is
the only information it uses.

#limitation(title: [The gap between validation and test accuracy])[
  Validation accuracy was 23.6% and test accuracy 17.6%, a drop of six
  percentage points. This suggests the hyperparameter search overfitted to the
  validation set. Picking the best of 36 configurations against only 500 images
  means some of the winning margin is noise specific to those 500, which does
  not carry over to the test set.
]

On whether the search was thorough enough: 18 values of k across two metrics is
a reasonable spread, and the curve is sampled densely enough at the low end where
the optimum turned out to be. The bigger limitation is the data, not the grid.
Training on 5,000 of the 11,959 available images is more likely to be what caps
performance here, so using the full training set would probably help more than
testing further values of k.
