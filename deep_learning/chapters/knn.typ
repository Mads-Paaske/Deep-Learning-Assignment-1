#import "../lib.typ": *

= k-Nearest Neighbors <knn>

== What it does

kNN is one of the simplest algorithms in machine learning. It memorizes the
training examples and classifies a new image by comparing it against them.
Nothing is learned in between: training just means storing the data.

All the work therefore happens at prediction time, which is why kNN is called a
lazy learner. Training is instant, prediction is slow, because every new image is
compared against every stored one.

Classifying one image means measuring the distance to every training image,
taking the k closest, and predicting whichever label appears most often among
them. If k is 13 and eight of the thirteen nearest are neutrophils, the answer is
neutrophil.

== Measuring distance

Images are just lists of 2,352 numbers, so distance means comparing two lists. L1
(Manhattan) takes the difference at each pixel, makes it positive, and sums:

#eq[$ d_1(a, b) = sum_(p=1)^(2352) abs(a_p - b_p) $]

L2 (Euclidean) squares each difference instead, sums, and takes the square root,
giving ordinary straight-line distance:

#eq[$ d_2(a, b) = sqrt(sum_(p=1)^(2352) (a_p - b_p)^2) $]

Squaring means one wildly different pixel counts for far more than several
slightly different ones, while L1 spreads the weight more evenly. We could not
say in advance which suits this data, so we tested both.

== What gets optimized

Nothing, in the usual sense. kNN has no weights and no loss function, so there is
no gradient descent. We only try settings and keep the best on validation.

*k* is how many neighbors vote. At k = 1 a single odd training image decides the
answer, which is overfitting: the model follows noise. At very high k the vote
includes genuinely different cells, which is underfitting: too coarse to catch
the real pattern.

*The distance metric* is L1 or L2.

== Implementation

The distance calculation is where nearly all the runtime goes. Written the
obvious way it is two nested loops, which is far too slow in Python. Expanding
the squared L2 distance gives a way around that:

#eq[$ norm(a - b)^2 = norm(a)^2 - 2 a dot b + norm(b)^2 $]

The middle term is a dot product, and dot products between every pair are exactly
what one matrix multiplication computes.

#codeblock(
  caption: [L2 distance for every test-train pair at once],
  explain: [Line 1 is $norm(a)^2$ per test image as a column, line 2 is
    $norm(b)^2$ per training image as a row, line 3 is the $a dot b$ term for all
    pairs in one multiplication. Line 4 combines them; because one is a column and
    one a row, NumPy broadcasts to the full matrix automatically. The
    `np.maximum(..., 0)` guards against rounding error pushing a value just below
    zero, which would break the square root.],
)[
```python
test_sq = np.sum(X_test ** 2, axis=1).reshape(num_test, 1)
train_sq = np.sum(self.X_train ** 2, axis=1).reshape(1, num_train)
cross_term = X_test @ self.X_train.T
dists = np.sqrt(np.maximum(test_sq - 2 * cross_term + train_sq, 0))
```
]

#trap(title: [The images must be cast to floating point first])[
  BloodMNIST loads as `uint8`, which holds whole numbers from 0 to 255. Squaring
  a pixel value of 255 gives 65,025, far outside that range, so the result wraps
  around instead of growing. The same happens in the dot product. Every distance
  then comes out wrong.

  Running the code above on the raw `uint8` arrays gave 17.6% test accuracy, and
  L2 returned the *same* accuracy for every value of k from 1 to 21, which is the
  clearest sign that the distances carried almost no information. Adding one cast
  before the reshape fixes it:

  ```python
  X_train = X_train.astype(np.float32)
  X_val = X_val.astype(np.float32)
  ```

  Test accuracy goes from 17.6% to 75.2%. Every number in this chapter is from
  the fixed version.
]

L1 gets no such trick, since absolute values do not expand into dot products.
That version loops over test images only, which stays fast enough. Doing both at
once would need 2,352 numbers per test-train pair and would run out of memory.

#codeblock(
  caption: [The majority vote],
  explain: [`argsort` sorts training images by distance and returns positions, so
    `[:k]` gives the k closest. `np.unique` with `return_counts` counts each label
    among them and `argmax` picks the most common. Ties go to the first, meaning
    the lower class number wins.],
)[
```python
nearest_idx = np.argsort(dists[i])[:k]
values, counts = np.unique(self.y_train[nearest_idx], return_counts=True)
y_pred[i] = values[np.argmax(counts)]
```
]

== Checking it works

#fig(
  caption: [Five validation images and the five training images closest to each
    under L1 distance. A tick means the neighbor has the same label as the query.],
  reading: [Most retrieved neighbors are the same cell type, which is what we want
    to see before trusting any accuracy number. This check is also what would have
    caught the overflow above: with the broken distances the neighbors were
    unrelated to the query.],
)[#image("../figures/knn-neighbours.png", width: 63%)]

== Tuning and results

There is no training loop, so tuning is just trying combinations. We tested 18
values of k (1, 3, 5, 7, 13, 21, 33, 45, 55, 67, 79, 91, 111, 131, 149, 167, 193,
201) with both metrics, giving 36 combinations, each scored once on the 500
validation images.

#fig(
  caption: [Validation accuracy against k for both distance metrics.],
  reading: [L1 beats L2 at every k. Both rise quickly to about k = 13 and then
    flatten, so the exact choice of k matters much less than the choice of metric.],
)[#image("../figures/knn-tuning.png", width: 60%)]

The best combination was *k = 13 with L1 distance* at *79.4%* on validation.
Running that once on the test set gave *75.2%*, and *72.5%* balanced accuracy.

The balanced figure is lower because it averages the eight classes equally
instead of letting the common ones dominate, so it is the more honest number on
an imbalanced dataset.

#fig(
  caption: [Confusion matrix for k = 13, L1, on the 1,000 test images. Rows are
    the true class, columns the prediction.],
  reading: [Platelets are almost perfect at 128 of 129. Basophils are worst at 28
    of 73, with 27 of them predicted as immature granulocytes. Monocytes have the
    same problem: 37 of 82 go to immature granulocytes.],
)[#image("../figures/knn-confusion.png", width: 48%)]

== Discussion

kNN does much better than a pixel-distance method might be expected to, which
tells us the cell images are more separable in raw pixel space than assumed. The
cells are centered and photographed under similar conditions, so two images of
the same type really do end up close together.

The per-class results line up with the visual analysis in @data-and-setup.
Platelets are the smallest and most distinct cell type and they are classified
almost perfectly. The errors cluster exactly where the cells look alike, and
immature granulocytes act as a sink: basophils, monocytes and neutrophils all
lose images to that class. That makes sense, since immature granulocytes are by
definition partly developed cells that resemble several mature types.

Basophils at 38.4% are the weakest class. They are also among the rarest in
training at 852 images, so the model has both the least data and the most visual
overlap working against it.

#limitation(title: [Validation scored higher than test])[
  79.4% on validation against 75.2% on test is a four-point drop. Some of that is
  the tuning fitting itself to the 500 validation images: pick the best of 36
  combinations and part of the winning margin is luck that does not transfer.
]

Eighteen values of k across two metrics is a reasonable spread, and since the
curve is flat past k = 13 a finer grid would not help. The data is the real
limit. We used 5,000 of the 11,959 available training images, and for a method
that works by finding similar examples, more examples to search through is what
helps most.
