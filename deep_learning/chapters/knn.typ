#import "../lib.typ": *

= k-Nearest Neighbors <knn>

== What it does

kNN is one of the simplest algorithms in machine learning. It memorizes the
training examples and classifies a new image by comparing it against them.
Nothing is learned in between: training just means storing the data.

All the work therefore happens at prediction time, which is why kNN is called a
lazy learner. Training is instant, prediction is slow, because every new image is
compared against every stored one in the dataset.

Classifying one image means measuring the distance to every training image,
taking the k closest, and predicting whichever label appears most often among
them. If k is 13 and eight of the thirteen nearest are neutrophils, the answer is
neutrophil.

== Measuring distance

Each image is a list of 2,352 numbers, so to measure how similar two images are, we compare the lists number by number. L1 (Manhattan) takes the difference at each pixel and adds it all up:

#eq[$ d_1(a, b) = sum_(p=1)^(2352) abs(a_p - b_p) $]

L2 (Euclidean) squares each difference, adds them up, and takes the square root:

#eq[$ d_2(a, b) = sqrt(sum_(p=1)^(2352) (a_p - b_p)^2) $]

The difference is that L2 punishes big differences harder. A single pixel that is very different counts a lot, while L1 treats all differences more equally.

== What gets optimized

Nothing, in the usual sense. kNN has no weights and no loss function, so there is
no gradient descent. We only try settings and keep the best on validation.

*k* is how many neighbors vote. At k = 1 a single odd training image decides the
answer, which is overfitting: the model follows noise. At very high k the vote
includes many different cells, which is underfitting: too coarse to catch
the real pattern.

== Implementation

The distance calculation is where nearly all the runtime goes. Written the
obvious way it is two nested loops, which is far too slow in Python. Expanding
the squared L2 distance gives a way around that:

#eq[$ norm(a - b)^2 = norm(a)^2 - 2 a dot b + norm(b)^2 $]

The middle term is a dot product, and dot products between every pair are exactly
what one matrix multiplication computes.

#codeblock(
  caption: [L2 distance for every test-train pair at once]
)[
```python
test_sq = np.sum(X_test ** 2, axis=1).reshape(num_test, 1)
train_sq = np.sum(self.X_train ** 2, axis=1).reshape(1, num_train)
cross_term = X_test @ self.X_train.T
dists = np.sqrt(np.maximum(test_sq - 2 * cross_term + train_sq, 0))
```
]
The first two lines compute the sum of squares for each test and training image. The third line computes all the dot products in one go. The last line puts the three parts together into a table with one distance for every test-train pair. np.maximum() stops small rounding errors from producing negative values, which would break the square root.



L1 has no such shortcut, because absolute values cannot be split up the same way. Instead we loop over the test images one at a time and compare each to all training images at once.

#codeblock(
  caption: [The majority vote],
)[
```python
nearest_idx = np.argsort(dists[i])[:k]
values, counts = np.unique(self.y_train[nearest_idx], return_counts=True)
y_pred[i] = values[np.argmax(counts)]
```
]

For each test image, we sort the training images by distance and keep the k closest. We then count how many of them belong to each class and pick the most common one. If two classes tie, the one with the lower class number wins.


== Tuning and results

There is no training loop, so tuning is just trying combinations. We tested 18
values of k (1, 3, 5, 7, 13, 21, 33, 45, 55, 67, 79, 91, 111, 131, 149, 167, 193,
201) with both metrics, giving 36 combinations, each scored once on the 500
validation images.

#fig(
  caption: [Validation accuracy against k for both distance metrics.],
)[#image("../figures/knn-tuning.png", width: 60%)]

The best combination was k = 13 with L1 distance at 79.4% on validation.
Running that once on the test set gave 75.2%, and 72.5% balanced accuracy.

The balanced figure is lower because it averages the eight classes equally
instead of letting the common ones dominate, so it is the more honest number on
an imbalanced dataset.

#fig(
  caption: [Confusion matrix for k = 13, L1, on the 1,000 test images. Rows are
    the true class, columns the prediction.],
)[#image("../figures/knn-confusion.png", width: 48%)]

== Discussion

kNN works better than we expected for a method that only compares pixels. This could be because the cells are
all centereed and photoographed in the same way, images of the same cells look more alike pixel by pixel.

In Section 2.3 we expected platelets, eosinophils and neutrophils to be easy, and basophils, monocytes
and immature granulocytes to be hard. This mostly holds.

Platelets are almost perfect at 128 of 129, and eosinophils are also good at 157 of 186. 
Neutrophils did worse than we expected at 147 of 195.

The three classes we expected to be hard are the three worst. Basophils get 28 of 73 right, 
monocytes 41 of 82 and immature granulocytes 121 of 176. This could come those 3 looking very similar.

#limitation(title: [Validation scored higher than test])[
  The model scored 79.4% on validation but only 75.2% on test. This is because
  we tried 36 combinations and picked the one that did best on the 500
  validation images. Part of why it won was luck with those specific images,
  and that luck does not carry over to the test set.
]

Trying more values of k would probably not help, since accuracy stops improving
after k = 13. The bigger problem is data. We only used 5,000 of the 11,959
training images, and kNN works better the more examples it has to compare with.
