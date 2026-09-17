#import "../lib.typ": *

== Data and experimental setup <data-and-setup>

=== The BloodMNIST dataset

BloodMNIST is part of the MedMNIST collection. It contains 17,092 images of
individual blood cells, each 28×28 pixels with three color channels, giving
2,352 values per image once flattened into a vector.

The dataset ships with a fixed split into training, validation and test sets:

#restable(
  columns: (auto, auto, auto),
  caption: [The predefined BloodMNIST split. The test set is used once, at the
    very end, and plays no part in choosing hyperparameters.],
)[*Split*][*Images*][*Share*]
[Training][11,959][70%]
[Validation][1,712][10%]
[Test][3,421][20%]
[*Total*][*17,092*][*100%*]

The training set is what the models learn from. The validation set is used to
choose hyperparameters. The test set is held back and used only to report final
performance.

=== Class distribution

The eight classes are not evenly represented. Counts in the training split:

#restable(
  columns: (auto, auto, auto),
  caption: [Class distribution in the training split. Neutrophils and
    eosinophils each appear more than twice as often as basophils or
    lymphocytes.],
)[*Class*][*Count*][*Share*]
[Basophil][852][7.1%]
[Eosinophil][2,181][18.2%]
[Erythroblast][1,085][9.1%]
[Immature granulocytes][2,026][16.9%]
[Lymphocyte][849][7.1%]
[Monocyte][993][8.3%]
[Neutrophil][2,330][19.5%]
[Platelet][1,643][13.7%]
[*Total*][*11,959*][*100%*]

#todo[The assignment asks for the distribution in every split, not just
training. Add the same counts for validation and test.]

This imbalance matters for how the results are read. Accuracy can look decent
while the model is quietly ignoring the rare classes, because the common classes
dominate the average. Per-class accuracy or a confusion matrix shows this where a
single overall number hides it.

=== Qualitative analysis of the classes

Four of the eight classes are granulocytes: basophils, eosinophils, immature
granulocytes and neutrophils. They share a broadly similar overall shape, which
makes them the most likely group to be confused with one another. Platelets sit
at the other extreme. They are much smaller than the other cell types and are
easy to pick out by eye.

#fig(
  caption: [Five random training examples from each of the eight classes.],
  reading: [#todo[Write one or two sentences on which classes you personally
    found hard to tell apart in this grid. This is a judgment call the grader
    wants in your own words.]],
)[#todo[figures/class-examples.png]]

=== Subsampling

#choice(title: [Design choice: subsampling the splits])[
  Both models were trained on 5,000 training images rather than the full 11,959,
  validated on 500 rather than 1,712, and tested on 1,000 rather than 3,421.

  The reason is purely computational. kNN in particular has to compute a full
  distance matrix between every test point and every training point, and the
  cost grows with the product of the two set sizes. Subsampling keeps the
  notebooks runnable in reasonable time on a laptop.

  The cost of this choice is accuracy. Using roughly 42% of the available
  training data leaves the models with less to learn from than they could have
  had, and it also means the validation set used to pick hyperparameters is only
  500 images, which is a small sample to choose a best configuration from. Both
  points are taken up again in @discussion.
]

=== Preprocessing

Each image is flattened from its 28×28×3 shape into a single vector of length
2,352, which is what both the distance calculations and the matrix
multiplications operate on.

#todo[Describe the preprocessing that is specific to the linear classifiers:
division by 255, subtracting the training mean, and appending the bias column.
Confirm from the notebook exactly which of these you applied.]

=== Metrics

Classification accuracy is the share of images assigned to the correct class:

#eq[$ "accuracy" = "number of correct predictions" / "total number of predictions" $]

With eight classes, random guessing gives roughly 12.5% accuracy. Any model has
to beat that number before it is doing anything useful.

#todo[The assignment explicitly asks for balanced accuracy and a confusion
matrix. Neither is computed in the notebooks yet. Given the class imbalance
described above, these are worth adding.]

=== Hyperparameter tuning strategy

#choice(title: [Design choice: tuning on validation, never on test])[
  Every hyperparameter was chosen by grid search evaluated on the validation
  set. The configuration scoring highest on validation was then run once on the
  test set to produce the reported figure.

  Keeping the test set untouched during the search is what makes the final
  number an honest estimate of performance on unseen data. If the test set had
  been used to pick the configuration, the reported accuracy would be
  optimistic, because the choice would have been fitted to the very data it is
  being measured on.
]
