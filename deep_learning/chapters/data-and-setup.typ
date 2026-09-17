#import "../lib.typ": *

= Data and setup <data-and-setup>

== The dataset

BloodMNIST is part of the MedMNIST collection and has 17,092 images. Each is 28
pixels wide, 28 tall, and 3 color channels, so 2,352 numbers per image. Both
methods work on that flat vector rather than on the image as a grid.

The data comes already split:

#restable(
  columns: (auto, auto, auto, 1fr),
  caption: [The three splits.],
  [*Split*], [*Images*], [*Share*], [*Used for*],
  [Training], [11,959], [70%], [What the model learns from],
  [Validation], [1,712], [10%], [Choosing hyperparameters],
  [Test], [3,421], [20%], [Final score, used once],
)

Hyperparameters are the settings that are not learned from the data and have to
be chosen by hand, such as how many neighbors to compare against. We try
different values and keep whichever scores best on validation. The test set stays
untouched until the end.

== Class distribution

#restable(
  columns: (auto, auto, auto, auto, auto),
  caption: [Class counts in each split. The classes are not balanced, and the
    imbalance is the same in all three.],
  [*Class*], [*Train*], [*Val*], [*Test*], [*Share*],
  [Basophil], [852], [122], [244], [7.1%],
  [Eosinophil], [2,181], [312], [624], [18.2%],
  [Erythroblast], [1,085], [155], [311], [9.1%],
  [Immature granulocytes], [2,026], [290], [579], [16.9%],
  [Lymphocyte], [849], [122], [243], [7.1%],
  [Monocyte], [993], [143], [284], [8.3%],
  [Neutrophil], [2,330], [333], [666], [19.5%],
  [Platelet], [1,643], [235], [470], [13.7%],
  [*Total*], [*11,959*], [*1,712*], [*3,421*], [*100%*],
)

== Which classes look alike

Looking at the example images, the classes fall into three groups.

Three of them are easy to pick out. Platelets are tiny specks, several times
smaller than any other cell. Neutrophils have a clearly lobed nucleus that splits
into separate segments. Eosinophils are the only class with pink or red granules
in the surrounding cytoplasm, so they stand out on color alone.

Three are hard. Basophils, immature granulocytes and monocytes all appear as one
large round purple mass filling most of the frame, and at 28×28 pixels there is
little to separate them.

The last two sit in between. Lymphocytes and erythroblasts both show a small,
round, dark nucleus, and they mostly differ in the surrounding cytoplasm, which is
exactly the detail this resolution loses.

This grouping predicts where the errors should fall, and @discussion checks it
against the confusion matrices.

#fig(
  caption: [Five random training images from each class.],
  reading: [The three easy classes (platelet, neutrophil, eosinophil) differ in
    size, nucleus shape and color. The three hard ones (basophil, immature
    granulocytes, monocyte) all look like a large round purple mass.],
)[#image("../figures/class-examples.png", width: 68%)]

== Preprocessing and subsampling

Every image is flattened from 28×28×3 into one vector of 2,352 numbers. The three
methods then differ slightly:

- *kNN* casts to `float32` and does nothing else. The cast matters more than it
  sounds; see the pitfall in @knn.
- *The linear classifiers* subtract the mean training image, which centers the
  data around zero, then append a constant 1 for the bias trick.
- *The network* subtracts the mean image and also divides by 255, which brings
  the values into a small range and keeps the gradients from blowing up early in
  training.

#choice(title: [Design choice: we used less data than we had])[
  We trained on 5,000 images instead of 11,959, validated on 500 instead of
  1,712, and tested on 1,000 instead of 3,421.

  The reason is runtime. kNN measures the distance from every test image to every
  training image, so the work grows with the two sizes multiplied together. Full
  runs took too long on a laptop.

  The cost is accuracy. The models see under half the training data. The
  500-image validation set is the bigger worry: picking the best setting from
  only 500 images means part of that choice is luck. @discussion returns to both.
]

== Metrics

Accuracy is the share of images the model got right:

#eq[$ "accuracy" = "number of correct predictions" / "total number of predictions" $]

With eight classes, random guessing gets about one in eight, so 12.5%. Any result
has to beat that to mean anything.

Because the classes are imbalanced we also report *balanced accuracy*, which
takes the accuracy of each class separately and averages those eight numbers.
Every class then counts equally regardless of how often it appears, so a model
that ignores basophils cannot hide behind its neutrophil score. Each model also
gets a confusion matrix showing how many images of each class were predicted as
each other class.

== How we tuned

#choice(title: [Design choice: tune on validation, never on test])[
  For each model we ran a grid search, meaning we tried every combination in a
  list and compared scores. The best on validation was kept and run once on test.

  The test set is never used while choosing. That is what makes the final number
  trustworthy. Picking the setting that scored best on test would mean reporting
  how well the model does on the data we tuned it against.
]
