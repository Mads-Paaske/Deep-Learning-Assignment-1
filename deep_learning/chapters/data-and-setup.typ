#import "../lib.typ": *

= Data and setup <data-and-setup>

== The dataset

BloodMNIST is part of the MedMNIST collection and has 17,092 images. Each is 28
pixels wide, 28 tall, and 3 color channels, so 2,352 numbers per image. Both
methods work on that flat vector rather than on the image as a grid.

The data has the following split:

#restable(
  columns: (auto, auto, auto, 1fr),
  caption: [The three splits.],
  [*Split*], [*Images*], [*Share*], [*Used for*],
  [Training], [11,959], [70%], [What the model learns from],
  [Validation], [1,712], [10%], [Choosing hyperparameters],
  [Test], [3,421], [20%], [Final score, used once],
)

Hyperparameters are the settings that are not learned from the data and have to
be chosen by hand, such as how many neighbors to compare against. We therefore adjust the code with different values and keep the ones with the best scores on validation.

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
smaller than any other cell. Neutrophils have a circled nucleus that splits
into separate nucleuses. Eosinophils are the only class with pink or red granules
in the surrounding cytoplasm, so they stand out on color alone.

Three are hard. Basophils, immature granulocytes and monocytes all appear as one
large round purple mass filling most of the frame, and at 28×28 pixels there is
little to separate them.

The last two is in between. Lymphocytes and erythroblasts both show a small,
round, dark nucleus, but differs from the surroundings.

This grouping gives a prediction of where the errors will fall, which we test against the results.

#fig(
  caption: [Five random training images from each class.],
  
)[#image("../figures/class-examples.png", width: 68%)]

== Preprocessing and subsampling

Every image is flattened from 28×28×3 into one vector of 2,352 numbers. The three
methods then differ slightly:

- *kNN* converts the pixel values to float32 and does nothing else. Without this step, squaring the values overflows and the distances come out wrong.
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

  The cost of this decision is accuracy. The models see under half the training data and is validated on under half of the data as well.
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
