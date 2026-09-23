#import "../lib.typ": *

= Discussion <discussion>

== Comparing the methods

#restable(
  columns: (auto, auto, auto, auto, auto),
  caption: [All four models on the same 1,000 test images. Guessing gives 12.5%.],
  [*Model*], [*Best setting*], [*Validation*], [*Test*], [*Balanced test*],
  [Neural network], [500 hidden, lr 0.1], [88.0%], [83.9%], [81.6%],
  [kNN], [k = 13, L1], [79.4%], [75.2%], [72.5%],
  [Linear, SVM], [lr 5e-7, reg 1e3], [75.0%], [71.4%], [68.6%],
  [Linear, softmax], [lr 1e-6, reg 1e3], [70.2%], [66.8%], [64.3%],
)

The ordering is the network first, kNN second, the linear classifiers last.

The network wins because it is the only model that can bend its decision
boundary. Cell types differ in shape and texture, which a straight line in pixel
space cannot separate well.

The more surprising result is kNN beating both linear classifiers. A linear
classifier compresses each class into a single template, so every image of a
class has to resemble one average picture of it. kNN has no such constraint and
can match any individual training image, which suits classes that vary in
appearance. Whatever a linear model gains by learning, it gives back by being
forced into one template per class.

== Cost, not just accuracy

Accuracy is not the only axis. kNN costs nothing to train but has to compare
every new image against all 5,000 stored ones, and it has to keep them all in
memory. The linear classifiers and the network take real time to train but then
predict with a couple of matrix multiplications.

For a tool meant to speed up lab work, the network is the clear choice: most
accurate and fast at prediction time. kNN would be the worst choice in
deployment despite beating the linear models on accuracy.

== Where the errors are

All three methods fail in the same places, which says the difficulty is in the
data rather than in any one model. Per-class accuracy for kNN:

#restable(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto),
  caption: [Per-class accuracy for kNN on the test set, ordered worst to best.],
  [*Baso.*], [*Mono.*], [*Imm. gran.*], [*Neutro.*], [*Erythro.*], [*Lympho.*], [*Eosino.*], [*Platelet*],
  [38.4%], [50.0%], [68.8%], [75.4%], [81.4%], [82.2%], [84.4%], [99.2%],
)

The visual grouping in @data-and-setup mostly holds. The three classes predicted
to be hard, basophils, monocytes and immature granulocytes, are exactly the three
worst, and their errors go where predicted: 27 of 73 basophils and 37 of 82
monocytes are classified as immature granulocytes. All three look like the same
large round purple mass at this resolution. Platelets, predicted easiest, are
classified almost perfectly at 128 of 129.

One part of the prediction was wrong. Lymphocytes and erythroblasts were expected
to be middling, since both show a small dark nucleus and differ mainly in the
cytoplasm. They came out at 82.2% and 81.4%, ahead of neutrophils at 75.4%, which
were predicted easy on the strength of their lobed nucleus. The lobes are
apparently harder to resolve at 28×28 than they look.

Basophils are also the rarest class in training at 852 images, so they have both
the least data and the most visual overlap working against them.

== Limitations

#limitation(title: [Subsampling])[
  Every model saw 5,000 of 11,959 training images, was tuned on 500 of 1,712 and
  tested on 1,000 of 3,421. Choosing the best setting from 500 validation images
  means part of that choice is luck, which is visible in the validation-to-test
  drop of 3 to 4 points for every model, as it could be the cause.
]

#limitation(title: [The grids were too narrow])[
  For the linear classifiers and the network, the winning setting sits at the
  edge of the grid rather than inside it. When that happens the true optimum is
  usually outside the range searched, so all three of those numbers should be read
  as lower bounds. Only kNN was searched widely enough that its curve clearly
  flattens.
]

#limitation(title: [The network is overfitting])[
  99.7% training accuracy against 88.0% validation. With over a million weights
  in the first layer and 5,000 training images, the model has enough capacity to
  memorize the training set, and regularization of 0.001 is too weak to prevent
  it.
]

== Strengths and weaknesses

#restable(
  columns: (auto, 1fr, 1fr),
  caption: [What each method is good and bad at.],
  [*Method*], [*Strengths*], [*Weaknesses*],
  [kNN], [Nothing to train; no assumption about the decision boundary; easy to implement], [Slow to predict and slower with more data; stores all training data; learns no features; needs a float cast to work at all],
  [Linear], [Fast to predict; small model; weights readable as images], [Only straight boundaries; one template per class is restrictive; weakest of the three],
  [Network], [Most accurate; bends the boundary; same fast prediction], [Overfits badly at this data size; many hyperparameters; slowest to train],
)

== What we would try next



*Use all the data.* Every result here comes from 42% of the training images. It
is the cheapest change, and the network should gain most since overfitting is its
main problem.

*Widen the grids* so the best setting is not at the edge.

*Cross-validation instead of one fixed validation set*, so the choice of settings
depends less on which 500 images we happened to draw.

*Handle the imbalance directly*, by weighting rare classes more heavily in the
loss or by resampling. Basophils would benefit most.

*Try a convolutional network.* This is a type of neural network made for
images. Unlike our models, it looks at the image as a picture instead of a long
list of numbers, so it can better recognize the shapes of the cells.
