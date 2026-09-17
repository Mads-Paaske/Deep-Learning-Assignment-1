#import "../lib.typ": *

== Discussion <discussion>

=== Comparison across methods

#restable(
  columns: (auto, auto, auto, auto),
  caption: [All models on the same 1,000-image test set. Random guessing across
    eight classes gives roughly 12.5%.],
)[*Model*][*Best configuration*][*Validation*][*Test*]
[kNN][k = 5, L1][23.6%][17.6%]
[Linear, SVM][lr = 1e-6, reg = 1e3][75.8%][70.6%]
[Linear, softmax][lr = 1e-6, reg = 1e2][71.8%][68.8%]
[#todo[1.3]][#todo[?]][#todo[?]][#todo[?]]

kNN costs almost nothing to train, since it only stores the data, but it is slow
at prediction time and scored worst by a wide margin. The linear classifiers
take real training time but then predict almost instantly, since prediction is a
single matrix multiplication, and they scored between 68% and 71% on the same
test setup.

For a clinical tool meant to speed up lab work this gap matters a lot. A model
that is both this much more accurate and this much faster at prediction time is
the obvious choice between the two.

=== Analysing the disparity in results

The roughly 50 percentage point gap between kNN and the linear classifiers comes
down to what each method is able to represent.

kNN never learns anything about the images. It compares them in raw pixel space,
where distance depends mostly on brightness, staining and where the cell sits in
the frame rather than on cell type. The linear classifiers learn a weighting over
the pixels, so they can suppress the parts of the image that carry no class
information and concentrate on the parts that do.

=== Relating results to the qualitative analysis

#todo[This section needs the confusion matrices to write properly. The prediction
from @data-and-setup is that the four granulocyte classes (basophil, eosinophil,
immature granulocytes, neutrophil) get confused with each other, and that
platelets are classified most reliably because they are visibly smaller. Check
whether the confusion matrices actually show this. If they do not, that is more
interesting than if they do, and worth writing up.]

=== Was the data sufficient?

#limitation(title: [Subsampling])[
  All models saw 5,000 of the 11,959 available training images, validated on 500
  of 1,712, and were tested on 1,000 of 3,421. This was done to keep runtimes
  manageable, but it has consequences.

  The 500-image validation set is the sharper problem. Choosing the best of 36
  configurations for kNN, and of 9 per loss function for the linear models,
  against so few images means part of the winning margin is noise. The
  six-point drop from validation to test accuracy for kNN is consistent with
  exactly that.
]

=== Was the hyperparameter search thorough enough?

For kNN, 18 values of k across two distance metrics is a reasonable spread, and
it samples densely at the low end where the optimum turned out to lie.

For the linear classifiers, 3 × 3 grids are thin. Only nine combinations per loss
function were tried, and both best configurations landed on the same learning
rate, which is a hint that the grid may not have bracketed the optimum. A wider
or finer search would be the obvious next step.

In both cases the amount of training data is the more likely limiting factor
than the density of the grid.

=== Strengths and weaknesses

#restable(
  columns: (auto, auto, auto),
  caption: [Trade-offs between the methods.],
)[*Method*][*Strengths*][*Weaknesses*]
[kNN][No training cost; no assumptions about the shape of the decision boundary; trivial to implement][Slow prediction, scaling with training set size; must store all training data; no feature learning; badly affected by high dimensionality]
[Linear (SVM / softmax)][Fast prediction; compact model; learns per-class templates; interpretable weights][Only linear decision boundaries; needs tuning; sensitive to feature scaling]
[#todo[1.3]][#todo[?]][#todo[?]]

=== Future work

Several things could be tried, roughly in order of expected payoff:

- *Use the full dataset.* Every result here comes from 42% of the available
  training images. This is the cheapest change with the clearest expected gain.
- *Cross-validation instead of a single split.* Averaging over folds would make
  hyperparameter selection far less sensitive to which 500 validation images
  happened to be drawn, and should close some of the validation-to-test gap.
- *Address the class imbalance directly*, through class weighting in the loss or
  resampling the rare classes, and report balanced accuracy alongside plain
  accuracy.
- *Wider hyperparameter grids*, particularly for the linear models, where the
  3 × 3 grid is thin.
- *Move beyond linear models.* A linear classifier can only draw straight
  decision boundaries in pixel space. A convolutional network would learn
  spatial features such as edges and texture, which is much better matched to
  distinguishing cell types that differ in shape and granularity rather than in
  overall brightness.
- *Data augmentation*, such as rotations and flips. Cell orientation in the
  frame is arbitrary, so a model should not be sensitive to it, and augmentation
  is a direct way to encode that.
