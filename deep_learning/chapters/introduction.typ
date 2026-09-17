#import "../lib.typ": *

= Introduction

The data comes from BloodMNIST: microscope images of single blood cells, each
belonging to one of eight groups numbered 0 to 7. The groups are basophil,
eosinophil, erythroblast, immature granulocytes, lymphocyte, monocyte,
neutrophil and platelet.

The model has to look at an image and say which group the cell belongs to. This
is multiclass classification: more than two possible answers, one answer per
image.

Today this is manual work. A lab technician looks at the cells and sorts them. A
model that does it accurately takes that task off the technician and speeds up
the clinical workflow.

We implemented three methods:

*kNN* (1.1) learns nothing. It stores the training images and classifies a new
image by finding the stored ones most similar to it.

*Linear classifiers* (1.2) do learn. They build one template per class and score
a new image against all eight. We trained one with hinge loss (SVM) and one with
softmax loss.

*Neural network* (1.3) stacks linear layers with a non-linear function between
them, so it can separate classes that a straight boundary cannot.

The dataset, splits, preprocessing and metrics are shared by all three, so they
are described once in @data-and-setup. Each method chapter then follows the same
order: what it does, what it optimizes, how we implemented it, how we tuned it,
and what it scored. @discussion compares them.
