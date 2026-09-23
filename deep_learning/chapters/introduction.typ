#import "../lib.typ": *

= Introduction

The data is collected from BloodMNIST: microscope images of single blood cells, each
belonging to one of eight groups numbered 0 to 7. The groups are basophil,
eosinophil, erythroblast, immature granulocytes, lymphocyte, monocyte,
neutrophil and platelet.

The model sorts each cell image into one of eight groups, a task lab technicians currently do by hand, which is insufissient.

We implemented three methods:

*kNN* learns nothing. It stores the training images and classifies a new
image by finding the stored ones most similar to it.

*Linear classifiers* do learn. They build one template per class and score
a new image against all eight. We trained one with hinge loss (SVM) and one with
softmax loss.

*Neural network* stacks linear layers with a non-linear function between
them, so it can separate classes that a straight boundary cannot.

The dataset, splits, preprocessing and metrics are shared by the three methods, so any difference in the results comes from the methods themselves.
