#import "../lib.typ": *

== Introduction

=== Problem and motivation

This report covers three image classification methods applied to healthy blood
cells. The data comes from the BloodMNIST dataset, which contains microscope
images of individual blood cells. Each cell belongs to one of eight groups,
represented by an integer from 0 to 7: basophil, eosinophil, erythroblast,
immature granulocytes, lymphocyte, monocyte, neutrophil, and platelet.

The task is to predict which of these eight groups a given cell image belongs
to. This is a multiclass classification problem on small RGB images.

Classifying blood cells is currently manual work done by lab technicians. A
model that can do it accurately would remove the need for technicians to perform
this work themselves, which makes the clinical workflow faster.

=== Scope

The report covers the three parts of Portfolio Assignment 1:

- *Assignment 1.1* implements k-Nearest Neighbors, a non-parametric baseline
  that classifies by comparing raw pixel values.
- *Assignment 1.2* implements two linear classifiers, one trained with hinge
  (SVM) loss and one with softmax loss, both optimized by mini-batch gradient
  descent.
- *Assignment 1.3* #todo[one sentence once the handout is available.]

=== Reading guide

The dataset, the splits, the preprocessing and the evaluation metrics are shared
across all three methods, so they are described once in @data-and-setup rather
than repeated in each method chapter. Each method chapter then follows the same
shape: theory, optimization objective and hyperparameters, implementation,
training process, and results. The methods are compared against each other in
@discussion.
