#import "../lib.typ": *

== Conclusion

Three approaches to classifying blood cells from the BloodMNIST dataset were
implemented and compared. k-Nearest Neighbors, working directly on raw pixel
distances, reached 17.6% test accuracy, only modestly above the 12.5% expected
from random guessing across eight classes. The two linear classifiers performed
substantially better, at 70.6% for hinge loss and 68.8% for softmax loss.

#todo[One sentence on the 1.3 result once it exists.]

The main finding is that learning a weighted template per class, even a purely
linear one, is worth far more on this task than comparing images pixel by pixel.
The largest remaining limitation is that all of these results come from a
subsample of the available data, so the figures reported here should be read as
a lower bound on what these methods can do.
