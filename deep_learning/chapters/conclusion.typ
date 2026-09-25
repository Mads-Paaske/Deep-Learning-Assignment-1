#import "../lib.typ": *

= Conclusion

We implemented and compared three ways of classifying blood cells from
BloodMNIST. The neural network was best at 83.9% on the test set, kNN reached
75.2%, and the two linear classifiers came last at 71.4% and 66.8%. All four beat
the 12.5% expected from guessing by a wide margin.

Two results are worth keeping. The network wins because it can bend its decision
boundary, and cell types differ in shape and texture rather than brightness. kNN
beating both linear classifiers is the less obvious one: compressing a class into
a single linear template costs more than the learning gains, for classes that
vary as much as these do.

The errors are the same everywhere. Platelets are easy because they look
different. Basophils and monocytes are hard, and they lose images to immature
granulocytes, which is what partly developed cells would be expected to do.

The main limitation is that all of this comes from 42% of the available training
data, and the network is clearly overfitting what it did see. These numbers
should be read as a floor rather than as the best these methods can do.



