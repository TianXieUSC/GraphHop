1. logisticRegression.py -- each batch iteration, the input is same proportion of training and pseudo-labels;
2. logisticRegression_2.py -- each batch iteration, fit the entire training labels and batch of pseudo-labels;
3. logisticRegression_3.py -- add pseudo-label uncertainty weight and class imbalance weight [1];
4. logisticRegression_4.py -- multilabels, binary cross entropy loss

1. model.py -- the vanilla model;
2. model_2.py -- add pseudo-label uncertainty weight and class imbalance weight [1]; (*) does not perform better;
3. model_3.py -- three-hops for Reddit dataset;
4. model_4.py -- only propagation, no classifier after the first step;
5. model_5.py -- add residential connection at each iteration;
6. model_6.py -- real large scale version used;
7. model_7.pu -- multilabels;

[1] "Label Propagation for Deep Semi-supervised Learning";