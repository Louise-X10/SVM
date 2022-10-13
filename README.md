# SVM

This code utilizes the `features.txt` file from HW2 which encodes the normalized color histograms with label appended. First the data is split into train and test set and converted into the format desired by SVM light (http://www.cs.cornell.edu/people/tj/svm_light/). This creates two files `train.dat` and `test.dat`, then one can run those with SVM light modules in the terminal. 
- `model`, `predictions`: results from default linear kernel SVM 
- `model_poly3`, `predictions_poly3`: results from polynomial with degree 3 kernel SVM

The second part of the code applies PCA transformation on all data, and the split into train and test set. The cumulative variance ratio vector is also printed to the console for reference. The `reduce_to_threshold` function reduces the dimension of the PCA-transformed data depending on the desired variance ratio, and performs creates corresponding `pca_train_n.dat` and `pc_test_n.dat`. Here n denotes the number of features kept in order to achieve that variance ratio. After looking at the cumulative variance ratio vector, I decided to try out variance ratio thresholds [0.7, 0.8, 0.9, 0.95, 0.98, 0.99]. One can then separately run SVM light modules for the different pairs of PCA train data and PCA test data created. One can also modify the code to test for other thresholds. One can also use the `keep_features` (commented out) to test by specifying the number of features kept, instead of specifying the desired threshold.
- `model_n4`, `predictions_n4`: results from default SVM on PCA-transformed data and reduced to 4 dimensions
