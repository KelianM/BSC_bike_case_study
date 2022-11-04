from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from bike_data_manager import BikeDataManager


filename = "Bike_Buyer_Data_edited.txt"
manager = BikeDataManager(filename)
# Show a summary of the data and columns
manager.show_summary()
# Handle missing values
manager.handle_missing()
# Encode target and categorical features
manager.encode()
# PLot the MI score for each feature and remove low MI features
manager.mutual_info(plot=True, thresh=1e-3)
# Standardise numeric data
manager.standardise()
# Evaluate and plot
fig, axes = plt.subplots(3, 2, figsize=(10, 10))
# Cross validation with 50 iterations and 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.1, random_state=0)
# Evaluate GaussianNB classifier
title = "Bike Performances NB"
classifier = GaussianNB()
manager.plot_learning_curves(classifier, title, axes=axes[:, 0], scoring='average_precision', cv=cv, train_sizes=np.linspace(0.2, 1.0, 9))

# Evaluate SVC classifier with RBF Kernel
title = "Bike Performances SVC"
classifier = SVC(gamma=0.2)
manager.plot_learning_curves(classifier, title, axes=axes[:, 1], scoring='average_precision', cv=cv, train_sizes=np.linspace(0.2, 1.0, 9))

plt.show()
