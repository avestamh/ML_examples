# ML_examples
In this repository, I will go through some short ML and DL examples. In some cases, I will explain how to fine-tune the model
in the LogesticRegression_with_numpy.py I used numpy to perform logistic regression.

################################################################################

Insied the **logestic_reg_random_forest_finetune.ipynb** I have some simple data processing using pandas
and then I tested some Ml algorithm on the data
# Logistic Regression 
  Perform a logistic regression to analyze the relationship between vehicle model year and the likelihood of being a plug-in hybrid.
# RandomForest
I Used RandomForest model to predict whether a new electric vehicle is a battery electric vehicle (BEV) or a plug-in hybrid electric vehicle (PHEV). 
It includes RandomForest **feature importance**, 
 * Model evaluation using **Cross-Validation** to get a better understanding of how well your model generalizes to unseen data
  * **Hyperparameter tuning**: Use RandomizedSearchCV or GridSearchCV to fine-tune the hyperparameters of the Random Forest model for better performance.

 *  **Handling Class Imbalance**
 *  **A/B Testin**
 *   **Perform Statistical Testing** 
To further validate the comparison between the two models, you can perform statistical testing. A common test is McNemar’s Test, which checks whether the differences in predictions between two models are statistically significant.
If the p-value is below a threshold (e.g., 0.05), it suggests that there is a statistically significant difference between the two models.
##############################################################################
# CNN examples
the **CNN_CIFAR10.py:** shows how to build and train a simple Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset using TensorFlow and Keras. it trains, evaluates the model and generates the predicted image.
############################################################################
# LSTM example
I used LSTM to predict stock price for the next 90 days based on the price of the last year to date.
