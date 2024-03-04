# -------------------------------------------------------------------------
# AUTHOR: Dhruv Patel
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

#Intialize Dataset
dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv']
test_data = ['cheat_test.csv']
encoder = OneHotEncoder()

for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)  # reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:, 1:]  # creating a training matrix without the id (NumPy library)
    features = df.drop(columns=['Tid'])


    # transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    # Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    # be converted to a float.
    # X =
    encoder.fit(features[['Marital Status']])
    x_encode = encoder.transform(features[['Marital Status']]).toarray()
    #taxable income to float
    X = np.hstack((x_encode, features[['Taxable Income']].astype(float).values))



    # transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    # --> add your Python code here
    # Y =
    Y = df['Class'].apply(lambda x: 1 if x == 'Yes' else 0).values

    accuracies = []
    # loop your training and test tasks 10 times here
    for i in range(10):

        # fitting the decision tree to the data by using Gini index and no max_depth
        clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=None)
        clf = clf.fit(X, Y)

        # plotting the decision tree
        tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'],
                       class_names=['Yes', 'No'], filled=True, rounded=True)
        plt.show()

        # read the test data and add this data to data_test NumPy
        # --> add your Python code here
        # noinspection PyTypeChecker
        data_test = pd.read_csv(test_data, sep=',', header=0)

        for data in data_test:
            test_feature = data_test.drop(columns=['Class'])

            # transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            # class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            # --> add your Python code here
            test_encode = encoder.transform(test_feature[['Marital Status']]).toarray()
            test_X = np.hstack((test_encode, test_feature[['Taxable Income']].astype(float).values))
            test_Y = data_test['Class'].apply(lambda x: 1 if x == 'Yes' else 0).values
            # compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
            # --> add your Python code here

        # find the average accuracy of this model during the 10 runs (training and test set)
        # --> add your Python code here

        # noinspection PyUnboundLocalVariable
        class_prediction = clf.predict(test_X)
        # noinspection PyUnboundLocalVariable
        accuracy = np.mean(class_prediction == test_Y)
        accuracies.append(accuracy)

    # print the accuracy of this model during the 10 runs (training and test set).
    # your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    # --> add your Python code here
    average_accuracy = np.mean(accuracies)
    print(f'Accuracy on{ds}:{average_accuracy:.2f}')
