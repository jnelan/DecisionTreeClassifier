# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 19:35:46 2021

@author: james
"""

#The below code will change the working directory and import the dataset into a dataframe
import os
os.chdir("//home/personal")
import pandas
cancer_data_frame = pandas.read_excel("cancerdata.xlsx")
# The below code will build a dictionary for our map to translate the data and look at the unique data in the "diagnosis column"_data_frame["diagnosis"].unique()
dictionary_for_our_map = {"benign":0, "malignant":1}
cancer_data_frame["diagnosis"] = cancer_data_frame["diagnosis"].map(dictionary_for_our_map)
# Using the pandas.set option, I will limit the columns to 4 and rows to 6 and only display 1 decimal point
pandas.set_option("display.max_rows",6)
pandas.set_option("display.max_columns",4)
pandas.set_option('display.float_format', "{:.1f}".format)
# I will print out the dataframe
print(cancer_data_frame)

# I will import the model selection from sklearn module
from sklearn import model_selection
# I will split the data into two dataframes
data_frame_for_x = cancer_data_frame.loc[:,cancer_data_frame.columns != "diagnosis"]
data_frame_for_y = cancer_data_frame.loc[:,cancer_data_frame.columns == "diagnosis"]
# these will convert the dataframes into numpy arrays
data_for_x = data_frame_for_x.to_numpy()
data_for_y = data_frame_for_y.to_numpy()
# The below will split the data into training and testing sets
training_data_for_x,testing_data_for_x,training_data_for_y,testing_data_for_y = model_selection.train_test_split(data_for_x,data_for_y,test_size = .25) 
# This will print out the shape of the arrays
print("trainingDataForX.shape: " + str(training_data_for_x.shape))
print("testingDataForX.shape: " + str(testing_data_for_x.shape))
print("trainingDataForY.shape: " + str(training_data_for_y.shape))
print("testingDataForY.shape: " + str(testing_data_for_y.shape))

# I will import the tree module from sklearn and also import the metrics module from sklearn
from sklearn import tree
from sklearn import metrics
#  I will know build the model using the training sets
decision_tree_model = tree.DecisionTreeClassifier().fit(training_data_for_x, training_data_for_y)
# Using the model I will now create predictions for diagnosis
predictions_for_y = decision_tree_model.predict(testing_data_for_x)
# the below code will calculate the accuracy of the model along with the classification report
accuracy = metrics.accuracy_score(testing_data_for_y,predictions_for_y)
print("The model's predictive accuracy is: " + "{:.2%}".format(accuracy))
print("The model's classification report: \n" + metrics.classification_report(testing_data_for_y,predictions_for_y))

# I will create a function to redefine the map in our dictionary
def get_key_from_value(dictionary,value_to_lookup):
    "The function will transform the data back to the original form based on the values"
    for key,value in dictionary.items():
        if(value_to_lookup==value):
            return key
    return "no data"
# The below will create lists for the for loop
lists_with_actual_values = []
lists_with_predicted_values = []
counter = 0
for each_prediction in predictions_for_y:
    lists_with_predicted_values.append(str(get_key_from_value(dictionary_for_our_map, each_prediction)))
    lists_with_actual_values.append(str(get_key_from_value(dictionary_for_our_map, testing_data_for_y[counter])))
    counter+=1 
# The below will create a new dataframe with the testing data for x and actual/predicted diagnosis
data_frame_with_testing_data_and_predictions = cancer_data_frame.loc[:,cancer_data_frame.columns != "diagnosis"]
data_frame_with_testing_data_and_predictions = pandas.DataFrame(columns = data_frame_with_testing_data_and_predictions.columns, data = testing_data_for_x)
data_frame_with_testing_data_and_predictions["actual diagnosis"] = lists_with_actual_values
data_frame_with_testing_data_and_predictions["predicted diagnosis"] = lists_with_predicted_values
# This will print out that new dataframe
pandas.set_option("display.max_rows",6)
pandas.set_option("display.max_columns",6)
print(data_frame_with_testing_data_and_predictions)


# The below will conditionally display data where actual diagnosis is malignant and a1 is greater than 12
a_one_data = 12.0
new_malignant_data_frame = data_frame_with_testing_data_and_predictions[(data_frame_with_testing_data_and_predictions["actual diagnosis"] == "malignant") & (data_frame_with_testing_data_and_predictions["a1"] > a_one_data)]
# This will print out the new dataframe of only maglignant and a1 greater than 12
print(new_malignant_data_frame)

# The below code will store that previous dataframe into a new dataframe but only with a1, b1, actual diagnosis, and predicted diagnosis
new_malignant_data_frame_new = new_malignant_data_frame[["a1","b1","actual diagnosis","predicted diagnosis"]]
# the below will print out the contents of the dataframe and will export the excel file without the index
print(new_malignant_data_frame_new)
# The below code will store that previous dataframe into a new dataframe but only with a1, b1, actual diagnosis, and predicted diagnosis
new_malignant_data_frame_new.to_excel("Dataset.xlsx",index = False)

# I will import the seaborn,matplotlib modules for visualizations
import seaborn
import matplotlib
# This first visualization will use a swarmplot of the "a1" against the actual diagnosis of the testing dataset 
seaborn.set(style="whitegrid")
seaborn.swarmplot(x="a1",y="actual diagnosis",data=data_frame_with_testing_data_and_predictions)
matplotlib.pyplot.title("Swarmplot of actual diagnosis vs a1")
matplotlib.pyplot.ylabel("actual diagnosis")
matplotlib.pyplot.xlabel("a1")
print("This visualization is a swarmplot of a1 vs actual diagnosis from the testing dataset.")
# This will display the first 5 rows and set the display to 10 columns
pandas.set_option("display.max_columns",10)
data_frame_with_testing_data_and_predictions.head(5)

# This visualization will produce a swarmplot on top of a violin plot using b1 and actual diagnosis
seaborn.set(style="darkgrid")
seaborn.swarmplot(x="b1",y="actual diagnosis",data=data_frame_with_testing_data_and_predictions,color = "white")
seaborn.violinplot(x="b1",y="actual diagnosis",data=data_frame_with_testing_data_and_predictions,inner=None)
matplotlib.pyplot.title("Swarmplot of actual diagnosis vs b1")
matplotlib.pyplot.ylabel("actual diagnosis")
matplotlib.pyplot.xlabel("b1")
print("This visualization is a swarmplot on top of a violinplot of b1 vs actual diagnosis from the testing dataset.")
# This will display the first 5 rows and set the display to 10 columns
pandas.set_option("display.max_columns",10)
data_frame_with_testing_data_and_predictions.head(5)

# This visualization will produce a scatterplot of a1 vs b1 
data_frame_with_testing_data_and_predictions.plot.scatter(x="a1",y="b1")
matplotlib.pyplot.title("Scatterplot of a1 vs b1")
matplotlib.pyplot.ylabel("b1")
matplotlib.pyplot.xlabel("a1")
print("This visualization is a scatterplot of a1 vs b1 of the testing dataset.  It is helpful to check for multicollinearity of predicter variables")
# This will display the first 5 rows and set the display to 10 columns
pandas.set_option("display.max_columns",10)
data_frame_with_testing_data_and_predictions.head(5)

# This visualization will produce a boxplot of c1 vs actual diagnosis horizontally
seaborn.set(style="whitegrid")
seaborn.boxplot(x="actual diagnosis", y = "c1",data=data_frame_with_testing_data_and_predictions)
matplotlib.pyplot.title("Boxplot of c1 vs actual diagnosis")
matplotlib.pyplot.ylabel("c1")
matplotlib.pyplot.xlabel("actual diagnosis")
print("This visualization will display a boxplot of the actual diagnosis vs c1 predicter of the testing dataset.")
# This will display the first 5 rows and set the display to 10 columns
pandas.set_option("display.max_columns",10)
data_frame_with_testing_data_and_predictions.head(5)


