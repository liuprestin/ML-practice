#data analysis
import pandas  as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"


#load the Titanic data - assumes will need to set directory
dir = "~/Public/week5/Titanic-kaggle"
train = pd.read_csv("~/Public/week5/Titanic-kaggle/train.csv")
test = pd.read_csv("~/Public/week5/Titanic-kaggle/test.csv")
#gender = pd.read_csv("~/Public/week5/Titanic-kaggle/gender_submission.csv") #this is a submission file

# ---------------------  Exploritory phase -------------------
train.head()
# what are the features that could affect surival?
# what features could be dependant on each other?

# categorical data - ticket , cabin , sex , Embark (not sure what that one means though)
# for the categorical data - ticket/cabin should be related - though - cabin has NaN , and I'm not sure how the ticket claffication works

# not sure if useful - Name - I don't think the name affects survival
# unless one adds history family data

# data description:

test.describe()
train.describe()

# using [] notation
# Passengers that survived vs passengers that passed away
survive = train["Survived"].value_counts()
print(survive)

# As proportions
survive_percent = train["Survived"].value_counts(normalize = True)
print(survive_percent)

# Males that survived vs males that passed away (subset)
male_survival = train["Survived"][train["Sex"] == 'male'].value_counts()
print(male_survival)

# Females that survived vs Females that passed away
female_survival = train["Survived"][train["Sex"] == 'female'].value_counts()
print(female_survival)

# Normalized male survival
male_survival_N = train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)
print(male_survival_N)

# Normalized female survival
female_survival_N = train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)
print(female_survival_N)

# testing age
# Create the column Child and assign to 'NaN'
train["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.

train["Child"][train["Age"] < 18] = 1
train["Child"][train["Age"] >= 18] = 0
print(train["Child"])

# Print normalized Survival Rates for passengers under 18
print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))

# Print normalized Survival Rates for passengers 18 or older

print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))

# female testing

# Create a copy of test: test_one

test_one = test

# Initialize a Survived column to 0

test_one["Survived"] = 0
test_one["Survived"]
# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`

test_one["Survived"][test_one["Sex"] == "female"] = 1

test_one["Survived"]

# decision trees
from sklearn import tree

# Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

#Print the Sex and Embarked columns
print(train.Embarked)
print(train.Sex)

# what information do we need for the decision tree?
