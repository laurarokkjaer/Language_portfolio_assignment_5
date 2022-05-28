
# Language Analytics - Spring 2022
# Portfolio Assignment 5 - MY SELF ASSIGNED PROJECT

This repository contains the code and descriptions from the last assigned project of the Spring 2022 module Language Analytics as part of the bachelor's tilvalg in Cultural Data Science at Aarhus University - whereas the overall Language Analytics portfolio (zip-file) consist of 5 projects, 4 class assignments + 1 self-assigned.

## Repo structure
### This repository has the following directory structure:

| **Folder** | **Description** |
| ----------- | ----------- |
| ```input``` | Contains the input data (will be empty) |
| ```output``` | Contains the results (outputs like plots or reports)  |
| ```src``` | Contains code for assignment 5 |
| ```utils``` | Contains utility functions written by [Ross](https://pure.au.dk/portal/en/persons/ross-deans-kristensenmclachlan(29ad140e-0785-4e07-bdc1-8af12f15856c).html), and which have been used in the assignments |

Also containing a ```MITLICENSE``` for guidelines of how to reproduce and use the data in this repository, as well as a ```.txt``` reqirements-file, where the required installments will be listed.

## Assignment description
This is my self-assigned project for which I have chosen to solve the following task:
How to train a model to predict/detect fake or real job postings online on the basis of different valuable informations abput the jobs. I want to create a classification model that can learn and distinguish the fraudulence of job postings and descriptions. To achieve this I will be:
- Performing some regex in order to gather the right informations
- Train test split
- Classifing using the MLP classifier 
- Some futher evaluation for analysing 
- Visualizing cross validation 

For this assignment i believe it is important to mention the change I made in the utils file called ```classifier_utils.py```. In the balance() function I had to change "label" to "fraudulent" in order to .groupby fraudulence, which was nessecary in my case. 

### The goal of the assignment 
The goal of this assignment was to demonstrate the use of the MLP classifier as well as other ```sklearn```methods which can be helpful in furter inspection and evaluation of a text data analysis like this one. 

### Data source
This dataset is from [kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction), and contains 18K job descriptions out of which about 800 are fake. The data consists of both textual information and meta-information about the jobs.


## Methods
To solve this assignment i have worked with ```TfidfVectorizer``` for text processing and most importantly the ```MLPClassifier``` which is my choise of classifier for this task. Furthermore i used ```LogisticRegression``` in order to do some further inspection of the MLP method and the overall classification. I used ```re```for regex when cleaning and sorting the datasey and at last ```matplotlib``` for visualisation.

## Usage (reproducing results)
These are the steps you will need to follow in order to get the script running and working:
- load the given data into ```input```
- make sure to install and import all necessities from ```requirements.txt``` 
- change your current working directory to the folder above src in order to get access to the input, output and utils folder as well 
- the following should be written in the command line:

      - cd src (changing the directory to the src folder in order to run the script)
      
      - python self_assigned_project.py (calling the function within the script)
      
- when processed, there will be a messagge saying that the script has succeeded and that the outputs can be seen in the output folder 



## Discussion of results
The classification accuracy is on 97% and the plot show a rather consistent graph which is not overfit or underfit too much. When running the script the features will also be shown in the terminal (also uploaded to the output folder in this repo), which shows the features that determines/predicts the classification on rather it is a real or a fake job posting. 


