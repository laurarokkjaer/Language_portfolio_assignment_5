
# Language Analytics - Spring 2022
# Portfolio Assignment 5 - MY SELF ASSIGNED PROJECT

This repository contains the code and descriptions from the last assigned project of the Spring 2022 module Language Analytics as part of the bachelor's tilvalg in Cultural Data Science at Aarhus University - whereas the overall Language Analytics portfolio (zip-file) consist of 5 projects, 4 class assignments + 1 self-assigned.

## Repo structure
### This repository has the following directory structure:

| **Folder** | **Description** |
| ----------- | ----------- |
| ```input``` | Contains the input data (will be empty) |
| ```output``` | Contains the results (outputs like plots or reports)  |
| ```src``` | Contains code for assignment 4 |
| ```utils``` | Contains utility functions written by [Ross](https://pure.au.dk/portal/en/persons/ross-deans-kristensenmclachlan(29ad140e-0785-4e07-bdc1-8af12f15856c).html), and which have been used in the assignments |

Also containing a ```MITLICENSE``` for guidelines of how to reproduce and use the data in this repository, as well as a ```.txt``` reqirements-file, where the required installments will be listed.

## Assignment description
For my self assigned project i wanted to build a model which can predict fake or real job posts. I want to create a classification model that can learn this fraudulence of job postings and descriptions. To achieve this i will be:
- 


### The goal of the assignment 


### Data source
This dataset is from [kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction), and contains 18K job descriptions out of which about 800 are fake. The data consists of both textual information and meta-information about the jobs.


## Methods
To solve this assignment i have worked with ```opencv``` in order to both calculate the histograms as well as for the general image processing (using the ```calcHist```, ```imread```, ```normalize``` and ```compareHist```). Futhermore i used the ```jimshow``` and ```jimshow_channel``` from the ```utils```-folder, along with the ```matplotlib``` for plotting and visualisation.

## Usage (reproducing results)
These are the steps you will need to follow in order to get the script running and working:
- load the given data into ```input```
- make sure to install and import all necessities from ```requirements.txt``` 
- change your current working directory to the folder above src in order to get access to the input, output and utils folder as well 
- the following should be written in the command line:

      - cd src (changing the directory to the src folder in order to run the script)
      
      - python image_search.py (calling the function within the script)
      
- when processed, there will be a messagge saying that the script has succeeded and that the outputs can be seen in the output folder 



## Discussion of results
The result of this script is an image which contains one target flower image and the calculated three similar images, as well as the calculated distance scores of the images. Furthermore, a csv file is made with the results (similar images). 

For further development, it could have been interesting to look at how to make the script run with a user defined input. Since this code have already been through a transision from jupiter notebook to .py script, it would not have been much change to do. For the user to parse an argument via the command line when running the code, the script would have been more reproduceble/reuseble, because of the fact that the user wpuld be able to define the target image themselves. 


