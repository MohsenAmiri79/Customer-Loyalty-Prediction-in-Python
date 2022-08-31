
# Customer Loyalty Prediction with Python (on bank customer data)

This project was done for practice and educational purposes.

In this project, I use a raw banking data file, and preprocess the data.
Then I created a simple feedforward neural network for the prediction task.

## Prerquisites
To replicate the training process or just evaluate the model, you need to have a cuda-enabled GPU.
you also need to have cuda in your python environment.

You also need to install all the dependencies listed in the "environment.yml" file.

For those who have conda on their computer, open anaconda prompt.
Proceed to the download directory and use the code below:
```
conda env create --file environment.yml
```
Then activate the environment and use it as your python environment in your preferred IDE.

## Usage Instructions
To try the model, you can download and run the 'evaluate.py' file and enter your own feature values.

To replicate the process of the project, you have to first, download the whole repository.
Then, run all the cells in 'preprocess.ipynb' and 'model.ipynb' consecutively.

## Description

This projects consists of 4 major parts that I am going to explain in this section.
1. Preprocessing
2. Model Training
3. Testing The model

### 1. Preprocessing

#### 1.1. Importing the data
The data used in this project, which is 700 records of bank customer information, is imported as a pandas dataframe.

#### 1.2. Outlier Management
We use plots and visual analysis to see which features have normal distribution.
The "debt 1" and "age" columns seem to have normal-like distributions.

- Noraml-distribution columns: in these columns, z-score was used to detect outliers in the data.
- Other columns: in these columns the IQR was used for outlier detection.

Since there isn't an abundance of data records, we can not remove outliers from the dataset.
Therefore, the outliers were set to blank (Nan) values to be imputed manually or automatically later.

#### 1.3. Missing Data Management
We had 4 levels missing data in the data columns.

- **High count**: "income" and "debt 2"
- **Medium count**: "credit", "age" and "edu"
- **Low count**: "debt 1"
- **No missings**: "exp", "red", "loyalty"

##### **Low count column:**
For this column, I used it's median to fill the missing cells.

##### **Medium count column:**
For these columns, the 'fitter' library was used to estimate each feature's distribution.
Then each column was filled using random numbers in their own distribution.

##### **High count columns:**
For these columns, I used scikit-learn's KNNImputer module and filled the columns using the pre-existing data.

#### 1.4. Data Scaling
The whole dataset was rescaled using scikit-learn's MinMaxScaler module.
All the values were set to be between zero and one.

The MinMaxScaler object is also saved to 'Model\scaler.gz' as we will need this scaler later to scale new test data.

#### 1.5. Feature Selection
Because of the low number of records available, feature selection is not necessarily needed.
But since this project was a means of practicing, I have added this step as well.

In this stage, the correlation matrix has been plotted and those features with correlation of 0.6 and higher were managed.
To do that, the correlation between the target feature and each of the highly correlated features were measured.
The one with lower correlation to the target feature was dropped.

Therefore, "*income*" was dropped for having a high correlation with the "*exp*" column.
And, "*debt 2*" was dropped for being highly dependant on the "*debt 1*" feature.

#### 1.6. Saving dataset
The preprocessed dataset was saved into a .csv file which can be find at 'Data/processed_customer_data.csv'.


### 2. Model Training
For the training of the model we used the *optuna* HPO library.
In the manual model, the training process iterated over 5 different counts if epochs (30, 60, 90, 120, 150) and 5 different batch sizes (4, 8, 16, 32, 64).
In the automatic model, a number in the [30, 150] range was selected by optuna for the number of epochs. and also a number between (2, 4, 8, 16, 32, 64) was also chosen by optuna for the size of each batch in each trial.

For hyperparameter optimization, we used optuna studies. each study was performed over one of the epoch counts and batch sizes.
each study contained 25 trials. each trial in a study differed in 4 types of parameters:
- Learning rate of the optimization method
- The optimization method itself
- number of layers in the neural network
- number of units in each layer of the network

The best model out of the whole 625 trials was extracted and saved to a .pt file at 'Models\\customer_loyalty_prediction.pt'.

### 3. Testing The Model

The saved model and the scaler are imported, and used for user input data to predict the loyalty of the hyptothetical customer
