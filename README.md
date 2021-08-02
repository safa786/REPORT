# REPORT

 LOAN REPAYMENT USING MACHINE LEARNING















                                                                                                                              SUBMITTED BY ,
                                                                                                                                         
                                                                                                                              Jatin Chokkar
                                                                                                                               Rajat Kaushik
                                                                                                                                Mrinalini R
                                                                                                                                     Safa N
                                                                                                                            Shewta Lakshmi R

                                                                                       


ABSTRACT

                                                            In the lending industry, investors provide loans to borrowers in exchange for the promise of repayment with interest. If the borrower repays the loan, then the lender would make profit from the interest. However, if the borrower fails to repay the loan, then the lender loses money. Therefore, lenders face the problem of predicting the risk of a borrower being unable to repay a loan. In this study, the data from Lending club is used to train two Machine Learning models,Artificial neural network and logistic regression , to determine if the borrower has the ability to repay its loan. In addition, we would analyze the performance of the models.






















CONTENTS

1.Introduction
2.Exploratory Data Analysis
   2.1 Data cleaning and pre-processing
   2.2.Importing the dataset 
   2.3 Strategies to deal with missing values 
   2.4 Strategies to deal with unwanted columns
   2.5.Categorical Feature extraction
   2.6.Visualization
   2.7.Dataset description and correlation
   2.8.Data scaling
3.Modeling
   3.1 Logistic Regression
      3.1.1 Importing the libraries
      3.1.2 Importing the dataset
      3.1.3 Training and testing
      3.1.4 Feature scaling
      3.1.5 Create and train the logistic regression model
      3.1.6 Create and train the MLR model
      3.1.7 Plotting the roc_curve
   3.2 Artificial Neural Network
     3.2.1 Importing the libraries
     3.2.2 Importing the dataset
     3.2.3 Training and testing
     3.2.4 Preprocessing
     3.2.5 Sequential model
     3.2.6 Summarize model
     3.2.7 Classification Report and Confusion Matrix
6.Deployment
7.Conclusion
 ACKNOWLEDGEMENT

We would like to express our sincere gratitude to our supervisor,Yaseen Shah and mentor,Subhajit Mondal,for their valuable and constructive suggestions during the development of this project work.Last but never the least ,each of the group member acknowledge their genuine efforts put in finalizing on a specific area of interest and working towards the completion of this research successfully.













1.INTRODUCTION 
The loan is one of the most important products of the financial institutes.  All the institutes are  trying  to  figure  out  effective  business  strategies  to  persuade  more  customers  to  apply their loans. Determining  whether  a  given  borrower  will  fully pay off the loan or cause it to be charged off (not fully pay off the loan) is difficult.Here,exploratory data analysis is applied to check and handle the missing values,and necessary data transformations is conducted to process the data
In this study, loan behaviors are analyzed with several machine learning models like ANN and Logistic regression models.The logistic regression is widely used to solve the classification problem.The  dataset  that  used  in  this  paper  is  from  Lending  Club.












2.Exploratory Data Analysis
  2.1.Data cleaning and pre-processing
 Dataset has around 32% of missing values with no duplicate records.
      

 We Import the libraries such as pandas ,numpy, matplotlib and seaborn.
 Numpy – is a python library used for working with arrays . It also has functions for working in 
the domain of linear algebra , fourier transform and matrices. It stands for numerical python .Here,Numpy is used as np.
 Pandas – in computer programming pandas is a software library written for python programming language for data manipulation and analysis . It offers data structures and operations for manipulating numerical tables and time series. .Here,Pandas is used as pd .
 Matplotlib.pyplot  -  is a plotting library for the python programming language and its numerical mathematics extension numpy.Here,Matplotlib.pyplot is used as plt . 
 Seaborn  -  is a python data visualisation library based on matplotlib . It provides a high level interface for drawing attractive and informative statistical graphics . It integrates closely with pandas data structure . It helps you explore and understand your data . Here,Seaborn is used as sns .
 Warnings - This is the base class of all warning category classes. It is a subclass of Exception. Warning messages are typically issued in situations where it is useful to alert the user of some condition in a program, where that condition (normally) doesn’t warrant raising an exception and terminating the program. For example, one might want to issue a warning when a program uses an obsolete module.
Python programmers issue warnings by calling the warn() function defined in this module.he determination whether to issue a warning message is controlled by the warning filter, which is a sequence of matching rules and actions. Rules can be added to the filter by calling filterwarnings() and reset to its default state by calling resetwarnings().
![Screenshot (428)](https://user-images.githubusercontent.com/67135174/127778394-a46bea95-1d0f-4746-86d5-45267adfe826.png)
     2.2 Importing the dataset 
 Our dataset used here is csv file ie, comma separated value . We use the pandas library to read our dataset "accepted_2007_to_2018Q4.csv".it reads the dataset and stores it in accepted_data.
Using shape() we find the number of rows in the dataset to be 2260701 and number of columns in the dataset to be 151  .The first few rows of the dataset is printed using head() .
  Pandas .info() function is used to get a concise summary of the dataframe. It comes really handy when doing exploratory analysis of the data. To get a quick overview of the dataset we use the dataframe.info() function.
![Screenshot (431)](https://user-images.githubusercontent.com/67135174/127778480-d7f7fe88-1092-4e6d-897c-8ee0b627606d.png)

![Screenshot (432)](https://user-images.githubusercontent.com/67135174/127778529-9caaa192-07ee-475f-ac08-a3ed3c8edc77.png)

2.3  Strategies to deal with missing values   
The isnull() function is used to detect missing values . It returns a boolean same sized object indicating if the values are NA . 
So here we find the total number of missing cells to be 108486249 and the percentage of the missing cells to be 31.78005318405443 %.
The model also checks for any duplicate record , and it finds none .

![Screenshot (435)](https://user-images.githubusercontent.com/67135174/127778566-6b79f109-0805-4b92-a4fc-ebdc88159f63.png)

![Screenshot (437)](https://user-images.githubusercontent.com/67135174/127778631-cc491ad1-06e8-442d-9ddd-157d0fa77b80.png)

 2.4  Strategies to deal with unwanted columns
Pandas sort_values() function sorts a data frame in Ascending or Descending order of passed Column. It’s different from the sorted Python function since it cannot sort a data frame and a particular column cannot be selected. Here it sorts considering the null percentage of each column , highest to the lowest and displays the result.
We then drop the columns having more than 70% of missing values as these columns are rendered to be useless,by referring to the sorted table above.
We also drop a few columns that are not required for the analysis.
The target column - We need to predict in the dataset if a new loan will get repaid or not. We consider the loan_status column for it.we find all the different variables in the loan_status and the variables charged off and default are combined and assigned as 1 and all the other variables are assigned 0 

![Screenshot (439)](https://user-images.githubusercontent.com/67135174/127778639-05fecbb3-a35a-44e0-8eb3-46313ac176f0.png)

2.5.Categorical Feature extraction

   Machine learning models can only work on dataset having numerical values and not on datasets having string. It’s crucial to develop the methods to deal with such variables.  If not, it is likely to miss out on finding the most important variables in a model.
We also Converting date object columns to integer years or months of columns 'issue_d' And 'last_credit_pull_d'

![Screenshot (441)](https://user-images.githubusercontent.com/67135174/127778655-fe154554-f5f3-4528-a6b0-bbf49dd76218.png)

2.6Visualization
Data Visualization for understanding of various features and their relation with other features Checking Distribution outliers and skewness of each attribute. 
By using sns.counterplot() ,it plots a bar graph for column emp_length i . giving x label ,y label and title as  "Distribution of Employment Length For Issued Loans" .

![Screenshot (443)](https://user-images.githubusercontent.com/67135174/127778673-c62d6ac1-0b70-46fb-a097-a54ed08d490a.png)

![Screenshot (445)](https://user-images.githubusercontent.com/67135174/127778704-abe2694c-cefe-4c51-a28f-318b04b535bd.png)
It can be seen that people who have worked for 10 or more years are more likely to take loans.A box plot is made for loan_amnt and loan_status.A barplot for grade
 ![Screenshot (447)](https://user-images.githubusercontent.com/67135174/127778757-4297d5c3-fad8-42c8-a9d3-620a71c35cbb.png)
![Screenshot (449)](https://user-images.githubusercontent.com/67135174/127778762-2f6fb925-721c-4a14-8437-14d468657152.png)

Since most of the loans are of B Grade, Let's take a look at their loan amounts too
A bar plot is find the average loan amount from the column grade and loan_amnt
 
![Screenshot (451)](https://user-images.githubusercontent.com/67135174/127778776-dd937d07-bf88-4fa1-a11e-9fe5a8bf84a0.png)

Average loan amount of B grade loans is the least of all grades.
![Screenshot (453)](https://user-images.githubusercontent.com/67135174/127778790-9ca8ec5b-e6ec-45f8-b7e8-f2e28044aeb4.png)
![Screenshot (455)](https://user-images.githubusercontent.com/67135174/127778792-fa2612d6-bc57-45eb-9cae-fedd1bb36cea.png)

A plot between interest rate and grade

![Screenshot (459)](https://user-images.githubusercontent.com/67135174/127778816-b3d32fb3-d35c-43be-8d8c-daaf79de4047.png)

Overall Distribution of interest rates

![Screenshot (461)](https://user-images.githubusercontent.com/67135174/127778830-77de6f4c-8625-458e-ada4-e25b36304296.png)


States with most default cases

![Screenshot (463)](https://user-images.githubusercontent.com/67135174/127778841-9806954e-4260-4e6f-80b1-5ad9ac689fa0.png)
States with non_default cases

![Screenshot (465)](https://user-images.githubusercontent.com/67135174/127778851-1e9b65ce-3c71-48b3-a419-05a5afc32018.png)


Home ownership for different loan status

![Screenshot (467)](https://user-images.githubusercontent.com/67135174/127778866-6f8e71dd-7c55-4a26-87aa-ba4c44926b2b.png)

Using distplot() for plotting loan amount distribution

![Screenshot (469)](https://user-images.githubusercontent.com/67135174/127778880-3e236ce0-cc09-4fea-b59a-a61907f7123a.png)


Installments and loan amount using scatterplot.

![Screenshot (476)](https://user-images.githubusercontent.com/67135174/127778895-833454aa-caa3-4604-a497-8061c62dc311.png)


2.7.Dataset description and correlation

Pandas describe() is used to view some basic statistical details like percentile, mean, std etc. of a data frame or a series of numeric values. Statistical description of dataframe was returned with the respective passed percentiles.
 
![Screenshot (481)](https://user-images.githubusercontent.com/67135174/127778932-48cf3369-7ea2-4b5b-8a9c-760f2550684e.png)


Pandas dataframe.corr() is used to find the pairwise correlation of all columns in the dataframe. Any na values are automatically excluded. For any non-numeric data type columns in the dataframe it is ignored.


![Screenshot (482)](https://user-images.githubusercontent.com/67135174/127778940-9de0a233-118b-43ed-9729-abae77896e77.png)


2.8.Data Scaling 
 Inorder to improve the result of the prediction model.  We will standardize the data first.










 3. Modeling
The goal is to identify if the lender would be able to repay the loan.Here, several supervised models will be applied on the loan repayment  dataset.

3.1Logistic Regression
Logistic regression is another supervised learning algorithm that is appropriate to conduct when the dependent variable binary. It is commonly used to obtain odds ratio in the presence of more than one explanatory variable.  The procedure is quite similar to linear regression, but its response variable is binomial.

3.1.1 Importing the libraries
Numpy – is a python library used for working with arrays . It also has functions for working in the domain of linear algebra , fourier transform and matrices . It is an open source project and you can use it freely . It stands for numerical python.Numpy is used as np.
Pandas – in computer programming pandas offers data structures and operations for manipulating numerical tables and time series ..Pandas is used as pd .
Matplotlib.pyplot  -  is a plotting library for the python programming language and its numerical mathematics extension numpy ..Matplotlib.pyplot is used as plt .
Sklearn -  is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.

![Screenshot (501)](https://user-images.githubusercontent.com/67135174/127779036-a17437d7-7d0a-46c7-8956-b05719cd7599.png)

3.1.2 Importing the dataset
 
 Our dataset used here is csv file ie, comma separated value . We use the pandas library to read our dataset “final_data.csv” . The read dataset is stored in a variable ,data. Next, we are printing the read dataset data using head() .
 
     
    	
![Screenshot (502)](https://user-images.githubusercontent.com/67135174/127779045-96dcb7cc-2591-4bcf-b423-6effa34e790c.png)


 
3.1.3 Training and testing 
The train_test_split splits arrays or matrices into random train and test subsets . 
The test size parameter represents the proportion of the dataset to include in the test split and the random_state parameter controls the shuffling applied to the data before applying the split . This function returns a list containing train-test splits of inputs .
 We drop the target column  and store the dataset as features .The target column stored as variable target.
We split the x and y that is the output and input as x train ,y train,x test and y test.we assign test_size as 0.2 and random state as 22.
We split the train set into validation and train set to check the model performance during the training .

![Screenshot (504)](https://user-images.githubusercontent.com/67135174/127779058-16637945-fc12-43f2-ad4d-ad15df6bc170.png)


3.1.4 feature scaling
The sklearn.preprocessing package provides the Standard Scaler.
 The StandardScaler assumes your data is normally distributed within each feature and will scale them such that the distribution is now centred around 0, with a standard deviation of 1.
StandardScaler uses a strict definition of standardization to standardize data. It purely centers the data by using the following formula, where u is the mean and s is the standard deviation.
x_scaled = (x — u) / s
We perform feature scaling on the input x.
 We Import the StandardScaler class and create a new instance.
 Then, fit and transform the scaler .
 And we print the x.
fit_transform() ,transform()- Both are the methods of class sklearn.preprocessing.StandardScaler() and used almost together while scaling or standardizing our training and test data.
The fit method is calculating the mean and variance of each of the features present in our data. The transform method is transforming all the features using the respective mean and variance.fit_transform is used on X test.
Using the transform method we can use the same mean and variance as it is calculated from our training data to transform our test data. Thus, the parameters learned by our model using the training data will help us to transform our test data.transform is used on X train.
 ![Screenshot (505)](https://user-images.githubusercontent.com/67135174/127779074-ba573d56-cc9a-4852-9237-7d477a9111d3.png)


3.1.5 Create and train the logistic regression model
Sklearn model selection is a set of methods intended for regression in which the target value is expected to be a linear combination of the features. 
LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation. 
We fit the x train and the y train into our model.
The predictions are always done on the x test.
Using metrics to calculate the accuracy .
Thus our model produces an accuracy of 99.641938%  
![Screenshot (506)](https://user-images.githubusercontent.com/67135174/127779090-da01493b-fc91-41dc-a4c4-46b9315cd587.png)

 
3.1.6.Creating Classification Report, Confusion Matrix and accuracy score
 
classification_report-For each class it is defined as the ratio of true positives to the sum of true positives and false negatives.Let us first have a look on the parameters of Classification Report:
              y_test : In this parameter we have to pass the true target values of the data.
            predictions : In this parameter we have to pass the predicted output of the model.
 Accuracy score-In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
 confusion matrix- also known as an error matrix, is a summarized table used to assess the performance of a classification model. The number of correct and incorrect predictions are summarized with count values and broken down by each class.
![Screenshot (508)](https://user-images.githubusercontent.com/67135174/127779101-f8a93eda-47c9-4576-a2c3-782ac20f7d1d.png)
![Screenshot (509)](https://user-images.githubusercontent.com/67135174/127779164-51793380-d4d1-4e3c-a9dd-9cdaaf1416d8.png)


3.2 Artificial neural network

 3.2.1 Importing the libraries
Numpy – is a python library used for working with arrays . It also has functions for                 working in the domain of linear algebra , fourier transform and matrices.It is an open source project and you can use it freely . It stands for numerical python.Numpy is used as np.
Pandas – in computer programming pandas offers data structures and operations for   manipulating numerical tables and time series.Pandas is used as pd .
Matplotlib.pyplot  -  is a plotting library for the python programming language and its numerical mathematics extension numpy . It provides an object oriented API for embedding plots into applications using general purpose GUI toolkits like Tkinter , wxPython , Qt or GTK+ .Matplotlib.pyplot is used as plt .
Tensorflow - TensorFlow is a Python library for fast numerical computing created and released by Google.It is a foundation library that can be used to create Deep Learning models directly or by using wrapper libraries that simplify the process built on top of TensorFlow.
 tf.function constructs a callable that executes a TensorFlow graph (tf.Graph) created by trace-compiling the TensorFlow operations in func, effectively executing func as a TensorFlow graph. tf.function to make graphs out of your programs. It is a transformation tool that creates Python-independent dataflow graphs out of your Python code.
Keras - Keras is a minimalist Python library for deep learning that can run on top of Theano or TensorFlow.It was developed to make implementing deep learning models as fast and easy as possible for research and development.
Sequential -A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
A Sequential model is not appropriate when:
Your model has multiple inputs or multiple outputs.Any of your layers has multiple inputs or multiple outputs.You need to do layer sharing.You want non-linear topology (e.g. a residual connection, a multi-branch model).
You can create a Sequential model by passing a list of layers to the Sequential constructor.
Dense- Dense layer is the regular deeply connected neural network layer. It is the most common and frequently used layer. Dense layer does the below operation on the input and returns the output.
                     output = activation(dot(input, kernel) + bias)
input represents the input data,kernel represents the weight data,dot represents numpy dot product of all input and its corresponding weights,bias represents a biased value used in machine learning to optimize the model,activation represents the activation function.
![Screenshot (508)](https://user-images.githubusercontent.com/67135174/127779101-f8a93eda-47c9-4576-a2c3-782ac20f7d1d.png)

3.2.2 Importing the dataset
 Our dataset used here is csv file ie, comma separated value . We use the pandas library to read our dataset “final_data.csv” . The read dataset is stored in a variable ,data. Next, we are printing the read dataset data using head() .
![Screenshot (486)](https://user-images.githubusercontent.com/67135174/127779187-6e1dbf00-d8ad-4c4b-9b7f-9de0af6d05ea.png)

we find the datatypes of each column in the dataset.
 ![Screenshot (488)](https://user-images.githubusercontent.com/67135174/127779194-861fba8b-9565-4902-90c4-eca584ffc9e7.png)

Displaying the count of 1’s and 0’s in the target column.
![Screenshot (490)](https://user-images.githubusercontent.com/67135174/127779206-6ed48346-6e25-4204-9c3e-3115374ed60e.png)

Displaying all the available columns in the dataset.
![Screenshot (492)](https://user-images.githubusercontent.com/67135174/127779217-790988af-e64c-4875-8e98-29067e08f573.png)
3.2.3 Training and testing 
The train_test_split splits arrays or matrices into random train and test subsets . The test size parameter represents the proportion of the dataset to include in the test split and the random_state parameter controls the shuffling applied to the data before applying the split . This function returns a list containing train-test splits of inputs .
 We drop the target column  and store the dataset as features .The target column stored as variable target.
We split the x and y that is the output and input as x train ,y train,x test,y test,x valid and y valid.we assign test_size as 0.2 and random state as 2.

![Screenshot (495)](https://user-images.githubusercontent.com/67135174/127779220-67744342-5cab-4f70-84c8-f949e6d67b12.png)

3.2.4 Preprocessing
sklearn.preprocessing  - this package provides several common utility functions and transformer classes to change raw feature vectors into a representation that is more suitable for the downstream estimators.
 In general, learning algorithms benefit from standardization of the data set. If some outliers are present in the set, robust scalers or transformers are more appropriate. The behaviors of the different scalers, transformers, and normalizers on a dataset containing marginal outliers is highlighted in Compare the effect of different scalers on data with outliers.
MinMaxScaler- Transform features by scaling each feature to a given range.This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.This transformation is often used as an alternative to zero mean, unit variance scaling.
fit_transform() ,transform()- Both are the methods of class sklearn.preprocessing.StandardScaler() and used almost together while scaling or standardizing our training and test data.
The fit method is calculating the mean and variance of each of the features present in our data. The transform method is transforming all the features using the respective mean and variance.fit_transform is used on Xtrain.
Using the transform method we can use the same mean and variance as it is calculated from our training data to transform our test data. Thus, the parameters learned by our model using the training data will help us to transform our test data.transform is used on Xvalid and Xtest.

![Screenshot (496)](https://user-images.githubusercontent.com/67135174/127779249-dbf8ba86-c279-4a8e-bac8-3d115568bc0f.png)

3.2.5 Sequential model
A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.The sequential API allows you to create models layer-by-layer for most problems. It is limited in that it does not allow you to create models that share layers or have multiple inputs or outputs.
Once you have an Input layer, the next step is to add a Dense layer.Dense layers learn a weight matrix, where the first dimension of the matrix is the dimension of the input data, and the second dimension is the dimension of the output data.The output generated by the dense layer is an ‘m’ dimensional vector. Thus, dense layer is basically used for changing the dimensions of the vector. Dense layers also applies operations like rotation, scaling, translation on the vector.
Input dim-Sometimes, though, you just have one dimension – which is the case with one-dimensional / flattened arrays, for example. In this case, you can also simply use input_dim: specifying the number of elements within that first dimension only. 
Activation function Relu - Relu or Rectified Linear Activation Function is the most common choice of activation function in the world of deep learning. Relu provides state of the art results and is computationally very efficient at the same time.
Activation function sigmoid - The sigmoid function takes in real numbers in any range and returns a real-valued output.The first derivative of the sigmoid function will be non-negative or non-positive.It appears in the output layers of the Deep Learning architectures, and is used for predicting probability based outputs and has been successfully implemented in binary classification problems, logistic regression tasks as well as other neural network applications.
There are 6 layers and the last layer is the output layer.

![Screenshot (498)](https://user-images.githubusercontent.com/67135174/127779266-846fecd0-4408-4b9e-a4f9-f9c205d47543.png)

3.2.6.Summarize Model
Keras provides a way to summarize a model.The summary is textual and includes information about:
The layers and their order in the model.
The output shape of each layer.
The number of parameters (weights) in each layer.
The total number of parameters (weights) in the model.
The summary can be created by calling the summary() function on the model that returns a string that in turn can be printed.Total params: 24,513 .Trainable params: 24,513 .Non-trainable params: 0 .
SGD, or stochastic gradient descent, is the "classical" optimization algorithm. In SGD we compute the gradient of the network loss function with respect to each individual weight in the network. Each forward pass through the network results in a certain parameterized loss function, and we use each of the gradients we've created for each of the weights, multiplied by a certain learning rate, to move our weights in whatever direction its gradient is pointing.
SGD's simplicity makes it a good choice for shallow networks. However, it also means that SGD converges significantly more slowly than other, more advanced algorithms that are also available in keras. It is also less capable of escaping locally optimal traps in the cost surface (see the next section). Hence SGD is not used or recommended for use on deep networks.
Keras model provides a method, compile() to compile the model. The argument and default value of the compile() method is as follows.the important arguments are as follows −
loss function is set as 'binary_crossentropy' which is a loss function that is used in binary classification tasks. These are tasks that answer a question with only two choices (yes or no, A or B, 0 or 1, left or right)

![Screenshot (512)](https://user-images.githubusercontent.com/67135174/127779460-bd16608e-a960-4a11-bac6-bb226d3c68ea.png)

Optimizer-Stochastic gradient descent is imported to optimization
Metrics-A metric is a function that is used to judge the performance of your model.
Metric functions are similar to loss functions, except that the results from evaluating a metric are not used when training the model. Note that you may use any loss function as a metric.
epochs − no of times the model is needed to be evaluated during training.neural_model.h5 is an extension which has the best epoch with loss fn less accuracy more
 Early stopping  - It is basically stopping the training once your loss starts to                     increase (or in other words validation accuracy starts to decrease).
The patience argument represents the number of epochs before stopping once your loss starts to increase (stops improving).

![Screenshot (500)](https://user-images.githubusercontent.com/67135174/127779282-e01d20ac-8324-487d-9323-70f5ab71d9d2.png)

Visualization

![Screenshot (514)](https://user-images.githubusercontent.com/67135174/127779498-882048af-6bc9-4998-ac76-099927318df0.png)

 
 
3.2.7.Creating Classification Report and Confusion Matrix
 
classification_report-For each class it is defined as the ratio of true positives to the sum of true positives and false negatives.
confusion matrix-Compute confusion matrix to evaluate the accuracy of a classification.
Class predictions - We can predict the class for new data instances using our finalized classification model in Keras using the predict_classes() function. ... This can be passed to the predict_classes() function on our model in order to predict the class values for each instance in the array.
 
      
Classification report - Let us first have a look on the parameters of Classification Report:
y_test : In this parameter we have to pass the true target values of the data.
predictions : In this parameter we have to pass the predicted output of the model.
print(classification_report(y_test, y_predict, target_names=class_names))

Confusion matrix - For the Confusion Matrix there are two parameters tested and predicted values of the data. print(confusion_matrix(y_test, y_predict))

 ![Screenshot (516)](https://user-images.githubusercontent.com/67135174/127779598-ff017607-4b1e-4860-81ad-bce59ff2ef82.png)

6.DEPLOYMENT
 
 
 
 
 
 
 
 
 
 
 
 
 
7.CONCLUSION
                                                  
                                                  Nowadays, the loan business has become more and more popular, and many people apply for loans for various reasons.  However, there are cases where people do not repay the bulk of the loan amount to the bank which results in huge financial loss.  Hence, if there is a way that can efficiently classify the loaners in advance, it would greatly prevent the financial loss.
                                                      In this study, the dataset was cleaned first, and the exploratory data analysis and feature engineering were performed.  The strategies to deal with both missing values and imbalanced data  sets  were  covered.   Then  we  propose  four  machine  learning  models  to  predict  if  the applicant  could  repay  the  loan,  which  are ANN,Logistic regression.
                                                      As we expected, borrowers with higher annual income are more likely to repay the loan fully.In addition, borrowers with lower interest rates and smaller installments are more likely to pay the loan fully.
 
 
 REFERENCES
 https://dataaspirant.com/how-logistic-regression-model-works/
 https://towardsdatascience.com/the-most-intuitive-and-easiest-guide-for-artificial-neural-network-6a3f2bc0eecb
 https://technocolabs-internship.gitbook.io/internship-prerequisites-learning-resources/
 
 
 
 
 




 
 
 



