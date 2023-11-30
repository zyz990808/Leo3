---
title: "```logreg``` Vignette"
author: "Yizhong Zhang"
date: "2023-11-29"
output: html_document
---



Source: [The github repository](https://github.com/R-4-Data-Science/Final_Project_Group7)

Package name: ```logreg```

The ```logreg``` package is based on the github repository in the given online address.

Firstly, we need to install the package and library it. 


```r
library(logreg)
```

This package implements logistic regression using numerical optimization. The functions in this package allow for you to:obtain the least squares estimators. This is used as a starting point for optimization, bootstrap confidence intervals for each predictor in the logistic regression, plot the fitted logistic curve to the responses, obtain a confusion matrix on the logistic regression and print metrics calculated from the matrix, and have the ability to plot the given metrics from 0.1 to 0.9. 

### Data Pre-Processing

In this section we use data of a csv document that was collected on canvas. The name of the csv document is ```expenses```. In this document, we get the information of 
some persons such as their age and bmi, and their expenses in medical insurance. We have already know there is a strong linear relationship between these variables. Now, we want to know the relationship between the variable ```somker``` and the other all the variables. It is obvious that ```smoker``` is a binary variable. So we could use the logistic regression method to get this fitted model.

First of all, we need deal with data: (The working directory is the author's directory)


```r
setwd("C:/Users/张易中/Desktop")
mydata <- read.csv("expenses.csv")
mydata$sex <- factor(mydata$sex, 
                     levels = c("male","female"), labels = c(0,1))
mydata$smoker <- factor(mydata$smoker, levels = c("yes","no"), labels = c(1,0))
mydata$region_northeast <- ifelse(mydata$region=="northeast",1,0)
mydata$region_northwest <- ifelse(mydata$region=="northwest",1,0)
mydata$region_southeast <- ifelse(mydata$region=="southeast",1,0)
mydata <- mydata[,-6]
head(mydata)
```

```
##   age sex    bmi children smoker   charges region_northeast region_northwest
## 1  19   1 27.900        0      1 16884.924                0                0
## 2  18   0 33.770        1      0  1725.552                0                0
## 3  28   0 33.000        3      0  4449.462                0                0
## 4  33   0 22.705        0      0 21984.471                0                1
## 5  32   0 28.880        0      0  3866.855                0                1
## 6  31   1 25.740        0      0  3756.622                0                0
##   region_southeast
## 1                0
## 2                1
## 3                1
## 4                0
## 5                0
## 6                1
```

We decide to use the data to create our predictors ```X``` and responses ```Y```. The format of ```X``` and ```Y``` are numeric matrix.


```r
mydata <- as.matrix(mydata)
mydata <- as.numeric(mydata)

mydata <- matrix(mydata, ncol=9)

X <- mydata[, c(1:4,6:9)]
Y <- mydata[, 5]

head(X)
```

```
##      [,1] [,2]   [,3] [,4]      [,5] [,6] [,7] [,8]
## [1,]   19    1 27.900    0 16884.924    0    0    0
## [2,]   18    0 33.770    1  1725.552    0    0    1
## [3,]   28    0 33.000    3  4449.462    0    0    1
## [4,]   33    0 22.705    0 21984.471    0    1    0
## [5,]   32    0 28.880    0  3866.855    0    1    0
## [6,]   31    1 25.740    0  3756.622    0    0    1
```

```r
head(Y)
```

```
## [1] 1 0 0 0 0 0
```

The dimension of matrix ```X``` is 1338 `\(\times\)` 8. The variables in matrix ```X``` are ```age```, ```sex```, ```bmi```, ```children```, ```charges```, ```region_northeast```, ```region_northwest``` and ```southeast```. The dimension of vector ```Y``` is 1338 `\(\times\)` 1, The elements in matrix ```Y``` is whether the person is a smoker.

### Initial Estimate Value of β

The function ```init``` gives the starting value for the predictors of the logistic regression. The specific calculation uses the least squares formula given by `\((X^TX)^{−1}X^Ty\)`. The estimated coefficient vector β
 which includes the independent variables plus the intercept. In this example, we get the estimated values of every elements in β and store them in vector ```init_beta_demo```. The first element is the intercept.
 
The input of this function is the predictor matrix and response vector. The row numbers of predictor and response should be equal to the number of our sample size. In this sample, it is 1338. The column number should be equal to the number of predicted variables in our model. In this example, it is 8.

```resp```: The binary response vector.

```pred```: The predictor matrix.


```r
(init_beta_demo <- init(resp = Y, pred= X))
```

```
##            [,1]
## c  4.648504e-01
##   -7.873921e-03
##   -2.118781e-02
##   -1.046147e-02
##   -1.329449e-02
##    2.997526e-05
##   -2.108683e-02
##   -1.855320e-02
##    2.314974e-02
```

### Bootstrap Confidence Intervals

This function ```Boot_CI``` will conduct a bootstrap routine for each predictor or β value for the logistic regression curve. The method to get the fitted values of every elements of β is the get the minimum value of loss function given in the canvas. The user is able to choose the amount of bootstraps ```n``` to conduct as well as the confidence level ```α``` for the intervals.If the users don't choose the value of amounts of bootstraps ```n``` and ```α```, the default value of ```n``` and ```α``` would be 20 and 0.05. 

The input are:

```resp```: The binary response vector.

```pred```: The predictor matrix.

```n```: The amount of bootstraps, the default value is 20.

```α```: The confidence interval level, the default value is 0.05.


```r
Boot_CI_demo <- Boot_CI(pred= X, resp= Y)
Boot_CI_demo
```

```
##                [,1]          [,2]
##  [1,]  0.3957196154  1.5429244622
##  [2,] -0.0847081204 -0.0528715087
##  [3,] -0.8296456863  0.2450634909
##  [4,] -0.2457846523 -0.1632114014
##  [5,] -0.2302458405  0.1076475916
##  [6,]  0.0003125329  0.0003923582
##  [7,] -0.8267359350  0.2868765318
##  [8,] -0.4745539258  0.2641183037
##  [9,] -0.2474437148  0.5939444701
```

Now, we get a matrix whose dimension is 9 `\(\times\)` 2. The number of rows is 9 because the vector ```β``` has 9 elements, they are from ```β_0``` to ```β_8```. The first column is the lower bound of the confidence interval. The second column is the upper bound of the confidence interval.

We can run another example and in this situation we can define the value of ```n``` and ```α```.


```r
Boot_CI_demo_new <- Boot_CI(pred= X, resp= Y, n =10, alpha = 0.5)
Boot_CI_demo_new
```

```
##                [,1]          [,2]
##  [1,]  0.3690177054  0.5413507061
##  [2,] -0.0792863789 -0.0589433280
##  [3,] -0.2929700214 -0.0769618808
##  [4,] -0.1955010382 -0.1724196268
##  [5,] -0.1727692192 -0.0599329749
##  [6,]  0.0003330328  0.0003636884
##  [7,] -0.1603423577  0.0959167643
##  [8,] -0.0365231405  0.2600556262
##  [9,]  0.2765045715  0.4737913338
```

Now, we get another matrix which is the confidence interval where ```n```=10 and ```α```=0.5

### Fitted Logistic Curve

This function ```log_plot``` will plot the optimized logistic curve for the given data,  where the y-axis is the binary response y while the x-axis represents a sequence of values from the range of fitted values from the logistic regression. In this function, we only need to input two data, they are:

```resp```: The binary response vector.

```pred```: The predictor matrix.

Then, we would like to get the plot of our example with a fitted curve line.


```r
log_plot(pred= X, resp= Y)
```

<img src="{{< blogdown/postref >}}index.en_files/figure-html/plot-1.png" width="672" />

Now, we get the plot of our example with the fitted curve line.

### Confusion Matrix

This function ```conf_mat``` will create a "confusion matrix" using a cut-off value for prediction at 0.5. It will also print out the ```Prevalence```, ```Accuracy```, ```Sensitivity```, ```Specificity```, ```False Discovery Rate```, and ```Diagnostic Odds Ratio``` metrics when run which can be calculated by values within the matrix. The matrix is 3x3 with values representing the performance of the logistic regression curve that we obtain.

The structure of our confusion matrix is:

|P+N|PP|PN|
|:---:|:---:|:---:|
|P  |TP|FN|
|N  |FP|TN|

The description of the elements in the matrix:

```P+N```: The total population of samples. In this example, the number is 1338.

```P```: The number of the responses whose value are 1.

```N```: The number of the responses whose value are 0.

```PP```: The number of the predicted responses whose values are greater than cut-off value.

```PN```: The number of the predicted responses whose values are smaller than cut-off value.

```TP```: The number of the situations where the responses are 1 and the predicted responses are greater than cut-off value.

```FN```: The number of the situations where the responses are 1 and the predicted responses are smaller than cut-off value.

```FP```: The number of the situations where the responses are 0 and the predicted responses are greater than cut-off value.

```TN```: The number of the situations where the responses are 0 and the predicted responses are smaller than cut-off value.

We would also calculate and print out the certain metrics:```Prevalence```, ```Accuracy```, ```Sensitivity```, ```Specificity```, ```False Discovery Rate```, and ```Diagnostic Odds Ratio```.

So we also use this example and the cut-off value is 0.5


```r
conf_mat(pred = X, resp = Y)
```

```
## The Prevalence is 0.204783
## The Accuracy is 0.949925
## The Sensitivity is 1.000000
## The Specificity is 0.937030
## The False Discovery Rate is 0.196481
## The Diagnostic Odds Ratio is Inf
```

```
##      [,1] [,2] [,3]
## [1,] 1338  341  997
## [2,]  274  274    0
## [3,] 1064   67  997
```

So in this situation, we get our confusion matrix and all the required metrics. However, with the condition of the cut-off value is 0.5, the situation of the number of the situations where the responses are 1 and the predicted responses are smaller than cut-off value does not exist. So the value of ```FN``` is 0. And the value of ```Diagnostic Odds Ratio``` does not exist because it converges to infinite. And the value of ```Sensitivity``` is 1 because sensitivity equals to TP/(TP+FN) and the value of ```FN``` is 0.

### Plot of Metrics

This function ```plot_metric``` conducts logistic regression on the given data and plots a given metric that visualizes the performance of the regression over a grid of cut-off values for prediction going from 0.1 to 0.9 with steps of 0.1

In this function, we also need input the predictors and responses. They are numeric matrix and numeric vector. We also have a "metric" option and we have six choices. 

```resp```: The binary response vector.

```pred```: The predictor matrix.

```metric```: The value we want to plot. The choices are: metric = c("Prevalence", "Accuracy", "Sensitivity", "Specificity", "False Discovery Rate", "Diagnostic Odds Ratio")

Now, we could use the ```plot_metric``` function to plot the Prevalence.


```r
plot_metric(resp=Y, pred=X, metric = "Prevalence")
```

<img src="{{< blogdown/postref >}}index.en_files/figure-html/Prevalence-1.png" width="672" />

We get this result because Prevalence=P/(P+N), and the values of P and N is the number of the actual responses whose value are 1 and 0. So in the nine steps, the responses are not changed. So the value of Prevalence would be a constant in this plot.

Now, we could use the ```plot_metric``` function to plot the Accuracy.


```r
plot_metric(resp=Y, pred=X, metric = "Accuracy")
```

<img src="{{< blogdown/postref >}}index.en_files/figure-html/Accuracy-1.png" width="672" />

Now, we could use the ```plot_metric``` function to plot the Sensitivity.


```r
plot_metric(resp=Y, pred=X, metric = "Sensitivity")
```

<img src="{{< blogdown/postref >}}index.en_files/figure-html/Sensitivity-1.png" width="672" />

We get this result because in this example, when the cut-off is smaller than 0.7, the value of ```FN``` is 0. And according to the formula: Sensitivity=TP/(TP+FN), when ```FN``` is 0, the value of Sensitivity should be 1. So in this situation, when cut-off value is smaller than 0.7, the Sensitivity is always 1.

Now, we could use the ```plot_metric``` function to plot the Specificity.


```r
plot_metric(resp=Y, pred=X, metric = "Specificity")
```

<img src="{{< blogdown/postref >}}index.en_files/figure-html/Specificity-1.png" width="672" />

Now, we could use the ```plot_metric``` function to plot the False Discovery Rate.


```r
plot_metric(resp=Y, pred=X, metric = "False_Discovery_Rate")
```

<img src="{{< blogdown/postref >}}index.en_files/figure-html/False_Discovery_Rate-1.png" width="672" />

Now, we could use the ```plot_metric``` function to plot the Diagnostic Odds Ratio.


```r
plot_metric(resp=Y, pred=X, metric = "Diagnostic_Odds_Ratio")
```

<img src="{{< blogdown/postref >}}index.en_files/figure-html/Diagnostic_Odds_Ratio-1.png" width="672" />

We get this result because when the cut-off is smaller than 0.7, the value of ```FN``` is 0. And according to the formula: Diagnostic_Odds_Ratio=LR+/LR-, when ```FN``` is 0, the value of Diagnostic_Odds_Ratio should converge to infinite and does not exist. So in this situation, the value of Diagnostic_Odds_Ratio could get its value only when the cut-off value is larger than 0.7

### Reference

This package is developed based on the text book: [Text Book](https://smac-group.github.io/ds/) 

The function ```log_plot``` is developed by the online instruction: [Fitted Curved Line](https://www.geeksforgeeks.org/how-to-plot-a-logistic-regression-curve-in-r/)

and [Fitted Curved Line 2](https://www.statology.org/plot-logistic-regression-in-r/)

The knowledge of confusion matrix is based on the online instruction: [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)

The structure of this file is developed by the sample document: [irg vignette](https://smac-group.github.io/irg/articles/vignette.html)
