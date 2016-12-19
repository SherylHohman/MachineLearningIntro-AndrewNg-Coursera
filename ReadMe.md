ReadMe.md

#Intro to Machine Learning Course
Andrew Ng of Stanford University
Coursera

week 01:

week 02:

##week 03: Logistical Regression
  _FILES_
  ex2.m         : runs Logistical Regression exercises
    X, y loaded from "ex2data1.txt"
  ex2data1.txt  : training data (X, y)
  plotData.m    : plots the training data for visual inspection

###_EXERCISES_
------------------------------------------------------------------------------
run: ex2.m
------------------------------------------------------------------------------
1 Logistic Regression (Linear)
  1.1 Visualizing the data  : Plots the Data for Visual Inspection
  1.2 Implementation
    1.2.1 Warmup Exercise: sigmoid function 
          (sigmoid.m)

          h_theta(x) = g(Theta_Transform * X)

          g(z)= 1/[1 + e ^-z]

    1.2.2 Cost Function and Gradient
          (costFunction.m)

          J(theta) = 1/m * SUM |i=1..m|
                               [     -yi  * log(    h_theta(xi)) 
                                 - (1-yi) * log(1 - h_theta(xi))
                               ]

          d J(theta)
          -----------  = 1/m * SUM [ (h_theta(xi) - yi) * xi ] |i=1..m|
          d theta_j   

          where h_theta(x) uses the sigmoid function above :logistic regression
          (otherwise the gradient (derivative) form is exactly like linear regression)

          
          RESULTS OF RUNNING THIS FUNCTION:   
          (costFunction.m)
          Calculates Cost of initial params of theta (zeros): 
                  .693174
          Calculates Gradient at initial params of theta (zeros): 
              [  -0.100000
                 -12.009217
                 -11.262842
              ]
    
    1.2.3 Learning Paramaters using fminunc
          (ex2.m, lines 77-108, utalizes Octave's built in fminunc)

          Uses fminunc function to find optimal values for theta:
               [ -25.161272
                0.206233
                0.201470
               ]
          And Cost at optimal values for theta:
                 0.203498
          And Plots Decision Boundry for this function as Figure 2

    1.2.4 Evaluating Logistic Regression
          (predict.m)

          Calculates the training accuracy of our classifier
          (where optimal theta values were determined by fminunc in 1.2.3)

          .. percentage of matches by our classifier on the training data:
          Train Accuracy: 89.000000

------------------------------------------------------------------------------
run: ex2_reg.m
------------------------------------------------------------------------------
2 Regularized Logistic Regression (Non-Linear, avoid overfitting)
  
  2.1 Visualizing the Data  : Plots the Data for Visual Inspection
      Clearly a straight line decision boundry will not fit our data set, so
        logistic regression is inadequate.
  
  2.2 Feature Mapping
      (mapFeature.m)

      map features to x1, x2, where x1 and x2 to the 0, 1, 2,..6th power
      28-dimension vector Transposed = [1, x1, x2, x1^2, x1x2, x2^2,...x2^6]

      observe the susiptibility of overfitting to such data, 
      in combating underfit of linear logistic regression
      via feature mapping of higher dimensional vectors.

      (no code to write)

  2.3 Cost Function and Gradient
      (costFunctionreg.m)

      m = number of examples (training data)
      n = number of Features

      COST FUNCTION:
      J(theta) = 1/m * SUM |i=1..m| 
                           [     -yi  * log(    h_theta(xi)) 
                             - (1-yi) * log(1 - h_theta(xi))
                           ]
                 + lambda/2m * SUM |j=1..n| 
                                   [theta^2] 

      NOTE: the lambda term is NOT to be computed on the first theta value 
        ie (theta_0 == Octave/Matlab's theta_1)


      
      GRADIENT:
**for j=0:**
 d J(theta)
----------- = 1/m * SUM|i=1..m| [ (h_theta(xi) - yi) * xj ]
 d theta_j   


**for j>=1:**
 d J(theta)
----------- = 1/m * SUM|i=1..m| [ (h_theta(xi) - yi) * xj ] + lambda/m * theta_j
 d theta_j          
          

      Results from running costFunctReg.m using 
        theta initialized to zeros,
        lambda = 1

      Cost: 0.693

    2.3.1 Learning Parameters
          (ex2_reg.m)

          As before, ex2_reg.m nowe uses fminunc to find optimized theta values




         

