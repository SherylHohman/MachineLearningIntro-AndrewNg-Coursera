ReadMe.md

Intro to Machine Learning Course
Andrew Ng of Stanford University
Coursera

week 01:

week 02:

week 03: Logistical Regression
  _FILES_
  ex2.m         : runs Logistical Regression exercises
    X, y loaded from "ex2data1.txt"
  ex2data1.txt  : training data (X, y)
  plotData.m    : plots the training data for visual inspection

  _EXERCISES_
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

         

