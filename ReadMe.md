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
          h_theta(x) = g(Theta_Transform * X)

          g(z)= 1/[1 + e ^-z]

    1.2.2 Cost Function and Gradient
          J(theta) = 1/m * SUM |i=1..m|
                               [     -yi  * log(    h_theta(xi)) 
                                 - (1-yi) * log(1 - h_theta(xi))
                               ]

          d J(theta)
          -----------  = 1/m * SUM [ (h_theta(xi) - yi) * xi ] |i=1..m|
          d theta_j   

          where h_theta(x) uses the sigmoid function above :logistic regression
          (otherwise the gradient (derivative) form is exactly like linear regression)


