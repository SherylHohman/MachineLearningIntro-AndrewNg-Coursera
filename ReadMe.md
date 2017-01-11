ReadMe.md

#Intro to Machine Learning Course
Andrew Ng of Stanford University
Coursera

graded results for HW are in "grades" folder

##week 01:

##week 02:

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

##week 04: Multiclass Classification and Neural Networks
  Recognize Hand-Written Digits
    by implementing one-vs-all logistic regression, and a neural network.

  _FILES_  
  ex3.m         : runs Classification exercises (part1)  
                  images loaded from "ex3data1.txt"  
  ex3data1.txt  : training data (set of hand-writtten digits  

  displayData.m : visual the dataset with this function  
  sigmoid.m     : Sigmoid function  
  fmincg.m      : function Minimization routing (similar to fminunc)  

  ex3_nn.m      : runs Neural Networks exercises (part2)
                  Theta1, Theta2 loaded from ex3weights.mat  
  ex3Weights.mat: initial weights for the neural network exercise  

  Files to Complete in exercises   
    lrCostFunction    : Logistic regression cost function  
    oneVsAll.m        : Trains the one-vs-all multi-class classifier  
    predictOneVsAll.m : Prediction from the one-vs-all multi-class classifier  

    predict.m         : neural network Prediction function

  ###_EXERCISES_
  ------------------------------------------------------------------------------
  run: ex3.m
  ------------------------------------------------------------------------------
  1 Multi-class Classification
    1.1 The Dataset
        X, y loads from ex3data1.mat
          5000 handwritten digits as training examples (MATLAB matrix format)
          each matrix is a 20x20 pixel grayscale image of a digit, 
            floating point number re[resemts pixel intensity
          this matrix is unrolled into a 400-dimensional vector
          Each digit's unrolled matrix then becomes a row in the training data
        
        X then is 5000 x 400 matrix training set of image data
        
        y is 5000-dimensional vector containing labels for the training set
          ie 1, 2, 3, 4, 5, 6, 7, 8, 9, 0
        
        since Octave/Matlab is 1-indexed instead of zero-indexed
          "0" digit is labeled as "10", so
          "1-9" can be labeled as expected

        to make the data more compatible with Octave/MATLAB, 

    1.2 Visualizing the data  : Plots the Data for Visual Inspection
        displayData.m
          picks 100 random rows from the training data
          and displays them on screen as an image (10x10 grid of images of digits)

    1.3 Vectorizing Logisitic Regression

        Need to train 10 logistic regression classifiers
          one for each class of the multi-class classifier (digits: 0-9)

      1.3.1 Vectorizing the cost function 
            (lrCostFunction.m)

            It should be realized that each row of X represents x_Transpose
              for that image.
            so a row in X * Theta == x_Transpose * Theta
                                  == Theta_Transpose * x

            Remember, 
            J(Theta) = 1/m * SUM |i=1..m|
                                 [     -yi  * log(    h_Theta(x_i)) 
                                   - (1-yi) * log(1 - h_Theta(x_i))
                                 ]
            h_Theta(x_i) = g(theta_Transpose * x_i) = g(z)
                         
                         where z = theta_Transpose * x_i

            sigmoid(z) = g(z)= 1/[1 + e ^-z], for classification problems




      1.3.2 Vectorizing the gradient
            (lrCostFunction.m)

      1.3.3 Vectorizing regularized logistic regression
            (lrCostFunction.m)

    1.4 One-vs-all Classfication
          (oneVsAll.m) (uses fminumc instead of minunc for efficiency)

          Train classifier (find optimum theta values) for each class, 
              ie each digit 0-9
          For this use a loop, to iterate through each of the 10 classes/digits
            - mask the y term to equal only matches for the class in question
            - y == digit_class, where y is vector, digit_class is a scaler.
              This zeroes out all columns except the one for the class in question
            - This masked term for y is then passed to our lrCostFunction
              to determine theta values for that digit.
            - Actually, lrCostFunction is passed to fmincg
              which then determines Theta values for that class
            - Now add theta for that class to all_theta, which stores out
              Theta values for all classes of digits.

      1.4.1 One-vs-all Prediction         
          (predictOneVsAll.m)
  ------------------------------------------------------------------------------
  run: ex3_nn
  ------------------------------------------------------------------------------
  2 Neural Networks
    2.1 Model representation
        X, y training data
        Theta1 and Theta2 from ex3weights.mat are loaded into ex3_nn.m  

    2.2 Feedforward Propagation and Prediction
        (predict.m)



