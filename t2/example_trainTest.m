clear all
load DTdata.mat
[n,d] = size(X);
Xtest = X (n/2 + 1 :n  , : );
ytest= y (n/2 +1 : n ) ;
X =  X ( 1:n/2 , : )  ;
y = y (1:n/2);
mindepth = -1 ; minError = Inf;
for depth =1 :15
    errorTrain = 0; errorTest = 0; 
    for i =1:2 
        [N,D] = size(X);
        T = length(ytest);
        model = decisionTree_InfoGain(X,y,depth);
        yhat = model.predictFunc(model,X);
        errorTrain = errorTrain +sum(yhat ~= y)/N;
        yhat = model.predictFunc(model,Xtest);
        errorTest = errorTest + sum(yhat ~= ytest)/T;
        [X, Xtest]=mySwap(Xtest, X); 
        [y,ytest] = mySwap(ytest,y) ;
    end
    disp(errorTest/2 ) ;     
    if errorTest < minError 
        minError= errorTest; 
        mindepth = depth;
    end
end
disp(minError); disp(mindepth); 

