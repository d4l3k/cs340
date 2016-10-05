function [model] = knn(X,y,k)
% [model] = knn(X,y,k)
%
% Implementation of k-nearest neighbour classifer

model.X = X;
model.y = y;
model.k = k;
model.c = max(y);
model.predict = @predict;
end

function [yhat] = predict(model,Xtest)
% Write me!
end