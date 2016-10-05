%% Clustering
load citiesSmall.mat

%% KNN Clustering

for k = [1 3 10]
  model = knn(X,y,k);

  % Evaluate training error
  [n,d] = size(X);
  yhat = model.predict(model,X);
  errorTrain = sum(yhat ~= y)/n;
  fprintf('Training error with k-%d knn: %.3f\n',k,errorTrain);

  % Evaluate test error
  t = size(Xtest,1);
  yhat = model.predict(model,Xtest);
  errorTest = sum(yhat ~= ytest)/t;
  fprintf('Test error with k-%d knn: %.3f\n',k,errorTest);

  if k == 10
    clustering2Dplot(X,y,model.X);
    print -dpng 1.1.3.png
  end
end
