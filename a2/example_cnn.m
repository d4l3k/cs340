%% Clustering
% load citiesBig1.mat
load citiesBig2.mat

for k = [1]
  model = cnn(X,y,k);
  fprintf('Number of objects in model %d of %d \n', rows(model.X), rows(X))

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

  if k == 1
    clustering2Dplot(X,y,model.X);
    % print -dpng 1.2.3.png
    print -dpng 1.2.6.png
  end
end
