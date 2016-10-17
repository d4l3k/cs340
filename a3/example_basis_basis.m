
% Clear variables and close figures
clear all
close all

% Load data
load basisData.mat % Loads X and y
[n,d] = size(X);

for p = 0:10
  % Fit least-squares model
  model = leastSquaresBasis(X,y,p);

  fprintf('p = %d:\n', p);

  % Compute training error
  yhat = model.predict(model,X);
  trainError = sum((yhat - y).^2)/n;
  fprintf('Training error = %.2f\n',trainError);

  % Compute test error
  t = size(Xtest,1);
  yhat = model.predict(model,Xtest);
  testError = sum((yhat - ytest).^2)/t;
  fprintf('Test error = %.2f\n',testError);
end

% Plot model
%figure(1);
%plot(X,y,'b.');
%title('Training Data');
%hold on
%Xhat = [min(X):.1:max(X)]'; % Choose points to evaluate the function
%yhat = model.predict(model,Xhat);
%plot(Xhat,yhat,'g');
%print -dpng 2.2.png