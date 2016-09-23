addpath ./catVersions

% Load training {X,y} and testing {Xtest,ytest} variables
load citiesSmall.mat
[n,d] = size(X);

depths = 15;

errors = [];

mid = floor(n/2);
validateX = X(1:mid, :);
validateY = y(1:mid, :);
testX = X(mid:n, :);
testY = y(mid:n, :);

for depth = 1:depths
  model = decisionTreeInfoGain(testX,testY,depth);

  % Evaluate training error
  yhat = model.predict(model,testX);
  errorTrain = sum(yhat ~= testY)/n;
  fprintf('Training error with depth-%d decision tree: %.3f\n',depth,errorTrain);

  % Evaluate test error
  t = size(validateX,1);
  yhat = model.predict(model,validateX);
  validateTest = sum(yhat ~= validateY)/t;
  fprintf('Validate error with depth-%d decision tree: %.3f\n',depth,validateTest);

  % Evaluate test error
  t = size(Xtest,1);
  yhat = model.predict(model,Xtest);
  errorTest = sum(yhat ~= ytest)/t;
  fprintf('Test error with depth-%d decision tree: %.3f\n',depth,errorTest);
  errors = [errors; [errorTrain validateTest errorTest]];
end

disp(errors);
plot(errors);
legend("Training Error", "Validation Error", "Test Error")
xlabel("Depth")
ylabel("Error")
print -dpng 3.2.png
