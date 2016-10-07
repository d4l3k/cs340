warning('off', 'Octave:divide-by-zero')

% Load X and y variable
load vowel.mat
[n,d] = size(X);
[t,d] = size(Xtest);

depths = 1:15;

training_error = [];
testing_error = [];

for depth = depths
  Xboot = X;
  yboot = y;
  for i = 1:n
    j = ceil(rand*n);
    Xboot(i,:) = X(j,:);
    yboot(i,:) = y(j,:);
  end

  %% Fit random tree and compute error
  model = decisionTree(Xboot,yboot,depth);

  % Evaluate training error
  yhat = model.predict(model,X);
  error = sum(yhat ~= y)/n;
  training_error = [training_error; error];
  fprintf('Training error with depth-%d random tree: %.2f\n',depth,error);

  % Evaluate training error
  yhat = model.predict(model,Xtest);
  error = sum(yhat ~= ytest)/t;
  testing_error = [testing_error; error];
  fprintf('Test error with depth-%d random tree: %.2f\n',depth,error);

  % Plot classifier
  %figure(1);
  %classifier2Dplot(X,y,model);
end

plot(depths, [training_error testing_error]);
ylabel('error')
xlabel('depth')
legend("Training", "Test")
print -dpng 2.1.4.png
