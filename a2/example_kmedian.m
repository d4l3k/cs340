%% Clustering
load clusterData2.mat

doPlot = 0; % Turn on visualization of the algorithm in action (2D data)

ks = 1:10;
errors = [];

for k = ks
  model = 0;
  error = Inf;

  %% K-Means Clustering
  for i = 1:50
    potentialModel = clusterKmedians(X,k,doPlot);
    e = potentialModel.error(potentialModel,X);
    if e < error
      fprintf('Old error %d, new %d\n', error, e)
      error = e;
      model = potentialModel;
      y = model.predict(model,X);
      %clustering2Dplot(X,y,model.W);
    end
  end

  fprintf('Final error %d\n', error)
  errors = [errors ; error];

  % Use model to cluster training data
  if k == 4
    y = model.predict(model,X);
    clustering2Dplot(X,y,model.W)
    %print -dpng 3.3.1.png
  end
end

clf
plot(ks, errors)
ylabel('error')
xlabel('k')
print -dpng 3.3.4.png
