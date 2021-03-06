%% Clustering
load clusterData.mat

doPlot = 0; % Turn on visualization of the algorithm in action (2D data)

ks = 1:10;
errors = [];

for k = ks
  model = 0;
  error = Inf;

  %% K-Means Clustering
  for i=1:50
    potentialModel = clusterKmeans(X,k,doPlot);
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
  errors = [errors ; error]
end

% Use model to cluster training data
%y = model.predict(model,X);
%clustering2Dplot(X,y,model.W)
%print -dpng 3.1.2.png

plot(ks, errors)
ylabel('error')
xlabel('k')
print -dpng 3.2.3.png
