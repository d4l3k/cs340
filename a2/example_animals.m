%% Animals with attributes data
load animals.mat

%% Density-Based Clustering
radius = 13;
minPts = 3;
doPlot = 0;
model = clusterDBcluster(X,radius,minPts,doPlot);

%% K-Means clustering
%k = 5;
%model = clusterKmeans(X,k,0);

for c = 1:model.k
    fprintf('Cluster %d: ',c);
    fprintf('%s ',animals{model.y==c});
    fprintf('\n');
end

