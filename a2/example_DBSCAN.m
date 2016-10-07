%% Clustering
load clusterData2.mat

%% Density-Based Clustering
radius = 256;
minPts = 3;
doPlot = 2;
model = clusterDBcluster(X,radius,minPts,doPlot);
title('Densty-Based clustering');
print -dpng 4.2.4.png
fprintf('K = %d\n', model.k)
