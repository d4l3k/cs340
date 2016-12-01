load arrowhead.mat

alpha = 0.1;
ranks = randomwalk(X, alpha, 1000000);
[largest, i] = max(ranks)
largest = names(i)
med = median(ranks)

names(64)
ranks(64)
