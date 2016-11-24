load animals.mat

[n,d] = size(X);
X = standardizeCols(X);

figure(1);
imagesc(X);
figure(2);
i = ceil(rand*d);
j = ceil(rand*d);
plot(X(:,i),X(:,j),'.');
gname(animals);
