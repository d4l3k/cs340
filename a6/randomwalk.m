function [ranks] = randomwalk(X, alpha, t)
n = rows(unique(X));

adj = zeros(n,n);
ranks = zeros(n,1);

for i = 1:rows(X)
  adj(X(i,1),X(i,2)) = 1;
  adj(X(i,2),X(i,1)) = 1;
end

current = ceil(rand()*n);
for i=1:t
  if rand < alpha
    current = ceil(rand()*n);
  else
    candidates = find(adj(current,:)==1);
    current = candidates(ceil(rand()*columns(candidates)));
  end
  ranks(current) += 1;
end

ranks = ranks./ sum(ranks);
end
