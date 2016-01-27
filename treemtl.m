
function [U W Tnleaf rho] = treemtl(taskFiles, treeHt, treeType)

K = length(taskFiles);
% load data
for t=1:K
	disp(taskFiles{t});
	load(taskFiles{t});
	X{t} = data.X;
	Y{t} = data.Y;
end
disp('Finished loading data..');

J = size(X{1},2);
W = zeros(J, K);
% center data and initialize Ws
for t=1:K
  [N] = size(X{t}, 1);
  Y{t} = (Y{t}-ones(N,1)*mean(Y{t},1));
  X{t} = (X{t}-ones(N,1)*mean(X{t},1));
	%W(:,t) = X{t} \ Y{t};
	W(:,t) = ridge(Y{t}, X{t}, 1);
end
disp('Finished centering data, initializing single task Ws..');

% construct tree
T = [];
Tnleaf = [];
U = [];
if strcmp(treeType,'cb')  % 'cb': complete binary
	Nnodes = 2^(treeHt+1) -1;
	Nnleaf = Nnodes-K;
	Tpa = 2^(treeHt-1);					% number of parents
	if(Tpa > K)
		error('Too few tasks for such a big tree\n');
		return;
	end
	T = zeros(Nnodes, K); % complete tree with leaves - used to pass info to TreeGroupLasso. Changes each iteration as parents of tasks change.
	Tnleaf = zeros(Nnleaf, Tpa); %non leaf part of tree. Records D(v): descendants of each node, that are direct parents of tasks. Remains constant always.
	Tnleaf(1:Tpa, :) = eye(Tpa, Tpa);
	for h=0:treeHt-2		% encode binary tree structure
		c = Tpa/2^h;
		last = Nnleaf-(2^(h+1)-1)+1;
		for nn=0:2^h-1
			curr = nn*c;
			Tnleaf(last+nn, :) = sum(Tnleaf(curr+1:curr+c,:));
		end
	end

	% initialize U using clusters of W
	U = zeros(Tpa, K);  % each row represents one Uv
	clusIdx=kmeans(W',Tpa);
	for clus=1:Tpa
		U(clus,:)=(clusIdx==clus);
	end

	% init T based on current values of U
	T(1:K, :) = eye(K,K);
	T(K+1:K+Tpa, :) = U;
	for v=K+Tpa+1:Nnodes
		T(v,:) = (sum(U( (Tnleaf(v-K,:)==1), : )) > 0);
	end
end
disp('Finished constructing tree structure and initializing..');


subplot(2,1,1);
imagesc(W');
subplot(2,1,2);
imagesc(U);
colormap(gray);
W0=W;

% call altmin
opts=[];
opts.eta=0.01;
opts.maxiter=20;
rho = 0.1*ones(Nnleaf,1);
%rho = rand(Nnleaf,1);
opts.rhoLeaf=1; % weights of leaves --- not used anywhere
[U W] = altmin(X, Y, T, Tnleaf, W, U, rho, opts);

figure;
subplot(2,1,1);
imagesc(W');
subplot(2,1,2);
imagesc(U);
colormap(gray);
norm(W-W0,'fro')
