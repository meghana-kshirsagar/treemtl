
function [U W] = groupmtl(taskFiles, numClus)

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
%W = randn(J,K);
disp('Finished centering data, initializing single task Ws..');

%W = W + 0.010*randn(size(W));

U = [];
Tpa = numClus;
if(Tpa > K)
	error('Too few tasks for such a big tree\n');
	return;
end
% initialize U using clusters of W
U = zeros(Tpa, K);  % each row represents one Uv
clusIdx=kmeans(W',Tpa);
for clus=1:Tpa
	U(clus,:)=(clusIdx==clus);
end
U = rand(Tpa, K);
%U = 1/K*ones(Tpa, K);

subplot(3,2,1);
imagesc(abs(W));
subplot(3,2,2);
imagesc(U);
colormap(gray);
W0=W;

% call altmin
opts=[];
opts.maxiter=1;
opts.lambda=1;
opts.rho = 1*ones(Tpa,1);
opts.mu=0.01;
opts.norm = 'l2';
% inner params
opts.eta_U=1.0000e-05;
opts.maxiter_U=5000;
opts.maxiter_W=300;
 
[U W] = altmin(X, Y, W, U, opts);

figure;
%subplot(2,1,1);
%imagesc(W');
%subplot(2,1,2);
imagesc(U);
colormap(gray);
norm(W-W0,'fro')
