
function [U W objVal] = groupmtl(taskFiles, numClus, params)

K = length(taskFiles);
% load data
for t=1:K
	load(taskFiles{t});
	X{t} = data.X;
	Y{t} = data.Y;
end
disp('Finished loading data..');

J = size(X{1},2);
W = zeros(J, K);
% center data and initialize Ws
regs = [1e-2,1e-1,1,10,50,100];
%regparam = regs(randi(size(regs,2)));		% regparam = 100;
for t=1:K
  [N] = size(X{t}, 1);
  %Y{t} = (Y{t}-ones(N,1)*mean(Y{t},1));
  %X{t} = (X{t}-ones(N,1)*mean(X{t},1));
	%W(:,t) = X{t} \ Y{t};
	%W(:,t) = ridge(Y{t}, X{t}, regparam);
	[B Fit] = lasso(X{t}, Y{t}, 'Lambda', regs, 'CV', 5);
	W(:,t) = B(:,Fit.IndexMinMSE);
end
%W = randn(J,K);
disp('Finished centering data, initializing single task Ws..');

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
imagesc(U);
U = rand(Tpa, K);
%U = 1/K*ones(Tpa, K);

W0=W;
figure; imagesc(W0');

% call altmin
opts=[];
opts.maxiter=3;		 % =1
opts.lambda=1;  % =1
opts.mu=0.1;				 % =0.1
opts.norm = 'l2';
opts.rho = 1*ones(Tpa,1);
% inner params
opts.eta_U=1.0000e-05;		% =1.0000e-05
opts.maxiter_U=20000;			% =5000
opts.maxiter_W=1000;				% =300

if isfield(params, 'lambda')
	opts.lambda=params.lambda;
end
if isfield(params, 'mu')
	opts.mu=params.mu;
end
if isfield(params, 'norm')
	opts.norm=params.norm;
end
 
[U W objVal] = altmin(X, Y, W, U, opts);

%figure;
%subplot(2,1,1);
%subplot(2,1,2);
%imagesc(U);
%colormap(gray);
norm(W-W0,'fro')
