
function [U W objVal] = groupmtl(taskNames, taskFiles, splitsFiles, numClus, params)

K = length(taskFiles);
nex = params.nex;
split=params.split;
mkdir(sprintf('merck_output/N%dJ3000_split%d',nex,split));

% load data
for t=1:K
	load(taskFiles{t});
	splits=load(splitsFiles{t});
	X{t} = [d(splits(1:nex),2:end)];
	Y{t} = d(splits(1:nex),1);
	Xtest{t} = [d(splits(nex+1:end),2:end)];
	Ytest{t} = d(splits(nex+1:end),1);
end
disp('Finished loading data..');

[N J] = size(X{1})
size(Y{1})
W = zeros(J+1, K);
% center data and initialize Ws
regs = [1e-2,1,100];
%regs = [1e-2,1e-1,1,10,1e-3,50,100,200,500,1e3];
multiplier = 1/sqrt(J);
for t=1:K
	[B Fit] = lasso(X{t}, Y{t}, 'Lambda', regs, 'CV', 3, 'Alpha', 0.4);  % alpha=1 --> lasso
	W(:,t) = [B(:,Fit.IndexMinMSE) ; Fit.Intercept(Fit.IndexMinMSE)];
	Xtest{t} = [Xtest{t} ones(size(Xtest{t},1),1)];
	Ypred = Xtest{t} * W(:,t);
	X{t} = [X{t} ones(size(X{t},1),1)];
	mse(t) = sum((Ypred-Ytest{t}).^2)/size(Ypred,1);
	disp(sprintf('Initial Task %d MSE: %f',t,mse(t)));
end

disp(sprintf('Initial Avg MSE: %f',mean(mse)));
W0=W;
disp('Finished centering data, initializing single task Ws..');

% print initial obj
for task=1:K
  task_obj(task) = sum(sum((Y{task} - X{task}*W(:,task)).^2))/2;
end
fprintf('Initial sqerr: %f\n',sum(task_obj));

U = [];
Tpa = numClus;
if(Tpa > K)
	error('Too few tasks for such a big tree\n');
	return;
end
% initialize U using clusters of W
U = zeros(Tpa, K);  % each row represents one Uv
U0=U;
figure(1);
subplot(1,2,1);
imagesc(U');
set(gca,'YTick',[1:K]);
set(gca,'YTickLabel',taskNames);
subplot(1,2,2);
imagesc(W0);
title('U0 and W0');
print(sprintf('merck_output/N%dJ3000_split%d/U_W0_lasso.jpg',nex,split),'-djpeg99');
close;

% call altmin
opts=[];
opts.maxiter=2;		 % =1,2
opts.lambda=1;  % =1
opts.mu=0.1;				 % =0.1
opts.norm = 'l2';
% inner params
opts.eta_U=1.0000e-05;		% =1.0000e-05
opts.maxiter_U=20000;			% =5000,20000
opts.maxiter_W=50000;				% =300 or 1000

if isfield(params, 'lambda')
	opts.lambda=params.lambda;
end
if isfield(params, 'mu')
	opts.mu=params.mu;
end
if isfield(params, 'norm')
	opts.norm=params.norm;
end
 

rng('shuffle');
for ll=[2.5 2 1.5 1 0.5 3] % 0.01 0.001 0.0001]; % 0.5 5 10 0.001 0.01 0.1 20] % 1 3
for mm=[0 0.1 0.5 1]; %0.1 0.01]; % 0.1 0.15 0.2 0.25 0.05 0.01 0.2 1 5] % 10 0.1 0.01 0.001] % 1 3
	opts.lambda = ll;
	opts.mu = mm; % best mu = 0.35
for i=3:3
	U = rand(Tpa, K);
	opts.rho = rand(Tpa,1);  %1*ones(Tpa,1);
	opts.rho
	[U W objVal] = altmin(X, Y, Xtest, Ytest, W0, U, opts);
	dlmwrite(sprintf('merck_output/N%dJ3000_split%d/clus%d_obj%d_lambda%g_mu%g.txt',nex,split,numClus,i,opts.lambda,opts.mu),objVal);
	figure(i);
	subplot(1,2,1);
	imagesc(U');
	set(gca,'YTick',[1:K]);
	set(gca,'YTickLabel',taskNames);
	subplot(1,2,2);
	imagesc(W);
	print(sprintf('merck_output/N%dJ3000_split%d/clus%d_U_W%d_lambda%g_mu%g.jpg',nex,split,numClus,i,opts.lambda,opts.mu), '-djpeg99');
	close;
	for t=1:K
		Ypred = Xtest{t} * W(:,t); 
		mse(t) = sum((Ypred-Ytest{t}).^2)/size(Ypred,1);
		r2(t) = rsquare(Ypred,Ytest{t});
		disp(sprintf('Lambda: %g Mu: %g MTL-Final Task-%d MSE: %f R2: %g -- %d',opts.lambda,opts.mu,t,mse(t),r2(t),i));
	end
	disp(sprintf('Lambda: %g Mu: %g MTL-Final Avg MSE: %f AvgR2: %g -- %d',opts.lambda,opts.mu,mean(mse),mean(r2),i));
end
end
end

%figure;
%subplot(2,1,1);
%subplot(2,1,2);
%imagesc(U);
%figure; imagesc(W0');
%colormap(gray);
norm(W-W0,'fro')
