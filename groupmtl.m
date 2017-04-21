
function [U W objVal] = groupmtl(taskNames, taskFiles, testFiles, numClus, params)

K = length(taskFiles);
ntr=100;
% load data
for t=1:K
	d=load(taskFiles{t});
	X{t} = [d(1:ntr,2:end)]; % ones(size(d,1),1)];
	X{t} = [X{t}; d((end-ntr+1):end,2:end)];
	Y{t} = d(1:ntr,1);
	Y{t} = [Y{t}; d((end-ntr+1):end,1)];
	d=load(testFiles{t});
	Xtest{t} = [d(:,2:end)]; % ones(size(d,1),1)];
	Ytest{t} = d(:,1);
end
disp('Finished loading data..');

[N J] = size(X{1})
W = zeros(J+1, K);
% center data and initialize Ws
%regs = [1e-2,1e-1,1,10,1e-3,50,100,200,500,1e3];
multiplier = 1/sqrt(J);
regs = [1e-3,1e-2,1,100,1e3];
%regparam = regs(randi(size(regs,2)));		% regparam = 100;
for t=1:K
	[B Fit] = lasso(X{t}, Y{t}, 'Lambda', regs, 'CV', 3, 'Alpha', 0.3);  % alpha=1 --> lasso
	W(:,t) = [B(:,Fit.IndexMinMSE) ; Fit.Intercept(Fit.IndexMinMSE)];
	Xtest{t} = [Xtest{t} ones(size(Xtest{t},1),1)];
	Ypred = Xtest{t} * W(:,t);
	X{t} = [X{t} ones(size(X{t},1),1)];
	%[B Fit] = lassoglm(X{t}, tempY, 'binomial', 'Lambda', regs, 'CV', 3, 'Alpha', 1);  % alpha=1 --> lasso
	%W(:,t) = B(:,Fit.Index1SE);
	%Ypred = glmval([Fit.Intercept(Fit.Index1SE); B(:,Fit.Index1SE)],Xtest{t},'logit');
	[Xpr,Ypr,Tpr,AUCpr] = perfcurve(Ytest{t}, Ypred, 1, 'xCrit', 'reca', 'yCrit', 'prec');
	auc(t)=AUCpr;
	disp(sprintf('Initial Task %d AUCPR: %f',t,AUCpr));
end

disp(sprintf('Initial Avg AUC-PR: %f',mean(auc)));
%W = randn(J,K);
W0=W;
disp('Finished centering data, initializing single task Ws..');

% print initial obj
for task=1:K
  task_obj(task) = sum(sum((Y{task} - X{task}*W(:,task)).^2))/2;
end
fprintf('Initial sqerr: %f\n',sum(task_obj));

%return;

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
U0=U;
figure(1);
subplot(1,2,1);
imagesc(U');
set(gca,'YTick',[1:K]);
set(gca,'YTickLabel',taskNames);
subplot(1,2,2);
imagesc(W0);
title('U0 and W0');
%print('realdata_output/N200J3000/U_W0_ridge_new.jpg','-djpeg99');
close;
%U = rand(Tpa, K);
%U = 1/K*ones(Tpa, K);

%figure;
%return;

% call altmin
opts=[];
opts.maxiter=2;		 % =1,2
opts.lambda=1;  % =1
opts.mu=0.1;				 % =0.1
opts.norm = 'l2';
opts.rho = 1*ones(Tpa,1);
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
for ll=[0.5] % [2.5 0.5] % 0.5 0.01] % 0.01 1 0.001] % 0.01 0.001 0.0001]; % 0.5 5 10 0.001 0.01 0.1 20] % 1 3
for mm=[0.001] %[0 0.01]; %0.1 0.01]; % 0.1 0.15 0.2 0.25 0.05 0.01 0.2 1 5] % 10 0.1 0.01 0.001] % 1 3
	opts.lambda = ll;
	opts.mu = mm; % best mu = 0.35
for i=1:1
	U = rand(Tpa, K);
	opts.rho = rand(Tpa,1);
	[U W objVal] = altmin_auc(X, Y, Xtest, Ytest, W0, U, opts);
	dlmwrite(sprintf('realdata_output/N200J3000/clus%d_obj%d_lambda%g_mu%g.txt',numClus,i,opts.lambda,opts.mu),objVal);
	save(sprintf('realdata_output/N200J3000/clus%d_W_lambda%g_mu%g.mat',numClus,opts.lambda,opts.mu), 'W');
	figure(i);
	subplot(1,2,1);
	imagesc(U');
	set(gca,'YTick',[1:K]);
	set(gca,'YTickLabel',taskNames);
	subplot(1,2,2);
	imagesc(W);
	saveas(gcf,sprintf('realdata_output/N200J3000/clus%d_U_W%d_lambda%g_mu%g.fig',numClus,i,opts.lambda,opts.mu));
	print(sprintf('realdata_output/N200J3000/clus%d_U_W%d_lambda%g_mu%g.jpg',numClus,i,opts.lambda,opts.mu), '-djpeg99');
	close;
	for t=1:K
		Ypred = Xtest{t} * W(:,t); 
		[Xpr,Ypr,Tpr,AUCpr] = perfcurve(Ytest{t}, Ypred, 1, 'xCrit', 'reca', 'yCrit', 'prec');
		auc(t)=AUCpr;
		disp(sprintf('Lambda: %g Mu: %g MTL-Final Task-%d AUCPR: %f -- %d',opts.lambda,opts.mu,t,auc(t),i));
	end
	disp(sprintf('Lambda: %g Mu: %g MTL-Final Avg AUC: %f -- %d',opts.lambda,opts.mu,mean(auc),i));
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
