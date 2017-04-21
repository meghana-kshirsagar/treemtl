
function [U W objVal] = altmin(X, Y, Xtest, Ytest, W0, U0, opts)

% X: 			centered data: cell array size K, each element Nt by J
% Y: 			cell array size K, each element vector size Nt

U = U0;
W = W0;
[Tpa K] = size(U);

disp('Initializing ...');
% precompute XX and XY
for task=1:K
  XX{task} = X{task}'*X{task};
  XY{task} = X{task}'*Y{task};
end

% print initial obj
for task=1:K
  task_obj(task) = sum(sum((Y{task} - X{task}*W(:,task)).^2))/2;
end
fprintf('[ALTMIN] Initial sqerr: %f\n',sum(task_obj));

%figure;
obj=zeros(1,opts.maxiter);
for iter=1:opts.maxiter

	% update U
	%if opts.type == 'fusion'
		[U change] = updateU_fusion(W, U, opts);
		if(change<=1e-3)
			break;
		end
	%else
	%U = updateU(W, U, opts);
	fprintf('Finished updating U.... \n Now updating W...\n');

	Wold=W;
	% update W
  	W = sqGroupLasso(W, Y, X, Ytest, Xtest, XX, XY, U, opts); 

	fprintf('Change in W: %f\n',norm(W-Wold,'fro'));

	% print objective
	for task=1:K
    	task_obj(task) = sum(sum((Y{task} - X{task}*W(:,task)).^2))/2;
		%fprintf('[Task %d] sq.err: %f\n',task,task_obj);
	end
	grpnorm = getGrpnorm(W, U, opts.rho, opts.norm);
	fusion = getFusion(U,K);
	fprintf('[GLOBAL] Lambda: %f Mu: %f Iter: %d  lsqerr: %f  reg: %f  fusion: %f\n', opts.lambda, opts.mu, iter, sum(task_obj), opts.lambda*grpnorm, opts.mu*fusion);
	obj(iter) = sum(task_obj) + opts.lambda*grpnorm + opts.mu*fusion;
	fprintf('[GLOBAL] Lambda: %f Mu: %f Obj: %f\n',opts.lambda,opts.mu,obj(iter));
	disp('----------------------------');

end

objVal = obj(iter);

nex=200;
for t=1:K
	Ypred = Xtest{t}(1:nex,:) * W(:,t); 
	mse(t) = sum((Ypred-Ytest{t}(1:nex)).^2)/size(Ypred,1);
	r2(t) = rsquare(Ypred,Ytest{t}(1:nex));
	disp(sprintf('Lambda: %g Mu: %g CV Task-%d Test-MSE: %f R2: %g',opts.lambda,opts.mu,t,mse(t),r2(t)));
end
disp(sprintf('Lambda: %g Mu: %g CV Avg Test-MSE: %f R2: %g',opts.lambda,opts.mu,mean(mse),mean(r2)));


end % -- end function
