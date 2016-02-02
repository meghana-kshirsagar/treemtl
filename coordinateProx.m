function [Beta, obj, iter] = coordinateProx( W, Y, X, XX, XY, U, option)
%U: n_group by K, group index
    
    if ~isfield(option, 'tol')
        option.tol=1e-7;
    end
    if ~isfield(option, 'threshold')
        option.threshold=1e-4;
    end  
		maxiter = option.maxiter_W;
		lambda = option.lambda;
		eta = option.eta;
    
    K = length(X); % number of tasks
    [J] = size(X{1},2); % number of features
		num_grps=size(U,1);
		lambdaG = lambda * ones(num_grps,1);

		grad_W = zeros(J, K);

		% create data structures
		UWsq = zeros(J, K, num_grps);
		Wsq = W.^2;
		for g=1:num_grps
			UWsq(:,:,g) = Wsq * diag(U(g,:)); % J x K
		end

		function gradL = getGradL(jidx, tidx)
			gradL = - W(jidx, tidx) * sum( (Y{tidx} - X{tidx}*W(:,tidx)) .* X{tidx}(:,jidx) );
		end

		function objVal = computeObj()
				for task=1:K
					task_obj(task) = sum(sum((Y{task} - X{task}*W(:,task)).^2))/2;
				end
				objVal = sum(task_obj) + lambda*getGrpnorm(W, U, ones(num_grps,1), option.norm); 
		end

    tic
    obj=zeros(1,maxiter);
    frac=zeros(1,maxiter);

    for iter=1:maxiter

			for j=1:J		% feats

				for t=1:K		% tasks

					notT = [1:(t-1) (t+1):K];
					notJ = [1:(j-1) (j+1):J];
					kappa = reshape( sum(UWsq(j,notT,:)), num_grps, 1);
					zeroG = (kappa == 0);

					sqTerm = sum(sqrt(sum(UWsq(notJ,:,~zeroG), 2)), 1);  % non-zero groups
					sqtDenom = sqrt(sum(UWsq(j,:,~zeroG), 2));
					sqTerm = sqTerm(:) .* U(~zeroG,t) ./ sqtDenom(:) ;
					gradR = 2 * W(j,t) * sum( lambdaG(~zeroG) .* (1 + sqTerm) );
					
					newW = W(j,t) - eta * (getGradL(j, t) + gradR);			% take a step

					softTh = sum(sqrt(sum(UWsq(notJ,:,zeroG), 2)), 1);  % zero groups
					multiplier = 2 * sum( lambdaG(zeroG) .* softTh(:) .* sqrt(U(zeroG,t)) );
        	W(j,t) = sign(newW).*max(0,abs(newW) - multiplier*eta); % soft-thresholding 
                
					% update UWsq
					myg = find(U(:,t)==1);
					UWsq(j,t,myg) = W(j,t).^2 * U(myg,t);

        	%if ((iter==1 || mod(iter,1)==0))
	           % fprintf('Iter %d: Obj: %g\n', iter, computeObj());    
  	      %end         
         
    	    %if (iter>10 && (abs(obj(iter)-obj(iter-1))/abs(obj(iter-1))<tol)) %increasing
      	   %   break;
        	%end        

				end 	% end task loop

      end		% end feats loop

			obj(iter) = computeObj();
			frac(iter) = mean(sum(abs(W)<1e-4)./J)*100;
	  	fprintf('Iter %d: Obj: %g\n', iter, obj(iter));    

    end
    
	  fprintf('Iter %d: Obj: %g\n', iter, computeObj());    
    
    %W(abs(W)<option.threshold) =0;
    Beta=W;
    
		figure;
		subplot(1,2,1);
		plot([1:iter],obj(1:iter));
		subplot(1,2,2);
		plot([1:iter],frac(1:iter));
		pause;


end
