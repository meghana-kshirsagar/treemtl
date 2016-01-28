function [Beta, obj, time, iter] = accgrad( bw, Y, X, lambda, T, XX, XY, C, g_idx, L, mu, option)

%Y Centered Matrix: N by K
%X Centered Matrix: N by J(p)
%lam: lambda
%T: sparse matrix: group info. rows: number of group, cols: number of tasks
%Tw: n_group by 1: weight for each group
%C: note \sum_|g| by K -- contains weights of each treenode (i.e the weights from Tw)
%g_idx: n_group by 2, group index
%L, Lipschitz cond
%TauNorm: \|\Tau\|_1,2^2 
%mu: mu in nesterov paper
%maxiter

    [J] = size(X{1},2); % number of features
    [K] = size(T,2); % number of tasks
    
    if isfield(option,'maxiter')
        maxiter=option.maxiter;
    else
        maxiter=1000;
    end
    
    if isfield(option, 'tol')
        tol=option.tol;
    else
        tol=1e-7;
    end
    
    if isfield(option, 'threshold')
        threshold=option.threshold;
    else
        threshold=1e-4;
    end  
    
    if isfield(option, 'verbose')
        verbose=option.verbose;
    else
        verbose=true;
    end  

    obj=zeros(1,maxiter);
    time=zeros(1,maxiter);

  	% Each row of C is non-zero at the entry corresponding to the leaf it ends in.
		% multiply by regularization parameter
    C=C*lambda;
    
    %bw=zeros(J,K);    % maps to W in our notation: J x m (m=#tasks) -- we will pass this
    bx=bw;    
    theta=1;
    tic
    for iter=1:maxiter
				% compute gradient
				% C*bw'/mu : passed to shrinking op, see eq 3.7  (= lambda*w_v*B/mu)
				% C: sum_|g|*K,    C*bw': sum_|g| * J
        R=shrink(C*bw'/mu, g_idx); 
                
				for task=1:K
        	grad_bw(:,task) = XX{task}*bw(:,task) - XY{task} + R'*C(:,task); % Step-1 from paper
				end
        
        bv=bw-1/L*grad_bw; % compute update
        
        bx_new=sign(bv).*max(0,abs(bv)-lambda/L); % soft-thresholding 
                
				for task=1:K
					task_obj = sum(sum((Y{task} - X{task}*bx_new(:,task)).^2))/2;
        	obj(iter) = obj(iter) + task_obj;
				end
				obj(iter) = obj(iter) + cal2norm(C*bx_new', g_idx);
        
        theta_new=2/(iter+2);  % Step-3 from paper
        
        bw=bx_new+(1-theta)/theta*theta_new*(bx_new-bx); % Step-4 from paper
        
        time(iter)=toc;
        
        if (verbose && (iter==1 || mod(iter,1)==0))
            fprintf('Iter %d: Obj: %g\n', iter, obj(iter));    
        end         
         
        theta=theta_new;
        bx=bx_new;
        
        if (iter>10 && (abs(obj(iter)-obj(iter-1))/abs(obj(iter-1))<tol)) %increasing
            break;
        end        
            
    end
    
    fprintf('In total: Iter: %d, Obj: %g\n', iter, obj(iter));
    
    bx(abs(bx)<threshold) =0;
    Beta=bx;
    obj=obj(1:iter);
    time=time(1:iter);
    
end
