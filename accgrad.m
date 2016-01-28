function [Beta, obj, time, iter] = accgrad( bw, Y, X, XX, XY, g_idx, lambda, rho, option)

%Y Centered Matrix: N by K
%X Centered Matrix: N by J(p)
%lam: lambda
%C: note \sum_|g| by K -- contains weights of each treenode (i.e the weights from Tw)
%g_idx: n_group by 2, group index
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
    
    obj=zeros(1,maxiter);
    time=zeros(1,maxiter);

    tic
		C = 2*lambda*rho;
		num_grps=size(g_idx,1);
    for iter=1:maxiter
				% compute gradient..  % bw: J x K, g_idx: num_grps x K
				featnorm = zeros(num_grps,J);
				for g=1:num_grps
					gbw = (bw*g_idx(g,:)');
					featnorm(g,:) = sqrt(sum(gbw.^2,2));
				end
				sumf = sum(featnorm,2); % size: num_grps x 1

				for task=1:K
					myg = find(g_idx(:,task)==1);
        	grad_bw(:,task) = XX{task}*bw(:,task) - XY{task} + C(task)*bw(:,task)*sumf(myg) ./ featnorm(myg,:)'; 
				end
        
        bv=bw-option.eta*grad_bw; % compute update
        
        bx_new=sign(bv).*max(0,abs(bv)-repmat(rho',J,1)*lambda*option.eta); % soft-thresholding 
                
				for task=1:K
					task_obj = sum(sum((Y{task} - X{task}*bx_new(:,task)).^2))/2;
        	obj(iter) = obj(iter) + task_obj;
				end
				obj(iter) = obj(iter) + cal2norm(bx_new, g_idx);
        
        bw=bx_new;
        
        time(iter)=toc;
        
        if (verbose && (iter==1 || mod(iter,1)==0))
            fprintf('Iter %d: Obj: %g\n', iter, obj(iter));    
        end         
         
        if (iter>10 && (abs(obj(iter)-obj(iter-1))/abs(obj(iter-1))<tol)) %increasing
            break;
        end        
            
    end
    
    fprintf('In total: Iter: %d, Obj: %g\n', iter, obj(iter));
    
    bw(abs(bw)<threshold) =0;
    Beta=bw;
    obj=obj(1:iter);
    time=time(1:iter);
    
end
