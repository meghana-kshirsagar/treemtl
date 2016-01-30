function [Beta, obj, time, iter] = accgrad( bw, Y, X, XX, XY, g_idx, option)

%Y Centered Matrix: N by K
%X Centered Matrix: N by J(p)
%C: note \sum_|g| by K -- contains weights of each treenode (i.e the weights from Tw)
%g_idx: n_group by K, group index

    K = length(X); % number of tasks
    [J] = size(X{1},2); % number of features
    
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
    
		maxiter = option.maxiter_W;
    obj=zeros(1,maxiter);
    time=zeros(1,maxiter);

    tic
		lambda = option.lambda;
		rho = option.rho;
		eta = option.eta;
		C = lambda*rho;
		num_grps=size(g_idx,1);
		grad_bw = zeros(J, K);
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
					if option.norm == 'l1'
	        	grad_bw(:,task) = XX{task}*bw(:,task) - XY{task} + C(myg)*bw(:,task) ./ featnorm(myg,:)'; 
					else
        		grad_bw(:,task) = XX{task}*bw(:,task) - XY{task} + 2*C(myg)*bw(:,task)*sumf(myg) ./ featnorm(myg,:)'; 
					end
					zeroes=find(featnorm(myg,:)==0);
					grad_bw(zeroes,task)=0;
				end

        bv=bw-eta*grad_bw; % compute update
        
        %bx_new=sign(bv).*max(0,abs(bv)-lambda*eta); % soft-thresholding 
				bx_new=bv;
                
				for task=1:K
					task_obj = sum(sum((Y{task} - X{task}*bx_new(:,task)).^2))/2;
        	obj(iter) = obj(iter) + task_obj;
				end
				obj(iter) = obj(iter) + lambda*getGrpnorm(bx_new, g_idx, rho, option.norm); % + lambda*sum(sum(abs(bx_new)));
        
        bw=bx_new;
        
        time(iter)=toc;
        
        if ((iter==1 || mod(iter,1)==0))
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
