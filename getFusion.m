function fval = getFusion(U, K)
		fval=0;	
    for t=2:K
      fval = fval + sum( sum( (repmat(U(:,t),1,(t-1)) - U(:,[1:t-1])).^2 )) ;
    end		
end

