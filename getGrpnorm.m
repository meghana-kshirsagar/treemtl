function s=getGrpnorm(A, g_idx, rho, normType)
         V=size(g_idx,1); % num groups
         s=0;
         for v=1:V
             gnorm=sqrt(sum((A*g_idx(v,:)').^2, 2)); % Jx1 .. l2norm over groups
						 if normType == 'l1'
	             s=s+rho(v)*sum(gnorm);
						 else
	             s=s+rho(v)*sum(gnorm).^2;
						 end
         end
end
