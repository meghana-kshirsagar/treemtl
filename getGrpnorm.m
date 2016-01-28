function s=getGrpnorm(A, g_idx, rho)
         V=size(g_idx,1);
         s=0;
         for v=1:V
             gnorm=sqrt(sum((A*g_idx(v,:)).^2, 2)); % Jx1
             s=s+rho(v)*sum(gnorm);
         end
end
