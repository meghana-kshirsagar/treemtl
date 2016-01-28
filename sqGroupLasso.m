
function [Wnew] = sqGroupLasso(W, Y, X, XX, XY, U, lambda, rho)
  option.maxiter=50;
  option.threshold=0;
  option.tol=1e-6;
	option.eta=1e-5;
	lambda=0.01; % regularization parameter
	g_idx = U;
  [Wnew, obj, time] = accgrad(W, Y, X, XX, XY, g_idx, lambda, rho, option);
end
