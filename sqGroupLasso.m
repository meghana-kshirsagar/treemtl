
function [Wnew] = sqGroupLasso(W, Y, X, XX, XY, U, opts)
  opts.threshold=0;
  opts.tol=1e-6;
	opts.eta=1e-6;
	g_idx = U;
  [Wnew, obj, time] = accgrad(W, Y, X, XX, XY, g_idx, opts);
end
