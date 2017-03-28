function [evals,modes] = tdmd(X,Y,r)
%TDMD Compute the "Total" DMD eigenvalues and modes.
%
% Given the unshifted and shifted snapshot data matrices 'X' and'Y',
% respectively, compute the TDMD modes and eigenvalues.
%
% 'r' specifies the reduced-rank level
%
% For further details, see
%
%   M.S. Hemati, C.W. Rowley, E.A. Deem, and L.N. Cattafesta
%   ``De-biasing the dynamic mode decomposition for
%     applied Koopman spectral analysis of noisy datasets,''
%   Theortical and Computational Fluid Dynamics (2017).
%
%   see also tdmd_run.m
%
% Reference page in Help browser:
%   <a href="matlab:doc tdmd">doc tdmd</a>
%


n = size(X,1); m = size(X,2);
assert(size(Y,1)==n);
assert(size(Y,2)==m);

[~,~,Q] = svd([X;Y]);
Q = Q(:,1:r);

Xhat = X*Q;
Yhat = Y*Q;

[U,S,V] = svd(Xhat,0);

Atilde = U'*Yhat*V*pinv(S);

[vtilde,evals] = eig(Atilde);

evals = diag(evals);
modes = U*vtilde;