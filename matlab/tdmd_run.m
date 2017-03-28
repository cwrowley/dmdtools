% tdmd_run  An example to demonstrate total DMD on noisy snapshot data
%
% Run a simple example of TDMD on a low-rank linear system with noisy
% snapshot measurements, based on Example 1 of
%
%   M.S. Hemati, C.W. Rowley, E.A. Deem, and L.N. Cattafesta
%   ``De-biasing the dynamic mode decomposition for 
%     applied Koopman spectral analysis of noisy datasets,''
%   Theortical and Computational Fluid Dynamics (2017).
%
% The example considers a low-rank n-dimensional linear system with 
% two undamped modes and one dampled mode.
%
% A total of m snapshots are measured sequentially and are subject to
% additive i.i.d. zero-mean Gaussian noise with covariance meas_cov.
%
% The example here runs TDMD and standard DMD based on noisy snapshot data
% from a single realization of the a random initial condition drawn
% from N(1,init_cov).
%
%   see also tdmd.m
%
% Reference page in Help browser:
%   <a href="matlab:doc tdmd_run">doc tdmd_run</a>
%

%% set example parameters
m = 100;   % number of snapshots
n = 250;   % number of states
r = 6;     % truncation level
dt = 0.01; % time step size
meas_cov = .05;  % Measurement covariance
init_cov = 0.1;

rng(0);  % Seed random number generator

%% Construct system to be studied

% specify characteristic frequencies and growth/decay rates
% associated with continuous-time dynamics
f = [1.0 2.5 5.5];
g = [0 0 -.3];

assert(length(f)==length(g))

% construct low-rank continuous-time operator (rank=k)
k = 2*length(f);  % (to perform DMD/TDMD with correct rank, set r=k)
A1 = [];
for ii = 1:length(f)
    A2 = [[g(ii) 2*pi*f(ii);-2*pi*f(ii) g(ii)]];
    A1 = [A1 A2];
end
Alowrank = [];
for ii = 1:length(f)
    Alowrank = blkdiag(Alowrank,A1(:,(ii-1)*2+1:2*ii));
end

% handle to low-dimensional operator for simulations
dynsys = @(t,x) Alowrank*x;  % handle to operator

% construct map to full state-dimension (Q: X^r --> X^n)
[Q,~] = qr(randn(n,k),0);

% handle to generate noisy snapshots from low-dimensional simulation output
getsnaps = @(x) Q*x + sqrt(meas_cov)*randn(n,m+1);

%% Generate snapshot data (with addititive synthetic noise)
x0 = 1+randn(k,1)*sqrt(init_cov);
t = dt*[0:m];
[~,y] = ode45(dynsys,t,x0);
snapshots = getsnaps(transpose(y));

X1 = snapshots(:,1:m);
X2 = snapshots(:,2:m+1);

%% Perform Standard DMD (with rank reduction)
[Qx, S, V] = svd(X1,0);
Qx = Qx(:,1:r);
S = S(1:r,1:r);
V = V(:,1:r);
Atilde = Qx' * X2 * V * pinv(S);
dmd_evals = eig(Atilde);
        
        
%% Perform TDMD
[tdmd_evals,~] = tdmd(X1,X2,r);

%% Plot results
true_evals = exp(eig(Alowrank)*dt);
th = [0:.01:2*pi 0.01];

[setfonts] = @(h) set(h,'FontSize',14);
[setmarkers] = @(h,col) set(h,'MarkerSize',8,'MarkerEdgeColor',...
                            col,'MarkerFaceColor',col);


h=plot(real(true_evals),imag(true_evals),'s'); setmarkers(h,'k'); hold on;
h=plot(real(dmd_evals),imag(dmd_evals),'o'); setmarkers(h,'g');
h=plot(real(tdmd_evals),imag(tdmd_evals),'^'); setmarkers(h,'m');
plot(cos(th),sin(th),'k--'); % unit circle
axis equal

h=xlabel('Re(\lambda)'); setfonts(h);
h=ylabel('Im(\lambda)'); setfonts(h);
h=legend('True','DMD','TDMD','Unit Circle'); setfonts(h);
setfonts(gca);