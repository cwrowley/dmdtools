%STDMD_RUN An example to demonstrate incrementally updated DMD.
%
%   Run a simple example of incrementally updated TDMD on a toy problem to
%   demonstrate the streaming TDMD algorithm, as described in:
%   
%       M.S. Hemati, M.O. Williams, C.W. Rowley,
%       ``Dynamic Mode Decomposition for Large and Streaming Datasets,''
%       Physics of Fluids, vol. 26, 111701 (2014).
%
%       M.S. Hemati, C.W. Rowley,
%       ``De-Biasing the Dynamic Mode Decomposition for Applied Koopman 
%       Spectral Analysis,'' submitted to PNAS (2015).
%
%   The example considers an arbitrary n-dimensional dynamical system with 
%   two characteristic frequencies.
%
%   A total of m snapshot pairs are measured sequentially and are subject 
%   to additive i.i.d. zero-mean Gaussian noise with covariance noise_cov.
%
%   Streaming versions of both DMD and TDMD are run side-by-side for
%   comparison.  The StreamingDMD and StreamingTDMD Matlab classes are
%   used to perform incremental updates to the DMD and TDMD computations,
%   respectively.  Both methods make use of a data compression procedure to
%   ensure that the analysis only retains up to a specified number of 
%   DMD modes, given by the truncation threshold `max_rank'.
%
% Tested using Matlab 8.1.0.604 (R2013a) for Mac OSX.
%
% Authors:
%   Maziar S. Hemati
%   Matthew O. Williams
%   Clarence W. Rowley
%
% Created:
%   March 2015.
%
%   see also StreamingTDMD
%
% Reference page in Help browser:
%   <a href="matlab:doc stdmd_run">doc stdmd_run</a>


% set example parameters here
max_rank = 0;      % maximum allowable rank of the DMD operator
                    %   (set to zero for unlimited)
m = 1001;           % total number of snapshots to be processed
n = 41;           % number of states
noise_cov = 1e-1;   % measurement noise covariance
rng(0)              % seed the random number generator

%% define the example system
% random state directions
v1 = randn(n,1);
v2 = randn(n,1);
v3 = randn(n,1);
v4 = randn(n,1);

% characteristic frequencies
f1 = 5.2;
f2 = 1.0;

% sampling time
dt = 1e-2;

dynsys = @(k) v1*cos(2*pi*dt*f1*k) + v2*cos(2*pi*f2*dt*k) + ...
    v3*sin(2*pi*dt*f1*k) + v4*sin(2*pi*f2*dt*k);

get_snapshot = @(k) dynsys(k) + sqrt(noise_cov) * randn(n,1);

%% collect the data

disp('Collecting data')
tic

% streaming algorithm
stdmd = StreamingTDMD(max_rank);
sdmd = StreamingDMD(max_rank);

yk = get_snapshot(0);
for k = 1:m
    xk = yk;
    yk = get_snapshot(k);
    stdmd = stdmd.update(xk, yk);
    sdmd = sdmd.update(xk, yk);
end

elapsed_time = toc;
fprintf('  Elapsed time: %f seconds\n', elapsed_time)


%% compute DMD spectrum
disp('Computing spectrum')
tic
% streaming algorithms
[modestls, evalstls] = stdmd.compute_modes();
[modes, evals] = sdmd.compute_modes();
elapsed_time = toc;
fprintf('  Elapsed time: %f seconds\n', elapsed_time)

% calculate corresponding frequencies
fdmd = abs(angle(evals)) ./ (2 * pi * dt);
ydmd = zeros(length(fdmd),1);
for ii = 1:length(fdmd)
    ydmd(ii) = norm(modes(:,ii)).*abs(evals(ii));
end
ydmd = ydmd./max(ydmd);

fdmdtls = abs(angle(evalstls)) ./ (2 * pi * dt);
ydmdtls = zeros(length(fdmdtls),1);
for ii = 1:length(fdmdtls)
    ydmdtls(ii) = norm(modestls(:,ii)).*abs(evalstls(ii));
end
ydmdtls = ydmdtls./max(abs(ydmdtls));

figure(1)
stem(fdmd,ydmd,'bo-')
hold on;
stem(fdmdtls,ydmdtls,'rx--')
xlabel('Frequency')
ylabel('Magnitude')
legend('Streaming DMD','Streaming TDMD')

figure(2);hold on;
plot(real(evals),imag(evals),'bo',real(evalstls),imag(evalstls),'rx');
hold on;
th = [0:.01:2*pi];
plot(cos(th),sin(th),'k--'); axis equal;
xlabel('Re')
ylabel('Im')
legend('Streaming DMD','Streaming TDMD')
title('DMD Eigenvalues')