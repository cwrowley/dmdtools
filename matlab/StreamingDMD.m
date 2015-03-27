%STREAMINGDMD  incrementally updated Dynamic Mode Decomposition
% Usage:
%   sdmd = StreamingDMD
%   sdmd = StreamingDMD(r)
%
%   r: the maximum dimension of the subspace on which DMD modes are
%      computed, using a POD-truncation version of the algorithm.
%      If r=0 (the default), use the direct algorithm, equivalent
%      to a batch-processed DMD.
%
%   A StreamingDMD object has two methods:
%     update         adds a pair of snapshots to the dataset
%     compute_modes  computes the DMD modes and eigenvalues
%
%   For details, see
%       M.S. Hemati, M.O. Williams, C.W. Rowley,
%       ``Dynamic Mode Decomposition for Large and Streaming Datasets,''
%       submitted to Physics of Fluids (2014).
%
% Tested using Matlab 8.1.0.604 (R2013a) for Mac OSX.
%
% Authors:
%   Maziar S. Hemati
%   Matthew O. Williams
%   Clarence W. Rowley
%
% Created:
%   August 2014.
%
%   See also: StreamingDMD/UPDATE, StreamingDMD/COMPUTE_MODES, sdmd_run
%
% Reference page in Help browser:
%   <a href="matlab:doc StreamingDMD">doc StreamingDMD</a>

classdef StreamingDMD
    properties
        count = 0;    % number of pairs of snapshots processed
        max_rank = 0; % maximum allowable rank of DMD operator
        Qx;           % orthonormal basis for the image of X  (see Eq. (3))
        Qy;           % orthonormal basis for the image of Y  (see Eq. (3))
        A;  % the matrix $A$ needed to compute $\tilde{K}$  (see Eq. (3))
        Gx; % the matrix $G_X$ needed to compute $\tilde{K}$  (see Eq. (3))
        Gy; % the matrix $G_Y$ needed to project onto a truncated POD basis
    end

    methods
        function obj = StreamingDMD(max_rank)
            % StreamingDMD  Create an object for incrementaly updated DMD.
            %   sdmd = StreamingDMD  or
            %   sdmd = StreamingDMD(0)  Use the direct algorithm
            %        (equivalent to batch-processed DMD)
            %
            %   obj = StreamingDMD(max_rank)  Use a POD-truncation
            %     version of the algorithm, with max_rank corresponding
            %     to the truncation threshold (i.e., the maximum number
            %     of POD modes retained).

            if nargin == 1
                obj.max_rank = max_rank;
            end
        end

        function obj = update(obj, x, y)
            % UPDATE  Update the DMD computation with a pair of snapshots
            %
            %   sdmd = sdmd.UPDATE(x,y) adds the pair of snapshots (x,y)
            %     to the data ensemble.  Here, if the (discrete-time)
            %     dynamics are given by z(n+1) = f(z(n)), then (x,y)
            %     should be measurements correponding to consecutive states
            %     z(n) and z(n+1).

            % parameters
            ngram = 5;      % number of times to reapply Gram-Schmidt
            epsilon = eps;  % tolerance for expanding the bases

            obj.count = obj.count + 1;

            % compute snapshot norms
            normx = norm(x);
            normy = norm(y);

            % ---- Process the First Iterate ----
            if obj.count == 1
                % construct bases
                obj.Qx = x / norm(x);
                obj.Qy = y / norm(y);

                % compute
                obj.Gx = normx^2;
                obj.Gy = normy^2;
                obj.A = normx * normy;
                return
            end

            % ---- ALGORITHM STEP 1 ----
            % classical Gram-Schmidt reorthonormalization
            xtilde = zeros(size(obj.Qx,2),1);
            ytilde = zeros(size(obj.Qy,2),1);
            ex = x;
            ey = y;
            for igram = 1:ngram
                dx = obj.Qx' * ex;
                dy = obj.Qy' * ey;
                xtilde = xtilde + dx;
                ytilde = ytilde + dy;
                ex = ex - obj.Qx * dx;
                ey = ey - obj.Qy * dy;
            end

            % ---- ALGORITHM STEP 2 ----
            % check basis for x and expand, if necessary
            if norm(ex) / normx > epsilon
                % update basis for x
                obj.Qx = [obj.Qx ex/norm(ex)];
                % increase size of Gx and A (by zero-padding)
                obj.Gx = [obj.Gx zeros(size(obj.Gx,1),1); zeros(1,size(obj.Gx,2)+1)];
                obj.A = [obj.A zeros(size(obj.A,1),1)];
            end

            % check basis for y and expand, if necessary
            if norm(ey) / normy > epsilon
                % update basis for y
                obj.Qy = [obj.Qy ey/norm(ey)];
                % increase size of Gy and A (by zero-padding)
                obj.Gy = [obj.Gy zeros(size(obj.Gy,1),1); zeros(1,size(obj.Gy,2)+1)];
                obj.A = [obj.A; zeros(1,size(obj.A,2))];
            end

            % ---- ALGORITHM STEP 3 ----
            % check if POD compression is needed
            r0 = obj.max_rank;
            if r0
                if size(obj.Qx,2) > r0
                    [evec, eval] = eig(obj.Gx);
                    [eval, indx] = sort(diag(eval),'descend');
                    qx = evec(:,indx(1:r0));
                    obj.Qx = obj.Qx * qx;
                    obj.A = obj.A * qx;
                    obj.Gx = diag(eval(1:r0));
                end
                if size(obj.Qy,2) > r0
                    [evec, eval] = eig(obj.Gy);
                    [eval, indx] = sort(diag(eval),'descend');
                    qy = evec(:,indx(1:r0));
                    obj.Qy = obj.Qy * qy;
                    obj.A = qy' * obj.A;
                    obj.Gy = diag(eval(1:r0));
                end
            end

            % ---- ALGORITHM STEP 4 ----
            xtilde = obj.Qx' * x;
            ytilde = obj.Qy' * y;

            % update A and Gx
            obj.A = obj.A + ytilde * xtilde';
            obj.Gx = obj.Gx + xtilde * xtilde';
            obj.Gy = obj.Gy + ytilde * ytilde';
        end

        function [modes, evals] = compute_modes(obj)
            % COMPUTE_MODES  Compute DMD modes and eigenvalues
            %
            % [V, lam] = sdmd.COMPUTE_MODES produces a vector lam
            %   of DMD eigenvalues, and a matrix V whose columns are
            %   DMD modes of the current dataset.

            Ktilde = obj.Qx' * obj.Qy * obj.A * pinv(obj.Gx);
            [evecK, evals] = eig(Ktilde);
            evals = diag(evals);
            modes = obj.Qx * evecK;
        end
    end
end
