%STREAMINGTDMD  incrementally updated TLS Dynamic Mode Decomposition
% Usage:
%   stdmd = StreamingTDMD
%   stdmd = StreamingTDMD(r)
%
%   r: the maximum dimension of the subspace on which DMD modes are
%      computed, using a POD-truncation version of the algorithm.
%      If r=0 (the default), use the direct algorithm, equivalent
%      to a batch-processed DMD.
%
%   A StreamingTDMD object has two methods:
%     update         adds a pair of snapshots to the dataset
%     compute_modes  computes the DMD modes and eigenvalues
%
%   For details, see
%       M.S. Hemati, M.O. Williams, C.W. Rowley,
%       ``Dynamic Mode Decomposition for Large and Streaming Datasets,''
%       Physics of Fluids, vol. 26, 111701 (2014).
%
%       M.S. Hemati, C.W. Rowley,
%       ``De-Biasing the Dynamic Mode Decomposition for Applied Koopman 
%       Spectral Analysis,'' submitted to PNAS (2015).
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
%   See also: StreamingTDMD/UPDATE, StreamingTDMD/COMPUTE_MODES, stdmd_run
%
% Reference page in Help browser:
%   <a href="matlab:doc StreamingTDMD">doc StreamingTDMD</a>

classdef StreamingTDMD
    properties
        count = 0;    % number of pairs of snapshots processed
        max_rank = 0; % maximum allowable rank of DMD operator
        Qz;
        Gz;
    end

    methods
        function obj = StreamingTDMD(max_rank)
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

        function obj = update(obj, xin, yin)
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

            % construct augmented snapshot
            z = [xin;yin];
            normz = norm(z);
            n = size(xin,1);
            
            
            % ---- Process the First Iterate ----
            if obj.count == 1
                % construct bases
                obj.Qz = z/normz;
                obj.Gz = normz^2; 
                return
            end
            
            % ---- TLS Stage, Step 1 ----
            % classical Gram-Schmidt reorthonormalization
            ztilde = zeros(size(obj.Qz,2),1);
            ez = z;
            for igram = 1:ngram
                dz = obj.Qz' * ez;
                ztilde = ztilde + dz;
                ez = ez - obj.Qz * dz;
            end
            
            % ---- TLS Stage, Step 2 ----
            % check basis for z and expand, if necessary
            if norm(ez) / normz > epsilon
                % update basis for z
                obj.Qz = [obj.Qz ez/norm(ez)];
                % increase size of Gz (by zero-padding)
                obj.Gz = [obj.Gz zeros(size(obj.Gz,1),1); zeros(1,size(obj.Gz,2)+1)];
            end
            
            
            ztilde = obj.Qz'*z;
            obj.Gz = obj.Gz + ztilde*ztilde';
            
            
            % ---- TLS Stage, Step 3 ----
            % check if POD compression is needed
            
            r0 = obj.max_rank;
            
            if r0
                if size(obj.Qz,2) > r0
                    [evec, eval] = eig(obj.Gz);
                    [eval, indx] = sort(diag(eval),'descend');
                    qz = evec(:,indx(1:r0));
                    obj.Qz = obj.Qz * qz;
                    obj.Gz = diag(eval(1:r0));
                end
            end
            
        end

        function [modes, evals] = compute_modes(obj)
            % COMPUTE_MODES  Compute DMD modes and eigenvalues
            %
            % [V, lam] = sdmd.COMPUTE_MODES produces a vector lam
            %   of DMD eigenvalues, and a matrix V whose columns are
            %   DMD modes of the current dataset.

            n = size(obj.Qz, 1)/2;
            [Qx, Rx] = qr(obj.Qz(1:n, :));
            r0 = obj.max_rank;
            if r0
                Qx = Qx(:,1:r0);
                Rx = Rx(1:r0,:);
            end
            Gx = Rx*obj.Gz*Rx';
            A = Qx'*obj.Qz(n+1:end, :)*obj.Gz*obj.Qz(1:n, :).'*Qx;
            
            Ktilde = A * pinv(Gx);
            [evecK, evals] = eig(Ktilde);
            evals = diag(evals);
            modes = Qx * evecK;
        end
    end
end
