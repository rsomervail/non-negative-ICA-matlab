%
%
%   whitening function adapted from the one in this link:
%       https://github.com/marcromani/cocktail/blob/master/src/cocktail/cocktail.py
%
%       - Richard Somervail
%
%
function  [Z, V] = whiten(X, num_components, center)

    r = num_components; 

    % Each observation is a row of X
    X = X'; 
    [n, p] = size(X);

    % Compute the mean of the observations (p-dimensional vector)
    mu = mean(X);

    % If p > n compute the eigenvectors efficiently
    if p > n % RS: this is rarely the case
        
        error 'rs_whiten: more channels than data'

    else
        % p x p matrix
        C = cov(X);

        % Eigenvector decomposition 
        [vecs, vals] = eig(C, 'vector');  %   vals, vecs = np.linalg.eig(C) 
        vals = real(vals); vecs = real(vecs); 

        % Sort the eigenvectors by "importance" and get the first r
        for i = 1:length(vals)
            pairs(i,:) = [ vals(i) vecs(:, i)' ];  % RS: could do this as a cell array instead and use cellfun? not sure
        end
        pairs = sortrows( pairs  , 1, 'descend'  ); % descending order
        pairs = pairs( abs(pairs(:,1)) > 1e-10, : ); % Remove the eigenvectors of 0 eigenvalue
        pairs = pairs(1:r,:); % RS: only get the first r eigenvectors (r = num_components)

        % pxr matrix of the first r eigenvectors of the covariance of X
        E = pairs(:,2:end)';

        % Eigenvalues of cov(X) to the -1/2
        diag =  1 ./ sqrt( pairs(:,1) )'; % diag = np.array([1/np.sqrt(p[0]) for p in pairs])
            
    end % p > n
    
    % Center the data
    if center
        X = X - mu; % RS: center data (not used for non-negative ICA)
    end
    
    % Whitening matrix
    V = E .* diag;

    % Whiten the data
    Z = X * V;
    
    % Transpose whitening matrix for output
    V = V';
    
end
