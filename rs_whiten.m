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

% RS: original code from Github (leave this in function as reference!!)
% 
% 
% def whiten(X, num_components=None, center=True, rowvar=True):
%     """Whiten the data in matrix X using PCA decomposition.
%     The data corresponds to n samples of a p-dimensional random vector. The shape
%     of the matrix can be either (n, p) if each row is considered to be a sample or
%     (p, n) if each column is considered to be a sample. How to read the matrix entries
%     is specified by the rowvar parameter. Before whitening, a dimensionality reduction
%     step can be applied to the data to reduce the p dimensions of each sample to
%     num_components dimensions. If num_components is None, the number of dimensions kept
%     is the maximum possible (nº of non-zero eigenvalues). For example, if X is full rank
%     (rank(X) = min(n, p)), then num_components = p if p < n, and num_components = n-1
%     if p >= n.
%     Args:
%         X: Data matrix.
%         num_components: Number of PCA dimensions of the whitened samples.
%         center: Whether to center the samples or not (zero-mean whitened samples).
%         rowvar: Whether each row of X corresponds to one of the p variables or not.
%     Returns:
%         (Z, V): The whitened data matrix and the whitening matrix.
%     """
%     r = num_components
% 
%     if rowvar:
%         X = X.transpose()
% 
%     # Data matrix contains n observations of a p-dimensional vector
%     # Each observation is a row of X
%     n, p = X.shape
% 
%     # Arbitrary (but sensible) choice. In any case, we remove the eigenvectors of 0 eigenvalue later
%     if r is None:
%         r = min(n, p)
% 
%     # Compute the mean of the observations (p-dimensional vector)
%     mu = np.mean(X, axis=0)
% 
%     # If p > n compute the eigenvectors efficiently
%     if p > n:
%         # n x n matrix
%         M = np.matmul((X-mu), (X-mu).transpose())
% 
%         # Eigenvector decomposition
%         vals, vecs = np.linalg.eig(M)
%         vals, vecs = vals.real, vecs.real
% 
%         # Sort the eigenvectors by "importance" and get the first r
%         pairs = sorted([(vals[i], vecs[:, i]) for i in range(len(vals))], key=lambda x: x[0], reverse=True)
%         pairs = [p for p in pairs if abs(p[0]) > 1e-10]  # Remove the eigenvectors of 0 eigenvalue
%         pairs = pairs[:r]
% 
%         # nxr matrix of eigenvectors (each column is an n-dimensional eigenvector)
%         E = np.array([p[1] for p in pairs]).transpose()
% 
%         # pxr matrix of the first r eigenvectors of the covariance of X
%         # Note that we normalize!
%         E = np.matmul((X-mu).transpose(), E)
%         E /= np.linalg.norm(E, axis=0)
% 
%         # Eigenvalues of cov(X) to the -1/2
%         # Note that we rescale the eigenvalues of M to get the eigenvalues of cov(X)!
%         diag = np.array([1/np.sqrt(p[0]/(n-1)) for p in pairs])
% 
%     else:
%         # p x p matrix
%         C = np.cov(X, rowvar=False)
% 
%         # Eigenvector decomposition
%         vals, vecs = np.linalg.eig(C)
%         vals, vecs = vals.real, vecs.real
% 
%         # Sort the eigenvectors by "importance" and get the first r
%         pairs = sorted([(vals[i], vecs[:, i]) for i in range(len(vals))], key=lambda x: x[0], reverse=True)
%         pairs = [p for p in pairs if abs(p[0]) > 1e-10]  # Remove the eigenvectors of 0 eigenvalue
%         pairs = pairs[:r]
% 
%         # pxr matrix of the first r eigenvectors of the covariance of X
%         E = np.array([p[1] for p in pairs]).transpose()
% 
%         # Eigenvalues of cov(X) to the -1/2
%         diag = np.array([1/np.sqrt(p[0]) for p in pairs])
% 
%     # Warn that the specified number of components is larger
%     # than the number of non-zero eigenvalues.
%     if num_components is not None:
%         if num_components > len(pairs):
%             warnings.warn(
%                 'The desired number of components (%d) is larger than the actual dimension'
%                 ' of the PCA subespace (%d)' % (num_components, len(pairs))
%             )
% 
%     # Center and whiten the data
%     if center:
%         X = X - mu
% 
%     # Whitening matrix
%     V = E * diag
% 
%     # White data
%     Z = np.matmul(X, V)
% 
%     if rowvar:
%         Z = Z.transpose()
% 
%     # Since X is assumed to be (n, p) through the computations, the current
%     # whitening matrix V is in fact the transpose of the actual whitening matrix.
%     # Observation: If z = V * x for random column vectors x, z, then Z = X * V
%     # for the (n, p) and (n, r) matrices X, Z of observations of x, z.
%     V = V.transpose()
% 
%     return Z, V