
function r = ricernd(v, s)
    
    if isscalar(v)
        dim = size(s);
    elseif isscalar(s)
        dim = size(v);
    elseif all(isequal(size(v), size(s)))
        % (both non-scalar, matching)
        dim = size(v); % == size(s)
    else
        error('ricernd:InputSizeMismatch','Sizes of s and v inconsistent.')
    end

    x = s .* randn(dim) + v;
    y = s .* randn(dim);
    r = sqrt(x.^2 + y.^2);
end


