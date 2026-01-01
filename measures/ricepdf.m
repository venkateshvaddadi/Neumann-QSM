function y = ricepdf(x, v, s)

    s2 = s.^2; % (neater below)
    try
        y = (x ./ s2) .*exp(-0.5 * (x.^2 + v.^2) ./ s2) .*besseli(0, x .* v ./ s2);
        y(x <= 0) = 0;
    catch
        error('ricepdf:InputSizeMismatch',...
            'Non-scalar arguments must match in size.');
end