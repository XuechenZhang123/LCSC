function im = upscaling2D(x, scale)

[row, col, ch] = size(x);
row_new = row * scale;
col_new = col * scale;

im = single(zeros(row_new, col_new, ch));

for i =  1 : scale
    for j = 1 : scale
       im(i:scale:end, j:scale:end, :) = x;
    end
end
    

end