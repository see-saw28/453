function imResult = showCorners(posr, posc, im)

if size(im,3) > 1
    fprintf('input image must be graylevel\n');
    return
end

imResult = cat(3, im, im, im);

for i = 1:length(posr)
    r = posr(i); c = posc(i);
    imResult(r-1:r+1,c,1) = 255;
    imResult(r-1:r+1,c,2) = 0;
    imResult(r-1:r+1,c,3) = 0;
    imResult(r,c-1:c+1,1) = 255;
    imResult(r,c-1:c+1,2) = 0;
    imResult(r,c-1:c+1,3) = 0;
end
imshow(uint8(imResult))


end