function showCircle(row,col,rad,im)

if size(im,3) > 1
    imResult = im;
else
    imResult = cat(3, im, im, im);
end

for i = 1:length(row)
    
    %la petite croix
    r = row(i); c = col(i);
    imResult(r-1:r+1,c,1) = 255;
    imResult(r-1:r+1,c,2) = 0;
    imResult(r-1:r+1,c,3) = 0;
    imResult(r,c-1:c+1,1) = 255;
    imResult(r,c-1:c+1,2) = 0;
    imResult(r,c-1:c+1,3) = 0;
end

imshow(uint8(imResult))
hold on;


    
    
for i = 1:length(row)
    ang=0:0.01:2*pi; 
    xp=rad(i)*cos(ang);
    yp=rad(i)*sin(ang);
    plot(col(i)+yp, row(i)+xp,'r');
end



end
