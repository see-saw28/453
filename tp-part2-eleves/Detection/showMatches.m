function showMatches(im1,im2,posr1,posc1,bestr,bestc)

    if size(im1,3) > 1 | size(im2,3) > 1 
        fprintf('input image must be graylevel\n');
        return
    end
    
    if length(posr1) ~= length(posc1) | length(bestc) ~= length(bestr)
        fprintf('different number of coordinates for 2D point list');
        return
    end
    
    if length(posr1) ~= length(bestr) 
        fprintf('not the same number of pairs in the two images');
        return
    end
    
    
    im1Temp = showCorners(posr1, posc1, im1);
    im2Temp = showCorners(bestr, bestc, im2);
    imResult = [im1Temp im2Temp];
    imshow(uint8(imResult));
    hold on;
    for i = 1:length(posr1)
        plot([posc1(i),bestc(i) + size(im1,2)],[posr1(i),bestr(i)],'Color','r','LineWidth',1)
    end
    
    imwrite(imResult, 'matches.png');
    
end