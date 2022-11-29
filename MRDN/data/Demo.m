% Demo script
% Uncomment each case to see the results
clc;
close all;

img_path = 'C:\\Users\\Tiger\\Desktop\\texture';
save_path = 'C:\\Users\\Tiger\\Desktop\\texture_f';

img_name = dir([img_path '\\' '*.jpg']);
for ii = 1:length(img_name)
    name = img_name(ii).name;
    I = (imread([img_path '\\' name]));
    I = imresize(I,[60 60]);
    S = uint8(tsmooth(I,0.05,6)*255);
    texture = rgb2gray(S)-rgb2gray(I);
    % figure, imshow(I), figure, imshow(texture);
    imwrite(texture, [save_path '\\' name]);
    
    fprintf('Processed: %d / %d\n', ii, length(img_name));
end

% I = (imread('imgs/graffiti.jpg'));
% S = tsmooth(I,0.015,3);
% figure, imshow(I), figure, imshow(S);

% I = (imread('imgs/crossstitch.jpg'));
% S = tsmooth(I,0.015,3);
% figure, imshow(I), figure, imshow(S);


% I = (imread('imgs/mosaicfloor.jpg'));
% S = tsmooth(I, 0.01, 3, 0.02, 5);
% figure, imshow(I), figure, imshow(S);