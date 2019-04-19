% --------------------------------------------------------
% This code is modified based on SphereFace project:
%   https://github.com/wy1iu/sphereface
% Copyright (c) Weiyang Liu, Yandong Wen
% Licensed under The MIT License [see LICENSE for details]
%
% Intro:
% This script is used to align the faces by similarity transformation.
% Here we only use five facial landmarks (two eyes, nose point and two mouth corners).
%
% --------------------------------------------------------


clear;clc;close all;


% imagepath x1 x2 ... x5 y1 y2 ... y5
mark_file = 'data/ldmark_casia_mtcnncaffe.txt';
output_dir = 'data/casia_mtcnncaffe_aligned';
prefix_img = '';


file = fopen(mark_file, 'r');
data = textscan(file, '%s %d %d %d %d %d %d %d %d %d %d %d %d');
imageList = data{1};
facial5points = [data{2:11}];

%% alignment settings
imgSize     = [112, 96];
coord5point = [30.2946, 51.6963;
               65.5318, 51.5014;
               48.0252, 71.7366;
               33.5493, 92.3655;
               62.7299, 92.2041];

%% face alignment
count = 1;
for i = 1:length(imageList)

    facial5point = double(reshape(facial5points(i,:), 5,2));

    % load and crop image
    img = imread([prefix_img '/' imageList{i}]);

    transf   = cp2tform(facial5point, coord5point, 'similarity');
    cropImg  = imtransform(img, transf, 'XData', [1 imgSize(2)],...
                                        'YData', [1 imgSize(1)], 'Size', imgSize);

    % save image
    [sPathStr, name, ext] = fileparts(imageList{i});
    parts = strsplit(imageList{i}, '/');
    tPathStr = [output_dir '/' parts{end-1}];
    if ~exist(tPathStr, 'dir')
       mkdir(tPathStr)
    end
    tPathFull = fullfile(tPathStr, [name ext]);
    imwrite(cropImg, tPathFull, 'jpg');
    if mod(count, 100) == 0
        fprintf('Aligning %dth image\n', count);
        disp(tPathFull);
    end
    count = count + 1;
end
disp('Finish Alignment.')

% end
