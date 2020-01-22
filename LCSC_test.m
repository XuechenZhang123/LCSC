close all;clear all;
run matconvnet/matlab/vl_setupnn;
addpath('utils')

%%Set parameter
scale = 3;
backbone_name = ['lcsc76_x', num2str(scale), '_backbone.mat'];
load(backbone_name);
backbone_model = lcsc;
fusion_name = ['lcsc76_x', num2str(scale), '_fusion.mat'];
load(fusion_name);
fusion_model = attention;


im_path = 'Data/Set5';
im_dir = dir( fullfile(im_path, '*bmp') );
im_num = length( im_dir );

for img = 1:im_num
    fprintf('imge %d\n', img);
    X = imread( fullfile(im_path, im_dir(img).name));
    grd = X;
    if size(X,3) == 3
        X = rgb2ycbcr(X);
        X = double(X(:,:, 1));
    else
        X = double(X);
    end
    X = modcrop(X, scale);
    grd = modcrop(grd, scale);
    X = double(X);
    [row, col, ~] = size(X);

 
%%Generate HR Y-channel results
    im_l = double((imresize(X, 1/scale, 'bicubic')))/255;
    im_bicubic = double((imresize(imresize(X, 1/scale, 'bicubic'), [row, col], 'bicubic')))/255;
    sr = double(LCSC(im_l, backbone_model, fusion_model, scale));
    sr = sr + im_bicubic;
    im_h = sr * 255;
    
%%Generate HR results
    lr = imresize(grd, 1/scale, 'bicubic');
    if size(lr, 3) == 3
        lr = rgb2ycbcr(lr);
        xcb = lr(:,:,2);
        xcr = lr(:, :, 3);
        bic(:, :, 1) = uint8(im_h);
        bic(:, :, 2) = imresize(xcb, scale, 'bicubic');
        bic(:, :, 3) = imresize(xcr, scale, 'bicubic');
        our = ycbcr2rgb(bic);
    else
        bic = imresize(lr, scale, 'bicubic');
        our = uint8(bic);
    end
    clear bic;
    grd = shave(grd, [scale, scale]);
    our = shave(our, [scale, scale]);
%%Save results
    filename = im_dir(img).name;
    filename = filename(1:end-4);
    imwrite(uint8(our), ['result/', filename, '_lcsc76_x', num2str(scale), '.bmp']);
%%Evaluation
    X = shave(uint8(X), [scale, scale]);
    im_h = shave(uint8(im_h), [scale, scale]);
    pp_psnr = compute_rmse(X, im_h);
    scores(img, 1) = pp_psnr;
    scores(img, 2) = ssim_index(X, im_h);
end
save result/scores scores;
mean(scores)


