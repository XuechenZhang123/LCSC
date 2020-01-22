function im_hr = LCSC(im_lr, lcsc_backbone, lcsc_attention, scale)

bb_weight = lcsc_backbone.weight;
bb_bias  = lcsc_backbone.bias;

att_weight = lcsc_attention.weight;
att_bias = lcsc_attention.bias;

len_list = [8, 8, 8, 8, 8, 8, 8, 8, 8];
block_num = size(len_list);
block_num = block_num(2);

inter_feature = {};
sub_hr = {};
cnv_indx = 1;
input_lr = single(im_lr);
convfea = vl_nnconv(input_lr, bb_weight{cnv_indx}, bb_bias{cnv_indx}, 'Pad', 1);
cnv_indx = cnv_indx + 1;

for i = 1 : block_num
    for j = 1 : len_list(i)
        linear_convfea = vl_nnconv(convfea, bb_weight{cnv_indx}, bb_bias{cnv_indx}, 'Pad', 0);
        cnv_indx = cnv_indx + 1;
        nonlinear_convfea = vl_nnrelu(convfea);
        nonlinear_convfea = vl_nnconv(nonlinear_convfea, bb_weight{cnv_indx}, bb_bias{cnv_indx}, 'Pad', 1);
        cnv_indx = cnv_indx + 1;
        convfea = vl_nnconcat({linear_convfea, nonlinear_convfea}, 3);
    end
    inter_feature{i} = convfea;
end
%fprintf('[%d]', cnv_indx);
for i = 1 : block_num
    sub_output = upscaling2D(inter_feature{i}, scale);
    for j = 0 : 1
        sub_output = vl_nnconv(sub_output, bb_weight{cnv_indx+j}, bb_bias{cnv_indx+j}, 'Pad', 1);
        sub_output = vl_nnrelu(sub_output);
    end
    %fprintf('[%d]', cnv_indx);
    sub_output = vl_nnconv(sub_output, bb_weight{cnv_indx+2}, bb_bias{cnv_indx+2}, 'Pad', 1);
    sub_hr{i} = sub_output;
end

output = sub_hr{1};
for i = 1 : (block_num-1)
    concat_output = vl_nnconcat({output, sub_hr{i+1}}, 3);
    merge_weight = vl_nnconv(concat_output, att_weight{i}, att_bias{i}, 'Pad', 0);
    merge_weight = vl_nnsigmoid(merge_weight);
    output = (output - sub_hr{i+1}) .* merge_weight + sub_hr{i+1};
end

im_hr = output;
end