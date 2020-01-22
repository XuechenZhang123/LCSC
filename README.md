# LCSCNet-Linear-Compressing-Based-Skip-Connecting-Network-for-Image-Super-Resolution
Codes for "LCSCNet: Linear Compressing Based Skip-Connecting Network for Image Super-Resolution", accepted by IEEE Transactions on Image Processing in 2019. The pytorch version (0.4.0) within Ubuntu 16.04 is built on [EDSR](https://github.com/LimBee/NTIRE2017). 
## Abstract
In this paper, we develop a concise but efficient network architecture called linear compressing based skip-connecting network (LCSCNet) for image super-resolution. Compared with two representative network architectures with skip connections, ResNet and DenseNet, a linear compressing layer is designed in LCSCNet for skip connection, which connects former feature maps and distinguishes them from newly-explored feature maps. In this way, the proposed LCSCNet enjoys the merits of the distinguish feature treatment of DenseNet and the parameter-economic form of ResNet. Moreover, to better exploit hierarchical information from both low and high levels of various receptive fields in deep models, inspired by gate units in LSTM, we also propose an adaptive element-wise fusion strategy with multi-supervised training. Experimental results in comparison with state-of-the-art algorithms validate the effectiveness of LCSCNet.
## Train code
1. Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) to 'dataset' file
2. Specify in 'src/demo.sh'. 
       For example, the scripts for training X2 BLCSCNet in the paper is 
### 
     python main.py --scale 2  --save  BELCSC_X2_B9U6 --model BELCSCNET --epochs 650 --batch_size 32 --loss '1*L1' --channels 64 --rate_list 0.75 0.71875 0.6875 0.65625 0.625 0.59375 0.5625 0.53125 0.5 --len_list 6 6 6 6 6 6 6 6 6 --multi_out False
The scripts for training X2 E-LCSCNet in the paper is 
### 
     python main.py --scale 2 --save FLDLCSC_X2_B9U16 --model FLDLCSC --epochs 650 --batch_size 32  --loss '1*L1+0.005*L1+0.005*L1+0.005*L1+0.005*L1+0.005*L1+0.005*L1+0.005*L1+0.005*L1+0.005*L1' --channels 128 --rate_list 0.75 0.71875 0.6875 0.65625 0.625 0.59375 0.5625 0.53125 0.5 --len_list 16 16 16 16 16 16 16 16 16 --multi_out True 
For training X3/X4 model, loading X2 models as pre-train model can significantly improve the performance. Please specify '--pre_train' to the corresponding X2 model in these cases. 

## Test code 
1. Download widely-used test dataset for deep learning SISR: [Set5](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html), [Set14](https://sites.google.com/site/romanzeyde/research-interests), [B100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) and [Urban100](https://sites.google.com/site/jbhuang0604/publications/struct_sr), and allocate them to 'dataset/benchmark'. All the test dataset have the structure as follows: 

                                       |---- HR 
                       Test_Dataset -- |                   | -- X2   
                                       |---- LR_bicubic -- | -- X3
                                                           | -- X4
                                                           
2. You can download the [pretrain models](https://pan.baidu.com/s/1IW1dagUj9GZSYFpVsnAn9g), the password is vct7.                                                    
3. Specify in 'src/demo.sh'. 
       For example, the scripts for testing X4 BLCSCNet in the paper is 
###      
        python main.py --data_test Set5 --scale 4 --pre_train ../experiment/BELCSC_X4_B9U6/model/model_best.pt --model BELCSCNET --channels 64 --rate_list 0.75 0.71875 0.6875 0.65625 0.625 0.59375 0.5625 0.53125 0.5 --len_list 6 6 6 6 6 6 6 6 6 --multi_out False --test_only
The scripts for testing X4 E-LCSCNet in the paper is 
###
        python main.py --data_test Set5 --scale 4 --pre_train ../experiment/FLDLCSC_X4_B9U16/model/model_best.pt --model FLDLCSC --channels 128 --rate_list 0.75 0.71875 0.6875 0.65625 0.625 0.59375 0.5625 0.53125 0.5 --len_list 16 16 16 16 16 16 16 16 16 --multi_out True --test_only       
        
4. Test codes for LCSC_76_291
Here light models trained with small training dataset (LCSC_76_291) are also provided for testing (These models for provided for comparing with those also trained with 291 dataset). You need to download [MatConvNet](http://www.vlfeat.org/matconvnet/) and then it can be implemented quite easily on any PC with MATLAB. 
        
### Citation
If you find our work is helpful, please cite our paper and EDSR
###
       @article{yang2019lcscnet,
        title={LCSCNet: Linear Compressing-Based Skip-Connecting Network for Image Super-Resolution},
        author={Yang, Wenming and Zhang, Xuechen and Tian, Yapeng and Wang, Wei and Xue, Jing-Hao and Liao, Qingmin},
        journal={IEEE Transactions on Image Processing},
        volume={29},
        pages={1450--1464},
        year={2019},
        publisher={IEEE}
       }
       
       @InProceedings{Lim_2017_CVPR_Workshops,
        author = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
        title = {Enhanced Deep Residual Networks for Single Image Super-Resolution},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
        month = {July},
        year = {2017}
       }

### Acknowledgement
The pytorch version of our paper is built on [EDSR](https://github.com/LimBee/NTIRE2017), we thank the authors for sharing their codes!
