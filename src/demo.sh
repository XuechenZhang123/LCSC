#Train BELCSCNet
#python main.py --scale 2  --save  BELCSC_X2_B9U6 --model BELCSCNET --epochs 650 --batch_size 32 --loss '1*L1' --channels 64 --rate_list 0.75 0.71875 0.6875 0.65625 0.625 0.59375 0.5625 0.53125 0.5 --len_list 6 6 6 6 6 6 6 6 6 --multi_out False

#Train E-LCSCNet
#python main.py --model FLDLCSC_X2_B9U16 --scale 2 --epochs 800 --batch_size 32  --loss '1*L1+0.005*L1+0.005*L1+0.005*L1+0.005*L1+0.005*L1+0.005*L1+0.005*L1+0.005*L1+0.005*L1' --channels 128 --rate_list 0.75 0.71875 0.6875 0.65625 0.625 0.59375 0.5625 0.53125 0.5 --len_list 16 16 16 16 16 16 16 16 16 --multi_out True

#Test BELCSCNet
#python main.py --data_test Set5 --scale 2 --pre_train ../experiment/BELCSC_X2_B9U6/model/model_best.pt --model BELCSCNET --channels 64 --rate_list 0.75 0.71875 0.6875 0.65625 0.625 0.59375 0.5625 0.53125 0.5 --len_list 6 6 6 6 6 6 6 6 6 --multi_out False --test_only


#Test E-LCSCNet
#python main.py --data_test Set5 --scale 2 --pre_train ../experiment/FLDLCSC_X2_B9U16/model/model_best.pt --model FLDLCSC --channels 128 --rate_list 0.75 0.71875 0.6875 0.65625 0.625 0.59375 0.5625 0.53125 0.5 --len_list 16 16 16 16 16 16 16 16 16 --multi_out True --test_only





