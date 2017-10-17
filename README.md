# Crowd Counting

This repository is an implementation of crowd counting described in the paper "Image Crowd Counting Using Convolutional Neural Network and Markov Random Field". The fully connected regress network is implemented by Keras (Tensorflow backend). Others are implemented by Matlab.

### Citation
If you find this code useful in your research, please cite:

	@article{han2017image,
	  title={Image Crowd Counting Using Convolutional Neural Network and Markov Random Field},
	  author={Han, Kang and Wan, Wanggen and Yao, Haiyan and Hou, Li},
	  journal={arXiv preprint arXiv:1706.03686},
	  year={2017}
	}
	
### Evalute

You can direct evalute the model's performance by running EvaluteUCF.m or EvaluateSHT.m using predicted patches' count. This process will apply Markov Random Field and get the global count.

0. Compiling the MRF code by running testMRF.m in MRF folder.
0. Running EvaluteUCF.m or EvaluateSHT.m.

### Training a new model
If you want to train a new regress model, follow these steps:

0. Installing MatConvNet and then runing ExtractFeatures.m to extract features. This step is not necessary if you use the extracted features in the data folder.
0. Installing Keras and runing regress_UCF.py or regress_SHT.py to train a new regress network. Also, you can run patch_predict_SHT.py or patch_predict_UCF.py to predict the patches' count using trained regress network.
0. Running EvaluteUCF.m or EvaluateSHT.m to evalute the model's performance.

### Results

1. UCF

	<table>
	    	<tr>
			<td>MAE</td>
			<td>MSE</td>
	    	</tr>
		<tr>
			<td>254.1</td>
			<td>352.5</td>
	    	</tr>
	</table>

1. Shanghaitech

	<table>
	    	<tr>
			<td colspan="2">Part_A</td> 
			<td colspan="2">Part_B</td> 
	   	</tr>
	    	<tr>
			<td>MAE</td>
			<td>MSE</td>
			<td>MAE</td>
			<td>MSE</td>
	    	</tr>
		<tr>
			<td>79.1</td>
			<td>130.1</td>
			<td>17.8</td>
			<td>26.0</td>
	    	</tr>
	</table>

