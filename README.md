<h2>TensorFlow-FlexUNet-Tiled-Image-Segmentation-Earth-Surface-Water (2025/11/30)</h2>

Toshiyuki Arai<br>
Software Laboratory antillia.com<br>
<br>
This is the first experiment of Image Segmentation for <b>Earth-Surface-Water</b> (Singleclass) based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>)
, and a 512x512  pixels PNG 
<a href="https://drive.google.com/file/d/1slsvlBkGOb2XspAI664OOklIcaBoul52/view?usp=sharing">
<b>Tiled-Earth-Surface-Water-ImageMask-Dataset.zip</b></a>
which was derived by us from <br><br>
<a href="https://zenodo.org/records/5205674">
<b>Earth Surface Water Dataset</b>
</a> on zenodo web site.
<br><br>
<b>Divide-and-Conquer Strategy</b><br>
Since the images and masks of the Earth-Surface-Water are large (1.5K to 2.5K pixels),
we adopted the following <b>Divide-and-Conquer Strategy</b> for building our segmentation model.
<br>
<br>
<b>1. Tiled Image Mask Dataset</b><br>
We generated a PNG image and mask datasets of 512x512 pixels tiledly-split dataset from
the  Earth-Surface-Water by our offline augmentation tool
<a href="./generator/TiledImageMaskDatasetGenerator.py">
TiledImageMaskDatasetGenerator.</a><br>
<br>
<b>2. Train Model by Tiled ImageMask Dataset</b><br>
We trained and validated the TensorFlowFlexUNet model for Earth-Surface-Water by using the 
Tiled-Earth-Surface-Water dataset.
<br><br>
<b>3. Tiled Image Segmentation</b><br>
We applied our Tiled-Image Segmentation inference method to predict mask regions for the mini_test images 
with the original resolution.
<br><br>
<hr>
<b>Actual Image Segmentation for the original Earth-Surface-Water Images of 1.5K to 2.5K pixels</b><br>
As shown below, the tiled_inferred masks predicted by our segmentation model trained on the 
Tiled dataset appear similar to the ground truth masks, but they lack precision in certain areas.<br>
However, in the second case, the groud truth (mask) seems to be slightly inappropriate.<br> 
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: tiled_inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test/images/101000.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test/masks/101000.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test_output_tiled/101000.png" width="320" height="auto"></td>
</tr>
</tr>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test/images/105000.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test/masks/105000.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test_output_tiled/105000.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test/images/111000.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test/masks/111000.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test_output_tiled/111000.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
1 Dataset Citation
</h3>
The dataset used here was derived from <b>dset-s2</b> 
<a href="https://zenodo.org/records/5205674">
<b>Earth Surface Water Dataset</b>
</a> on zenodo web site.
<br><br>
<b>Author</b><br>
Xin Luo 
<br><br>
<b>Description</b><br>
We provided a new dataset for deep learning of surface water features on Sentinel-2 satellite images. <br>
This dataset contains 95 scenes that are globally distributed.<br>

The related source code can be found at: <a href="https://github.com/xinluo2018/WatNet">https://github.com/xinluo2018/WatNet</a>. <br>
<br>
And the full paper can be found at: <a href="https://doi.org/10.1016/j.jag.2021.102472">https://doi.org/10.1016/j.jag.2021.102472</a>.<br>
<br>
<b>License</b><br>
<a href="https://creativecommons.org/licenses/by/4.0/legalcode">
Creative Commons Attribution 4.0 International
</a>
<br>
<br>
<h3>
2 Tiled Earth-Surface-Water ImageMask Dataset
</h3>
<h4>2.1 Download Tiled Earth-Surface-Water</h4>
 If you would like to train this Earth-Surface-Water Segmentation model by yourself,
 please download <a href="https://drive.google.com/file/d/1slsvlBkGOb2XspAI664OOklIcaBoul52/view?usp=sharing">
 <b>Tiled-Earth-Surface-Water-ImageMask-Dataset.zip</b></a>
on the google drive, expand the downloaded, and put it under dataset folder to be:
<pre>
./dataset
└─Earth-Surface-Water
    ├─test
    │  ├─images
    │  └─masks
    ├─train
    │  ├─images
    │  └─masks
    └─valid
        ├─images
        └─masks
</pre>
<b>Tilled-Earth-Surface-Water Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Earth-Surface-Water/Earth-Surface-Water_Statistics.png" width="512" height="auto"><br>
<br>

As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br><br> 

<h4>2.2 Tiled Earth-Surface-Water Derivation</h4>
The folder structure of the original <b>dset-s2</b> is the following.
<pre>
./dset-s2
├─tra_scene
├─tra_truth
├─val_scene
└─val_truth
</pre>
Firstly, we compiled all the TIF files from folders <b>tra_scene</b> and <b>val_scene</b> and saved them as PNG files in a single folder
<b>./Earth-Surface-Water/masks</b>, 
and all TIF ground truth files in from <b>tra_truth</b> and <b>val_truth</b> and saved them to a folder <b>./Earth-Surface-Water/masks</b>. 
<br>
<pre>
./Earth-Surface-Water
├─images
└─masks
</pre>

<img src="././projects/TensorFlowFlexUNet/Earth-Surface-Water/asset/Earth_Suface_Water_Images_64.png" width="1024" height="auto"><br>
<img src="././projects/TensorFlowFlexUNet/Earth-Surface-Water/asset/Earth_Suface_Water_Masks_64.png"  width="1024" height="auto"><br>
<br>
Secondly, we generated our augmented Tiled Earth-Surface-Water from <b>Earth-Surface-Water</b> dataset by using an offline
augmentation tool <a href="./generator/TiledImageMaskDatasetGenerator.py">TiledImageMaskDatasetGenerator.py</a>, because
the <b>images</b> and <b>masks</b> under <b>Earth-Surface-Walter</b> folder contains only 64 files respectively. <br><br>

<h4>2.3 Tiled Earth-Surface-Water Samples</h4>

<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained Earth-Surface-Water TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Earth-Surface-Water/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Earth-Surface-Water and, and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16</b> and small <b>base_kernels = (5,5)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3

num_classes    = 2

base_filters   = 16
base_kernels   = (5,5)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)

</pre>

<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00008
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorFlowFlexUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
You may specify other loss and metrics function names.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for Earth-Surface-Water 1+1 classes.
<pre>
[mask]
mask_datatype    = "categorized"
mask_file_format = ".png"
;                      Water: blue
rgb_map = {(0,0,0):0, (0,0,255):1,}
</pre>

<b>Tiled inference parameters</b><br>
<pre>
[tiledinfer] 
overlapping = 128
images_dir    = "./mini_test/images/"
output_dir    = "./mini_test_output_tiled/"
</pre>

<b>Epoch change tiled inference callback</b><br>
Enabled <a href="./src/EpochChangeTiledInferencer.py">epoch_change_tiled_infer callback (EpochChangeTiledInferencer.py)</a></b>.<br>
<pre>
[train]
poch_change_infer     = False
epoch_change_infer_dir =  "./epoch_change_infer"
epoch_change_tiled_infer     = True
epoch_change_tiled_infer_dir =  "./epoch_change_tiled_infer"
</pre>

By using this callback, on every epoch_change, the tiled inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 38,39,40)</b><br>
<img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 77,78,79)</b><br>
<img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>


In this experiment, the training process was stopped at epoch 79 by EarlyStoppingCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/asset/train_console_output_at_epoch79.png" width="880" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Earth-Surface-Water/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Earth-Surface-Water/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Earth-Surface-Water</b> folder,<br>
and run the following bat file to evaluate TensorFlowUNet model for Earth-Surface-Water.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/asset/evaluate_console_output_at_epoch79.png" width="880" height="auto">
<br><br>Image-Segmentation-Aerial-Imagery

<a href="./projects/TensorFlowFlexUNet/Earth-Surface-Water/evaluation.csv">evaluation.csv</a><br>

The loss (categorical_crossentropy) to this Earth-Surface-Water/test was not low, and dice_coef_multiclass not high as shown below.
<br>
<pre>
categorical_crossentropy,0.0784
dice_coef_multiclass,0.9638
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Earth-Surface-Water</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for Earth-Surface-Water.<br>
<pre>
./4.tiled_infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetTiledInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Tiled_inferred test masks</b><br>
 
<img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/asset/mini_test_output_tiled.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for the original Earth-Surface-Water Images of 1.5K to 2.5K pixels</b><br>
As shown below, the tiled_inferred masks predicted by our segmentation model trained on the 
Tiled dataset appear similar to the ground truth masks, but they lack precision in certain areas.<br>
<table>
<tr>

<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Tiled-inferred-mask</th>
</tr>

<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test/images/102000.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test/masks/102000.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test_output_tiled/102000.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test/images/105000.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test/masks/105000.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test_output_tiled/105000.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test/images/109000.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test/masks/109000.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test_output_tiled/109000.png" width="320" height="auto"></td>
</tr>


<tr>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test/images/112000.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test/masks/112000.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test_output_tiled/112000.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test/images/118000.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test/masks/118000.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test_output_tiled/118000.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test/images/124000.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test/masks/124000.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Earth-Surface-Water/mini_test_output_tiled/124000.png" width="320" height="auto"></td>
</tr>


</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. Satellite Imagery Aerial-Imagery Segmentation</b><br>
Nithish<br>
<a href="https://medium.com/@nithishmailme/satellite-imagery-road-segmentation-ad2964dc3812">
https://medium.com/@nithishmailme/satellite-imagery-road-segmentation-ad2964dc3812
</a>
<br>
<br>
<b>2. Deep Learning-based Aerial-Imagery Segmentation Using Aerial Images: A Comparative Study</b><br>
Kamal KC, Alaka Acharya, Kushal Devkota, Kalyan Singh Karki, and Surendra Shrestha<br>
<a href="https://www.researchgate.net/publication/382973365_Deep_Learning-based_Aerial-Imagery_Segmentation_Using_Aerial_Images_A_Comparative_Study">
https://www.researchgate.net/publication/382973365_Deep_Learning-based_Aerial-Imagery_Segmentation_Using_Aerial_Images_A_Comparative_Study</a>
<br>
<br>
<b>3. A Comparative Study of Deep Learning Methods for Automated Aerial-Imagery Network<br>
Extraction from High-Spatial-ResolutionRemotely Sensed Imagery</b><br>
Haochen Zhou, Hongjie He, Linlin Xu, Lingfei Ma, Dedong Zhang, Nan Chen, Michael A. Chapman, and Jonathan Li<br>
<a href="https://uwaterloo.ca/geospatial-intelligence/sites/default/files/uploads/documents/march2025_zhou_10.14358_pers_24-00100r2.pdf">
https://uwaterloo.ca/geospatial-intelligence/sites/default/files/uploads/documents/march2025_zhou_10.14358_pers_24-00100r2.pdf
</a>
<br>
<br>
<b>4. An applicable and automatic method for earth surface water mapping based on multispectral images</b><br>
Xin Luo, Xiaohua Tong, Zhongwen Hu <br>
<a href="https://www.sciencedirect.com/science/article/pii/S0303243421001793?via%3Dihub">
https://www.sciencedirect.com/science/article/pii/S0303243421001793?via%3Dihub
</a>
<br>
<br>
<b>5. WatNet</b><br>
Xin Luo<br>
<a href="https://github.com/xinluo2018/WatNet">
https://github.com/xinluo2018/WatNet
</a>
<br>
<br>
<b>6. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
<b>7. TensorFlow-FlexUNet-Tiled-Image-Segmentation-Concrete-Crack</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Concrete-Crack">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Concrete-Crack
</a>

