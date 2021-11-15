# EEG encoding using Deep Neural Networks
Here we provide the code to reproduce the results of the paper:</br>
["Paper title"][paper_link].</br>
Alessandro T. Gifford, Kshitij Dwivedi, Gemma Roig, Radoslaw M. Cichy
</br>


## Environment setup
To run the code first install [Anaconda][conda], then create and activate a dedicated Conda environment by typing the following into your terminal:
```shell
curl -O https://raw.githubusercontent.com/gifale95/eeg_encoding_model/main/environment.yml
conda env create -f environment.yml
conda activate eeg_encoding
```
Alternatively, after installing Anaconda you can download the [environment.yml][env_file] file, open the terminal in the download directory and type:
```shell
conda env create -f environment.yml
conda activate eeg_encoding
```
</br>

## Data availability
The raw and preprocessed EEG dataset, the training and test images and the DNN feature maps are available on [OSF][osf]. The ILSVRC-2012 validation and test images can be found on [ImageNet][imagenet]. Tu run the code, the data must be downloaded and extracted into the following directories:

* **Raw EEG data:** `~/project_dir/eeg_dataset/raw_data/`.
* **Preprocessed EEG data:** `~/project_dir/eeg_dataset/preprocessed_data/`.
* **Training and test images; ILSVRC-2012 validation and test images:** `~/project_dir/image_set/`.
* **DNN feature maps:** `~/project_dir/dnn_feature_maps/pca_feature_maps`.
</br>


## Code description
* **01_eeg_preprocessing:** preprocessing of the raw EEG data.
* **02_dnn_feature_maps_extraction:** extracting the feature maps of all images using four DNN architectures (AlexNet, ResNet-50, CORnet-S, MoCo), and downsampling them using principal component analysis (PCA).
* **03_synthetic_data_prediction:** training of a linear regression to predict the synthetic EEG responses to images using the DNN feature maps of those same images as predictors. !!!!!!!!!!!!!!!!!!!! Edit
* **04_synthetic_data_analyses:** performing several analyses on the synthetic EEG data. !!!!!!!!!!!!!!!!!!!! Edit
* **05_plotting:** plotting the analyses results. !!!!!!!!!!!!!!!!!!!! Edit
</br>


## Cite
If you use our code, partly or as it is, please cite the paper:

```
Paper citation !!!!!!!!!!!!!!!!!!!! Edit
```

[paper_link]: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
[conda]: https://www.anaconda.com/
[env_file]: https://github.com/gifale95/eeg_encoding_model/blob/main/environment.yml
[osf]: https://osf.io/3jk45/s
[imagenet]: https://www.image-net.org/download.php