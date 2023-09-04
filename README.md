# Building and evaluating encoding models of EEG visual responses using DNNs

Here we provide the code to reproduce the results of our data resource paper:</br>
"[A large and rich EEG dataset for modeling human visual object recognition][paper_link]".</br>
Alessandro T. Gifford, Kshitij Dwivedi, Gemma Roig, Radoslaw M. Cichy

If you experience problems with the code, please create a pull request or report the bug directly to Ale via email (alessandro.gifford@gmail.com).

Please visit the [dataset page][dataset_page] for the data, paper, dataset tutorial and more.

[Here][videos] you will find some useful videos on our EEG dataset.



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


## Data availability
The raw and preprocessed EEG dataset, the training and test images and the DNN feature maps are available on [OSF][osf]. The ILSVRC-2012 validation and test images can be found on [ImageNet][imagenet]. To run the code, the data must be downloaded and placed into the following directories:

* **Raw EEG data:** `../project_directory/eeg_dataset/raw_data/`.
* **Preprocessed EEG data:** `../project_directory/eeg_dataset/preprocessed_data/`.
* **Training and test images; ILSVRC-2012 validation and test images:** `../project_directory/image_set/`.
* **DNN feature maps:** `../project_directory/dnn_feature_maps/pca_feature_maps`.



## Code description
* **01_eeg_preprocessing:** preprocess the raw EEG data.
* **02_dnn_feature_maps_extraction:** extract the feature maps of all images using four DNN architectures (AlexNet, ResNet-50, CORnet-S, MoCo), and downsample them using principal component analysis (PCA).
* **03_synthesizing_eeg_data:** synthesize the EEG responses to images through linearizing and end-to-end encoding models.
* **04_synthetic_data_analyses:** perform the correlation, pairwise decoding and zero-shot identification analyses on the synthetic EEG data.
* **05_plotting:** plot the analyses results.



## Interactive dataset tutorial
[Here][colab] you will find a Colab interactive tutorial on how to load and visualize the preprocessed EEG data and the corresponding stimuli images.

[colab]: https://colab.research.google.com/drive/1i1IKeP4cK3ViscP4b4kNOVo4kRoL8tf6?usp=sharing



## Cite
If you use any of our data or code, partly or as it is, please cite our paper:

Gifford AT, Dwivedi K, Roig G, Cichy RM. 2022. A large and rich EEG dataset for modeling human visual object recognition. _NeuroImage_, 264:119754. DOI: [https://doi.org/10.1016/j.neuroimage.2022.119754][paper_link]


[dataset_page]: https://www.alegifford.com/publications/eeg_dataset/
[videos]: https://www.youtube.com/playlist?list=PLAkLSNuCebPPv_S3gTjYIFvQ82hyezIld
[paper_link]: https://doi.org/10.1016/j.neuroimage.2022.119754
[conda]: https://www.anaconda.com/
[env_file]: https://github.com/gifale95/eeg_encoding_model/blob/main/environment.yml
[osf]: https://osf.io/3jk45/
[imagenet]: https://www.image-net.org/download.php
