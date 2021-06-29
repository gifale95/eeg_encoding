# EEG encoding using Deep Neural Networks
Here we provide the code to reproduce the results of the paper:</br>
["Paper title"][paper_link].</br>
Alessandro T. Gifford, Kshitij Dwivedi, Radoslaw M. Cichy
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
The raw and preprocessed EEG dataset, along with the used image set, are available on [OSF][osf]. Tu run the code, the data must be downloaded and extracted into the following directories:

* **Raw EEG data:** `~/project_dir/eeg_dataset/raw_data/`.
* **Preprocessed EEG data:** `~/project_dir/eeg_dataset/preprocessed_data/`.
* **Stimuli images:** `~/project_dir/stimuli_images/`.
</br>


## Code description
* **01_eeg_preprocessing:** preprocessing of the raw EEG data.
* **02_dnn_feature_maps_extraction:** extraction of the image-related feature maps from four DNN architectures (AlexNet, ResNet-50, CORnet-S, MoCo).
* **03_encoding_model:** training of a linear regression to predict the EEG responses to images using the DNN feature maps of those same images as predictors.
* **04_predicted_data_analyses:** assessing the quality of the predicted EEG data using correlation and decoding analyses.
* **05_stats:** assessing the statistical significance of the analyses results.
* **06_plotting:** plotting the analyses results.
</br>


## Acknowledgements
The code to extract the feature maps of the different DNN architectures is borrowed from [where from?][fmaps_code].
</br>


## Cite
If you use our code, partly or as it is, please cite the paper:

```
Paper citation
```

[paper_link]: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
[conda]: https://www.anaconda.com/
[env_file]: https://github.com/gifale95/eeg_encoding_model/blob/main/environment.yml
[osf]: https://osf.io/3jk45/
[fmaps_code]: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!