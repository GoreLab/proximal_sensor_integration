# proximal_sensor_integration

The code presented here will process, curate, and quality check the aerial multispectral images (MSIs) and ground-based LiDAR scans used for all model implementations. These data include over 50,000 plot-level MSIs and over 30,000 plot-level LiDAR scans collected for maize hybrids over 5 growing seasons from 2020 through 2024 at Musgrave Research Farm in Aurora, NY.

We also provide the code needed to train the autoencoder model and extract latent phenotypes ("autoencoder/"), available here for application to the published MSI and LiDAR datasets, as well as new datasets. The framework for model training and execution was implemented with Pytorch-VAE, available in GitHub at https://github.com/AntixK/PyTorch-VAE (Subramanian, 2020). The License for this repository is available in its source form, and all utilized scripts have been modified from the source form of the work. We further provide custom model architecture and dataset classes, as well as custom scripts for extracting the latent space. The autoencoder accepts .png and .tif files, though the dataset.py file may be modified for other image formats, and requires a config file following the below template:

## Autoencoder training config:

```yaml
model_params:
  name: 'ResnetAEVarLS'
  in_channels: <number of input image channels>
  model_name: "resnet18"
  num_latent_dims: <size of the latent space>


data_params:
  data_path: "<path to input images>"
  annotations_file: "<path to annotations file>"
  data_type: "<data type: msi or lidar>"
  train_batch_size: 8
  val_batch_size: 8
  patch_size: 64
  num_workers: 16

exp_params:
  LR: 0.00001
  weight_decay: 0.0
  scheduler_gamma: 0.95
  kld_weight: 0.0
  manual_seed: 1265

trainer_params:
  gpus: [0]
  max_epochs: 150

logging_params:
  save_dir: "logs/"
  name: "ResnetAEVarLS"
```

Extracting the latent space from a trained autoencoder model also requires a config file following the below template:

## Extract latent space config:

```yaml
AE_model_params:
  config_path: '<path to config used to train autoencoder model'
  checkpoint_path: '<path to checkpoint for loading weights of trained autoencoder model>'

AE_data_params:
  labels_path: '<path to annotations file>'
  file_name_column: '<name of column in annotations file where image file names are stored>'
  data_path: '<path to input images>'
  data_type: '<data type: msi or lidar>'
  in_channels: <number of input image channels>
  in_height: <height of input images in pixels>
  in_width: <width of input images in pixels>
  out_channels: <size of the latent space>
  out_height: <height of latent space>
  out_width: <width of latent space>
```

Finally, we also provide the code for predicting crop traits from the latent spaces extracted from each proximal sensing data stream, as well as their integration, and figure generation. The full workflow, ordering of each script, and their corresponding inputs and outputs are presented below:

## Workflow:

1.	lidar_preprocess_post2020.py/lidar_preprocess_2020.py
   * Input:
     i.	Lidar point clouds (h5 or las);
     ii.	EarthSense plot splits for each lidar collection (2021-2024) (csv);
     iii.	Dataframe of 2020 plot splits unnested (2020 only) (csv)
   * Output:
     i.	Voxelized & denoised lidar density maps flipped and not flipped (png);
     ii.	Filenames of poor quality point clouds (txt);
     iii.	Plot splits with plant height information added (2020 only) (csv)
2.	move_files_by_ID.sh used to copy lidar density maps to new folder for all of NYH2 and NYH3 2020-24 and remove border ranges
3.	lidar_pad.py
   * Input:
     i.	Output 1bi (with border ranges removed);
     ii.	Output 1bii
   * Output:
     i.	Flipped lidar density maps padded to same size (png), with all poor quality files removed
4.	tif_preprocess.py
   * Input:
     i.	Plot-level MSIs (tif)
   * Output:
     i.	Plot-level MSIs with values clipped and NAs set to average of surrounding pixels (tif);
     ii.	Filenames of tifs with NAs filled & number of iterations
5.	get_VIs.py
   * Input:
     i.	Plot-level MSIs (tif)
   * Output:
     i.	Dataframe of filenames and VIs, thresholded as specified (csv)
6.	get_pixel_stats.py
   * Input:
     i.	Plot-level MSIs (tif)
   * Output:
     i.	Dataframe of filenames and pixel stats, including clipped values, NAs, etc. (csv)
7.	quality_checks.Rmd
   * Input:
     i.	Dataframe of all plot splits for NYH2 and NYH3 2020-24, removing border ranges (csv);
     ii.	Output 1bii;
     iii.	Output 1biii;
     iv.	Output 5bi;
     v.	Output 6bi;
     vi.	G2F labels
   * Output:
     i.	Scatterplots of trends in ES plant height and NDVI by year;
     ii.	Scatterplots of trends in heritabilities of ES plant height & NDVI by year;
     iii.	Bar plot of correlation between ES plant height and manually measured plant height;
     iv.	Histograms of ratios of clipped and NA values, bar plots of summary statistics of clipped and NA values
8.	get_GDD.Rmd
   * Input:
     i.	Dataframes of daily max and min temps downloaded from NASA Power (csv)
   * Output:
     i.	Dataframe of cumulative GDD (csv)
9.	get_lidar_md.Rmd/get_msi_md.Rmd
   * Input:
     i.	Dataframes of filenames and filename information (csv);
     ii.	G2F labels
   * Output:
     i.	Dataframe of filenames, plot split information, and plot meta data (csv)/dataframe of filenames and plot meta data (csv)
10.	get_lidar_labels.Rmd/get_msi_labels.Rmd
   * Input:
     i.	Output 8bi;
     ii.	Output 9bi;
     iii.	G2F labels
   * Output:
     i.	Dataframes of filenames, plot meta data, phenotypes, DAP, and GDD (csv)
11.	train_test_split_CVBP.Rmd/train_test_split_CV1.Rmd/train_test_split_LO1Y.Rmd/train_test_split_CV00.Rmd/train_test_split_Test.Rmd
   * Input:
     i.	Outputs 10bi;
     ii.	G2F labels
   * Output:
     i.	Dataframes of CVBP (csv);
     ii.	Dataframes of CV1 (csv);
     iii.	Dataframes of CV00 (csv);
     iv.	Dataframes of LO1Y (csv);
     v.	Dataframes of Test (csv)
12.	run.py
   * Input:
     i.	config file (lidar_resnet_vae_all_varLS.yaml/msi_resnet_vae_all_varLS.yaml);
     ii.	dataset.py;
     iii.	resnet50_ae_varLS.py;
     iv.	experiment.py;
     v.	Outputs 11b;
     vi.	Output 3bi/Output 4bi
   * Output:
     i.	Trained autoencoder model;
     ii.	Sample original and reconstructed images (png)
13.	extract_z.py
   * Input:
     i.	config file (extract_z_lidar.yaml/extract_z_msi.yaml);
     ii.	dataset.py;
     iii.	resnet50_ae_varLS.py;
     iv.	experiment.py;
     v.	Output 3bi/Output 4bi;
     vi.	Output 11b;
     vii.	Input 12ai;
     viii.	Output 12bi
   * Output:
     i.	Dataframe of latent space extracted from trained model in order of annotations file (csv)
14.	get_lsp_dfs.Rmd
   * Input:
     i.	Output 11b;
     ii.	Output 13bi;
     iii.	Output 5bi
   * Output:
     i.	Dataframes of labels and latent spaces (csv);
     ii.	Dataframes of labels and latent spaces for equivalent GDD time points across all data (csv)
15.	cullis_heritability.Rmd
   * Input:
     i.	Output 14bii;
     ii.	G2F labels
   * Output:
     i.	Dataframes of Cullis heritabilities for manually measured phenotypes and latent phenotypes (csv)
16.	lsp_predictions.Rmd
   * Input:
     i.	Output 14bii;
     ii.	Output 5bi;
     iii.	G2F labels
   * Output:
     i.	Dataframes of predicted accuracies from trained prediction model(s) (csv)
17.	figures.Rmd
   * Input:
     i.	Output 8bi;
     ii.	Output 10bi;
     iii.	Output 15bi;
     iv.	Output 16bi
   * Output:
     i.	Data visualizations


