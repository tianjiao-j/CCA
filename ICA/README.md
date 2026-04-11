# ICA Unmixing Matrix

Steps to compute the ICA unmixing matrix:

1. **Download** the ICA training samples and pre-extracted features from [Google Drive](https://drive.google.com/drive/folders/1lHpYCVutV_nNVV1LhbdfGO5fDChHsh86?usp=sharing) and place them in the `ICA/` directory.

2. **Extract CLIP image features** from the images listed in `ICA/ica_samples.json`, using the same method as `utils.build_cache_model`, or skip this step and use the pre-extracted features at `ICA/img_features.pt`.

3. **Run the ICA script** from the project root:

   ```bash
   python ica.py
   ```

   This fits a `FastICA` model with 1024 components on the extracted features and saves the unmixing matrix to `mat_unmix_1024d.pt`.
