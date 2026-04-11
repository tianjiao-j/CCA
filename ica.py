import torch
import clip
from sklearn.decomposition import FastICA
import random

if __name__ == "__main__":
    random.seed(1)
    torch.manual_seed(1)
    clip_model, preprocess = clip.load("RN50")
    clip_model.eval()

    n_components = 1024
    mat_unmix_path = f"mat_unmix_{n_components}d.pt"

    img_features = torch.load("ICA/img_features.pt")

    transformer = FastICA(
        n_components, random_state=1, whiten="unit-variance", max_iter=10000
    )
    transformer.fit(img_features.T.cpu())
    mat_unmix = torch.from_numpy(transformer.components_).to(torch.float32)

    print("Saving unmixing matrix to {}".format(mat_unmix_path))
    torch.save(mat_unmix, mat_unmix_path)
