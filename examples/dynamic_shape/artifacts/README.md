# Trained digits CNN

`digits_cnn.onnx` is a 6,218-parameter CNN trained by this project with
`../export_digits.py` (seed 2026, 80 epochs). It declares dynamic batch, height
and width. No external pretrained checkpoint is used.

The dataset is the real optical-recognition handwritten digits dataset bundled
with [scikit-learn load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).
It contains 1,797 images at 8×8 resolution. The stratified split used 1,437
training images and 360 held-out images. Test accuracy at 8×8 was 96.11%.

`digits_samples.npz` contains eight held-out images normalized to [0,1], their
labels and PyTorch reference logits. They are used for cross-framework checks;
the dynamic-resolution tests resize these images by nearest-neighbor sampling.
Accuracy at other resolutions is not claimed.

`training.json` records package versions, split, seed, metric and SHA256. These
small artifacts allow CPU/CUDA/ORT regression tests without installing a
training framework or downloading data.
