# Movie Revenue Prediction

This repository predicts movie box office revenue from poster images using a ResNet50 backbone with a custom regression head. The project focuses on whether visual features alone can recover a useful revenue signal and documents both the achieved performance and the limits of that approach.

## Repository layout

- `train.py`: model definition and training loop for the poster-only revenue regressor
- `clean_data.py`: dataset filtering and deduplication pipeline for the poster dataset
- `full_predict.py`: sample evaluation script for reviewing predictions on held-out movies
- `large_10M_torch2/`: saved filtered dataset metadata
- `model+results/`: final metrics, model metadata, and logs
- `sample_posters/`: poster examples and companion metadata for qualitative inspection
- `archived_experiments/`: older exploratory scripts and architecture tests kept for reference
- `Final_Draft.pdf`: project write-up and analysis

## Setup

Create an environment and install the libraries used by the training scripts:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision datasets pillow numpy
```

Prepare the filtered dataset:

```bash
python clean_data.py
```

Train the poster-only model:

```bash
python train.py
```

Review a held-out sample with the saved model:

```bash
python full_predict.py
```

## Methodology

The model uses transfer learning on top of an ImageNet-pretrained ResNet50. Poster images are normalized with ImageNet statistics, revenue targets are log-transformed, and a multi-layer regression head maps the visual embedding to a scalar revenue prediction. Training uses AdamW with early stopping on validation loss, and evaluation reports both regression quality and a simpler above-versus-below-median classification view.

## Results

The best saved run reached a validation loss of `0.2274` and test loss of `0.2399`. On the held-out set, predictions landed within 25% of realized revenue for `17.8%` of films and within 50% for `35.6%`. The binary above/below-median revenue framing reached roughly `57.8%` accuracy, which suggests the model extracts some market signal from poster imagery but remains far from production-grade forecasting.

## Known limits

- Poster imagery alone is an incomplete input for box office forecasting
- The model struggles with extreme outliers and franchise-scale releases
- The repository assumes access to local dataset directories rather than a fully packaged data download flow
- Archived experiments are retained for transparency and may not match the final model path exactly

**Note**: Movie information in the additional [sample_posters](https://github.com/Aroesler1/movie-revenue-prediction/tree/main/sample_posters) dataset was taken from [IMDB](https://www.imdb.com/)
