# LAVA-test-generation
Codebase of the BSc thesis by Shangsu Feng "XAI guided Test Generation from Latent Space"

## Environment
Needed environments are in the `requirements.yaml`.

## Introduction
- `measure_*.py` modules to measure the metrics of the algorithms. Mainly used in `compare_adversarial.py`.

- `train_vae.py` to train a VAE as a benchmark for latent space analysis.
- `train_cae.py` to train a CAE as a benchmark for latent space analysis.
- `train_cvae.py` to train a CVAE as a benchmark for latent space analysis.
- `view_latent_space.py` check the latent space of Encoders.

- `xai.py` build a framework to analysis the latent space.
- `xai_condition.py` build a framework to analysis the latent space targeting a specific label.
- `adversarial.py` generate confusing images to attack cnn. (compare it with original image)

- `compare_adversarial.py` build a framework to compare the generation of confusing images and generate metrics comparison tables.
- `compare_adversarial_lava.py` ... targeting several variation of LAVA methods.

- `load_*.ipynb` scripts to load the saved results and draw figures for the report.
- `*.ipynb` playgrounds.

## Baseline
Baseline approaches are in folder `adversarial_methods`.

## Results
- `trained_models` saved all trained cae, vae and cvae models with epoch 50, 100, 150 and 200.
- `results` saved all results displayed in thesis, such as average time and success rate. In addition, several generated adversarial images are also saved.
- `survey` saved the survey results of human assessor.