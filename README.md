# CinC2025

[![docker-ci-and-test](https://github.com/wenh06/cinc2025/actions/workflows/docker-test.yml/badge.svg?branch=docker-test)](https://github.com/wenh06/cinc2025/actions/workflows/docker-test.yml)
[![format-check](https://github.com/wenh06/cinc2025/actions/workflows/check-formatting.yml/badge.svg)](https://github.com/wenh06/cinc2025/actions/workflows/check-formatting.yml)

<p align="left">
  <img src="images/cinc2025-banner.png" width="40%" />
</p>

Detection of Chagas Disease from the ECG: The George B. Moody PhysioNet Challenge 2025

[Challenge Website](https://moody-challenge.physionet.org/2025/)

:skull::skull::skull: **BIG MISTAKE**: Forgot to add (z-score) normalization in the preprocessing pipeline in the [config](cfg.py)!!! :skull::skull::skull:

<!-- toc -->

- [The Conference](#the-conference)
- [Description of the files/folders(modules)](#description-of-the-filesfoldersmodules)
- [Key problem to solve](#key-problem-to-solve)
- [Background knowledge](#background-knowledge)
- [Post-Conference Thoughts](#post-conference-thoughts)

<!-- tocstop -->

## The Conference

[Conference Website](https://cinc2025.org/) |
[Unofficial Phase Leaderboard](https://docs.google.com/spreadsheets/d/e/2PACX-1vQIDwTRtZc7goD10lScYl20J0xfjaPb1tHVyeqr5zmgZPMDhXj034S6w7fW8SJwzlAgKezxd5w9vS2i/pubhtml?gid=173721180&single=true&widget=true&headers=false)[^1] |
[Official Phase Leaderboard](https://docs.google.com/spreadsheets/d/e/2PACX-1vQtPv9dEP-aC1N7Vds-zy63Jy8XftSeftCbtpdBDXZ7ubKT7eHHKLjSrFyaJ7v881PlHDXrnPiYZwbU/pubhtml?gid=1127050801&single=true) |
[Final Test Results](https://moody-challenge.physionet.org/2025/results/)

[^1]: As clarified by the organizers, the validation set for the official phase was updated, hence the unofficial and official phase leaderboards are not comparable.

<p align="middle">
  <img src="images/badge-of-recognition-1.png" width="40%" />
  &nbsp; &nbsp; &nbsp;
  <img src="images/badge-of-recognition-2.png" width="40%" />
</p>

### The Conference Poster

<details>
<summary>Click to view the details</summary>

<div style="text-align:center; margin:20px 0; position:relative;">
  <img src="images/cinc2025-poster.svg"
       alt="The Conference Poster"
       style="width:80%; max-width:800px; border:1px solid #eee; box-shadow:0 2px 8px rgba(0,0,0,0.1);"
       onerror="this.style.display='none';">
  <div style="display:none; width:80%; max-width:800px; margin:0 auto; padding:40px; border:1px solid #eee; background:#f9f9f9; text-align:center;"
       id="fallback-text">
    The Conference Poster
  </div>
</div>

</details>

:point_right: [Back to TOC](#cinc2025)

## Description of the files/folders(modules)

### Files

<details>
<summary>Click to view the details</summary>

- [README.md](README.md): this file, serves as the documentation of the project.
- [cfg.py](cfg.py): the configuration file for the whole project.
- [const.py](const.py): constant definitions.
- [Dockerfile](Dockerfile): docker file for building the docker image for submissions.
- [requirements.txt](requirements.txt), [requirements-docker.txt](requirements-docker.txt), [requirements-no-torch.txt](requirements-no-torch.txt):
  requirements files for different purposes.
- [evaluate_model.py](evaluate_model.py), [helper_code.py](helper_code.py), [prepare_code15_data.py](prepare_code15_data.py),
  [run_model.py](run_model.py), [train_model.py](train_model.py): scripts inherited from the
  [official baseline](https://github.com/physionetchallenges/python-example-2025.git) and
  [official scoring code](https://github.com/physionetchallenges/evaluation-2025.git).
  Modifications on these files are invalid and are immediately overwritten after being pulled by the organizers (or the submission system).
- [sync_official.py](sync_official.py): script for synchronizing data from the official baseline and official scoring code.
- [team_code.py](team_code.py): entry file for the submissions.
- [submissions](submissions): log file for the submissions, including the key hyperparameters, the scores received,
  commit hash, etc. The log file is updated after each submission and organized as a YAML file.

</details>

### Folders(Modules)

<details>
<summary>Click to view the details</summary>

- [official_baseline](official_baseline): the official baseline code, included as a submodule.
- [official_scoring_metric](official_scoring_metric): the official scoring code, included as a submodule.
- [models](models): folder for model definitions, typically we used a [CRNN model](models/crnn.py).
  Some [custom loss functions](models/loss.py) are also defined in this module.
- [utils](utils): various utility functions, including [custom scoring functions](utils/scoring_metrics.py),
  and some training-validation split files.
- [results](results): folder containing some typical experiment log files, for reproducibility.

</details>

:point_right: [Back to TOC](#cinc2025)

## Key problem to solve

The data is highly imbalanced, with only approximately 2% of the data being positive.
Dealing with the imbalanced data is the key problem to solve in this challenge. Possible solutions include:

- Upsampling the positive data
- Downsampling the negative data
- Using Focal Loss, Asymmetric Loss, etc.
- Using class weights
- Using data augmentation, including Mixup, Cutmix, etc.

## Background knowledge

### Chagas Disease and ECG

According to a [review paper](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0006567)
about ECG abnormalities in Chagas Disease, the most common ECG abnormalities are:

- Prevalence of overall ECG abnormalities was higher in participants with CD (40.1%; 95%CIs=39.2-41.0)
  compared to non-CD (24.1%; 95%CIs=23.5-24.7) (OR=2.78; 95%CIs=2.37-3.26).
- Among specific ECG abnormalities, prevalence of
  - complete right bundle branch block (RBBB) (OR=4.60; 95%CIs=2.97-7.11),
  - left anterior fascicular block (LAFB) (OR=1.60; 95%CIs=1.21-2.13),
  - combination of complete RBBB/LAFB (OR=3.34; 95%CIs=1.76-6.35),
  - first-degree atrioventricular block (A-V B) (OR=1.71; 95%CIs=1.25-2.33),
  - atrial fibrillation (AF) or flutter (OR=2.11; 95%CIs=1.40-3.19),
  - ventricular extrasystoles (VE) (OR=1.62; 95%CIs=1.14-2.30)

  was higher in CD compared to non-CD participants

## Post-Conference Thoughts

### Thoughts on the Final Results

- High-performing teams often accepted very low accuracy in exchange for better recall/risk ranking.
- Foundation/self-supervised ECG encoders (ViT/Transformer backbones, distilled/foundation models pretrained on large ECG corpora) are widely used.

### Miscellaneous

[CinC2020](https://github.com/DeepPSP/cinc2020) | [CinC2021](https://github.com/DeepPSP/cinc2021) | [CinC2022](https://github.com/DeepPSP/cinc2022) | [CinC2023](https://github.com/wenh06/cinc2023) | [CinC2024](https://github.com/wenh06/cinc2024)
