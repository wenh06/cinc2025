# CinC2025

[![docker-ci-and-test](https://github.com/wenh06/cinc2025/actions/workflows/docker-test.yml/badge.svg?branch=docker-test)](https://github.com/wenh06/cinc2025/actions/workflows/docker-test.yml)
[![format-check](https://github.com/wenh06/cinc2025/actions/workflows/check-formatting.yml/badge.svg)](https://github.com/wenh06/cinc2025/actions/workflows/check-formatting.yml)

<p align="left">
  <img src="images/cinc2025-banner.png" width="40%" />
</p>

Detection of Chagas Disease from the ECG: The George B. Moody PhysioNet Challenge 2025

[Challenge Website](https://moody-challenge.physionet.org/2025/)

<!-- toc -->

- [The Conference](#the-conference)

<!-- tocstop -->

## The Conference

[Conference Website](https://cinc2025.org/) |
[Unofficial Phase Leaderboard](https://docs.google.com/spreadsheets/d/e/2PACX-1vQIDwTRtZc7goD10lScYl20J0xfjaPb1tHVyeqr5zmgZPMDhXj034S6w7fW8SJwzlAgKezxd5w9vS2i/pubhtml?gid=173721180&single=true&widget=true&headers=false)

## Description of the files/folders(modules)

### Files

<details>
<summary>Click to view the details</summary>

- [README.md](README.md): this file, serves as the documentation of the project.
- [cfg.py](cfg.py): the configuration file for the whole project.
- [const.py](const.py): constant definitions.
- [Dockerfile](Dockerfile): docker file for building the docker image for submissions.
- [requirements.txt](requirements.txt), [requirements-docker.txt](requirements-docker.txt), [requirements-no-torch.txt](requirements-no-torch.txt): requirements files for different purposes.
- [evaluate_model.py](evaluate_model.py), [helper_code.py](helper_code.py), [prepare_code15_data.py](prepare_code15_data.py), [run_model.py](run_model.py), [train_model.py](train_model.py): scripts inherited from the [official baseline](https://github.com/physionetchallenges/python-example-2025.git) and [official scoring code](https://github.com/physionetchallenges/evaluation-2025.git). Modifications on these files are invalid and are immediately overwritten after being pulled by the organizers (or the submission system).
- [sync_official.py](sync_official.py): script for synchronizing data from the official baseline and official scoring code.
- [team_code.py](team_code.py): entry file for the submissions.
- [submissions](submissions): log file for the submissions, including the key hyperparameters, the scores received, commit hash, etc. The log file is updated after each submission and organized as a YAML file.

</details>

### Folders(Modules)

<details>
<summary>Click to view the details</summary>

- [official_baseline](official_baseline): the official baseline code, included as a submodule.
- [official_scoring_metric](official_scoring_metric): the official scoring code, included as a submodule.

</details>

:point_right: [Back to TOC](#cinc2025)
