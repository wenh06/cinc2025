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

[Conference Website](https://cinc2025.org/) | more to be added...

## Description of the files/folders(modules)

### Files

<details>
<summary>Click to view the details</summary>

- [README.md](README.md): this file, serves as the documentation of the project.
- [cfg.py](cfg.py): the configuration file for the whole project.
- [const.py](const.py): constant definitions.
- [Dockerfile](Dockerfile): docker file for building the docker image for submissions.
- [requirements.txt](requirements.txt), [requirements-docker.txt](requirements-docker.txt), [requirements-no-torch.txt](requirements-no-torch.txt): requirements files for different purposes.

</details>

### Folders(Modules)

<details>
<summary>Click to view the details</summary>

- [official_baseline](official_baseline): the official baseline code, included as a submodule.
- [official_scoring_metric](official_scoring_metric): the official scoring code, included as a submodule.

</details>

:point_right: [Back to TOC](#cinc2025)
