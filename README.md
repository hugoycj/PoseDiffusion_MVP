---
title: PoseDiffusion_MVP
emoji: 🐠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 3.38.0
app_file: app.py
pinned: false
license: apache-2.0
---

# An Out-Of-The-Box Version of PoseDiffusion
[![Hugging Face Spaces](splash_sample2_record.gif)](https://huggingface.co/spaces/chongjie/PoseDiffusion_MVP)

## Introduction
Camera pose estimation is a critical task in computer vision, traditionally relying on classical methods such as keypoint matching, RANSAC, and bundle adjustment. [PoseDiffusion](https://posediffusion.github.io/) introduces a novel approach to this problem by formulating the Structure from Motion (SfM) problem within a probabilistic diffusion framework. 

[![Demo Video](https://posediffusion.github.io/resources/qual_co3d.png)](https://posediffusion.github.io/resources/splash_sample2.mp4 "Demo Video")

## Usage

There are several ways you can use or interact with this project:

* **Direct Use**: If you want to use the space directly without any modifications, simply click [here](https://huggingface.co/spaces/chongjie/PoseDiffusion_MVP). This will take you to the live application where you can interact with it as is.

* **Duplicate the Space**: If you want to create a copy of this space for your own use or modifications, click [here](https://huggingface.co/spaces/chongjie/co-tracker?duplicate=true). This will create a duplicate of the space under your account, which you can then modify as per your needs.

* **Run with Docker**: If you prefer to run the application locally using Docker, you can do so with the following command:

    ```bash
    docker run -it -p 7860:7860 --platform=linux/amd64 \
    registry.hf.space/chongjie-posediffusion-mvp:latest python app.py
    ```

## Acknowledgments
This repository is based on original [PoseDiffusion](https://posediffusion.github.io/)
