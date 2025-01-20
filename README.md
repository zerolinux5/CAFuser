# CAFuser: Condition-Aware Multimodal Fusion for Robust Semantic Perception of Driving Scenes

**[[Paper]](https://arxiv.org/pdf/2410.10791)**

:bell: **News:**

* [2025-01-11] We are happy to announce that CAFuser was accepted in the **IEEE Robotics and Automation Letters**.


### Overview

This repository contains the official code for the **RA-L 2025** paper [CAFuser: Condition-Aware Multimodal Fusion for Robust Semantic Perception of Driving Scenes](https://arxiv.org/pdf/2410.10791).
CAFuser is a condition-aware multimodal fusion architecture designed to enhance robust semantic perception in autonomous driving. It employs a **Condition Token (CT)**, dynamically guiding the fusion of multiple sensor modalities to optimize performance across diverse scenarios. To train the CT, a **verbo-visual contrastive loss** aligns it with semantic environmental descriptors, enabling direct prediction from RGB features. 
The **Condition-Aware Fusion** module uses the CT to adaptively fuse sensor data based on environmental context.
Further, CAFuser introduces modality-specific feature adapters, aligning inputs from different sensors into a shared latent space, and integrates these without loss of performance with a single shared backbone.
CAFuser ranks **first on the public [MUSES](https://muses.vision.ee.ethz.ch/) benchmarks**, achieving 59.7 PQ for multimodal panoptic and 78.2 mIoU for semantic segmentation and also sets the new state of the art on [DeLiVER](https://github.com/jamycheung/DELIVER).

![CAFuser Overview Figure](resources/cafuser_teaser.png)

### Code
Coming soon...
