# KE-RT-DETR: Knowledge-Enhanced RT-DETR for Infrared Small Target Detection

**This repository contains the core code implementation of our manuscript.** 

**Due to the ongoing review process, we have currently uploaded:**
- **Core model definitions (three key modules)**
- **Complete YAML configuration**

**The full project (training scripts, evaluation pipelines, and pre-trained weights) will be made publicly available upon paper acceptance.**

## Overview

This repository contains the official implementation of **KE-RT-DETR**, a knowledge-enhanced RT-DETR framework for infrared small target detection. The model integrates three domain-prior modules:

| Module                                   | Abbr.         | Description                                                  |
| ---------------------------------------- | ------------- | ------------------------------------------------------------ |
| Statistical-Physical Context Enhancement | **SPCE**      | Input-level enhancement via multi-directional gradients and statistical priors |
| Soft-Gated SCConv                        | **SG-SCConv** | Feature-level target-background decoupling via soft gating   |
| Gradient-Adaptive Reparameterized CDC    | **GA-RepCDC** | Representation-level contrast enhancement with zero inference overhead |

---

## Datasets

[DAUB](https://www.scidb.cn/en/detail?dataSetId=720626420933459968)
[IRSTD-1k](https://drive.google.com/file/d/1JoGDGF96v4CncKZprDnoIor0k1opaLZa/view)

We thank the providers of the DAUB and IRSTD-1k datasets for making their data available.

