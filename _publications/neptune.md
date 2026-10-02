---
title: "NePTune: A Neuro-Pythonic Framework for Tunable Compositional Reasoning on Vision-Language"
collection: publications
permalink: /publication/neptune
excerpt: 'NePTune translates a question into a Python program that combines imperative control flow with soft logic over scores from a vision-language model, and runs it without training.'
date: 2026-01-26
venue: 'International Conference on Learning Representations (ICLR)'
authors: 'Danial Kamali, Parisa Kordjamshidi'
award: 'Best Paper Award, MSLD 2026'
highlight: 'Oral, SpaVLE Workshop @ NeurIPS 2025'
image: "files/neptune/img.jpg"
header: "files/neptune/overview.webp"
paperurl: 'https://arxiv.org/pdf/2509.25757'
arxivurl: 'https://arxiv.org/abs/2509.25757'
openreviewurl: 'https://openreview.net/forum?id=8H0TkSusWI'
codeurl: "https://github.com/HLR/NePTune"
bibtex: '@inproceedings{kamali2026neptune,
title={Ne{PT}une: A Neuro-Pythonic Framework for Tunable Compositional Reasoning on Vision-Language},
author={Danial Kamali and Parisa Kordjamshidi},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=8H0TkSusWI}
}'
---

<p style="text-align: center; font-size: 0.9em; color: #6c757d;"><em>NePTune overview (figure from the paper). Given an image and a query, (1) LLM-based program generation converts the query into a Pythonic program, (2) perceptual grounding extracts object bounding boxes, and (3) the symbolic executor runs the program, combining concepts scored by the VLM with soft composition and imperative logic to derive the answer.</em></p>

## **Abstract**

Modern Vision-Language Models (VLMs) have achieved impressive performance in various tasks, yet they often struggle with compositional reasoning, the ability to decompose and recombine concepts to solve novel problems. While neuro-symbolic approaches offer a promising direction, they are typically constrained by crisp logical execution or predefined predicates, which limit flexibility. In this work, we introduce NePTune, a neuro-symbolic framework that overcomes these limitations through a hybrid execution model that integrates the perception capabilities of foundation vision models with the compositional expressiveness of symbolic reasoning. NePTune dynamically translates natural language queries into executable Python programs that blend imperative control flow with soft logic operators capable of reasoning over VLM-generated uncertainty. Operating in a training-free manner, NePTune, with a modular design, decouples perception from reasoning, yet its differentiable composition operations support fine-tuning. We evaluate NePTune on multiple visual reasoning benchmarks and various domains, utilizing adversarial tests, and demonstrate a significant improvement over base models, as well as its effective compositional generalization and adaptation capabilities in novel environments.

NePTune was accepted to ICLR 2026 and was presented as an oral at the [SpaVLE workshop](https://space-in-vision-language-embodied-ai.github.io/) at NeurIPS 2025.

```bibtex
@inproceedings{kamali2026neptune,
title={Ne{PT}une: A Neuro-Pythonic Framework for Tunable Compositional Reasoning on Vision-Language},
author={Danial Kamali and Parisa Kordjamshidi},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=8H0TkSusWI}
}
```
