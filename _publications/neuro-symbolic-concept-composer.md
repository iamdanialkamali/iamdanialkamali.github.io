---
title: "NeSyCoCo: A Neuro-Symbolic Concept Composer for Compositional
Generalization"
collection: publications
permalink: /publication/neuro-symbolic-concept-composer
image: "https://reascan.github.io/assets/data-browser-examples/138-no-target.png"
excerpt: 'Compositional generalization is crucial for artificial intelligence agents tackling intricate reasoning over vision and language (V\&L) problems. While neuro-symbolic methods have demonstrated potential in understanding compositional structures, they face challenges such as the need for symbolic domain representations that typically involve a set of predefined predicates, difficulties in deriving domain predicates from raw data, and the requirement for differentiable operations to compose primitive concepts. To address these issues, we propose NeSyCoCo, which is built on the existing neuro-symbolic frameworks that leverage large language models (LLMs) for obtaining symbolic representations of the domain and map them to differentiable neural computations for V\&L reasoning. Our approach a) augments the natural language inputs with their dependency structure to improve the accuracy of symbolic representations, b) utilizes distributed word representations for handling the variety of linguistically motivated logical predicates that are linked to neural modules, and c) utilizes soft composition of normalized predicate scores for better semantic alignment between symbolic compositions and differentiable operations. NeSyCoCo achieves state-of-the-art results on the ReaSCAN and CLEVR-CoGenT compositional generalization benchmarks, as well as the CLEVR vision-language benchmark. It also maintains high accuracy with new, similar concepts in the CLEVR-SYN benchmark.'
date: 2024-12-12
venue: 'Preprint'
paperurl: 'github.com'
citation: 'Kamali, Danial, and Parisa Kordjamshidi. "Syntax-Guided Transformers: Elevating Compositional Generalization and Grounding in Multimodal Environments." GenBench: The first workshop on generalisation (benchmarking) in NLP. 2023.'
---

Compositional generalization, the ability of intelligent models to extrapolate understanding of components to novel compositions, is a fundamental yet challenging facet in AI research, especially within multimodal environments. In this work, we address this challenge by exploiting the syntactic structure of language to boost compositional generalization. This paper elevates the importance of syntactic grounding, particularly through attention masking techniques derived from text input parsing. We introduce and evaluate the merits of using syntactic information in the multimodal grounding problem. Our results on grounded compositional generalization underscore the positive impact of dependency parsing across diverse tasks when utilized with Weight Sharing across the Transformer encoder. The results push the state-of-the-art in multimodal grounding and parameter-efficient modeling and provide insights for future research.

```bibtex
@inproceedings{kamali2023syntax,
  title={Syntax-Guided Transformers: Elevating Compositional Generalization and Grounding in Multimodal Environments},
  author={Kamali, Danial and Kordjamshidi, Parisa},
  booktitle={GenBench: The first workshop on generalisation (benchmarking) in NLP},
  pages={130},
  year={2023}
}
```
