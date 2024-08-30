---
title: "Using Persuasive Writing Strategies to Explain and Detect Health Misinformation"
collection: publications
permalink: /publication/misinformation-detection
excerpt: 'In this work we introduce a persuasive strategy detection dataset and show using their labels can improve misinformation detection and explanation.'
date: 2024-05-01
venue: 'Joint International Conference on Computational Linguistics, Language Resources and Evaluation'
image: "files/misinformation-detection/img.png"
paperurl: 'https://aclanthology.org/2024.lrec-main.1501.pdf'
slidesurl: "files/misinformation-detection/slides.pdf"
posterurl: "files/misinformation-detection/poster.pdf"
codeurl: "https://github.com/HLR/Misinformation-Detection"
citation: 'Kamali, D., Romain, J., Liu, H., Peng, W., Meng, J., & Kordjamshidi, P. (2023). Using Persuasive Writing Strategies to Explain and Detect Health Misinformation. arXiv preprint arXiv:2211.05985.'
bibtex: '@inproceedings{kamali-etal-2024-using,
    title = "Using Persuasive Writing Strategies to Explain and Detect Health Misinformation",
    author = "Kamali, Danial  and
      Romain, Joseph D.  and
      Liu, Huiyi  and
      Peng, Wei  and
      Meng, Jingbo  and
      Kordjamshidi, Parisa",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1501",
    pages = "17285--17309",
    abstract = "Nowadays, the spread of misinformation is a prominent problem in society. Our research focuses on aiding the automatic identification of misinformation by analyzing the persuasive strategies employed in textual documents. We introduce a novel annotation scheme encompassing common persuasive writing tactics to achieve our objective. Additionally, we provide a dataset on health misinformation, thoroughly annotated by experts utilizing our proposed scheme. Our contribution includes proposing a new task of annotating pieces of text with their persuasive writing strategy types. We evaluate fine-tuning and prompt-engineering techniques with pre-trained language models of the BERT family and the generative large language models of the GPT family using persuasive strategies as an additional source of information. We evaluate the effects of employing persuasive strategies as intermediate labels in the context of misinformation detection. Our results show that those strategies enhance accuracy and improve the explainability of misinformation detection models. The persuasive strategies can serve as valuable insights and explanations, enabling other models or even humans to make more informed decisions regarding the trustworthiness of the information.",
}'
---

The spread of misinformation is a prominent problem in today's society, and many researchers in academia and industry are trying to combat it. Due to the vast amount of misinformation that is created every day, it is unrealistic to leave this task to human fact-checkers. Data scientists and researchers have been working on automated misinformation detection for years, and it is still a challenging problem today. The goal of our research is to add a new level to automated misinformation detection; classifying segments of text with persuasive writing techniques in order to produce interpretable reasoning for why an article can be marked as misinformation. To accomplish this, we present a novel annotation scheme containing many common persuasive writing tactics, along with a dataset with human annotations accordingly. For this task, we make use of a RoBERTa model for text classification, due to its high performance in NLP. We develop several language model-based baselines and present the results of our persuasive strategy label predictions as well as the improvements these intermediate labels make in detecting misinformation and producing interpretable results.

