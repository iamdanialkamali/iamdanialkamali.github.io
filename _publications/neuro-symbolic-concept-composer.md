---
title: "NeSyCoCo: A Neuro-Symbolic Concept Composer for Compositional Generalization"
collection: publications
permalink: /publication/neuro-symbolic-concept-composer
image: "files/neuro-symbolic-concept-composer/nesycoco-framework.png"
date: 2024-12-12
venue: 'AAAI 2025'
paperurl: 'https://github.com/HLR/NeSyCoCo'
citation: |
  @inproceedings{kamali2025nesycoco,
    title={NeSyCoCo: A Neuro-Symbolic Concept Composer for Compositional Generalization},
    author={Kamali, Danial and Barezi, Elham J. and Kordjamshidi, Parisa},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    year={2025}
  }
---

## **Abstract**
NeSyCoCo combines **symbolic reasoning** with **neural networks** to tackle **compositional generalization** in vision-language tasks. The framework introduces:
1. **Dependency-parsing for natural language augmentation**.
2. **Distributed linguistic embeddings** for predicate representation.
3. **Soft composition** of predicate scores for enhanced reasoning.

**Highlights**:
- **State-of-the-Art Results**: Demonstrated on ReaSCAN, CLEVR-CoGenT, and CLEVR-SYN.
- **Novel Contribution**: Bridged symbolic and neural reasoning with a unified framework.

---

## **Framework Overview**

![Framework Overview](files/neuro-symbolic-concept-composer/framework.png)  
*Figure 1: NeSyCoCo framework, comprising three components: Language-to-Program module, Perception module, and Differentiable Neuro-Symbolic Executor.*

### Key Components:
- **Language-to-Program Module**: Converts natural language queries into symbolic programs using **dependency parsing**.  
  *Figure 2 illustrates the process.*
- **Perception Module**: Extracts visual features and relationships from images via models like Mask RCNN.  
- **Differentiable Executor**: Executes symbolic programs with soft composition for robust generalization.  

---

## **Key Results**

### **1. Compositional Generalization**
- **ReaSCAN Benchmark (Table 2)**:  
  - Outperformed baselines with **97.3% accuracy** on relative clause and spatial reasoning splits.  
- **CLEVR-CoGenT (Table 3)**:  
  - Achieved **78.8% accuracy** on unseen attribute combinations in Split B.  

**Figure 4** compares NeSyCoCo’s normalized predicate scores with the previous LEFT method.

---

### **2. Vision-Language Reasoning**
- **CLEVR Extensions**:
  - **CLEVR-RPM**: Perfect performance (**100% accuracy**) on relational reasoning tasks.  
  - **CLEVR-Puzzle**: High accuracy (**95%**), demonstrating robustness in multi-step reasoning.  
  - *Detailed results are shown in Table 5.*

---

### **3. Handling Synonyms**
- **CLEVR-SYN Benchmark (Table 7)**:
  - Showed strong zero-shot generalization to unseen synonyms (e.g., "huge" for "large").
  - Maintained **73.4% accuracy** on the hardest splits.  

*Figure 5 demonstrates the relationship between predicate embeddings’ similarity and generalization accuracy.*

---

## **How NeSyCoCo Differs**
1. **Soft Predicate Composition**:
   - Unlike scalar scores in LEFT, NeSyCoCo employs **normalized predicate scores**, improving robustness (Figure 4).  

2. **Dependency Parsing**:
   - Enhances program accuracy, especially for **nested queries** (Figure 2).

3. **Distributed Embeddings**:
   - Addresses language variability using word representations from pre-trained encoders.

---

## **Figures and Tables**

### Figures:
- **Figure 1**: NeSyCoCo framework (already in Framework Overview).
- **Figure 2**: Language-to-Program module (place under **Language-to-Program Module**).
- **Figure 4**: Comparison of predicate scores for NeSyCoCo and LEFT (place under **Key Results**).
- **Figure 5**: Relationship between embedding similarity and accuracy (place in **Handling Synonyms**).

### Tables:
- **Table 1**: Logical forms and differentiable implementations (place under **Framework Overview**).
- **Table 2**: Accuracy on the ReaSCAN benchmark (in **Compositional Generalization** section).
- **Table 3**: CLEVR-CoGenT benchmark results (same section).
- **Table 5**: CLEVR extension results (in **Vision-Language Reasoning** section).
- **Table 7**: CLEVR-SYN benchmark accuracy (in **Handling Synonyms** section).

---

## **Resources**
- **Paper PDF**: [Link](https://github.com/HLR/NeSyCoCo)  
- **Code Repository**: [GitHub Repository](https://github.com/HLR/NeSyCoCo)  
- **Slides**: Coming soon!  

---