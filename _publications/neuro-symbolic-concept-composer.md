---
title: "NeSyCoCo: A Neuro-Symbolic Concept Composer for Compositional Generalization"
collection: publications
permalink: /publication/neuro-symbolic-concept-composer
image: "files/neuro-symbolic-concept-composer/nesycoco-framework.png"
date: 2025-01-01
venue: 'Association for the Advancement of Artificial Intelligence (AAAI)'
image: "files/neuro-symbolic-concept-composer/img.png"
header: "files/neuro-symbolic-concept-composer/pipeline.svg"
paperurl: 'https://iamdanialkamali.github.io/publication/neuro-symbolic-concept-composer'
slidesurl: "https://iamdanialkamali.github.io/publication/neuro-symbolic-concept-composer"
posterurl: "https://iamdanialkamali.github.io/publication/neuro-symbolic-concept-composer"
codeurl: "https://github.com/HLR/NeSyCoCo"
citation: |
  @inproceedings{kamali2025nesycoco,
    title={NeSyCoCo: A Neuro-Symbolic Concept Composer for Compositional Generalization},
    author={Kamali, Danial and Barezi, Elham J. and Kordjamshidi, Parisa},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    year={2025}
  }
---
## **Abstract**
NeSyCoCo is a neuro-symbolic visual reasoning framework that tackles generalization in vision-language tasks and more specially **compositional generalization**.

**Highlights**:
- **State-of-the-Art Results**: Demonstrated on ReaSCAN, CLEVR-CoGenT, and CLEVR-SYN.
- **Novel Contributions**: Alleviating three main issues in neuro-symbolic vision-language reasoning.
    - **Predefined Predicates**: Generalizable predicate function to **handle language variety** and **concept generalization**.
    - **Language-to-Program Bottleneck**: Utilizing syntactic information for **improved language-to-symbolic program** generation.
    - **Concept Composition**: Proposing an improved set of functions for **more effective composition** in first-order logic.

---

### **Key Components**
- **Language-to-Program Module**: Converts natural language queries into symbolic programs using **dependency parsing**.
  <div style="text-align: center;">
    <img style="width: 50%; margin: 2em 0em 1em 0em;" src="../files/neuro-symbolic-concept-composer/language_to_program.svg" alt="Figure 2 illustrates the language-to-program process">
    <p><em>Language-to-Program process</em></p>
  </div>
- **Perception Module**: Extracts visual features and relationships from images via models like Mask RCNN.
- **Differentiable Executor**: Executes symbolic programs with soft composition for robust generalization.
  - **Predicate Functions**:
    <div style="text-align: center;">
      <img style="width: 80%; margin: 2em 0em 1em 0em;" src="../files/neuro-symbolic-concept-composer/predicate_function.svg" alt="Figure 3 illustrates the First-Order Logic functions">
      <p><em>Generalizable predicate function compared to LEFT</em></p>
    </div>
  - **First-Order-Logic Function**
    <div style="text-align: center;">
      <img style="width: 80%; margin: 2em 0em 1em 0em;" src="../files/neuro-symbolic-concept-composer/fol_functions.png" alt="Figure 3 illustrates the First-Order Logic functions">
      <p><em>FOL functions</em></p>
    </div>
---

## **Key Results**

### **1. Compositional Generalization**
- **ReaSCAN Benchmark (Table 2)**:
    - Outperformed baselines with **97.3% accuracy** on relative clause and spatial reasoning splits.
- **CLEVR-CoGenT (Table 3)**:
    - Achieved **78.8% accuracy** on unseen attribute combinations in Split B.

---

### **2. Vision-Language Reasoning**
- **CLEVR Extensions**:
    - **CLEVR-RPM**: Perfect performance (**100% accuracy**) on relational reasoning tasks.
    - **CLEVR-Puzzle**: High accuracy (**95%**), demonstrating robustness in multi-step reasoning.
    - *Detailed results are shown in Table 5.*

---

### **3. Handling Predicate Language Variety**
- **CLEVR-SYN Benchmark (Table 7)**:
    - Showed strong zero-shot generalization to unseen synonyms (e.g., "huge" for "large").
    - Maintained **73.4% accuracy** on the hardest splits.

<div style="text-align: center;">
  <img style="width: 70%" src="../files/neuro-symbolic-concept-composer/correlation.svg" alt="Figure 5 demonstrates the relationship between predicate embeddings’ similarity and generalization accuracy">
  <p><em>Figure 5: Predicate Embeddings Similarity vs. Pearson Correlation of Predicate Score</em></p>
</div>

### 4. Why soft composition?
- **Comparison of Concept Scores**:
  - Presented boxplots compare concept scores of LEFT and NeSyCoCo using 10k CLEVR validation samples.

- **Variability in Score Ranges**:
  - Different concepts exhibit significantly varying score ranges, leading to inconsistencies.

- **Challenges with Min Function**:
  - **Undervaluation of Specific Concepts**:
    - When composing concepts (e.g., `red` and `rubber`) using a *min* function, the score for one concept (e.g., `red`) is often undervalued.
  - **Biased Composition**:
    - This undervaluation results in a biased composition that fails to accurately reflect the true relationship between the combined concepts.

- **Advantages of Soft Composition**:
    - **Normalized Predicate Scores**: Soft composition utilizes normalized predicate scores, mitigating the issues caused by varying score ranges.


<div>
<div style="display: inline-flex; justify-content: center; gap: 20px; text-align: center;">
  <div style="text-align: center; width: 100%;">
    <img src="../files/neuro-symbolic-concept-composer/box_plot_left.svg" alt="Figure 4.1 compares NeSyCoCo’s normalized predicate scores with the previous LEFT method.">
  </div>
  <div style="text-align: center; width: 100%;">
    <img src="../files/neuro-symbolic-concept-composer/box_plot_nesycoco.svg" alt="Figure 4.2 compares NeSyCoCo’s normalized predicate scores with the previous LEFT method.">
  </div>
</div>
<div style="text-align: center;">
<p><em>NeSyCoCo predicate score box plot compared to LEFT (Dotted and solid lines represent the mean and median, respectively) </em></p>
</div>
</div>

---

## **How NeSyCoCo Differs**
1. **Soft Predicate Composition**:
    - Unlike scalar scores in LEFT, NeSyCoCo employs **normalized predicate scores**, improving robustness (Figure 4).

2. **Dependency Parsing**:
    - Enhances program accuracy, especially for **nested queries** (Figure 2).

3. **Distributed Predicate Representation**:
    - Addresses language variability using word representations from pre-trained encoders.

---

