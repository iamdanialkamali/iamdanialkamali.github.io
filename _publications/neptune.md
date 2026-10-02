---
title: "NePTune: A Neuro-Pythonic Framework for Tunable Compositional Reasoning on Vision-Language"
collection: publications
permalink: /publication/neptune
excerpt: 'NePTune translates a question into a Python program that combines imperative control flow with soft logic over scores from a vision-language model, and runs it without training.'
date: 2026-01-26
venue: 'International Conference on Learning Representations (ICLR)'
authors: 'Danial Kamali, Parisa Kordjamshidi'
affiliations: 'Michigan State University'
award: 'Best Paper Award, MSLD 2026'
highlight: 'Oral, SpaVLE Workshop @ NeurIPS 2025'
image: "files/neptune/img.jpg"
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

<link rel="stylesheet" href="{{ site.baseurl }}/files/neptune/neptune.css">

<div class="npt">

<div class="npt-tldr"><b>TL;DR</b> Vision-language models often fail at questions that combine several concepts. NePTune has an LLM translate the question into a short Python program. Inside the program, a <b>declarative</b> first-order-logic formula combines soft concept scores from a vision-language model, while <b>imperative</b> Python handles loops, conditionals and counting. NePTune works zero-shot, and because the logic operators are differentiable, the same programs can also be used to fine-tune the vision-language model.</div>

<h2 id="tree">From question to program to tree</h2>

<p>This walkthrough follows the example in Figure&nbsp;2 of the paper. The program is shown first as Python code and then as a tree. <span class="npt-k-con">Green</span> nodes are concept scores from the vision-language model (VLM), <span class="npt-k-sl">blue</span> nodes are soft logic, and <span class="npt-k-imp">purple</span> nodes are ordinary Python. The declarative part scores every detected box at once. The imperative part then loops over the boxes, branches on their scores and counts the answers. Press <b>&#9654;</b> to play or pause, use the numbered steps to jump, and hover over a line of code or a node to see its counterpart.</p>

<div id="npt-fwd" class="npt-demo" data-src="{{ site.baseurl }}/files/neptune/">
<p class="nd-loading">Loading the example&hellip;</p>
<noscript><p>The walkthrough needs JavaScript. The paper's overview figure is shown below in the method section.</p></noscript>
</div>


<h2 id="tunable">Tunable end to end</h2>

<p>Every soft-logic operator in NePTune has a differentiable implementation, so a program also carries gradients. This walkthrough uses a referring expression from Ref-GTA, Expression&nbsp;1 of Figure&nbsp;3 in the paper, whose program is a single first-order-logic formula. After the forward pass, a binary cross-entropy loss on the program&rsquo;s answer flows back through <code>iota</code> and the soft AND into the concept scores, and from there into the VLM. This is how the paper fine-tunes a VLM for Ref-GTA, a video game domain that is new to VLMs trained on natural images.</p>

<div id="npt-ref" class="npt-demo" data-src="{{ site.baseurl }}/files/neptune/">
<p class="nd-loading">Loading the example&hellip;</p>
<noscript><p>The walkthrough needs JavaScript.</p></noscript>
</div>


<h2 id="abstract">Abstract</h2>

<p>Modern Vision-Language Models (VLMs) have achieved impressive performance in various tasks, yet they often struggle with compositional reasoning, the ability to decompose and recombine concepts to solve novel problems. While neuro-symbolic approaches offer a promising direction, they are typically constrained by crisp logical execution or predefined predicates, which limit flexibility. In this work, we introduce NePTune, a neuro-symbolic framework that overcomes these limitations through a hybrid execution model that integrates the perception capabilities of foundation vision models with the compositional expressiveness of symbolic reasoning. NePTune dynamically translates natural language queries into executable Python programs that blend imperative control flow with soft logic operators capable of reasoning over VLM-generated uncertainty. Operating in a training-free manner, NePTune, with a modular design, decouples perception from reasoning, yet its differentiable composition operations support fine-tuning. We evaluate NePTune on multiple visual reasoning benchmarks and various domains, utilizing adversarial tests, and demonstrate a significant improvement over base models, as well as its effective compositional generalization and adaptation capabilities in novel environments.</p>

<h2 id="method">How NePTune works</h2>

<figure>
<img src="{{ site.baseurl }}/files/neptune/compositional.webp" alt="A query is decomposed into the concepts behind, brown, red and sphere, which are scored and composed into a logical form" loading="lazy" width="1800" height="479">
<figcaption>Figure 1 of the paper. The query is decomposed into symbolic concepts such as <i>red</i> and <i>sphere</i>, which are then composed to reason explicitly over objects and their relations.</figcaption>
</figure>

<div class="npt-steps">
<div class="npt-step"><b><span class="npt-n">1</span>Program generation</b>An LLM (DeepSeek-V3 in the paper) acts as a few-shot parser. In one call, it writes a Python program for the question and lists the object names that the detector should look for.</div>
<div class="npt-step"><b><span class="npt-n">2</span>Perceptual grounding</b>Grounding DINO proposes boxes for those objects. A VLM then grounds each atomic concept through visual prompting: <code>score</code> draws the box in red and returns p(Yes) from the logits of &ldquo;Yes&rdquo; and &ldquo;No&rdquo;, and <code>query</code> returns a short text answer.</div>
<div class="npt-step"><b><span class="npt-n">3</span>Symbolic executor</b>A standard Python interpreter runs the program. Overloaded operators such as <code>&amp;</code> and <code>|</code> compose the score tensors with soft logic, and loops, conditionals and variables give the program the full power of Python.</div>
</div>

<h3>Soft logic operators</h3>
<div class="npt-tablewrap">
<table class="npt-t">
<thead><tr><th>Syntax</th><th>Logical form</th><th>Description</th><th>Differentiable implementation</th></tr></thead>
<tbody>
<tr><td><code>&alpha;<sub>x</sub>.exists()</code></td><td>&exist;x &alpha;<sub>x</sub></td><td>Existential quantification</td><td>max(&alpha;<sub>x</sub>)</td></tr>
<tr><td><code>&alpha;<sub>x</sub>.forall()</code></td><td>&forall;x &alpha;<sub>x</sub></td><td>Universal quantification</td><td>min(&alpha;<sub>x</sub>)</td></tr>
<tr><td><code>&alpha;<sub>x</sub> &amp; &alpha;<sub>y</sub></code></td><td>&alpha;<sub>x</sub> &and; &alpha;<sub>y</sub></td><td>Logical conjunction</td><td>min(&alpha;<sub>x</sub>, &alpha;<sub>y</sub>)</td></tr>
<tr><td><code>&alpha;<sub>x</sub> &amp; &beta;<sub>xy</sub></code></td><td>&alpha;<sub>x</sub> &and; &beta;<sub>xy</sub></td><td>Relational conjunction</td><td>&sum;<sub>y</sub> &alpha;<sub>x</sub> &middot; &beta;<sub>xy</sub></td></tr>
<tr><td><code>&alpha;<sub>x</sub> | &alpha;<sub>y</sub></code></td><td>&alpha;<sub>x</sub> &or; &alpha;<sub>y</sub></td><td>Logical disjunction</td><td>max(&alpha;<sub>x</sub>, &alpha;<sub>y</sub>)</td></tr>
<tr><td><code>&alpha;<sub>x</sub>.implies(&alpha;<sub>y</sub>)</code></td><td>&alpha;<sub>x</sub> &rarr; &alpha;<sub>y</sub></td><td>Logical implication</td><td>max(1 &minus; &alpha;<sub>x</sub>, &alpha;<sub>y</sub>)</td></tr>
<tr><td><code>not &alpha;<sub>x</sub></code></td><td>&not;&alpha;<sub>x</sub></td><td>Logical negation</td><td>1 &minus; &alpha;<sub>x</sub></td></tr>
<tr><td><code>&alpha;<sub>x</sub>.iota(var)</code></td><td>&iota;(var, &alpha;<sub>x</sub>)</td><td>Best match</td><td>softmax(&alpha;<sub>x</sub>)</td></tr>
<tr><td><code>&alpha;<sub>x</sub>.count()</code></td><td>count(&alpha;<sub>x</sub>)</td><td>Counting elements</td><td>&sum; &alpha;<sub>x</sub></td></tr>
<tr><td><code>s<sub>1</sub> == s<sub>2</sub></code></td><td>s<sub>1</sub> = s<sub>2</sub></td><td>Scalar equality</td><td>&sigma;(&tau;(&gamma; &minus; |s<sub>1</sub> &minus; s<sub>2</sub>|) / &gamma;)</td></tr>
<tr><td><code>s<sub>1</sub> &gt; s<sub>2</sub></code></td><td>s<sub>1</sub> &gt; s<sub>2</sub></td><td>Scalar inequality</td><td>&sigma;(&tau;(s<sub>1</sub> &minus; s<sub>2</sub> &minus; 1 + &gamma;))</td></tr>
</tbody>
</table>
</div>
<p class="npt-tcap">Table 2 of the paper. &alpha; is an object-centric or scalar probabilistic score, &beta; is a relation probability score, &tau; = 0.25 is a temperature and &gamma; = 0.25 is a margin.</p>

<h2 id="results">Key results</h2>

<h3>CLEVR</h3>
<div class="npt-tablewrap">
<table class="npt-t">
<thead><tr><th></th><th>InternVL2.5</th><th>NePTune</th><th>ViperGPT</th><th>NeSyCoCo</th><th>LEFT</th></tr></thead>
<tbody>
<tr class="npt-grp"><td colspan="6">Zero-shot: the backbone VLM (end-to-end), NePTune and ViperGPT (neuro-symbolic). Trained: NeSyCoCo and LEFT</td></tr>
<tr><td>Final accuracy</td><td>90.25</td><td><b>92.65</b></td><td>36.05</td><td>99.68</td><td>99.50</td></tr>
<tr><td>Exist</td><td>87.10</td><td><b>93.19</b></td><td>48.75</td><td>99.28</td><td>98.92</td></tr>
<tr><td>Query attribute</td><td><b>98.26</b></td><td>96.81</td><td>29.42</td><td>100.00</td><td>99.86</td></tr>
<tr><td>Compare attribute</td><td><b>98.61</b></td><td>91.94</td><td>53.06</td><td>99.44</td><td>99.72</td></tr>
<tr><td>Count</td><td>74.60</td><td><b>87.10</b></td><td>21.37</td><td>99.79</td><td>98.99</td></tr>
<tr><td>Compare number</td><td>90.86</td><td><b>92.57</b></td><td>48.57</td><td>100.00</td><td>100.00</td></tr>
</tbody>
</table>
</div>
<p class="npt-tcap">Table 3 of the paper, showing accuracy (%) by question category. Bold marks the best zero-shot result. NePTune is the strongest zero-shot neuro-symbolic method, and its largest gain over the backbone is on counting (+12.50).</p>

<div class="npt-twocol">
<div>
<h3>Declarative and imperative together</h3>
<div class="npt-tablewrap">
<table class="npt-t">
<thead><tr><th>Ablation setting</th><th>CLEVR-Humans</th></tr></thead>
<tbody>
<tr><td>Declarative + trained concepts</td><td>56.12</td></tr>
<tr><td>+ VLM concepts</td><td>68.48 <small>(+12.36)</small></td></tr>
<tr class="npt-ours"><td>+ Imperative reasoning</td><td>87.67 <small>(+19.19)</small></td></tr>
</tbody>
</table>
</div>
<p class="npt-tcap">Table 10 of the paper, showing accuracy (%). Starting from a purely declarative reasoner, scoring concepts with a VLM adds 12.36 points, and adding imperative reasoning adds another 19.19.</p>
</div>
<div>
<h3>CLEVR-Humans</h3>
<div class="npt-tablewrap">
<table class="npt-t">
<thead><tr><th>Method</th><th>Accuracy</th></tr></thead>
<tbody>
<tr class="npt-grp"><td colspan="2">Trained</td></tr>
<tr><td>LEFT</td><td>56.69</td></tr>
<tr><td>NeSyCoCo</td><td>56.12</td></tr>
<tr><td>MDETR</td><td>81.73</td></tr>
<tr class="npt-grp"><td colspan="2">Zero-shot</td></tr>
<tr><td>Qwen2VL-7B</td><td>84.12</td></tr>
<tr><td>InternVL2.5-8B</td><td>85.95</td></tr>
<tr><td>Ovis1.6-9B</td><td>79.96</td></tr>
<tr><td>ViperGPT</td><td>31.05</td></tr>
<tr class="npt-ours"><td>NePTune</td><td>87.67</td></tr>
</tbody>
</table>
</div>
<p class="npt-tcap">Table 5 of the paper, showing accuracy (%) on human-written questions.</p>
</div>
</div>

<h3>CLEVR extensions</h3>
<div class="npt-tablewrap">
<table class="npt-t">
<thead><tr><th>Method</th><th>Ref</th><th>Puzzles</th><th>RPM</th></tr></thead>
<tbody>
<tr class="npt-grp"><td colspan="4">Trained</td></tr>
<tr><td>NeSyCoCo&dagger;</td><td>100.00</td><td>95.00</td><td>100.00</td></tr>
<tr><td>NeSyCoCo</td><td>94.00</td><td>94.00</td><td>74.00</td></tr>
<tr><td>LEFT</td><td>94.00</td><td>85.00</td><td>87.00</td></tr>
<tr class="npt-grp"><td colspan="4">Zero-shot</td></tr>
<tr><td>Qwen2VL-7B</td><td>21.00</td><td>43.00</td><td>53.00</td></tr>
<tr><td>InternVL2.5-8B</td><td>27.00</td><td>52.00</td><td>47.00</td></tr>
<tr><td>Ovis1.6-9B</td><td>4.00</td><td>47.00</td><td>49.00</td></tr>
<tr><td>ViperGPT</td><td>8.00</td><td>34.00</td><td>4.00</td></tr>
<tr><td>VisProg</td><td>35.00</td><td>27.00</td><td>51.00</td></tr>
<tr><td>NePTune&dagger;</td><td>99.00</td><td>65.00</td><td>99.00</td></tr>
<tr class="npt-ours"><td>NePTune</td><td>91.00</td><td>60.00</td><td>80.00</td></tr>
</tbody>
</table>
</div>
<p class="npt-tcap">Table 4 of the paper, showing accuracy (%) on referring expressions (Ref), visual puzzles and Raven&rsquo;s progressive matrices (RPM) built on CLEVR. &dagger; marks methods that use ground-truth programs.</p>

<div class="npt-twocol">
<div>
<h3>Domain shift and fine-tuning: Ref-GTA</h3>
<div class="npt-tablewrap">
<table class="npt-t">
<thead><tr><th>Method</th><th>Ref-GTA</th></tr></thead>
<tbody>
<tr><td>GroundingDINO-B</td><td>27.90</td></tr>
<tr><td>Florence2-L</td><td>58.65</td></tr>
<tr><td>Ovis1.6-9B</td><td>2.78</td></tr>
<tr><td>InternVL2.5-8B</td><td>6.95</td></tr>
<tr><td>InternVL2.5-1B</td><td>1.64</td></tr>
<tr><td>&nbsp;&nbsp;+ Fine-tuning</td><td>32.61 <small>&plusmn;0.35</small></td></tr>
<tr><td>ViperGPT</td><td>1.40</td></tr>
<tr><td>NAVER&dagger;</td><td>54.84</td></tr>
<tr><td>NAVER</td><td>58.73</td></tr>
<tr><td>NePTune&Dagger;</td><td>62.73</td></tr>
<tr class="npt-ours"><td>NePTune</td><td>69.69</td></tr>
<tr><td>NePTune (1B)</td><td>34.92</td></tr>
<tr class="npt-ours"><td>&nbsp;&nbsp;+ Fine-tuning</td><td>69.90 <small>&plusmn;1.16</small></td></tr>
</tbody>
</table>
</div>
<p class="npt-tcap">Table 7 of the paper, showing grounding accuracy (%) on images from a game engine, a domain shift that the VLMs were not trained on. Fine-tuning a 1B VLM through NePTune&rsquo;s differentiable computations with only 1,000 samples reaches 69.90, while standard fine-tuning of the same VLM reaches 32.61.</p>
</div>
<div>
<h3>Real images: Ref-Adv</h3>
<div class="npt-tablewrap">
<table class="npt-t">
<thead><tr><th>Method</th><th>Ref-Adv</th></tr></thead>
<tbody>
<tr><td>Grounding DINO-B</td><td>60.85</td></tr>
<tr><td>Florence2-L</td><td>71.73</td></tr>
<tr><td>Ovis1.6-9B</td><td>30.70</td></tr>
<tr><td>InternVL2-8B</td><td>72.92</td></tr>
<tr><td>InternVL2.5-8B</td><td>76.13</td></tr>
<tr><td>ViperGPT</td><td>60.66</td></tr>
<tr><td>NAVER&dagger;</td><td>36.45</td></tr>
<tr><td>NAVER</td><td>65.13</td></tr>
<tr><td>NePTune&Dagger;</td><td>63.71</td></tr>
<tr><td>&nbsp;&nbsp;+ Verification</td><td>75.54</td></tr>
<tr><td>NePTune</td><td>71.57</td></tr>
<tr class="npt-ours"><td>&nbsp;&nbsp;+ Verification</td><td>78.08</td></tr>
<tr><td>NePTune (1B)</td><td>60.69</td></tr>
<tr><td>&nbsp;&nbsp;+ Fine-tuning</td><td>68.06 <small>&plusmn;0.56</small></td></tr>
<tr><td>&nbsp;&nbsp;+ Verification</td><td>74.59 <small>&plusmn;0.12</small></td></tr>
</tbody>
</table>
</div>
<p class="npt-tcap">Table 6 of the paper, showing grounding accuracy (%) on RefCOCO-Adversarial. NePTune&Dagger; uses the same backbones as NAVER, and NAVER&dagger; is NAVER&rsquo;s execution step on its own.</p>
</div>
</div>

<p>NePTune was accepted to ICLR 2026 and was presented as an oral at the <a href="https://space-in-vision-language-embodied-ai.github.io/">SpaVLE workshop</a> at NeurIPS 2025.</p>

<h2 id="bibtex">BibTeX</h2>

<pre class="npt-bib">@inproceedings{kamali2026neptune,
title={Ne{PT}une: A Neuro-Pythonic Framework for Tunable Compositional Reasoning on Vision-Language},
author={Danial Kamali and Parisa Kordjamshidi},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=8H0TkSusWI}
}</pre>

</div>

<script src="{{ site.baseurl }}/files/neptune/neptune-demo.js"></script>
