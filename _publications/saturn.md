---
title: "SATURN: Symbolic Spatial Reasoning for Multi-Perspective Grounding"
collection: publications
permalink: /publication/saturn
redirect_from:
  - /saturn/
excerpt: 'SATURN answers frame-of-reference questions by running a short program of soft spatial predicates over an estimated 3D scene, and we also release 3D FORCE, a diagnostic benchmark for composing spatial relations across perspectives.'
date: 2026-10-02
venue: 'Conference on Empirical Methods in Natural Language Processing (EMNLP)'
authors: 'Danial Kamali, Tanawan Premsri, Shreya Rajpal, Amir Zadeh, Chuan Li, Parisa Kordjamshidi'
authors_page: 'Danial Kamali<sup>1</sup>, Tanawan Premsri<sup>1</sup>, Shreya Rajpal<sup>1</sup>, Amir Zadeh<sup>2</sup>, Chuan Li<sup>2</sup>, Parisa Kordjamshidi<sup>1</sup>'
affiliations: '<sup>1</sup>Michigan State University &nbsp;&nbsp; <sup>2</sup>Lambda Labs'
image: "files/saturn/img.jpg"
paperurl: 'https://arxiv.org/pdf/2606.22694'
arxivurl: 'https://arxiv.org/abs/2606.22694'
codeurl: "https://github.com/HLR/SATURN"
dataseturl: "https://huggingface.co/datasets/iamdanialkamali/3D-FORCE-Zip"
projecturl: "publication/saturn"
bibtex: '@misc{kamali2026saturnsymbolicspatialreasoning,
      title={SATURN: Symbolic Spatial Reasoning for Multi-Perspective Grounding},
      author={Danial Kamali and Tanawan Premsri and Shreya Rajpal and Amir Zadeh and Chuan Li and Parisa Kordjamshidi},
      year={2026},
      eprint={2606.22694},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.22694},
}'
---

<link rel="stylesheet" href="{{ site.baseurl }}/files/saturn/saturn.css">

<div class="sat">

<div class="sat-tldr"><b>TL;DR</b> Vision-language models often fail when a spatial question requires someone else's point of view. SATURN estimates a 3D scene from the images and answers with a short program of <em>soft</em> spatial predicates. Each predicate uses the appropriate frame of reference to judge left, right, front, or behind from a given position and facing direction. We also release <b>3D FORCE</b>, a benchmark that controls reasoning depth, number of views, and how frames of reference are mixed.</div>

<h2 id="example">One example, step by step</h2>

<p class="sat-lead">This walkthrough follows one question from 3D FORCE that SATURN answers correctly, while GPT-5.1, Gemini-3.1-Pro, Qwen3.5-9B, and Qwen3-VL-8B all answer incorrectly. Answering the question requires two frames of reference: the bus's own view and camera&nbsp;0's view. Press <b>&#9654;</b> to play or pause, use the numbered steps to jump, and drag the 3D view to rotate the scene while the animation is paused.</p>

<div id="sat-demo" class="sat-demo" data-src="{{ site.baseurl }}/files/saturn/example/">
<p class="sd-loading">Loading the example&hellip;</p>
<noscript><p>The walkthrough needs JavaScript. Camera 0 of the example:</p><img src="{{ site.baseurl }}/files/saturn/img.jpg" alt="Camera 0 view of the example scene with SATURN's answer boxed in green"></noscript>
</div>

<p class="sat-note">The detections, the reconstructed scene and cameras, and the program come from one SATURN run of the released code (seed 0; Qwen3-VL-8B as the VLM, SAM3, VGGT and Orient Anything V2 for perception), and every score comes from the execution trace of that run. The ground shading shows only which side each frame calls &ldquo;behind&rdquo;, while the numbers are the predicate values.</p>
<p class="sat-note">The baseline answers are the models' own outputs from the paper's evaluation, scored by the paper's rule (IoU &gt; 0.5). This question comes from the &ldquo;Obj + One-Cam&rdquo; REF setting, which combines one object frame with one camera frame. In that setting, SATURN reaches 73% and Gemini-3.1-Pro reaches 41% (Figure 4 of the paper).</p>

<h2 id="perspectives">The same scene from each perspective</h2>

<p>To make the two frames of reference concrete, we rendered the example scene again with the 3D FORCE generator. The camera visits the three input views, rises to a bird's-eye view with the bus facing the top of the picture, and then returns to camera&nbsp;0. The clip uses the benchmark's ground-truth scene to show what the question means, not what SATURN estimated. SATURN only sees the three input views.</p>

<div id="sat-video" class="sat-video" data-segments='[[0.0, 0.5833, "Camera 0, one of the three input views"], [0.5833, 2.5833, "Moving to camera 1"], [2.5833, 3.0, "Camera 1"], [3.0, 5.0, "Moving to camera 2"], [5.0, 5.4167, "Camera 2"], [5.4167, 7.4167, "Moving back to camera 0"], [7.4167, 7.8333, "Camera 0"], [7.8333, 10.3333, "Rising and turning into the bus&#39;s frame"], [10.3333, 11.8333, "Bus facing up: the lower half is behind the bus"], [11.8333, 14.3333, "Returning to camera 0"], [14.3333, 15.2, "From camera 0, behind the SUV means farther away than the SUV"]]'>
<video muted playsinline loop controls preload="metadata" poster="{{ site.baseurl }}/files/saturn/perspective-change.jpg" aria-label="Rendered fly-through of the example scene from camera 0 to camera 1, camera 2, the bus's own frame, and back to camera 0"><source src="{{ site.baseurl }}/files/saturn/perspective-change.webm" type="video/webm"><source src="{{ site.baseurl }}/files/saturn/perspective-change.mp4" type="video/mp4"></video>
<div class="sat-video-cap"><span></span></div>
</div>
<p class="sat-note">In the bus's frame, the green tank on the left in camera&nbsp;0 (the answer) lies in the lower half of the picture, behind the bus. The green tank on the right in camera&nbsp;0 lies in the upper half, so that tank is not behind the bus.</p>

<h2 id="abstract">Abstract</h2>

<p>Vision-Language Models (VLMs) remain unreliable when spatial reasoning requires composing relations whose meanings depend on frames of reference. Existing tool-augmented spatial reasoning methods make reasoning more explicit, but often rely on low-level geometric procedures and hard binary decisions over noisy perception. We propose SATURN, a neuro-symbolic framework for perspective-aware compositional spatial reasoning. SATURN reconstructs an approximate 3D scene, derives soft perspective-aware spatial predicates, and composes them with a training-free Pythonic symbolic executor, separating perception from reasoning while preserving uncertainty through multi-hop inference. We also introduce 3D FORCE, a diagnostic benchmark that controls reasoning depth, view, and perspective composition for spatial arrangement grounding (SAG) and referring expression grounding (REF). On 3D FORCE, VLMs and spatially trained models degrade sharply as depth and perspective complexity increase, whereas SATURN degrades the least and outperforms every baseline at each depth. On the real-world MindCube benchmark, SATURN achieves 78.06% overall accuracy, outperforming the strongest baseline by 14 percentage points.</p>

<h2 id="method">How SATURN works</h2>

<p>SATURN uses a VLM to identify which objects the question needs and a code LLM to write a short program over an estimated 3D scene. The program uses soft predicates in frames defined by a position and facing direction at an object or camera. A soft-logic engine runs the program and selects the object or option with the highest score for the claim.</p>

<figure>
<img src="{{ site.baseurl }}/files/saturn/pipeline.webp" alt="SATURN overview: query-guided scene estimation, spatial engine, and program generation and execution" loading="lazy" width="1800" height="654">
<figcaption>SATURN overview (Figure 2 of the paper).</figcaption>
</figure>

<div class="sat-steps">
<div class="sat-step"><b><span class="sat-n">1</span>Query-guided scene estimation</b>A VLM parser reads the question and lists the required objects. SAM3 grounds those objects in every view, VGGT reconstructs the objects' 3D positions and the cameras, and Orient Anything V2 estimates which way each object faces. SATURN also uses any facts the question states about the cameras to refine the camera poses, such as two views taken from the same spot or a fixed rotation between views.</div>
<div class="sat-step"><b><span class="sat-n">2</span>Spatial engine</b>Every frame of reference is a local coordinate system attached to a camera, an object, or a virtual viewer. The spatial engine computes relations such as left, front, or behind in that frame and assigns soft scores in [0, 1] rather than hard yes/no decisions.</div>
<div class="sat-step"><b><span class="sat-n">3</span>Program and soft execution</b>A code LLM writes a short Python program that uses the VLM to score semantic concepts and calls the frame-aware predicates to score spatial relations. The executor combines these scores with soft logic (AND is the minimum), allowing uncertainty from perception to carry through every hop rather than being thresholded away.</div>
</div>

<h2 id="benchmark">The 3D FORCE benchmark</h2>

<p>3D FORCE isolates the reasoning needed to compose spatial relations across perspectives. High-resolution rendered scenes keep perception and object grounding simple, while the benchmark controls reasoning depth, relation topology, view count, and the frame of reference for each relation. Every question comes with a formal logical form, and answers come from the scene graph used to render the images.</p>

<p>The benchmark has two subsets. <b>SAG</b> (spatial arrangement grounding) asks whether a described arrangement of objects exists in the scene. <b>REF</b> (referring expression grounding) asks for a box around the object identified by a multi-hop description.</p>

<div class="sat-facts">
<div class="sat-fact"><b>2,088</b>REF questions</div>
<div class="sat-fact"><b>1,150</b>SAG questions</div>
<div class="sat-fact"><b>1&ndash;4</b>views, plus partial views</div>
<div class="sat-fact"><b>0&ndash;6</b>relation hops; chain, star, hybrid</div>
</div>

<figure>
<img src="{{ site.baseurl }}/files/saturn/benchmark.webp" alt="Overview of the 3D FORCE benchmark with its SAG and REF subsets" loading="lazy" width="1800" height="650">
<figcaption>Overview of 3D FORCE (Figure 3 of the paper). Download the benchmark from <a href="https://huggingface.co/datasets/iamdanialkamali/3D-FORCE-Zip">Hugging Face</a>.</figcaption>
</figure>

<h2 id="results">Key results</h2>

<h3>3D FORCE</h3>
<div class="sat-tablewrap">
<table class="sat-t">
<thead><tr><th>Method</th><th>Micro Avg.</th><th>SAG</th><th>REF</th></tr></thead>
<tbody>
<tr><td>Random choice</td><td>18.80</td><td>50.17</td><td>1.53</td></tr>
<tr class="sat-grp"><td colspan="4">General-purpose VLMs</td></tr>
<tr><td>Qwen3-VL-8B</td><td>24.31</td><td>65.91</td><td>1.39</td></tr>
<tr><td>Qwen3-VL-235B</td><td>26.44</td><td>73.22</td><td>0.67</td></tr>
<tr><td>Qwen3.5-9B</td><td>57.75</td><td>74.87</td><td>48.32</td></tr>
<tr><td>Qwen3.5-35B</td><td>57.63</td><td>76.35</td><td>47.32</td></tr>
<tr><td>InternVL3.5-38B</td><td>30.82</td><td>58.26</td><td>15.71</td></tr>
<tr><td>GPT-5.1</td><td>29.80</td><td>71.39</td><td>6.90</td></tr>
<tr><td>Gemini-3.1-Pro</td><td>57.63</td><td>75.30</td><td>47.89</td></tr>
<tr class="sat-grp"><td colspan="4">Spatially trained VLMs</td></tr>
<tr><td>SpaceOm-3B</td><td>17.79</td><td>49.39</td><td>0.38</td></tr>
<tr><td>Cosmos-Reason1-7B</td><td>13.47</td><td>37.04</td><td>0.48</td></tr>
<tr class="sat-grp"><td colspan="4">Tool-augmented</td></tr>
<tr><td>GCA</td><td>24.24</td><td>51.00</td><td>9.50</td></tr>
<tr><td>pySpatial</td><td>44.84</td><td>48.35</td><td>42.91</td></tr>
<tr><td>TIGeR</td><td>20.38</td><td>30.43</td><td>14.85</td></tr>
<tr class="sat-ours"><td>SATURN (ours)</td><td>82.89 <small>&plusmn;0.40</small></td><td>85.88 <small>&plusmn;0.13</small></td><td>81.24 <small>&plusmn;0.68</small></td></tr>
<tr><td>SATURN with ground-truth boxes</td><td>88.85 <small>&plusmn;0.55</small></td><td>87.57 <small>&plusmn;0.97</small></td><td>89.56 <small>&plusmn;0.67</small></td></tr>
<tr><td>SATURN with oracle 3D</td><td>95.94 <small>&plusmn;0.35</small></td><td>94.40 <small>&plusmn;0.68</small></td><td>96.79 <small>&plusmn;0.39</small></td></tr>
</tbody>
</table>
</div>
<p class="sat-tcap">Selected rows from Table 1 in the paper, showing accuracy (%). REF counts a box as correct at IoU &gt; 0.5, and SATURN rows report the mean and standard deviation over three runs. The full table lists all 19 VLMs.</p>

<div class="sat-twocol">
<figure>
<img src="{{ site.baseurl }}/files/saturn/perspective.webp" alt="Accuracy on SAG and REF by perspective setting for InternVL3.5-38B, Gemini-3.1-Pro, Qwen3.5-35B, and SATURN" loading="lazy" width="921" height="851">
<figcaption>Accuracy by perspective setting (Figure 4). Every VLM performs best with a single camera frame, with accuracy dropping when an object frame or a second camera frame is involved. Across settings, SATURN varies by about 7 points on SAG and 13 on REF, while each VLM varies by at least 22 points.</figcaption>
</figure>
<figure>
<img src="{{ site.baseurl }}/files/saturn/hops.webp" alt="REF accuracy by number of reasoning hops" loading="lazy" width="921" height="418">
<figcaption>REF accuracy by number of reasoning hops (Figure 5). From zero to six hops, accuracy falls from 94% to 67% for SATURN, from 80% to 33% for Gemini-3.1-Pro, and from 74% to 29% for Qwen3.5-35B. SATURN outperforms every baseline at each hop count.</figcaption>
</figure>
</div>

<h3>Why soft predicates</h3>
<p>The paper compares three variants using the same scenes, built from the benchmark's ground-truth boxes. Writing low-level geometry code directly reaches 75.43% on REF, while using the declarative predicate interface with crisp 0/1 scores reaches 79.07%. Using the same predicates with continuous scores raises accuracy to 89.6%, so preserving uncertainty accounts for the larger gain (+10.5 points).</p>

<h3>Real-world benchmarks: MindCube and MMSI</h3>
<div class="sat-tablewrap">
<table class="sat-t">
<thead><tr><th>Method</th><th>MindCube</th><th>Rotation</th><th>Among</th><th>Around</th><th>MMSI</th></tr></thead>
<tbody>
<tr><td>Qwen3-VL-8B-Instruct</td><td>38.86</td><td>43.50</td><td>32.83</td><td>49.60</td><td>29.40</td></tr>
<tr><td>Qwen3-VL-235B-Thinking</td><td>47.30</td><td>87.00</td><td>35.00</td><td>47.30</td><td>32.60</td></tr>
<tr><td>Gemini-2.5-Pro</td><td>57.50</td><td>89.50</td><td>48.80</td><td>54.50</td><td>36.90</td></tr>
<tr><td>GCA (235B-Think)</td><td>64.20</td><td>82.00</td><td>59.80</td><td>61.80</td><td>41.90</td></tr>
<tr><td>pySpatial</td><td>53.14</td><td>38.50</td><td>50.30</td><td>71.60</td><td>28.20</td></tr>
<tr><td>Qwen3VL + scene state</td><td>60.86</td><td>49.50</td><td>60.50</td><td>70.80</td><td>39.40</td></tr>
<tr class="sat-ours"><td>SATURN (ours)</td><td>78.06 <small>&plusmn;1.05</small></td><td>85.67 <small>&plusmn;2.93</small></td><td>77.00 <small>&plusmn;1.02</small></td><td>74.53 <small>&plusmn;2.31</small></td><td>48.77 <small>&plusmn;0.61</small></td></tr>
</tbody>
</table>
</div>
<p class="sat-tcap">Selected rows from Table 2 in the paper, showing accuracy (%). The MMSI result is the overall score; the paper also reports the four MMSI categories and paired confidence intervals.</p>

<h2 id="bibtex">BibTeX</h2>

<pre class="sat-bib">@misc{kamali2026saturnsymbolicspatialreasoning,
      title={SATURN: Symbolic Spatial Reasoning for Multi-Perspective Grounding},
      author={Danial Kamali and Tanawan Premsri and Shreya Rajpal and Amir Zadeh and Chuan Li and Parisa Kordjamshidi},
      year={2026},
      eprint={2606.22694},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.22694},
}</pre>

</div>

<script type="importmap">{"imports": {"three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js", "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module" src="{{ site.baseurl }}/files/saturn/saturn-demo.js"></script>
<script src="{{ site.baseurl }}/files/saturn/saturn-video.js"></script>
