---
title: 'Aligning AI with Humanity: The Role of Reinforcement Learning in Language Model Alignment'
date: 2024-03-01
permalink: /posts/2023/03/alignment/
tags:
  - cool posts
  - category1
  - category2
---

In this work, we look into the prominent applications of Reinforcement Learning (RL) in the field of Natural Language Processing (NLP) with a focus on Language Models (LM). First, we examine one of the initial applications of Reinforcement Learning with Human Feedback (RLHF) in NLP. Then, we discuss how this method evolves to be applied in a more general AI and becomes a fundamental aspect of Large Language Model (LLM) training. Also, we discuss the risks, challenges, and potential problems associated with RLHF, offering insights into how these issues might be addressed and mitigated. Furthermore, we explore the emerging field of Reinforcement Learning with AI Feedback (RLAIF), assessing its position in current research. Our investigation shows that RLHF training is a very effective tool for language model alignment. This method cannot only improve the performance of the overall model in NLP benchmarks but also help with problems such as hallucination. In addition, we showed that methods like Constitutional AI can improve the LLMs' safety by increasing harmlessness while keeping high levels of helpfulness.

## Introduction

Current Artificial Intelligence tools have shown impressive performance in Computer Vision and NLP, especially in recent years, surpassing human capabilities in various tasks. Language models are the core of most modern NLP tools. They have evolved from initial n-gram statistical models to neural models using transformer architecture. These language models are typically trained on extensive data corpora, using either masked token or next token prediction while using cross-entropy loss. In recent years, we have seen the emergence of Large Language Models with billions of parameters trained on terabytes of data.

These models can be utilized for various problems using zero-shot and meta-learning at inference time or by fine-tuning the LLMs on task-specific datasets. While fine-tuning appears advantageous, it poses risks and limitations, especially for modern large language models (≥50B parameters), such as data size, model size, and costs associated with data and spurious correlations in new training data. For instance, one problem that may arise in the pre-training + fine-tuning is that the model gets overly tuned on the small and narrowly distributed fine-tuning dataset. Recent studies have shown that zero or few-shot learning on a pre-trained LLM can achieve state-of-the-art performance comparable to fine-tuned models. However, for tasks like summarization, where cross-entropy loss may not be ideal, methods such as BLEU or ROUGE are often used to capture summary quality better. However, these methods only partially encapsulate writing quality as judged by human standards. Since we need something that best assimilates human standards, we can use data from actual human evaluations.

We need to align the training objective with human preferences to train models that generate human-preferred responses. At this point, we need to apply preference-based learning in our training procedure. This is where Reinforcement Learning techniques become crucial. Using RL, models can be trained to maximize rewards derived from human feedback, leading to Reinforcement Learning with Human Feedback. Famous tools such as InstructGPT and Claude, which are trained extensively on human preference data, showcase this approach's effectiveness and real-world applicability, producing outputs of higher quality than those trained solely on supervised learning.

Human feedback can be accurate and effective. However, it is not easily scalable and can be time-consuming, with NLP models increasingly matching or surpassing human performance in many tasks. We can use these models to label preference data, which leads to the emergence of RLAIF methods. Our work will also discuss these methods' positive and negative implications, highlighting their impact on various NLP tasks.

In this work, we focus on Reinforcement Learning and its applications in Natural Language Processing, with a particular emphasis on preference-based learning methods. The second section provides an overview of the background and foundational concepts in RL, Deep Reinforcement Learning (DeepRL), and Language Modeling. In the third section, we explore Reinforcement Learning from the Human Feedback method, discussing its necessity and how models are trained using this approach. The fourth section presents a summary of the reviewed papers. In the fifth section, we analyze the differences between each approach and their impact on the outcomes. Finally, in the sixth section, we examine the prospects of RL in language models and explore potential directions for this field.

### Background

#### Reinforcement Learning

Reinforcement Learning is a machine learning method next to unsupervised and supervised learning. Unlike supervised learning methods that rely on labeled outputs, Reinforcement Learning uses the environment's reward to learn. In particular, RL can be used in scenarios where an agent interacts with an environment and receives a reward (feedback) based on its action.

The RL problem can be formulated as an agent interacting with an environment modeled as an MDP. As shown in Figure 1, in an RL scenario, at each state \(S_t\), action \(A_t\) can result in going to a new state \(S_{t+1}\) while receiving a reward \(R_{t+1}\) based on the consequences of the action \(A_t\). These rewards are positive for favorable actions and negative for detrimental ones, which will help the model decide which action to choose at each state.

Understanding the theoretical framework and objectives of Reinforcement Learning provides a foundation for appreciating its practical applications. This method, aimed at maximizing cumulative rewards, has not only been a theoretical concept but has also found significant real-world applications. One of the most prominent areas where RL has been extensively tested and proven is in video games, as highlighted by (citation needed). This benchmark testing in gaming has paved the way for RL's adoption in more diverse and complex fields. Nowadays, principles of RL are being applied across various sectors, such as autonomous vehicles, NLP, and robotics.

(Figure 1: Schema of an RL model)

There are three main approaches in RL:

- Value-Based: Agent learns a value function \(V(S_t)\) or Q-function \(Q(S_t, A)\) that estimates the potential reward of being in state \(S_t\) or taking an action in the current state respectively. Usually, we keep the \(Q\) or \(V\) values in a table and use a greedy method to choose the highest potential reward action.

- Policy-Based: Agent directly learns a policy function \(\pi(A|S_t)\) that gives a probability distribution over actions based on the state \(S_t\) and can dictate the best action in the current state.

- Hybrid: Combines elements of both Value-Based and Policy-Based methods.

##### Deep Reinforcement Learning

Reinforcement Learning is commonly applied in simpler domains such as classic control or basic games. However, it faces limitations as the complexity of the environment increases. For instance, complex games like chess, which have numerous possible states, pose significant challenges to traditional RL methods. In scenarios like these, there is a need for functions that can accurately approximate the \(V\) (value) and \(Q\) (quality) values (In Q-learning). Neural Networks, recognized as universal function approximators, are well-suited to meet this challenge. They can effectively approximate complex value functions or policy functions required in RL. Deep Reinforcement Learning integrates these Deep Neural Networks with traditional RL principles. DeepRL aims to maximize the reward obtained from the environment. This is achieved using gradient ascent, an optimization technique that contrasts with gradient descent.

#### Language Modeling

Language models serve as the backbone of various AI applications, ranging from information retrieval to resume evaluation and educational support. These models are typically trained using techniques such as masked token prediction or next token prediction, which have proven powerful and effective in both understanding and generating natural language. However, despite their effectiveness, these models often fail to accurately represent more nuanced aspects of language, such as sarcasm, humor, or ethical considerations. This shortfall poses a significant challenge: How can a language model be trained to capture these complex language nuances, i.e., how can a loss function be devised that effectively evaluates such aspects?

Human preference or rating is the most suitable metric for addressing such intricacies, which better aligns with the nuanced understanding of language. Therefore, there is a crucial need to develop methods that can align the outputs of language models with human preference ratings, ensuring that the models not only perform well in standard tasks but also accurately reflect the subtleties and complexities of human language and communication.

## Reinforcement Learning from Human Feedback

The application of human feedback in intelligent models has evolved significantly, particularly with the advent of Deep Reinforcement Learning. This evolution reflects a growing emphasis on integrating human preferences into agent decision-making processes. RLHF, in this context, represents a paradigm shift in model training, focusing on human-centric model development by aligning AI outputs with human preferences. This approach is instrumental in creating AI systems that are not only efficient but also resonate with the complexities of human judgment and decision-making. Here, we overview the steps in the RLHF training procedure based on the formulation from (citation needed).

### Step 0 (Optional) - Pre-training

RLHF often begins with an initial base model \(\pi_{\theta}\), where \(\theta\) are the parameters, which generates a distribution of examples. For instance, when performing RLHF with large language models, the base model is typically a pre-trained language generator. This step can include a wide range of methods, from unsupervised language modeling on web text or another dataset to training on task-specific datasets. We aim to teach the model some general knowledge that is going to be utilized in other steps. (Figure 2: Diagram of the RLHF step 0)

### Step 1 - Data Collection

In this step, data is collected by obtaining samples from the base model and using human feedback to label those samples. For example, RLHF with LLMs might involve task samples (\(x_i\)) consisting of conversation pairs and feedback (\(y_i\)) in the form of preferences expressed within each pair. Equation 1 represents the formal representation of this feedback process.

\[
y_i = f(H, x_i, \epsilon_i)
\]

Here, \(H\) is a human evaluator, and \(\epsilon_i\) is random noise. An example of steps 1 and 2 is shown in Figure 3.

### Step 2 - Fitting the Reward Model

The second step involves fitting a reward model \(\hat{r}_{\phi}\) to approximate evaluations from human feedback as closely as possible. Reward fitting is achieved by minimizing a loss function, often a cross-entropy loss, defined over a dataset of examples and preferences \(D = \{(x_i, y_i)\}\) as shown in Equation 2.

\[
L(D, \phi) = \sum_{i=1}^{n} \ell(\hat{r}_{\phi}(x_i), y_i) + \lambda_r(\phi)
\]

where \(\ell\) is the loss function and \(\lambda_r\) is a regularize. In some cases, the reward model is trained on a comparative approach in which it tries the increase to maximize \(\ell(\hat{r}_{\phi}(x_i), y_1) - \ell(\hat{r}_{\phi}(x_i), y_2)\) if \(y_1\) is preferred. The loss function for this training is shown in Equation 3.

\[
L(D, \phi) = \sum_{i=1}^{n} \left( \ell(\hat{r}_{\phi}(x_i), y_1) - \ell(\hat{r}_{\phi}(x_i), y_2) \right) + \lambda_r(\phi)
\]

(Figure 3: Diagram of the RLHF step 1-2)

### Step 3 - Optimizing the Policy with RL

The final step is to use the trained reward model \(\hat{r}_{\phi}\) to fine-tune the base model using Reinforcement Learning. The new parameters \(\theta_{\text{new}}\) of the policy \(\pi\) are trained to maximize the reward function shown in Equation 4.

\[
R(\theta_{\text{new}}) = E_{x \sim \pi_{\theta_{\text{new}}}}[\hat{r}_{\phi}(x) + \lambda_p(\theta, \theta_{\text{new}}, x)]
\]

This step ensures that the model generates outputs that align with human values and preferences.

RLHF has shown to be a critical advancement in the field of AI, offering a nuanced approach to integrating human insights into model training. By focusing on human feedback and preferences, RLHF ensures the development of AI systems that are not only task-efficient but also better reflective of human values and decision-making processes. This human-centric approach represents a significant step in developing more responsive, ethical, and effective AI systems. (Figure 4: Diagram of the RLHF step 3)

## An Overview of Three RLHF Applications

Having gained a basic understanding of RLHF, this section will comprehensively explore each paper selected for our report. We begin by examining summarization from Human Feedback, an early application of RLHF. Subsequently, we delve into the widely-utilized instruction-based model, InstructGPT. Following this, we explore RLAIF, highlighting its role as a distinctive variant within the RLHF paradigm.

### Learning to Summarize from Human Feedback

(citation needed) tackle the challenge of text summarization models. Authors show that simply maximizing the likelihood of human-written text does not guarantee effective summarization. Also, they show the limitations of traditional metrics like ROUGE and underscore the importance of human input for quality assessment.

To address this, the authors incorporated human preferences into the training process. They collect data on human preferences for summaries and train a reward model based on these preferences. The reward model is trained using the loss function shown in Equation 5.

\[
\text{loss}(r) = -\mathbb{E}_{(x;y_0;y_1;i) \sim D} \left[ \log \left( r(x; y_i) - r(x; y_{1-i}) \right) \right]
\]

Here, \(r(x; y_i)\) and \(r(x; y_{1-i})\) represent the scores given by the reward model to two different summaries of \(x\), with the expectation \(\mathbb{E}\) taken over a dataset of human judgments.

Reinforcement Learning and Proximal Policy Optimization (PPO) are used to train models that generate summaries based on this reward model. The models start with a policy fine-tuning on the Reddit TL;DR dataset. The total reward function \(R\) used in training is shown in Equation 6.

\[
R(x; y) = r(x; y) - \gamma \log\left[\frac{\pi_{\text{RL}}(y|x)}{\pi_{\text{SFT}}(y|x)}\right]
\]

This includes a term that penalizes the KL divergence between the learned RL policy and the original supervised model, promoting exploration and preventing the model from straying too far from familiar outputs.

The introduction of human feedback has markedly improved summarization quality, as shown in Figure 5. In addition, models trained with human feedback show enhanced generalization abilities, which is evident in their performance on diverse datasets, including news articles, beyond the Reddit TL;DR, their training dataset. In addition, an extensive analysis of model size, data size, and optimization of reward models reveals that these reward models are more effective than metrics like ROUGE in predicting human preferences, leading to higher-quality summaries.

(Figure 5: Results of summarization from human feedback across four axes compared to different methods. Results are scores assigned by human labelers from 7.)

### Training language models to follow instructions with human feedback

(citation needed) aim to improve the alignment of GPT-3, a large language model, with human intentions. The primary challenge addressed is the misalignment between the model's objective of next token prediction and the actual intentions of users. The researchers seek to fine-tune GPT-3 to better align with human values and preferences, focusing on three key areas: explicit intentions (following instructions) and implicit intentions (being helpful, honest, and harmless).

The initial dataset used consists of different types of prompts, including plain, few-shot, and user-based, which are being used in three distinct datasets for supervised fine-tuning (SFT), reward model (RM) training, and PPO fine-tuning. The models are evaluated based on how well they align with the criteria of being helpful, honest, and harmless. The evaluation also includes performance on public NLP datasets and user-generated API prompts.

Significant findings include improvements in truthfulness and reductions in toxicity compared to the original GPT-3 model. The InstructGPT models demonstrate a better understanding of user instructions and show promising generalization capabilities. However, they still exhibit limitations such as generating biased or toxic outputs, making up facts, and sometimes failing to generate reasonable outputs for certain inputs.

The study raises important questions about alignment research, including the cost of increasing model alignment, generalization of instruction-following, and the challenges in designing a fair and accountable alignment process. It highlights the limitations of the current models and presents open questions for future research, such as methods to decrease toxicity, ensure harmlessness despite user instructions, and explore alternatives to RLHF.

### Constitutional AI Harmlessness from AI Feedback

(citation needed) delves into the development of AI models that are both harmless and efficient, with a specific focus on making human involvement more efficient. This is achieved through a novel approach called "Constitutional AI" (CAI), which revolves around governing AI behavior with a set of principles or "constitution." This approach stands out for its emphasis on training AI models to self-supervise by critiquing and revising their responses in line with these principles, thereby reducing the need for extensive human feedback.

A significant aspect of this research is the integration of Supervised Learning and Reinforcement Learning in the training process. In the Supervised Learning phase, AI models are tasked with generating responses, self-critiques, and subsequent revisions. These revised responses are then utilized to fine-tune the original model. In the RL stage, AI models evaluate responses based on constitutional principles, using this feedback to refine further and enhance their behavior.

The utilization of Chain-of-Thought (CoT) reasoning is a critical element in both the SL and RL stages. CoT reasoning enhances the performance and transparency of the AI models, making the AI's decision-making process explicit and more comprehensible. This approach is mainly instrumental in achieving the dual objectives of harmlessness (avoiding engagement in or promotion of harmful behavior) and non-evasiveness (willingness to engage thoughtfully with sensitive topics).

One of the groundbreaking contributions of this research is the significant reduction in human involvement in AI supervision. This is primarily accomplished by leveraging AI-generated critiques and preference labels, which not only enhance the AI's behavior as evaluated by humans but also foster greater self-sufficiency and efficiency in the AI models. As shown in Figure 6, a model trained with CAI shows much better harmlessness at the same level of helpfulness.

(Figure 6: Helpfulness and Harmlessness trade-off.)

Despite these advancements, the research acknowledges specific challenges, such as the AI's potential to adopt an overly strict or accusatory tone in applying rules. To address these challenges, (citation needed) suggests several solutions, including the rewriting of constitutional principles, the implementation of ensembling techniques, and the adjustment of the strictness of preference labels.

The paper also emphasizes the future direction of this research, particularly in exploring the effectiveness of natural language feedback compared to human preference labels. Moreover, it points toward increasing the robustness of AI models against red-team attacks and further aligning the concepts of helpfulness and harmlessness. This future direction is critical in ensuring that AI models can robustly respond to various challenges and adapt to various scenarios while maintaining ethical and harmless behavior.

## Comparison and Discussion

In this section, we evaluate and analyze the methodologies employed in the referenced works. We aim to examine each of the methods and highlight how their distinct methodologies influence their outcomes and effectiveness. The core of this discussion revolves around the impact of data collection and labeling, the design of reward models, and the intricate balance between policy learning and AI alignment. Furthermore, this evaluation not only contrasts the methods but also delves into their implications.

### Data Collection

#### Labeling

Using human judgment has traditionally been the primary source of ground-truth information in machine-learning research. Recently, human labelers have been instrumental in labeling preferences, as highlighted in (citation needed) and (citation needed). (citation needed) asks human labelers to choose between two sample outputs; on the other hand, (citation needed) asks humans to rank a batch size \(K\) of outputs, which is more efficient. We can get \({K \choose 2}\) different comparison in one labeling step. (citation needed) data collection is more complex. Considering this, the current advancements in AI systems suggest a shift in this paradigm. Many AI systems now have the capability to match or even surpass human abilities in specific tasks. For instance, while AI systems might not independently generate less toxic or harmful content, they can effectively discern which of two text pieces is more toxic. Also, as shown in Figure 7, LLMs can surpass the capability of reward models trained on human feedback. This insight forms the foundation of using AI systems to label preferences, either wholly or partially, leading to a new methodology known as RL from AI Feedback. In (citation needed), leverages this approach to generate a harmlessness comparison dataset and mixes that data with the human-generated helpfulness dataset to train their model.

(Figure 7: Performance LLMs with Anthropic’s helpful, honest, and harmless models trained from human feedback (orange), 5-shot pre-trained language models (black), and Claude models (bar chart).)

#### Labelers

Training models with RLHF requires an immense amount of human labelers and evaluators. Human evaluations can bring about some problems. The bias in human opinions can be transferred to the labels, which results in the increase of bias in LLMs. Summarization may be a task that is simple and easily understood by all humans, but more complex labeling, such as instruction following datasets or code generations datasets, requires expert knowledge and clear intention recognition. Therefore, (citation needed) chose their labelers based on their performance, provided clear instructions, and closely monitored the contractors to answer their questions.

In the RLAIF framework, the role of humans transitions from direct labeling to setting a series of guidelines or a "constitution" for the AI. The AI system then evaluates its outputs against these rules and makes necessary adjustments. The goal of RLAIF is not to remove humans from the labeling process but to enhance the efficiency of their involvement. Human oversight is primarily in setting the constitutional principles to which the AI system adheres. This method allows for more precise control of AI behavior with significantly fewer human labels and enhances AI systems' transparency and decision-making process.

In conclusion, adopting RLAIF in labeling preferences marks a significant shift in RL applications. RLAIF offers a more efficient, controlled, and transparent approach to developing advanced AI systems by integrating AI systems into the labeling process under human-defined constitutional guidelines.

### Reward Model

One of the essential parts of the RLHF is the Reward Model. Reward models learn to estimate human preference based on the number of samples comparing pairs of samples with each other. Therefore, it is crucial that it does not overfit the training data. (citation needed) Employed Two main strategies to enhance the generalization and prevent overfitting in reward models (RMs) for RLHF, which is followed by (citation needed). First, comparisons between pairs of model outputs are grouped and treated as a single batch element during training rather than shuffling them into one dataset. This approach minimizes the risk of overfitting from multiple gradient updates on the same data within a single epoch. Second, the use of a cross-entropy loss function aligns the model's predictions more closely with human preferences. Together, these strategies ensure that RMs accurately estimate human preferences while maintaining robustness against overfitting.

There is an implicit problem with the use of RM models; a single function cannot best represent the whole society's preference. All of the papers we are investigating (citation needed), (citation needed), and (citation needed) have reported human-annotator and annotator-annotator agreement rates are which are in the range of 63% to 77%. It is important to note here that while using cross-entropy to train these models, the model considers minority votes, which can lead to under-representing minority groups whose opinions are not represented well in the training.

#### Statistical Metrics as Rewards

One task in which we have a famously defined statistical measure is summarization. ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a widely used metric in natural language processing, particularly for evaluating the quality of text summaries. However, when it comes to using ROUGE as a reward signal in Reinforcement Learning, there are two notable limitations:

First, studies show that ROUGE has approximately 57% agreement with human labelers. This level of agreement indicates a relatively low sampling quality. The disparity between ROUGE scores and human judgment highlights the metric's inability to fully capture the nuances that human evaluators consider in text quality. Second, empirical studies (as shown in Figure 8) show that when ROUGE is used as a reward signal, the final model tends to peak both sooner and at a substantially lower quality rate compared to optimization against reward models that incorporate human feedback. This early peaking suggests that models trained solely on ROUGE may converge to suboptimal solutions, missing out on the richer quality assessments that human feedback can provide.

(Figure 8: Summary quality in comparison with the reference summaries.)

In essence, while ROUGE can be a useful tool for preliminary evaluations, its limitations underscore the necessity of human feedback in Reinforcement Learning for language models. The human judgment remains crucial in capturing the qualitative aspects of language that automated metrics like ROUGE may overlook.

### Policy Learning

All of our main papers use similar methods to train their final model using RL and PPO as shown in Equation 7.

\[
R(\theta_{\text{new}}) = E_{x,y \sim \pi_{\theta_{\text{new}}}}[\hat{r}_{\phi}(x,y) - \beta \log\frac{\pi^{\text{RL}}(y | x)} {\pi^{\text{SFT}}(y | x)}] + \gamma E_{x \sim \text{pre-train}} [\text{log}(\pi^{\text{RL}}(x))]
\]

The key distinction among these studies is that in (citation needed), the second term is included, whereas it is absent in others. This term is used to mitigate the 'alignment tax,' a term denoting the decrease in benchmark performance observed in models post-training with RLHF. In contrast, (citation needed) did not report a performance drop after RLHF training. This discrepancy can be attributed to two factors: firstly, Claude's initial accuracy was not sufficiently high to be significantly impacted by RLHF. Secondly, the variance in side effects from training with RLHF could be due to the different human feedback data used in training these models.

#### Reward Hacking

This problem happens when language models tend to generate nonsense words to satisfy the non-globally optimum reward model. To avoid reward hacking problems, RLHF models add negative KL-divergence from the probability of the pre-trained model to the reward function. In addition, Experiments in (citation needed) show that RL-CAI models can be over-trained, exhibiting Goodharting behavior in which the model sometimes responded with a harsh/accusatory tone to the harmful prompts.

### Honesty, Helpfulness, Harmlessness,

So far, we have discussed the explicit alignment of these methods; here, we are going to focus on the three main implicit alignment intentions:

- Honesty (It should not fabricate information or mislead the user)
- Helpfulness (It should help the user solve their task)
- Harmlessness (It should not cause physical, psychological, or social harm to people or the environment).

#### Honesty in AI Systems

The aim is to develop AI systems that maintain honesty, especially as they achieve or surpass human capabilities. CAI suggests a move towards more autonomous AI decision-making while ensuring robustness against harmful behaviors. The goal is to encode AI behavior in simple, transparent forms to understand better and evaluate AI decision-making. The result of this approach on the TruthfulQA benchmark is shown in Table 1. This result shows that models trained with CAI are less prone to hallucination.

(Table 1: The result of models trained w/ without CAI on TruthfulQA.)

#### Helpfulness vs. Harmlessness

A key challenge in AI development is balancing helpfulness with harmlessness. AI models trained to be harmless may become evasive and less helpful, particularly in avoiding controversial or potentially harmful queries. The CAI methodology aims to train AI assistants who are non-evasive and engage in explaining their refusal to comply with unethical requests, thereby maintaining helpfulness without compromising harmlessness. Human evaluation by (citation needed) showed that CAI methods almost never give evasive responses to harmful prompts. As shown in Figure 9, the CAI method shows better harmlessness in the same helpfulness level in comparison to normal RLHF.

(Figure 9: Responses of models trained with/without CAI to a harmful prompt.)

#### Evasiveness vs. Transparency

Earlier models, particularly those trained with human feedback, often resorted to evasive responses when confronted with sensitive topics. CAI models, however, are trained to be less evasive and more transparent. Human evaluations show that CAI is almost never evasive. By avoiding evasiveness, these models can provide more nuanced and thoughtful responses, enhancing both the perception of helpfulness and the actual usefulness of the AI in various scenarios. Examples of the responses of different models to a harmful prompt are shown in Figures 10 and 11.

(Figure 10: An example of how models respond to harmful prompts.)

In addition to CAI, Chain-of-thought (CoT) prompting is used in the feedback model to generate labels for training. This method involves reformatting feedback principles in a conversational manner, allowing for higher-quality reasoning and decision-making in the AI models. CoT prompting leads to more transparent AI decision-making, as the AI explains its reasoning step-by-step, thereby making the training objectives and the AI's thought process more transparent.

### Alignment

The advancements in RLHF and RLAIF, especially through the development of CAI, mark a substantial leap in creating AI systems that balance honesty, helpfulness, and harmlessness. These systems show lower levels of inclination for harmful behavior and higher levels of transparency in decision-making. This methodology resolves the conflict between being helpful and harmlessness, leading to more ethically aligned AI systems.

However, it is crucial to recognize that AI-generated feedback fundamentally depends on human input, primarily because (1) the models providing feedback are trained on human-generated data, and (2) human controllers direct prompts and guide the integration of this feedback into the final model.

In discussing alignment, it is important to consider the literature's emphasis on 'human preferences' or 'human values.' In practice, alignment often translates to aligning with the preferences of a specific set of labelers. These labelers' input, shaped by their unique experiences and the context of their employment, directly influences the data used to fine-tune our models.

This alignment process presents challenges in fairness, transparency, and accountability. While the authors demonstrate that alignment can target specific human reference groups for certain applications, it is not an assertion that the preferences of researchers, hired labelers, or customers are the universally correct sources for alignment. The broader issue involves balancing the preferences of various stakeholders: the organization or individual training the model, the customers using it, the end-users, and the general public.

A possible direction could involve developing models that can be conditioned on the preferences of specific groups or easily fine-tuned to represent diverse values. However, such models, while being more representative, might still impact broader society, raising complex decisions about whose preferences to prioritize, how to ensure fair representation, and the option for groups to opt out of potentially harmful processes.

Returning to the development of CAI, this complexity underscores the potential and challenges of creating more autonomous AI systems. These systems must manage complex tasks with greater honesty, helpfulness, and harmlessness while navigating the intricate web of human preferences and values that guide their training and application.

## How does the future look like?

### Moving From RL

Reinforcement Learning methods, while effective and powerful, have been found to exhibit instability. As of January 2023, numerous papers are emerging on RLHF models, with some aiming to enhance their stability while others are working on transitioning from unstable RL to a more stable algorithm. One of the most famous models that are challenging RLHF is Direct Preference Optimization (DPO), which allows the model to solve the standard RLHF problem with only a simple classification loss. However, there are still ongoing debates on how effective this method is, but the community is actively re-evaluating its choice of RL and is looking for ways to improve or replace it.

### Open-source RLHF models

One of the problems with large RLHF models is that they are mostly owned by large companies. Thanks to Meta Llama 2, code and pre-trained weights are available to the community. However, this is work, and openness is required. Doing active and influential research in this field requires access to data and open-source code, so it would be feasible for the community to make well-evaluated experiments and draw experiment-based conclusions. However, it is essential to note that the openness of a large language model trained with RLHF poses potential risks. Recent research demonstrates that safety layers implemented through RLHF are vulnerable. Attackers can undermine these protections through further fine-tuning, requiring as few as 340 examples to achieve a 95% success rate in removing RLHF safeguards.

### Reasonable Computation Cost

As LLMs increase in size, training and fine-tuning them require more time and resources. For instance, even training/evaluating methods like DPO on real-world models requires a considerable amount of computation resources that are simply unavailable for every researcher or even the research team. We need to develop new techniques that enable us to train such huge models or find a way to fit them into limited resources.

## Conclusion

This report presents the findings and compares the methodologies of three primary papers in the realm of Reinforcement Learning and Language Modeling. We discussed the misalignment of language model training with its real objective. We showed how these approaches differ from each other and what are their shortcomings. Empirical results show that RLHF increases the performance of tasks that do not have precise metrics and how human preference outperforms statistical metrics. In addition, we presented InstructGPT, one of the most widely used chat-based language models, and discussed how it is trained by human preference. Moreover, We elaborated on the problems with the alignment of RLHF models and how methods such as CAI can help the model improve harmlessness while keeping high levels of helpfulness and transparency.


References:
- (Your references here)
