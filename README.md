# From Time Series Analysis to Question Answering: A Survey in the LLM Era

[![GitHub Stars](https://img.shields.io/github/stars/Leeway-95/Aligning-TSD-with-LLM?style=social)](https://github.com/Leeway-95/Aligning-TSD-with-LLM/stargazers)
![Topic](https://img.shields.io/badge/Time%20Series-Alignment--LLMs-blueviolet)
[![How to Cite](https://img.shields.io/badge/Cite-bibtex-orange)](#citation)

✨ If you find our <em>position</em> useful for your research, please consider giving it a <strong>star ⭐ on GitHub</strong> to stay updated with future releases.

## Abstract
Recently, Large Language Models (LLMs) have emerged as a novel paradigm for Time Series Analysis (TSA), leveraging strong language capabilities to support tasks such as forecasting and anomaly detection. However, a fundamental objective gap persists between TSA and LLMs. LLMs are pre-trained to optimize semantic relevance for question answering, rather than objectives specialized for TSA. To bridge this gap, Time Series Question Answering (TSQA) reframes TSA as answering natural language queries over numerical sequences, enabling users to interact with time series through language instead of predefined analytical pipelines. This survey provides an up-to-date overview of economical, flexible, and generalizable alignment paradigms to support evolution from TSA to TSQA. We first propose a taxonomy that organizes existing literature into three alignment paradigms: Injective Alignment, Bridging Alignment, and Internal Alignment, and provide practical guidance for selecting among them. We then analyze representative TSQA datasets from the perspectives of data characteristics and domains. Finally, we discuss challenges and future directions.

<img width="550" alt="image" src="https://github.com/user-attachments/assets/10fcb059-af69-4573-9ad4-b8fcde8bc27f" />


The horizontal axis indicates whether LLM parameters are trained, and the vertical axis indicates whether temporal modifications are required. Temporal modification refers to adapting time series for LLMs, including both modifications outside the LLM and adjustments to the internal LLM architecture. These two dimensions determine the three alignment paradigms.
(a) Injective Alignment involves no temporal modification and adopts frozen LLMs. This design preserves the original LLM parameters.
(b) Bridging Alignment introduces temporal modification while still employing frozen LLMs. This design enables joint processing of time series and textual inputs while preserving all parameters of the original LLM.
(c) Internal Alignment combines temporal modification with training LLMs through parameter updating to provide native support for time series.

<!--
- [Taxonomy](#taxonomy)
  - [Prompting](#prompting)
  - [Quantization](#quantization)
  - [Aligning](#aligning)
  - [Vision](#vision)
  - [Tool](#tool)
- [Datasets](#datasets)
- [Citation](#citation)
-->

### Relevant Survey:

Date|Paper|Institute|Publication
---|---|---|---
5 <br>May <br>2025|[Towards Cross-Modality Modeling for Time Series Analytics: A Survey in the LLM Era](https://arxiv.org/abs/2505.02583)|Nanyang Technological University|IJCAI'25
14 <br>Mar <br>2025|[How Can Time Series Analysis Benefit From Multiple Modalities? A Survey and Outlook](https://arxiv.org/abs/2503.11835)|Georgia Institute of Technology|Preprint
12 <br>Mar <br>2025|[Foundation Models for Spatio-Temporal Data Science: A Tutorial and Survey](https://arxiv.org/abs/2503.13502)|The Hong Kong University of Science and Technology (Guangzhou)|KDD'25
3 <br>Feb <br>2025|[Position: Empowering Time Series Reasoning with Multimodal LLMs](https://arxiv.org/abs/2502.01477)|University of Oxford|Preprint
21 <br>Mar <br>2024|[Foundation Models for Time Series Analysis: A Tutorial and Survey](https://arxiv.org/abs/2403.14735)|The Hong Kong University of Science and Technology (Guangzhou)|KDD'24
5 <br>Feb <br>2024|[Empowering Time Series Analysis with Large Language Models: A Survey](https://arxiv.org/abs/2402.03182)|University of Connecticut, USA|IJCAI'24
5 <br>Feb <br>2024|[Position: What Can Large Language Models Tell Us about Time Series Analysis](https://arxiv.org/abs/2402.02713)|Griffith University|ICML'24
2 <br>Feb <br>2024|[Large Language Models for Time Series: A Survey](https://arxiv.org/abs/2402.01801)|University of California, San Diego|IJCAI'24
16 <br>Oct <br>2023|[Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook](https://arxiv.org/abs/2310.10196)|Monash University|Preprint
18 <br>May <br>2023|[A Survey on Time-Series Pre-Trained Models](https://arxiv.org/abs/2305.10716)|South China University of Technology|TKDE'24
3 <br>May <br>2023|[A Survey of Time Series Foundation Models: Generalizing Time Series Representation with Large Language Model](https://arxiv.org/abs/2405.02358)|Hong Kong University of Science and Technology|Preprint

### Relevant Datasets and Benchmarks:

Date|Paper|Institute|Publication|Domain|LLMs
---|---|---|---|---|---
7 <br>Nov <br>2025|[QuAnTS: Question Answering on Time Series](https://arxiv.org/abs/2511.05124)**[**[**Code**](https://github.com/mauricekraus/quants-generate)**]**|TU Darmstadt|Preprint|IoT|Llama3.1-8B
21 <br>Mar <br>2025|[MTBench: A Multimodal Time Series Benchmark for Temporal Reasoning and Question Answering](https://arxiv.org/abs/2503.16858)**[**[**Code**](https://github.com/Graph-and-Geometric-Learning/MTBench)**]**|Yale University|Preprint|Financial|GPT-4o, <br>Gemini, <br>Claude, <br>DeepSeek, <br>Llama3.1
25 <br>Jun <br>2025|[ITFormer: Bridging Time Series and Natural Language for Multi-Modal QA with Large-Scale Multitask Dataset](https://arxiv.org/abs/2506.20093)**[**[**Code**](https://pandalin98.github.io/itformer_site/)**]**|Shanghai Jiao Tong University|ICML'25|General|GPT-4o, <br>Gemini-Pro
5 <br>Jun <br>2025|[Context is Key: A Benchmark for Forecasting with Essential Textual Information (CiK)](https://arxiv.org/abs/2410.18959)**[**[**Code**](https://servicenow.github.io/context-is-key-forecasting/v0/)**]**|ServiceNow Research|ICML'25|General|Qwen-2.5-7B, <br>Llama-3-70B, <br>Llama-3.1-405B
3 <br>Mar <br>2025|[SensorQA: A Question Answering Benchmark for Daily-Life Monitoring](https://arxiv.org/abs/2501.04974)**[**[**Code**](https://github.com/benjamin-reichman/SensorQA)**]**|Georgia Institute of Technology|SenSys'25|IoT|GPT-3.5-Turbo, <br>GPT-4-Turbo 
18 <br>Oct <br>2024|[TimeSeriesExam: A time series understanding exam](https://arxiv.org/abs/2410.14752)**[**[**Code**](https://huggingface.co/datasets/AutonLab/TimeSeriesExam1)**]**|Carnegie Mellon University|NeurIPS'24 Workshop|General|GPT-4o, Gemini, Phi3.5
12 <br>Jun <br>2024|[Time-MMD: Multi-Domain Multimodal Dataset for Time Series Analysis](https://arxiv.org/abs/2406.08627)**[**[**Code**](https://github.com/AdityaLab/Time-MMD)**]**|Georgia Institute of Technology|NeurIPS'24|General|LLaMA-3, <br>GPT-2
17 <br>Apr <br>2024|[Language Models Still Struggle to Zero-shot Reason about Time Series (TSandLanguage)](https://arxiv.org/abs/2404.11757)**[**[**Code**](https://github.com/behavioral-data/TSandLanguage)**]**|University of Washington|EMNLP'24 (Findings)|General|GPT-4 
28 <br>Oct <br>2023|[Insight Miner: A Time Series Analysis Dataset for Cross-Domain Alignment with Natural Language](https://openreview.net/forum?id=E1khscdUdH&referrer=%5Bthe%20profile%20of%20Ming%20Zheng%5D(%2Fprofile%3Fid%3D~Ming_Zheng2))|University of California, Berkeley|NeurIPS'23 Workshop|General|LLaVA, <br>GPT-4
27 <br>Oct <br>2023|[JoLT: Jointly Learned Representations of Language and Time-Series](https://openreview.net/forum?id=UVF1AMBj9u&referrer=%5Bthe%20profile%20of%20Yifu%20Cai%5D(%2Fprofile%3Fid%3D~Yifu_Cai1))|Carnegie Mellon University|NeurIPS'23 Workshop|Medical|GPT-2, <br>OPT

### Injective Alignment:

Date|Paper|Institute|Publication|Domain|LLMs
---|---|---|---|---|---
11 <br>May<br>2025|[Can LLMs Understand Time Series Anomalies? (AnomLLM)](https://arxiv.org/abs/2410.05440)|University of California|Preprint|General|Qwen-VL-Chat, <br>InternVL2-Llama3-76B, <br>GPT-4o-mini, <br>Gemini-1.5-Flash
25 <br>Apr <br>2025|[A Picture is Worth A Thousand Numbers: Enabling LLMs Reason about Time Series via Visualization (TimerBed)](https://arxiv.org/abs/2411.06018)**[**[**Code**](https://github.com/AdityaLab/DeepTime/)**]**|Georgia Institute of Technology|NAACL'25|General|GPT-4o-mini, <br>Qwen2-VL-72B
16 <br>Feb <br>2025|[TableTime: Reformulating Time Series Classification as Training-Free Table Understanding with Large Language Models](https://arxiv.org/abs/2411.15737)**[**[**Code**](https://anonymous.4open.science/r/TableTime-5E4D)**]**|University of Science and Technology of China|CIKM'25|General|Llama-3.1
24 <br>Jan <br>2025|[Argos: Agentic Time-Series Anomaly Detection with Autonomous Rule Generation via Large Language Models](https://arxiv.org/abs/2501.14170)|University of Washington|Preprint|General|GPT-3.5-Turbo, <br>GPT-4o
24 <br>Nov <br>2024|[LeMoLE: LLM-Enhanced Mixture of Linear Experts for Time Series Forecasting](https://arxiv.org/abs/2412.00053)**[**[**Code**](https://github.com/RogerNi/MoLE)**]**|Hong Kong University of Science and Technology (Guangzhou)|Preprint|General|GPT2
18 <br>Oct <br>2024|[XForecast: Evaluating Natural Language Explanations for Time Series Forecasting](https://arxiv.org/abs/2410.14180)|Salesforce AI Research|Preprint|General|GPT-4
14 <br>Aug <br>2024|[MedTsLLM: Leveraging LLMs for Multimodal Medical Time Series Analysis](https://arxiv.org/abs/2408.07773)**[**[**Code**](https://github.com/flixpar/med-ts-llm)**]**|Johns Hopkins University|MLHC'24|Medical|LLaMA
24 <br>May <br>2024|[Large Language Models can Deliver Accurate and Interpretable Time Series Anomaly Detection (LLMAD)](https://arxiv.org/abs/2405.15370)|University of Chinese Academy of Sciences China|Preprint|General|GPT-4
2 <br>Apr <br>2024|[TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting](https://arxiv.org/abs/2310.04948)**[**[**Code**](https://github.com/DC-research/TEMPO)**]**|University of Southern California, Google|ICLR'24|General|GPT-2
6 <br>Mar <br>2024|[K-Link: Knowledge-Link Graph from LLMs for Enhanced Representation Learning in Multivariate Time-Series Data](https://arxiv.org/abs/2403.03645)|Nanyang Technological University|Preprint|General|CLIP, <br>GPT-2
25 <br>Feb <br>2024|[LSTPrompt: Large Language Models as Zero-Shot Time Series Forecasters by Long-Short-Term Prompting](https://arxiv.org/abs/2402.16132)**[**[**Code**](https://github.com/AdityaLab/lstprompt)**]**|Georgia Institute of Technology, Microsoft Research Asia|ACL'24 Findings|General|GPT-3.5, <br>GPT-4
16 <br>Feb <br>2024|[Time Series Forecasting with LLMs: Understanding and Enhancing Model Capabilities (TSFLLMs)](https://arxiv.org/abs/2402.10835)**[**[**Code**](https://github.com/MingyuJ666/Time-Series-Forecasting-with-LLMs)**]**|Rutgers University|KDD'25 Explorations Newsletter|General|GPT-3.5, <br>GPT-4, <br>LLaMA-2
10 <br>Feb <br>2024|[REALM: RAG-Driven Enhancement of Multimodal Electronic Healthcare Records Analysis via Large Language Models](https://arxiv.org/abs/2402.07016)|Beihang University, China Mobile Research Institute|Preprint|Medical|GPT-4, <br>Qwen
10 <br>Dec <br>2023|[PromptCast: A New Prompt-based Learning Paradigm for Time Series Forecasting](https://arxiv.org/abs/2210.08964)**[**[**Code**](https://github.com/HaoUNSW/PISA)**]**|University of New South Wales|TKDE'24|General|BART, <br>BERT, <br>ChatGPT 
14 <br>Nov <br>2023|[TENT: Connect Language Models with IoT Sensors for Zero-Shot Activity Recognition](https://arxiv.org/abs/2311.08245)|Nanyang Technological University|Preprint|IoT|CLIP, <br>GPT-2
11 <br>Oct <br>2023|[Large Language Models Are Zero-Shot Time Series Forecasters (LLMTime)](https://arxiv.org/abs/2310.07820)**[**[**Code**](https://github.com/ngruver/llmtime)**]**|New York University|NeurIPS'23|General|GPT-3, <br>LLaMA-2 
22 <br>Jun <br>2023|[Instruct-FinGPT: Financial Sentiment Analysis by Instruction Tuning of General-Purpose Large Language Models](https://arxiv.org/abs/2306.12659)|Columbia University|IJCAI'23 <br>FinLLM <br>Symposium|Financial|LLaMA, ChatGPT
24 <br>May <br>2023|[Large Language Models are Few-Shot Healthcare Learners](https://arxiv.org/abs/2305.15525)**[**[**Code**](https://github.com/marianux/ecg-kit)**]**|Google|Preprint|Medical|PaLM
10 <br>Apr <br>2023|[The Wall Street Neophyte: A Zero-Shot Analysis of ChatGPT Over MultiModal Stock Movement Prediction Challenges](https://arxiv.org/abs/2304.05351)|Wuhan University|Preprint|Financial|ChatGPT
1 <br>Jan <br>2023|[Unleashing the Power of Shared Label Structures for Human Activity Recognition (SHARE)](https://arxiv.org/abs/2301.03462)|University of California|CIKM'23|IoT|GPT-4

### Bridging Alignment:
Date|Paper|Institute|Publication|Domain|LLMs
---|---|---|---|---|---
27 <br>Dec <br>2025|[Chain-of-thought Reviewing and Correction for Time Series Question Answering (T3LLM)](https://www.arxiv.org/abs/2512.22627)|University of Science and Technology of China|Preprint|General|DeepSeek-R1, <br>Qwen2.5-14B
19 <br>Dec <br>2025|[Hierarchical Multimodal LLMs with Semantic Space Alignment for Enhanced Time Series Classification (HiTime)](https://arxiv.org/abs/2410.18686)**[**[**Code**](https://github.com/Xiaoyu-Tao/HiTime)**]**|University of Science and Technology of China|TIST'25|General|LLaMA3.1-8B <br>GPT-2
25 <br>Oct <br>2025|[TimeXL: Explainable Multi-modal Time Series Prediction with LLM-in-the-Loop](https://arxiv.org/abs/2503.01013)|University of Connecticut|NeurIPS'25|General|GPT-4o, <br>GPT-4o-mini, <br>Gemini-2.0-Flash
6 <br>Aug <br>2025|[CLaSP: Learning Concepts for Time-Series Signals from Natural Language Supervision](https://arxiv.org/abs/2411.08397)|Hosei University|EUSIPCO'25|General|BERT
19 <br>May<br>2025|[Decoding Time Series with LLMs: A Multi-Agent Framework for Cross-Domain Annotation (TESSA)](https://arxiv.org/abs/2410.17462)|The Pennsylvania State University |Preprint|General|GPT-4o, <br>LLaMA3.1-8B, <br>Qwen2-7B
13 <br>May <br>2025|[MADLLM: Multivariate Anomaly Detection via Pre-trained LLMs](https://arxiv.org/abs/2504.09504)|Huazhong University of Science and Technology|ICME'25|General|Unknown
12 <br>May <br>2025|[MedualTime: A Dual-Adapter Language Model for Medical Time Series-Text Multimodal Learning](https://arxiv.org/abs/2406.06620)**[**[**Code**](https://github.com/start2020/MedualTime)**]**|Hong Kong University of Science and Technology|IJCAI'25|General|GPT-2, <br>BERT
4 <br>May <br>2025|[Efficient Multivariate Time Series Forecasting via Calibrated Language Models with Privileged Knowledge Distillation (TimeKD)](https://arxiv.org/abs/2505.02138)**[**[**Code**](https://github.com/ChenxiLiu-HNU/TimeKD)**]**|Nanyang Technological University|ICDE'25|General|BERT, <br>GPT-2, <br>LLaMA-3.2
8 <br>Apr <br>2025|[CALF: Aligning LLMs for Time Series Forecasting via Cross-modal Fine-Tuning](https://arxiv.org/abs/2403.07300)**[**[**Code**](https://github.com/Hank0626/CALF)**]**|Tsinghua University|AAAI'25|General|GPT-2
5 <br>Apr <br>2025|[Context-Alignment: Activating and Enhancing LLMs Capabilities in Time Series (DECA)](https://arxiv.org/abs/2501.03747)**[**[**Code**](https://github.com/tokaka22/ICLR25-FSCA)**]**|The Hong Kong Polytechnic University|ICLR'25|General|GPT-2
20 <br>Feb <br>2025|[LLM4TS: Aligning Pre-Trained LLMs as Data-Efficient Time-Series Forecasters](https://arxiv.org/abs/2308.08469v6)|National Yang Ming Chiao Tung University|TIST'25|General|GPT-2
19 <br>Feb <br>2025|[Adapting Large Language Models for Time Series Modeling via a Novel Parameter-efficient Adaptation Method (Time-LlaMA)](https://arxiv.org/abs/2502.13725)|Nanyang Technological University|ACL'25|General|Llama-2
17 <br>Feb <br>2025|[TimeCAP: Learning to Contextualize, Augment, and Predict Time Series Events with Large Language Model Agents](https://arxiv.org/abs/2502.11418)**[**[**Code**](https://github.com/geon0325/TimeCAP)**]**|Korea Advanced Institute of Science and Technology|AAAI'25|General|GPT-4, <br>BERT
6 <br>Feb <br>2025|[Time-VLM: Exploring Multimodal Vision-Language Models for Augmented Time Series Forecasting](https://arxiv.org/abs/2502.04395)|Hong Kong University of Science and Technology (Guangzhou)|ICML'25|General|ViLT, <br>CLIP, <br>BLIP-2
5 <br>Feb <br>2025|[SensorChat: Answering Qualitative and Quantitative Questions during Long-Term Multimodal Sensor Interactions](https://arxiv.org/abs/2502.02883)**[**[**Code**](https://github.com/benjamin-reichman/SensorQA)**]**|University of California San Diego|IMWUT'25|IoT|GPT-3.5-Turbo, <br>LLaMA
27 <br>Jan <br>2025|[Smarter Together: Combining Large Language Models and Small Models for Physiological Signals Visual Inspection (ConMIL)](https://arxiv.org/abs/2501.16215)**[**[**Code**](https://github.com/HuayuLiArizona/Conformalized-Multiple-Instance-Learning-For-MedTS)**]**|University of Arizona|J. Heal. Informatics Res.|Medical|GPT-4, <br>Qwen2-VL
8 <br>Jan <br>2025|[TS-TCD: Triplet-Level Cross-Modal Distillation for Time-Series Forecasting Using Large Language Models](https://arxiv.org/abs/2409.14978v1)|East China Normal University|ICASSP'25|General|GPT-2
3 <br>Jan <br>2025|[Time Series Language Model for Descriptive Caption Generation (TSLM)](https://arxiv.org/abs/2501.01832)|Nokia Bell Labs|Preprint|General|LLaMA-2
23 <br>Dec <br>2024|[VITRO: Vocabulary Inversion for Time-series Representation Optimization](https://arxiv.org/abs/2412.17921)**[**[**Code**](https://github.com/thuml/Time-Series-Library)**]**|University of Michigana|ICASSP'25|General|GPT-2, <br>LLaMA
27 <br>Nov <br>2024|[LLM-ABBA: Understanding time series via symbolic approximation](https://arxiv.org/abs/2411.18506)**[**[**Code**](https://github.com/inEXASCALE/llm-abba)**]**|Charles University|Preprint|General|Llama2, <br>Mistral
24 <br>Nov <br>2024|[LeRet: Language-Empowered Retentive Network for Time Series Forecasting](https://www.ijcai.org/proceedings/2024/0460.pdf)**[**[**Code**](https://github.com/hqh0728/LeRet)**]**|University of Science and Technology of China|IJCAI'24|General|LLaMA
18 <br>Nov <br>2024|[Understanding the Role of Textual Prompts in LLM for Time Series Forecasting: an Adapter View](https://arxiv.org/abs/2311.14782)|Alibaba|Preprint|General|GPT-2, <br>LLaMA
5 <br>Nov <br>2024|[Learning Transferable Time Series Classifier with Cross-Domain Pre-training from Language Model (CrossTimeNet)](https://arxiv.org/abs/2403.12372)**[**[**Code**](https://github.com/Mingyue-Cheng/CrossTimeNet)**]**|University of Science and Technology of China|Preprint|General|BERT, <br>GPT-2
31 <br>Oct <br>2024|[AutoTimes: Autoregressive Time Series Forecasters via Large Language Models](https://arxiv.org/abs/2402.02370)**[**[**Code**](https://github.com/thuml/AutoTimes)**]**|Tsinghua University|NeurIPS'24|General|LLaMA, <br>GPT-2, <br>OPT
21 <br>Oct <br>2024|[LLM-TS Integrator: Integrating LLM for Enhanced Time Series Modeling](https://arxiv.org/abs/2410.16489)|Borealis AI|Preprint|General|LLaMA
14 <br>Oct <br>2024|[SensorLLM: Aligning Large Language Models with Motion Sensors for Human Activity Recognition](https://arxiv.org/abs/2410.10624)**[**[**Code**](https://github.com/zechenli03/SensorLLM)**]**|University of New South Wales, Sydney|EMNLP'25|IoT|Llama3
8 <br>Oct <br>2024|[Time-FFM: Towards LM-Empowered Federated Foundation Model for Time Series Forecasting](https://arxiv.org/abs/2405.14252)|The Hong Kong University of Science and Technology (Guangzhou)|NeurIPS 2024|General|GPT-2
23 <br>Sep <br>2024|[TS-HTFA: Advancing Time Series Forecasting via Hierarchical Text-Free Alignment with Large Language Models](https://arxiv.org/abs/2409.14978)|East China Normal University|Preprint|General|GPT-2
30 <br>Jul <br>2024|[A federated large language model for long-term time series forecasting (FedTime)](https://arxiv.org/abs/2407.20503)|Concordia Universit|Preprint|General|LLaMA
7 <br>Jul <br>2024|[S2IP-LLM: Semantic Space Informed Prompt Learning with LLM for Time Series Forecasting](https://arxiv.org/abs/2403.05798)|University of Connecticut|ICML'24|General|GPT-2
3 <br>Jun <br>2024|[TimeCMA: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment](https://arxiv.org/abs/2406.01638)**[**[**Code**](https://github.com/ChenxiLiu-HNU/TimeCMA)**]**|Nanyang Technological University|AAAI'25|General|GPT-2
4 <br>May <br>2024 |[Can Brain Signals Reveal Inner Alignment with Human Languages? (MATM)](https://arxiv.org/abs/2208.06348v5)**[**[**Code**](https://github.com/Jason-Qiu/EEG_Language_Alignment)**]**|Carnegie Mellon University|EMNLP'23 Findings|Medical|BERT
24 <br>Mar <br>2024|[GPT4MTS: Prompt-Based Large Language Model for Multimodal Time-Series Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/30383)|University of Southern California|AAAI'24|General|GPT-2, <br>BERT
12 <br>Mar <br>2024|[Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)**[**[**Code**](https://github.com/amazon-science/chronos-forecasting)**]**|AWS AI Labs|TMLR'24|General|GPT-2
23 <br>Feb <br>2024|[UniTime: A Language-Empowered Unified Model for Cross-Domain Time Series Forecasting](https://arxiv.org/abs/2310.09751)**[**[**Code**](https://github.com/liuxu77/UniTime)**]**|National University of Singapore|WWW'24|General|GPT-2
7 <br>Feb <br>2024|[Multi-Patch Prediction: Adapting LLMs for Time Series Representation Learning (aLLM4TS)](https://arxiv.org/abs/2402.04852)|The Chinese University of Hong Kong|ICML'24|General|GPT-2
22 <br>Feb <br>2024|[TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series](https://arxiv.org/abs/2308.08241)**[**[**Code**](https://github.com/SCXsunchenxi/TEST)**]**|Peking University|ICLR'24|General|BERT, <br>GPT-2, <br>ChatGLM
30 <br>Jan <br>2025|[Large Language Models are Few-shot Multivariate Time Series Classifiers (LLMFew)](https://arxiv.org/abs/2502.00059)|University of Technology Sydney|Preprint|General|GPT-2, <br>GPT-4
29 <br>Jan <br>2024|[Time-LLM: Time Series Forecasting by Reprogramming Large Language Models](https://arxiv.org/abs/2310.01728)**[**[**Code**](https://github.com/KimMeen/Time-LLM)**]**|Monash University|ICLR'24|General|LLaMA
26 <br>Jan <br>2024|[Large Language Model Guided Knowledge Distillation for Time Series Anomaly Detection (AnomalyLLM)](https://arxiv.org/abs/2401.15123)|Zhejiang University|IJCAI'24|General|GPT-2
9 <br>Oct <br>2023|[Integrating Stock Features and Global Information via Large Language Models for Enhanced Stock Return Prediction (SCRL-LG)](https://arxiv.org/abs/2310.05627)|Hithink Royal Flush Information Network|IJCAI'23|Financial|LLaMA
25 <br>Sep <br>2023|[DeWave: Discrete EEG Waves Encoding for Brain Dynamics to Text Translation](https://arxiv.org/abs/2309.14030)**[**[**Code**](https://github.com/duanyiqun/DeWave)**]**|University of Technology Sydney|NeurIPS'23|Medical|BART
22 <br>Mar <br>2023|[Frozen Language Model Helps ECG Zero-Shot Learning (METS)](https://arxiv.org/abs/2303.12311)|Jilin University|MIDL'23|Medical|BERT
6 <br>Sep <br>2023|[ETP: Learning Transferable ECG Representations via ECG-Text Pre-training](https://arxiv.org/abs/2309.07145)|Imperial College London|ICASSP'24|Medical|BERT
21 <br>Jan <br>2023|[Transfer Knowledge from Natural Language to Electrocardiography: Can We Detect Cardiovascular Disease Through Language Models? (ECG-LLM)](https://arxiv.org/abs/2301.09017)|Carnegie Mellon University|EACL'23 Findings|Medical|BERT, <br>BART

### Internal Alignment:

Date|Paper|Institute|Publication|Domain|LLMs
---|---|---|---|---|---
29 <br>Dec <br> 2025|[Alpha-R1: Alpha Screening with LLM Reasoning via Reinforcement Learning](https://www.arxiv.org/abs/2512.23515)**[**[**Code**](https://github.com/FinStep-AI/Alpha-R1)**]**|Shanghai Jiao Tong University|Preprint|General|Qwen3-8B
9 <br>Nov <br> 2025|[TimeSense: Making Large Language Models Proficient in Time-Series Analysis](https://arxiv.org/abs/2511.06344)|Tsinghua University|Preprint|General|Qwen3-8B, <br>GPT-5
28 <br>Jun <br> 2025|[Time-MQA: Time Series Multi-Task Question Answering with Context Enhancement](https://arxiv.org/abs/2503.01875)**[**[**Code**](https://huggingface.co/datasets/Time-MQA/TSQA)**]**|University of Oxford|ACL'25|General|Mistral-7B
16 <br>Apr <br>2025|[ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning](https://arxiv.org/abs/2412.03104)**[**[**Code**](https://github.com/NetManAIOps/ChatTS)**]**|Tsinghua University|VLDB'25|General|QWen-2.5
16 <br>Apr <br>2025|[ECG-Chat: A Large ECG-Language Model for Cardiac Disease Diagnosis](https://arxiv.org/abs/2408.08849)**[**[**Code**](https://github.com/YubaoZhao/ECG-Chat)**]**|China University of Geosciences|ICME'25|Medical|GPT-4, <br>Vicuna-13B
16 <br>Dec <br>2024|[ChatTime: A Unified Multimodal Time Series Foundation Model Bridging Numerical and Textual Data](https://arxiv.org/abs/2412.11376)**[**[**Code**](https://github.com/ForestsKing/ChatTime)**]**|Beijing University of Posts and Telecommunications|AAAI'25|General|LLaMA-2
13 <br>Aug <br>2024|[GenG: An LLM-Based Generic Time Series Data Generation Approach for Edge Intelligence via Cross-Domain Collaboration](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10620716)|Future Network Research Center, Purple Mountain Laboratories|INFOCOM'24|IoT|LLaMA
19 <br>Mar <br>2024|[Advancing Time Series Classification with Multimodal Language Modeling (InstructTime)](https://arxiv.org/abs/2403.12371)**[**[**Code**](https://github.com/Mingyue-Cheng/InstructTime)**]**|University of Science and Technology of China|WSDM'25|General|GPT-2
21 <br>Dec <br>2023|[BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564)|Bloomberg|Preprint|Financial|GPT-NeoX, <br>OPT, <br>BLOOM

### Dataset

Dataset|Domain|Characteristic|Representation|Statistic
---|---|---|---|---
[ECG-QA](https://github.com/Jwoo5/ecg-qa)|Medical|Multivariate, <br>stationarity, noise, trend|Text(Number)|16,054 samples
[PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)|Medical|Multivariate, <br>noise|Number+Table|60h data
[JoLT](https://physionet.org/content/ptb-xl/1.0.3/)|Medical|Multivariate, <br>stationarity, trend, noise|Number+Text|21,837 samples
[Zuco 2.0](https://osf.io/2urht/)|Medical|Multivariate, <br>noise|Number+Text|739 samples
[PIXIU](https://github.com/chancefocus/PIXIU)|Financial|Multivariate, trend, noise|Number+Text<br>+Table|136K samples
[StockNet](https://github.com/yumoxu/stocknet-dataset)|Financial|Multivariate, noise, trend|Number+Text|26,614 samples
[Finst](https://github.com/Zdong104/FNSPID_Financial_News_Dataset)|Financial|Multivariate, <br>periodicity, trend, noise|Text(Number)|29.7M samples
[Ego4D](https://ego4d-data.org/)|IoT|Multivariate, <br>stationarity, periodic, noise|Video+Audio|3,670h data, <br>3.85M samples
[DeepSQA](https://github.com/nesl/DeepSQA)|IoT|Multivariate, <br>noise, trends, periodicity|Text(Number)|1K samples
[TS-Insights](https://drive.google.com/drive/folders/1qGXigxE5GvmF1oLuGXaqLMkRgwoQfZ7V)|General|Multivariate, <br>trend, seasonality|Number+Text<br>+Image|100,000 samples
[TimeMMD](https://github.com/AdityaLab/Time-MMD)|General|Multivariate, <br>stationarity, trends|Text(Number)|496 samples
[CiK](https://github.com/ServiceNow/context-is-key-forecasting)|General|Univariate, multivariate, <br>stationarity, trend, <br>noise, periodicity|Number+Text|2,644 samples

## Contact Us
For inquiries or further assistance, contact us at [leeway@ruc.edu.cn](mailto:leeway@ruc.edu.cn).

<!--
## Citation

If you find this useful, please cite our paper: "Aligning Time Series Data with Large Language Models: A Survey".

```
@article{zhang2024large,
  title={Large Language Models for Time Series: A Survey},
  author={Zhang, Xiyuan and Chowdhury, Ranak Roy and Gupta, Rajesh K and Shang, Jingbo},
  journal={arXiv preprint arXiv:2402.01801},
  year={2024}
}
```
-->
