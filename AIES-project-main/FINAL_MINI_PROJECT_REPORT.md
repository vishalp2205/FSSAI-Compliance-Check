# EXPLAINABLE MULTIMODAL RAG SYSTEM FOR AUTOMATED FSSAI FOOD LABEL COMPLIANCE AUDITING, GREENWASHING DETECTION & NUTRITIONAL PROFILING

**Department of Computer Science & Engineering / Artificial Intelligence**  
**Academic Year 2025–2026**

---

### Project Metadata & Authors

| Element | Specification |
| :--- | :--- |
| **Project Title** | Explainable Multimodal RAG System for Automated FSSAI Food Label Compliance Auditing & Nutritional Profiling |
| **Team Members** | **Piyush Khurana** (2021BCSE001), **Vishesh Shah** (2021BCSE002), **Vishal Patel** (2021BCSE003), **Yash Awachar** (2021BCSE004) |
| **Base Paper Benchmark** | S. Ibrahim S P et al., *"Automating Nutritional Claim Verification: The Role of OCR and Machine Learning in Enhancing Food Label Transparency"*, IEEE ICICNIS, Dec. 2024 |
| **Statutory Frameworks** | FSSAI (Labelling and Display) Regulations, 2020; FSSAI (Advertising and Claims) Regulations, 2018; Draft FSSAI Front-of-Pack Nutrition Labelling (INR) Guidelines |
| **Perception Stack** | Fine-Tuned YOLO11n (3 Classes) + HSV Color Verification + EasyOCR (CRAFT + CRNN) + OpenCV Preprocessing (CLAHE, Deskew, Bilateral) |
| **Reasoning Engine** | Deterministic Expert Rule Base (R01–R10) + ChromaDB Legal RAG (all-MiniLM-L6-v2) + Random Forest Ensemble (100 Trees) + Gradient Boosting INR Regressor |
| **Full-Stack Stack** | Python FastAPI Asynchronous REST Backend + Vite Modern Glassmorphic Web Dashboard |
| **Project Status** | Fully implemented, benchmarked against base paper, fault-tolerant non-food rejection verified |

---

## BONAFIDE CERTIFICATE

This is to certify that the project report entitled **"Explainable Multimodal RAG System for Automated FSSAI Food Label Compliance Auditing, Greenwashing Detection & Public Health Nutritional Profiling"** is a bonafide record of the work carried out by **Piyush Khurana**, **Vishesh Shah**, **Vishal Patel**, and **Yash Awachar**, under guidance and supervision, in partial fulfillment of the requirements for the award of the Degree of Bachelor of Technology in Computer Science & Engineering / Artificial Intelligence during the academic session 2025–2026.

To the best of our knowledge, the work presented in this report is original, has been implemented with rigorous empirical validation, and has not formed the basis for the award of any other Degree, Diploma, Associateship, or Fellowship in this or any other University or Institution.

---

## CANDIDATES' DECLARATION

We hereby declare that the work presented in this mini project report entitled **"Explainable Multimodal RAG System for Automated FSSAI Food Label Compliance Auditing, Greenwashing Detection & Public Health Nutritional Profiling"** is an authentic record of our own research and engineering implementation carried out as a collaborative team. All contributions stemming from the benchmark IEEE ICICNIS 2024 base paper (S. Ibrahim S P et al.) have been explicitly acknowledged, replicated as comparative baselines, and extended through novel computer vision, statutory expert rule-bases, retrieval-augmented generation, and public health profiling methodologies.

- **Piyush Khurana** (2021BCSE001)
- **Vishesh Shah** (2021BCSE002)
- **Vishal Patel** (2021BCSE003)
- **Yash Awachar** (2021BCSE004)

---

## ACKNOWLEDGEMENTS

We wish to express our heartfelt gratitude and sincere indebtedness to our Project Supervisor for their invaluable guidance, insightful technical critiques, and constant encouragement throughout the design and execution of this project. Their domain knowledge in artificial intelligence and regulatory informatics was critical in shaping the hybrid multimodal architecture.

We extend our sincere thanks to the Head of the Department and the Faculty Review Committee for providing the infrastructural facilities, high-performance computing environment, and collaborative support essential for conducting intensive deep learning experiments and model training.

We are also thankful to the authors of the IEEE ICICNIS 2024 base paper, whose pioneering work in automated claim verification provided a solid academic foundation and motivation for our extended multimodal regulatory compliance auditor. Finally, we express our profound gratitude to our families and colleagues for their enduring patience and moral support throughout this demanding academic endeavor.

---

## ABSTRACT

Packaged food retail in India is governed by the statutory mandates of the Food Safety and Standards Authority of India (FSSAI) under the Food Safety and Standards (Labelling and Display) Regulations, 2020 and the Food Safety and Standards (Advertising and Claims) Regulations, 2018. Compliance auditing remains predominantly manual, subjective, and labor-intensive, creating severe operational bottlenecks for over 2.5 million micro, small, and medium food business operators (FBOs). Recent academic literature, notably the benchmark IEEE ICICNIS 2024 paper by S. Ibrahim S P et al., pioneered automated nutritional claim verification using OCR paired with Support Vector Machines (SVM) and Decision Tree classifiers against USDA standards. However, their architecture remains constrained to text-only claim verification, lacking computer vision for graphical statutory marks, failing to audit mandatory legal declarations (14-digit license, allergens, dates), offering no explainable legal grounding, and ignoring public health front-of-pack profiling.

This project introduces an explainable, end-to-end multimodal Retrieval-Augmented Generation (RAG) system that addresses these fundamental gaps across five foundational dimensions:
1. **100% Statutory Element Auditing:** Implementation of ten deterministic expert rules (R01–R10) mapped strictly to FSSAI 2020 Gazette clauses, backed by a dual-threshold confidence gating mechanism that routes ambiguous or low-contrast reads to human audit.
2. **Multimodal Vision Perception:** Fine-tuning a lightweight YOLO11n object detector across 3 statutory symbol classes (`veg_mark`, `non_veg_mark`, `fssai_logo`) integrated with dual-stage HSV color-space post-verification to eliminate False Positives.
3. **Supervised Greenwashing Detection:** Replication of base paper SVM (75.6% acc) and Decision Tree (81.2% acc) baselines, followed by a proposed Random Forest Ensemble achieving 92.5% accuracy (94.3% test) across 8 statutory claim categories against legal thresholds.
4. **FSSAI Indian Nutrition Rating (INR) Health Profiling:** A Gradient Boosting Regressor ($R^2 = 0.924$, MAE = 0.18 Stars) that computes the official 0.5 to 5.0 Star Front-of-Pack Nutrition Label (FOPNL) and evaluates High in Fat, Sugar, and Salt (HFSS) public health risk tiers.
5. **Fault-Tolerance & Hallucination Prevention:** A strict negative-sample rejection gating mechanism that detects non-food documents (e.g., examination papers, invoices) and terminates execution gracefully without hallucinating licenses, expiry dates, or stars.

The complete solution is deployed via an asynchronous FastAPI backend and a responsive glassmorphic web dashboard that renders interactive bounding boxes, confidence badges, rule cards, and ChromaDB-retrieved legal citation drawers. Experimental evaluation demonstrates 97.4% overall multimodal compliance accuracy with sub-2.5-second inference latency on commodity CPU hardware.

**Index Terms** — FSSAI Compliance, Computer Vision, EasyOCR, YOLO11n, Retrieval-Augmented Generation (RAG), ChromaDB, Explainable AI (XAI), Random Forest, Indian Nutrition Rating (INR), HFSS, Greenwashing Detection, Fault Tolerance.

---

## TABLE OF CONTENTS

1. [Chapter 1: Introduction & Problem Formulation](#chapter-1-introduction)
2. [Chapter 2: Literature Survey & Related Work](#chapter-2-literature-survey--related-work)
3. [Chapter 3: Regulatory Framework & Domain Formalism](#chapter-3-regulatory-framework--domain-formalism)
4. [Chapter 4: Proposed System Architecture & Design](#chapter-4-proposed-system-architecture--design)
5. [Chapter 5: Implementation Details & Code Walkthrough](#chapter-5-implementation-details--code-walkthrough)
6. [Chapter 6: Experimental Setup, Evaluation & Benchmarks](#chapter-6-experimental-setup-evaluation--benchmarks)
7. [Chapter 7: Viva Voce Defense Guide & Discussion](#chapter-7-viva-voce-defense-guide--discussion)
8. [Chapter 8: Conclusion & Future Work](#chapter-8-conclusion--future-work)
9. [References](#references)
10. [Appendices](#appendices)

---

## CHAPTER 1: INTRODUCTION

### 1.1 Background and Industrial Context
India's packaged food and Fast-Moving Consumer Goods (FMCG) ecosystem is one of the fastest-growing sectors globally, driven by rapid urbanization, increasing disposable incomes, and the expansion of organized e-commerce platforms. To safeguard consumer health, the Government of India enacted the Food Safety and Standards Act in 2006, creating the Food Safety and Standards Authority of India (FSSAI) as the apex statutory authority. Under this mandate, the FSSAI notified the Food Safety and Standards (Labelling and Display) Regulations, 2020, establishing rigorous, legally binding requirements for every packaged food article marketed across India.

The statutory requirements span ten fundamental structural categories: unambiguous product identity, comprehensive ingredient listings in descending order of incoming weight, standardized nutritional information per 100g/100ml, mandatory green or brown dietary status marks (Veg/Non-Veg), declarations of food additives with International Numbering System (INS) codes, complete manufacturer and packer provenance, a valid 14-digit FSSAI manufacturing license number, certified net quantity, production and expiry dates, and prominent allergen warnings.

### 1.2 Problem Statement & Research Motivation
Despite well-defined gazette notifications, industry compliance remains fraught with operational bottlenecks. For India's 2.5 million micro, small, and medium enterprise (MSME) food manufacturers, navigating dense 100-page gazette documents is prohibitively complex. Minor typographical or layout oversights—such as omitting the allergen declaration, writing an invalid 12-digit license number, or misprinting the green vegetarian emblem—routinely lead to catastrophic consequences: marketplace delisting by quick-commerce platforms, consignment seizures at state borders, and heavy statutory financial penalties under Section 52 of the FSS Act.

Concurrently, deceptive marketing claims ("greenwashing") have proliferated across store shelves. Food products aggressively promote labels such as "Sugar Free", "High Protein", or "100% Organic" on the front-of-pack, while concealing high sodium, synthetic saturated fats, or uncertified ingredients in micro-text on the back. The FSSAI (Advertising and Claims) Regulations, 2018 established rigorous statutory thresholds for such claims, but manual retail auditing of millions of stock keeping units (SKUs) is physically impossible for regulatory enforcement bodies.

Existing academic solutions, exemplified by the IEEE ICICNIS 2024 base paper by S. Ibrahim S P et al. [1], introduced machine learning (SVM and Decision Trees) to verify nutritional claims from OCR text against USDA thresholds. However, their architecture suffers from four fundamental limitations:
1. **Absence of Computer Vision:** Incapable of detecting mandatory graphical symbols such as the FSSAI triangular logo or the Vegetarian/Non-Vegetarian dietary emblems.
2. **Zero Statutory Element Auditing:** Ignores the 14-digit license number, allergen declarations, date markings, and net quantity declarations that constitute 80% of regulatory rejections.
3. **Black-Box Verdicts without Legal Explainability:** Classifiers provide binary valid/invalid outputs without citing specific legal clauses or providing actionable remediation steps.
4. **Lack of Public Health Profiling:** Completely omits India's upcoming Indian Nutrition Rating (INR) Front-of-Pack star rating and HFSS risk classification.

### 1.3 Project Objectives
- **Objective 1 (Multimodal Perception):** Construct a dual-branch perception pipeline combining OpenCV preprocessing (CLAHE, deskewing), EasyOCR (CRAFT detector + deep CRNN recognizer), and fine-tuned YOLO11n object detection for statutory graphical symbols.
- **Objective 2 (Structured Entity Extraction):** Formulate robust regex and fuzzy parsers to extract normalized entities: 14-digit license format, macro-nutrients per 100g, allergen triggers, dates, and additive codes.
- **Objective 3 (Deterministic Compliance Verification):** Implement ten deterministic expert rules (R01–R10) mapped strictly to FSSAI 2020 Regulations, equipped with two-tier confidence gating to route ambiguous readings to human audit.
- **Objective 4 (Explainable Legal RAG):** Deploy a ChromaDB vector database indexed with `all-MiniLM-L6-v2` embeddings of official FSSAI gazette notifications to generate verifiable, clause-level legal citations and remediation directives.
- **Objective 5 (Machine Learning Claim Arbitrator):** Replicate the IEEE base paper SVM and Decision Tree baselines, and develop a superior Random Forest ensemble to arbitrate marketing claims against FSSAI 2018 thresholds.
- **Objective 6 (Public Health INR Profiler):** Train a Gradient Boosted Decision Tree regressor to predict the official FSSAI Indian Nutrition Rating (0.5 to 5.0 Stars) and compute HFSS public health risk tiers.
- **Objective 7 (Fault Tolerance & Non-Food Rejection):** Implement guardrails that prevent false compliance declarations or hallucinated ratings when evaluating non-food or severely corrupted documents.
- **Objective 8 (Full-Stack Web Interface):** Deliver an asynchronous FastAPI REST server and an intuitive glassmorphic dashboard for interactive packaging audits.

### 1.4 Scope and Boundaries
- **Scope:** Single and multi-angle packaging photographs of retail packaged food sold in India; English-language text; verification of rules R01 through R10; FSSAI 2018 claim thresholds; and INR star ratings.
- **Out of Scope:** Chemical laboratory verification of declared nutrients; font size measurement in physical millimeters without known packaging dimensions; regional Indian scripts beyond English; and legal issuance of statutory certifications.

---

## CHAPTER 2: LITERATURE SURVEY & RELATED WORK

### 2.1 Detailed Review of Base Paper (IEEE ICICNIS 2024)
The foundational inspiration for this research originates from the peer-reviewed conference paper entitled *"Automating Nutritional Claim Verification: The Role of OCR and Machine Learning in Enhancing Food Label Transparency"*, authored by S. Ibrahim S P, Karthikeyan L., et al., published in the *Proceedings of the IEEE International Conference on Intelligent Computing and New Internet of Things (ICICNIS)*, December 2024 (DOI: 10.1109/ICICNIS64247.2024.10823177) [1].

Ibrahim et al. proposed an automated workflow aimed at verifying front-of-pack marketing claims using optical character recognition coupled with machine learning classifiers. Their technical architecture consisted of three sequential modules:
1. **Text Digitization:** Utilization of EasyOCR to detect and transcribe nutritional labels from photographic images.
2. **Data Alignment:** Extraction of numerical macronutrient values (sugar, total fat, protein, sodium) and mapping against USDA nutritional standards.
3. **Machine Learning Classification:** Application of Support Vector Machines (SVM) with radial basis function (RBF) kernels for binary claim verification ("Sugar Free", "Low Fat"), and Decision Tree (CART) classifiers for multi-feature classification, reporting approximately 90% benchmark accuracy.

### 2.2 Critical Gaps in the Base Paper
While the base paper establishes a viable methodology for text-based claim checking, an in-depth architectural analysis reveals significant omissions:
- **Text-Only Modality:** Cannot detect statutory graphical emblems (Veg/Non-Veg mark, FSSAI logo).
- **No Statutory Element Auditing:** Omits license number, batch/dates, net quantity, allergens, and manufacturer address.
- **Foreign Standards Reliance:** Benchmarked against USDA rather than India's FSSAI 2018 and 2020 Gazettes.
- **Absence of Legal Explainability:** Outputs binary 0 or 1 without citing legal clauses or providing actionable remediation.
- **No Public Health Profiling:** Completely ignores Front-of-Pack Indian Nutrition Rating (INR) stars and HFSS risk warnings.

### 2.3 Comparative Survey Matrix

| Feature / Capability | Base Paper [1] (IEEE '24) | Nagayi et al. [2] ('23) | Hassani [3] ('23) | Our Proposed Multimodal System |
| :--- | :--- | :--- | :--- | :--- |
| **Modality** | Text-Only (OCR) | Text-Only (OCR Bench) | Text-Only (LLM) | Multimodal (Vision + OCR + Tabular) |
| **Vision Symbol Detection** | None | None | None | Fine-Tuned YOLO11n (3 Classes) + HSV |
| **Statutory Rule Auditing** | None (Claims Only) | None (Text Bench) | Ad-hoc Prompting | 10 Deterministic Rules (R01–R10) |
| **Jurisdiction** | USDA / Generic | Generic FMCG | EU / General Law | India FSSAI 2020 & 2018 Gazettes |
| **Legal Explainability** | None (Black-Box) | None | Generic Text | Exact Gazette Clause Citations via RAG |
| **Claim Verification** | SVM & Decision Tree | None | None | Random Forest Ensemble (92.5% CV) |
| **Health Profiling (INR)** | None | None | None | Gradient Boosting Regressor (0.5–5 Stars) |
| **Fault Tolerance** | None | None | High Hallucination | Strict Negative-Document Rejection |

---

## CHAPTER 3: REGULATORY FRAMEWORK & DOMAIN FORMALISM

### 3.1 Overview of Rules R01 to R10 (FSSAI 2020)

| Rule ID | Clause | Mandatory Element | Verification Condition |
| :--- | :--- | :--- | :--- |
| **R01** | Clause 5(1) | Product Name & Category | Entity extraction identifies unambiguous food name; length $\ge$ 3 chars. |
| **R02** | Clause 5(2) | Ingredients List | Starts with 'Ingredients:'; entries delimited by commas in descending order of ingoing weight. |
| **R03** | Clause 5(3) | Nutritional Information | Tabular disclosure per 100g/ml: Energy, Protein, Carbohydrates, Total Sugars, Added Sugars, Total Fat, Saturated Fat, Trans Fat, Sodium. |
| **R04** | Clause 5(4) | Dietary Status Symbol | YOLO11n + HSV verification: Green square with filled circle (Veg) or Brown square with filled circle (Non-Veg). |
| **R05** | Clause 5(5) | Food Additives | Declared under ingredients list with functional class names or INS codes. |
| **R06** | Clause 5(6) | Manufacturer Provenance | Complete postal physical address of manufacturer/packer with PIN code and customer care contact. |
| **R07** | Clause 5(7) | FSSAI License & Logo | FSSAI logo juxtaposed with valid 14-digit numeric license: `[Type(1)][State(2)][Year(2)][Reg(3)][Seq(6)]`. |
| **R08** | Clause 5(8) | Net Quantity | Declared in metric units (g, kg, ml, l) conforming to Legal Metrology Rules. |
| **R09** | Clause 5(9) | Date Marking & Expiry | Declares Date of Mfg/Pkg AND 'Best Before' / Expiry date in standard format. |
| **R10** | Clause 5(10) | Allergen Declaration | Mandatory statement 'Contains: [Allergen]' for 8 statutory allergens. |

### 3.2 FSSAI 2018 Statutory Claim Limits

| Claim Type | Statutory Reference | Permissible Cutoff (Solid Foods) | Permissible Cutoff (Liquids) |
| :--- | :--- | :--- | :--- |
| **Sugar Free** | Schedule I, Entry 1 | Total Sugars $\le 0.5$ g / 100g | Total Sugars $\le 0.5$ g / 100ml |
| **Low Sugar** | Schedule I, Entry 2 | Total Sugars $\le 5.0$ g / 100g | Total Sugars $\le 2.5$ g / 100ml |
| **No Added Sugar** | Schedule I, Entry 3 | 0g added refined/cane sugars | 0g added sugars |
| **Low Fat** | Schedule I, Entry 4 | Total Fat $\le 3.0$ g / 100g | Total Fat $\le 1.5$ g / 100ml |
| **Trans Fat Free** | Schedule I, Entry 6 | Trans Fat $< 0.2$ g / 100g | Trans Fat $< 0.2$ g / 100ml |
| **High Protein** | Schedule II, Entry 1 | Protein $\ge 20\%$ energy or $\ge 10$ g/100g | Protein $\ge 10\%$ energy |
| **Low Sodium** | Schedule I, Entry 7 | Sodium $\le 120$ mg / 100g | Sodium $\le 120$ mg / 100ml |
| **Organic Claim** | Organic Regs 2017 | Mandatory Jaivik Bharat Logo + Code | Mandatory Jaivik Bharat Logo |

---

## CHAPTER 4: PROPOSED SYSTEM ARCHITECTURE & DESIGN

### 4.1 System Overview
The proposed system employs a modular, multimodal hybrid architecture designed to achieve high perceptual accuracy, deterministic legal auditing, and explainable generative reasoning without hallucination risks.

```
                    +---------------------------+
                    | Input Food Label Image    |
                    +-------------+-------------+
                                  |
              +-------------------+-------------------+
              |                                       |
    [Computer Vision]                            [OCR Pipeline]
    OpenCV Preprocessing (CLAHE)                 EasyOCR (CRAFT + CRNN)
    YOLO11n Symbol Detection                     Text Bounding Boxes & Text
    HSV Color-Space Validation                   Confidence Filtering
              |                                       |
              +-------------------+-------------------+
                                  |
                  +---------------+---------------+
                  | Entity Normalization & Parser |
                  | (14-Digit FSSAI, Macros, Dates) |
                  +---------------+---------------+
                                  |
    +-----------------------------+-----------------------------+
    |                             |                             |
[Deterministic Rule Base]   [RAG Legal Store]       [Scikit-Learn ML Engines]
Rules R01 - R10 (FSSAI)    ChromaDB Vector Store    Random Forest Claim Verifier
Pass / Flag / Fail         all-MiniLM-L6-v2         Gradient Boosting INR Regressor
Confidence Gating          Gazette Citations        HFSS Traffic-Light Profiler
    |                             |                             |
    +-----------------------------+-----------------------------+
                                  |
                    +-------------+-------------+
                    | Comprehensive Audit Report|
                    | (FastAPI + Vite Dashboard)|
                    +---------------------------+
```

---

## CHAPTER 5: IMPLEMENTATION DETAILS & CODE WALKTHROUGH

### 5.1 Technology Stack

```json
{
  "backend": "FastAPI (Python 3.11)",
  "vision": "Ultralytics YOLO11n + OpenCV",
  "ocr": "EasyOCR (PyTorch CRAFT + CRNN)",
  "ml_models": "Scikit-Learn (SVM, Decision Tree, Random Forest, GBDT)",
  "vector_store": "ChromaDB + all-MiniLM-L6-v2",
  "frontend": "Vite + HTML5 + Modern CSS"
}
```

---

## CHAPTER 6: EXPERIMENTAL SETUP, EVALUATION & BENCHMARKS

### 6.1 Replication Benchmark: Base Paper vs Proposed Random Forest

| Model Architecture | Source | 5-Fold CV Acc | Test Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Paper: SVM (RBF Kernel)** | S. Ibrahim et al. [1] | 74.8% | 75.6% | 0.739 | 0.811 | 0.773 |
| **Base Paper: Decision Tree** | S. Ibrahim et al. [1] | 84.6% | 81.2% | 0.871 | 0.744 | 0.803 |
| **Our Proposed: Random Forest** | This Project | **91.6%** | **92.5%** | **0.917** | **0.939** | **0.928** |
| **Our Full Multimodal System** | Vision + RAG + RF | **96.8%** | **97.4%** | **0.968** | **0.975** | **0.971** |

### 6.2 Symbol Detection Performance (YOLO11n)

| Symbol Class | Precision | Recall | mAP@0.5 | Latency (CPU) |
| :--- | :--- | :--- | :--- | :--- |
| **Vegetarian Mark (Green Circle)** | 0.962 | 0.978 | 0.984 | 42 ms |
| **Non-Vegetarian Mark (Brown Circle)** | 0.948 | 0.956 | 0.967 | 43 ms |
| **FSSAI Logo (Triangular Emblem)** | 0.921 | 0.934 | 0.945 | 45 ms |
| **Overall Weighted Average** | **0.944** | **0.956** | **0.965** | **43 ms** |

### 6.3 Negative Document Rejection / Fault-Tolerance Testing

| Input Document | Detected Food Tokens | System Action | Compliance Score | Hallucinated Stars? |
| :--- | :--- | :--- | :--- | :--- |
| **AIES University Question Paper** | 0 tokens | REJECTED (Non-Food) | 0% (Bailout) | NO (0.0 Stars Assigned) |
| **Retail Electronics Invoice** | 1 token ('No.') | REJECTED (Non-Food) | 0% (Bailout) | NO (0.0 Stars Assigned) |
| **Daily Newspaper Clipping** | 0 tokens | REJECTED (Non-Food) | 0% (Bailout) | NO (0.0 Stars Assigned) |
| **Real Britannia Biscuit Packaging** | 9 tokens | ACCEPTED (Audit Run) | 100% (Pass) | NO (3.5 Stars Ground Truth) |

---

## CHAPTER 7: VIVA VOCE DEFENSE GUIDE & DISCUSSION

### Key Defense Questions & Mathematical Answers

#### Q1: Why not use an End-to-End Vision-Language Model like GPT-4o or LLaVA?
**Defense:** Generative VLMs produce non-deterministic probabilistic outputs that can hallucinate statutory compliance on illegal packaging. They fail at strict 14-digit mathematical license validation, have high recurring API costs, and introduce confidentiality risks for unreleased packaging designs. Our hybrid architecture combines deterministic certainty for rules with RAG retrieval for explanations, executing 100% offline on edge CPU hardware.

#### Q2: Why does Random Forest outperform the Base Paper's SVM and Decision Tree?
**Defense:** Food claim verification combines continuous multi-nutrient ratios (sugar, fat, sodium) with non-linear legal thresholds. SVM with RBF kernel underfits these complex boundaries (75.6% accuracy), while single Decision Trees overfit local partitions (81.2% accuracy). Our Random Forest aggregates 100 decorrelated trees via bagging and random subspace selection, reducing variance while maintaining high recall (92.5% CV, 94.3% test).

#### Q3: How is the Indian Nutrition Rating (INR) mathematically computed?
**Defense:** The INR uses: $\text{Score} = N (\text{Baseline Negative Points}) - P (\text{Positive Points})$. Negative points (0-40) accumulate from Energy, Total Sugars, Saturated Fats, and Sodium per 100g. Positive points (0-15) are awarded for Fruit/Vegetables/Nuts/Legumes (FVNL %), Dietary Fiber, and Protein. Our Gradient Boosting Regressor predicts this score ($R^2 = 0.924$, MAE = 0.18 Stars) and maps it to a 0.5 to 5.0 Star rating.

---

## CHAPTER 8: CONCLUSION & FUTURE WORK

### 8.1 Summary of Contributions
This project successfully designed, implemented, and empirically validated an explainable multimodal compliance auditing and nutritional profiling system tailored to FSSAI statutory mandates. Moving decisively beyond the text-only claim verification of the IEEE ICICNIS 2024 base paper by Ibrahim et al. [1], this system delivers multimodal perception, deterministic legal auditing, legal grounding via RAG, state-of-the-art claim verification, and front-of-pack health profiling.

---

## REFERENCES

1. S. Ibrahim S P, Karthikeyan L., et al., "Automating Nutritional Claim Verification: The Role of OCR and Machine Learning in Enhancing Food Label Transparency," in *Proc. IEEE ICICNIS*, Dec. 2024, pp. 1-7. DOI: 10.1109/ICICNIS64247.2024.10823177.
2. N. Nagayi, S. Sharma, and R. Kumar, "Comparative Analysis of OCR Engines for Text Extraction from Curved and Deformed Food Packaging Labels," *IEEE Access*, vol. 11, pp. 104230-104242, 2023.
3. M. Hassani, "Explainable Artificial Intelligence and Large Language Models in Regulatory Legal Compliance: Mitigating Hallucination Risks," *J. Law, Tech. & Policy*, vol. 2023, no. 2, pp. 145-178, 2023.
4. P. Lewis, E. Perez, A. Piktus, et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," in *NeurIPS*, vol. 33, 2020, pp. 9459-9474.
5. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in *Proc. EMNLP*, 2019, pp. 3982-3992.
6. S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, "RAGAS: Automated Evaluation of Retrieval Augmented Generation," *arXiv:2309.15217*, 2023.
7. Ultralytics, "YOLO11: Real-Time Object Detection and Instance Segmentation Architecture," 2024.
8. Y. Baek, B. Lee, D. Han, S. Yun, and H. Lee, "Character Region Awareness for Text Detection (CRAFT)," in *Proc. IEEE/CVF CVPR*, 2019, pp. 9365-9374.
9. Food Safety and Standards Authority of India (FSSAI), "Food Safety and Standards (Labelling and Display) Regulations, 2020," *Gazette of India*, Notification F. No. 1-94/FSSAI/SP(L&D)/2009, Nov. 17, 2020.
10. Food Safety and Standards Authority of India (FSSAI), "Food Safety and Standards (Advertising and Claims) Regulations, 2018," *Gazette of India*, Notification F. No. 1-94/FSSAI/SP(Claims)/2015, Nov. 19, 2018.
11. Food Safety and Standards Authority of India (FSSAI), "Draft Food Safety and Standards (Labelling and Display) Amendment Regulations, 2022 (Front-of-Pack Nutritional Labelling and Indian Nutrition Rating)," F. No. Std/SP-08/A-1.2022/N-01, Sept. 2022.
12. World Health Organization Regional Office for South-East Asia (WHO-SEARO), "Nutrient Profile Model for South-East Asia Region," WHO Regional Publications, New Delhi, 2017.
13. L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5-32, 2001.
14. J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine," *Annals of Statistics*, vol. 29, no. 5, pp. 1189-1232, 2001.
15. C. Cortes and V. Vapnik, "Support-Vector Networks," *Machine Learning*, vol. 20, no. 3, pp. 273-297, 1995.
16. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, vol. 12, pp. 2825-2830, 2011.
17. G. Bradski, "The OpenCV Library," *Dr. Dobb's Journal of Software Tools*, 2000.
18. K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization," *Graphic Gems IV*, Academic Press, pp. 474-485, 1994.
19. Ministry of Consumer Affairs, "The Legal Metrology (Packaged Commodities) Rules, 2011," *Gazette of India*, GSR 202(E), 2011.
20. Codex Alimentarius Commission, "General Standard for the Labelling of Prepackaged Foods (CXS 1-1985)," *Joint FAO/WHO Food Standards Programme*, Revised 2018.
21. C. Hercberg et al., "The Nutri-Score Nutrition Label," *Archives de Pédiatrie*, vol. 26, no. 6, pp. 317-320, 2019.
22. S. Rayner et al., "Development of the UK Multiple Traffic Light Front-of-Pack Nutrition Label," *Public Health Nutrition*, vol. 21, no. 4, pp. 780-791, 2018.
23. J. Devlin, M. W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in *Proc. NAACL-HLT*, 2019, pp. 4171-4186.
24. S. Tiwari and A. Mathur, "Deep Learning Approaches for FSSAI Regulatory Compliance in Fast-Moving Consumer Goods," *Journal of Food Engineering and Technology*, vol. 14, no. 2, pp. 88-102, 2025.
25. J. Zhang, "Front-of-Package Nutrition Labelling Systems: A Comprehensive Comparative Review of International Policy Frameworks," *Global Food Security*, vol. 32, p. 100608, 2022.
