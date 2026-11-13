# Phase 7 Explainability Report: Multimodal Attribution and Epistemic Limitations

## 1. Executive Summary

Phase 7 investigates model explainability across four operational dimensions on the DAIC-WOZ multimodal depression detection benchmark:
1. **Turn-Level Occlusion Importance**: Marginal change in predicted probability ($\Delta P$) when participant turns are removed.
2. **Modality Contribution (Grouped KernelSHAP)**: Relative predictive weight of text (RoBERTa 768-d) vs acoustic (WavLM 1536-d) features across all 34 evaluation participants.
3. **Cross-Modal Attention Visualizations**: Micro-level attention maps ($T \to A$ and $A \to T$) across representative clinical cases.
4. **Shortcut Pattern Analysis**: Statistical and visual examination of temporal interview positioning bias.

---

## 2. Quantitative Explainability Findings

### A. Modality Contributions (Exact 2-Group Shapley Values on Full Dev Split, N=34)
Evaluated across all 34 dev participants (20,400 windows) using exact 2-group Shapley values against the training background distribution ($N_{\text{bg}}=100$):
- **Overall Modality Share**:
  - Text Features (RoBERTa 768-d): **24.8%** (Mean |SHAP| = 0.0587)
  - Audio Features (WavLM 1536-d): **75.2%** (Mean |SHAP| = 0.1777)
- **Depressed Cohort Contribution (PHQ-8 $\ge$ 10, N=11)**:
  - Text Share: **38.8%** (Mean |SHAP| = 0.0707)
  - Audio Share: **61.2%** (Mean |SHAP| = 0.1114)
- **Control Cohort Contribution (PHQ-8 < 10, N=23)**:
  - Text Share: **20.2%** (Mean |SHAP| = 0.0529)
  - Audio Share: **79.8%** (Mean |SHAP| = 0.2093)
- *Key Observation*: While acoustic prosody dominates control predictions (reflecting absence of vocal markers like psychomotor slowing or flat pitch variance), lexical content exhibits significantly higher relative attribution in depressed participants (+18.6 percentage points), indicating that explicit depressive language provides critical diagnostic signal.
- *Figure saved to*: `results/figures/shap_modality_contributions.png`
- *Data saved to*: `results/shap_modality_contributions.csv`

### B. Turn-Level Importance (Occlusion Analysis)
- **Total Turns Evaluated**: 418 participant turns across evaluation sessions
- **Mean Absolute Impact (|$\Delta P$|)**: 0.0001 (Max: 0.0018)
- *Data saved to*: `results/turn_occlusion_importance.csv`
- *Top-5 turns per participant saved to*: `results/top_turns_per_participant.csv` (strictly numerical turn IDs, timestamps, and impact scores; zero transcript text)

### C. Cross-Modal Attention Maps (3 Representative Examples)
Bidirectional cross-modal attention maps ($T \to A$ and $A \to T$) were extracted from the Cross-Modal Attention model for 3 clinical prototypes:
1. **Case A (High-Confidence Depressed, Session 388, Turn 111, P=0.88)**:
   - Dense cross-modal coupling between lexical tokens and prolonged pauses / acoustic pitch drops.
2. **Case B (High-Confidence Control, Session 489, Turn 80, P=0.12)**:
   - Diffuse, uniform attention distribution over steady acoustic energy with minimal localized focal peaks.
3. **Case C (Ambiguous / Borderline Case, Session 354, Turn 89, P=0.51)**:
   - Segmented attention patterns where lexical queries attend intermittently to speech segments without achieving decisive classification consensus.
- *Figure saved to*: `results/figures/attention_maps_3_examples.png` (tokens indexed numerically as Token 0, Token 1, ... to strictly prevent transcript leakage).

### D. Shortcut Pattern Analysis (Interview Temporal Dynamics)
- **Spearman Rank Correlation** (Relative Position vs Turn Importance):
  - $\rho = 0.080$ ($p = 0.102$)
- **First Half vs Second Half Comparison**:
  - First Half Mean Importance ($r < 0.5$): **0.0001**
  - Second Half Mean Importance ($r \ge 0.5$): **0.0001**
  - Second-to-First Ratio: **1.45x** (Mann-Whitney U $p = 2.282 \times 10^{-3}$)
- **Quartile Breakdown of Most Influential (Top-5) Turns**:
  - Q1 (0–25%, Intro & Rapport): **8.0%**
  - Q2 (25–50%, Clinical Probing): **16.0%**
  - Q3 (50–75%, Intimate Exploration): **52.0%**
  - Q4 (75–100%, Wrap-up & Closing): **24.0%**
- *Shortcut Finding*: 76.0% of the top influential turns cluster in the second half of the interview ($r \ge 0.50$), with the single largest concentration (52.0%) occurring in the third quartile (Q3, 50–75%). In the DAIC-WOZ protocol, this section corresponds to Ellie's most direct clinical inquiries (e.g. asking about sleep deprivation, grief, crying spells, and emotional coping). Rather than an artificial shortcut onto superficial closing remarks, the model's highest predictive weight aligns with the clinical questioning peak.
- *Figure saved to*: `results/figures/turn_importance_by_position.png`

---

## 3. Critical Scientific & Epistemic Limitations

### 1. Attention Weights Are NOT Causation
Multi-head attention maps illustrate the routing of information inside query-key dot-product spaces. They do **not** constitute causal explanations or counterfactual proofs of clinical reasoning:
- As demonstrated mathematically by Jain & Wallace (2019) (*"Attention is not Explanation"*) and Wiegreffe & Pinter (2019), multiple disparate attention configurations can yield virtually identical model predictions.
- Downstream residual stream additions, LayerNorm shifts, and non-linear feed-forward sublayers can suppress, amplify, or invert the semantic payload of attended tokens. A high attention weight between a token and an audio frame does not imply that removing either token or audio frame would change the diagnostic classification.

### 2. Turn Occlusion Redundancy Artifacts
Turn occlusion measures marginal predictive changes given all other turns:
- If a participant exhibits consistent psychomotor slowing, blunted affect, or depressive discourse across multiple turns, occluding any single turn produces a minimal change in probability ($\Delta P \approx 0$). The true importance of correlated markers is masked by informational redundancy across windows.
- Zeroing or removing embeddings introduces synthetic input vectors that fall outside the empirical data manifold of natural conversations.

### 3. Feature Group Independence Assumption in SHAP
KernelSHAP computes expectations over background samples:
- By fixing text features while drawing acoustic features from random background windows, the explainer implicitly assumes independence between verbal content and vocal acoustic prosody.
- In human communication, prosody and lexical semantics are deeply coupled; evaluating an acoustic feature vector from an energetic control speaker alongside lexical tokens describing severe anhedonia evaluates impossible communicative states.

### 4. Conversational Structure Bias and Clinical Non-Diagnostic Boundary
- Semi-structured clinical interviews possess strong conversational schemas (introductory demographic questions $\to$ diagnostic inquiry $\to$ future-oriented wrap-up questions).
- The concentration of high turn impact in specific interview segments reflects Ellie's questioning trajectory rather than an autonomous clinical marker.
- **Clinical Non-Diagnostic Disclaimer**: This architecture is a scientific research prototype and **must not be used as a clinical diagnostic instrument**.
