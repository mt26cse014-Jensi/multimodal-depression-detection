PROJECT: Multimodal (voice + text) depression detection with deep learning.
Datasets: EATD-Corpus (Mandarin, 162 participants, short answers) and DAIC
WOZ
(English, 189 interview sessions). They are evaluated SEPARATELY with the 
same
architecture. Never merge them into one training set.
HARDWARE: [CPU model, RAM, GPU or none]
PYTHON: [Python 3.12.4]
GOAL: compare text-only, audio-only and fused models on both datasets, with
explainability, as a research prototype (not a diagnostic tool).
ARCHITECTURE: frozen pretrained encoders (text: BERT/MentalBERT, Chinese BERT 
for
EATD; audio: Wav2Vec2 or WavLM, 16 kHz) -> cached embeddings -> small fusion 
head
(baseline: concatenation; advanced: cross-modal attention) -> binary label 
and,
on DAIC-WOZ, PHQ-8 regression.
HARD RULES:
1. Never modify or delete anything in data/raw/. Write outputs to 
data/processed/.
2. Split by PARTICIPANT, never by window or segment. No participant may 
appear in
more than one of train/dev/test (or more than one CV fold's test set).
3. DAIC-WOZ: use only participant speech for headline results. Interviewer 
(Ellie)
text and audio must be excluded, except in the explicit ablation.
4. Never tune hyperparameters on the test set. Use the official DAIC-WOZ splits.
5. Fix random seeds, save every config to configs/, and log all metrics to 
results/.
6. DAIC-WOZ data is license-restricted: inspect file structure and at most a 
few
lines of text. Never print or paste full transcripts. Never upload data 
anywhere.
7. Report macro F1 AND depressed-class F1, accuracy, AUC-ROC (and RMSE for 
PHQ-8).
8. Before any full run, do a smoke test on 3-5 participants and show me the 
result.
9. Write small, documented, reusable functions in src/. Add a short test for 
every
data-loading or splitting function.
10. If something about the data is unclear (file names, label columns, 
thresholds),
STOP and ask me. Do not guess.
WORKFLOW: for each phase, first write an implementation plan, wait for 
approval,
then implement, then report what you did, what you verified, and any open 
issues.