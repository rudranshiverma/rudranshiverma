# Rudranshi Verma

Final-year B.Tech Computer Science student (graduating 2027), building applied AI/ML systems with a focus on audio, NLP, and reliability of ML pipelines in production. 


![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![NLP](https://img.shields.io/badge/NLP-4B8BBE?style=flat-square)
![Audio ML](https://img.shields.io/badge/Audio%20ML-4B8BBE?style=flat-square)
![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white)



## Projects

### AudioInsight - Deepfake Audio Detection
Dual-model system detecting AI-generated speech, built to compare a spectral approach against a self-supervised one on the same task rather than just ensembling them.

- ResNet18 on mel spectrograms + Wav2Vec2 fine-tuned on raw waveforms
- Trained on ASVspoof 2019 LA, extended with fine-tuning on modern TTS-generated audio
- GradCAM and attention-weight visualization for interpretability
- EER ≈ 0.045 (ResNet18), ≈ 0.041 (Wav2Vec2), PR-AUC ≈ 0.9983
- Deployed on Hugging Face Spaces

**[Repo](https://github.com/rudranshiverma/audioinsight-deepfake-audio-detection)** · **[Demo](https://huggingface.co/spaces/rudranshiverma/audioinsight-deepfake-audio-detection)**


### Fraud-Spike Detector (Razorpay Buildathon)
One-week build distinguishing card-testing attacks from legitimate festive demand spikes (e.g. Diwali, Raksha Bandhan) on UPI/e-commerce payment volume.

- Gradient-boosted classifier on engineered features: calendar-aware seasonal residuals, CUSUM/EWMA statistics, decline rate, BIN/VPA concentration
- CUSUM/EWMA-only detection kept as an ablation baseline
- SHAP for attribution, isotonic/Platt calibration on output scores
- Synthetic data generator calibrated for realism against public fraud datasets (ULB, IEEE-CIS)

**[Repo](https://github.com/rudranshiverma/surge-shield)**

### CertainRAG - Runtime Uncertainty Quantification for RAG
Python library that attaches a trust signal to RAG outputs via retrieval quality and faithfulness scores. Work in progress.

- Two signals: retrieval confidence (rank-weighted aggregation) and faithfulness (LLM-as-judge, evidence-before-verdict prompting)
- Faithfulness AUC ≈ 0.82 on SQuAD, also evaluated on PubMedQA

**[Repo](https://github.com/rudranshiverma/CertainRAG)**

### Book Recommendation System
Content-based recommender using semantic similarity instead of keyword/genre matching.

- Sentence Transformer embeddings for meaning-level similarity across descriptions, genres, and authors
- Popularity-aware re-ranking (rating average, log-scaled rating count) to avoid obscure/low-signal picks
- Intent-alignment step to reduce genre leakage, with optional strict genre filter
- Rule-based explanation per recommendation instead of a black-box score
- Streamlit UI with search-as-you-type and preference toggles

**[Repo](https://github.com/rudranshiverma/book-recommendation-system)** · **[Demo](https://book-recommendation-app0310.streamlit.app/)**
  

## Contact

**[Linkedin](https://www.linkedin.com/in/rudranshi-verma/)**
[Email](mailto:verma.rudranshi03@gmail.com)
