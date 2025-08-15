# HoneyAI 🐝

HoneyAI is a **distributed honeypot threat analysis platform** designed to automatically collect, ingest, and analyze cybersecurity attack data from honeypots. The platform detects malicious behavior, classifies attacker types, and produces actionable threat intelligence. HoneyAI combines **automated data ingestion, static and dynamic analysis, and machine learning** to provide a comprehensive view of cyber threats.  

---

## 🚀 Features  

- **Honeypot Deployment**: Supports multiple honeypots (e.g., Cowrie) on distributed EC2 instances.  
- **Automated Ingestion**: Processes JSON, TTY logs, and metadata into structured datasets.  
- **Static & Dynamic Analysis**: Extracts commands, session behavior, and attacker fingerprints.  
- **Machine Learning Pipeline**:  
  - Recon vs. Attack Classifier (multi-class with risk scoring).  
  - Attacker Profile Identifier using unsupervised clustering (KMeans, HDBSCAN).  
  - Next-action prediction based on historical patterns.  
- **Data Visualization**: Session replay, frequency heatmaps, and behavioral summaries.  
- **Extensible & Modular**: Add new honeypot sources, analysis scripts, or ML models with minimal effort.  



---

## ⚙️ Installation  

1. Clone the repository:  
bash
git clone https://github.com/<username>/HoneyAI.git
cd HoneyAI


## Dependencies:

python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
pip install -r requirements.txt

📊 Visualization

Session Replays: outputs/tty_analysis/
Heatmaps & Frequency Plots: outputs/static_features/
Behavior Clusters: outputs/predictions/
Jupyter notebooks provide interactive visualization for rapid analysis.

🧩 Code Design

Modular: Separate modules for ingestion, analysis, ML, and prediction.
Scalable: Designed to process multiple EC2 honeypots in parallel.
Reproducible: Metadata and logging included for every session and feature.
Extensible: Add new models, scripts, or honeypot sources without changing the core pipeline.

⚡ Future Work

LLM Integration: Natural language summarization of attack patterns and automated recommendations.
Real-Time Detection: Streaming ingestion from honeypots with on-the-fly classification.
Threat Graphs: Visual representation of attacker tactics, techniques, and procedures.
Distributed Processing: Multi-node, GPU-accelerated analysis for large-scale deployments.
Cross-Honeypot Fusion: Combine data from multiple honeypot types to improve coverage and detection accuracy.
