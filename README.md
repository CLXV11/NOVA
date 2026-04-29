Nova - Facebook Engagement Analyzer

https://img.shields.io/badge/Python-3.8%2B-blue.svg
https://img.shields.io/badge/License-MIT-yellow.svg
https://img.shields.io/badge/version-3.0.0-brightgreen.svg

Hybrid ML-Powered Fake Engagement Detection - Professional tool that combines Facebook's official Graph API with machine learning to identify suspicious accounts interacting with posts.

📸 Screenshots

```
╔══════════════════════════════════════════════════════════════╗
║   ███╗   ██╗ ██████╗ ██╗   ██╗ █████╗                       ║
║   ████╗  ██║██╔═══██╗██║   ██║██╔══██╗                      ║
║   ██╔██╗ ██║██║   ██║██║   ██║███████║                      ║
║   ██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║                      ║
║   ██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║                      ║
╚══════════════════════════════════════════════════════════════╝
   Facebook Engagement Analyzer - Professional Edition v3.0
   Developed by: Emaf-png | Hybrid ML + Graph API Approach
```

🎯 Overview

Nova is a professional-grade tool designed for security researchers, social media analysts, and penetration testers to analyze Facebook post engagement patterns. It goes beyond simple rule-based detection by employing:

· Machine Learning: Isolation Forest anomaly detection on 22 behavioral features
· NLP Analysis: Natural language processing for name authenticity verification
· Graph API Integration: Official Facebook API for reliable data access (no scraping)
· Weighted Scoring Engine: Multi-dimensional risk assessment with confidence metrics

✨ Key Features

🔍 Intelligent Analysis

· 22-Dimensional Feature Extraction: Profile completeness, network metrics, temporal patterns, linguistic quality
· ML Anomaly Detection: Automatically identifies statistical outliers using Isolation Forest
· NLP Name Verification: Checks if profile names contain real words using NLTK
· Behavioral Pattern Recognition: Posting frequency, friend-to-follower ratios, activity regularity

📊 Professional Scoring

· Weighted Risk Score (0-100): Combines 6 analysis categories with configurable weights
· Confidence Metrics: Reports how reliable each classification is
· Risk Levels: 5-tier classification (Legitimate → Critical)
· ASCII Visualization: Real-time risk distribution charts in terminal

🚀 Performance

· Batch API Processing: Efficient bulk profile fetching (50 profiles/request)
· Intelligent Caching: Avoids redundant API calls
· Graceful Degradation: Works without ML/NLP libraries with reduced features
· Rate Limit Handling: Respects Facebook API limits automatically

🛡️ Security & Privacy

· Official API Only: No scraping, no cookies, no password required
· Local Analysis: All data processed on your machine
· Minimal Permissions: Only requires pages_read_engagement scope
· No Data Storage: Only saves reports if explicitly requested

📋 Requirements

Core Dependencies

```bash
pip install requests colorama tqdm
```

Enhanced Features (Recommended)

```bash
pip install numpy scikit-learn nltk
python -c "import nltk; nltk.download('words'); nltk.download('punkt')"
```

Component Required Purpose
requests ✅ Required HTTP client for API calls
colorama ✅ Required Terminal coloring
tqdm ✅ Required Progress bars
numpy ⭐ Recommended Numerical operations
scikit-learn ⭐ Recommended ML anomaly detection
nltk ⭐ Recommended NLP name analysis

🔧 Installation

```bash
# Clone repository
git clone https://github.com/Emaf-png/nova.git
cd nova

# Install dependencies
pip install -r requirements.txt

# For full ML capabilities
pip install -r requirements-full.txt
```

🚀 Quick Start

1. Get Facebook Access Token

1. Visit Graph API Explorer
2. Select API Version: v18.0
3. Add permissions: pages_read_engagement, pages_read_user_content
4. Click "Generate Access Token"

2. Run Analysis

Basic Scan (Rule-based only):

```bash
python nova.py --token "YOUR_ACCESS_TOKEN" --url "https://www.facebook.com/post_url"
```

Deep Scan (With ML + NLP):

```bash
python nova.py \
  --token "YOUR_ACCESS_TOKEN" \
  --url "https://www.facebook.com/post_url" \
  --deep \
  --max 1000 \
  --output report.json
```

3. Read Results

```
📊 RISK DISTRIBUTION
   ✅ LEGITIMATE     [ 342] ████████████████████████████████
   🟢 LOW_RISK       [  98] ████████
   🟡 MEDIUM_RISK    [  45] ████
   🟠 HIGH_RISK      [  12] ██
   🔴 CRITICAL       [   3] █

⚠️  TOP SUSPICIOUS ACCOUNTS
   🔴 #1 | Score: 92/100 | Confidence: 87%
   Name: xX_DarkShadow_Xx
   Profile: https://facebook.com/profile?id=123456789
   Flags: No profile picture, Suspicious name pattern, Account 2 days old
```

📖 Usage

Command Line Arguments

Argument Required Description Default
--token ✅ Yes Facebook Graph API Access Token -
--url ✅ Yes Facebook post URL to analyze -
--deep No Enable deep ML/NLP analysis False
--max No Maximum reactions to analyze 500
--output No Save report as JSON file -

Examples

Quick scan of recent post:

```bash
python nova.py --token "EAAB..." --url "https://fb.com/photo.php?fbid=123"
```

Comprehensive analysis with export:

```bash
python nova.py --token "EAAB..." --url "https://fb.com/posts/456" --deep --output analysis.json
```

Batch analysis (via script):

```bash
#!/bin/bash
while read url; do
  python nova.py --token "$TOKEN" --url "$url" --output "report_$(date +%s).json"
done < posts.txt
```

🏗️ Architecture

```
Nova v3.0 Architecture
├── Config Layer
│   └── Config, RiskLevel, ProfileFeatures
├── Data Access Layer  
│   └── FacebookGraphAPI (Official API)
├── Feature Engineering
│   ├── MLAnalyzer (22 features)
│   └── NLP Analysis (Name verification)
├── Scoring Engine
│   ├── Weighted Scoring (6 categories)
│   ├── Isolation Forest (Anomaly detection)
│   └── Confidence Calculation
└── Reporting
    ├── Terminal UI (ASCII charts)
    └── JSON Export
```

🔬 Methodology

Analysis Categories & Weights

Category Weight Features
Profile Completeness 25% Picture, cover, bio, education, work, location, relationship
Account Age 15% Days since creation, temporal patterns
Network Metrics 20% Friends, followers, ratios
Name Quality 15% Entropy, numbers, real words (NLP)
Activity Patterns 15% Posts, photos, frequency
ML Anomaly 10% Statistical outlier detection

Risk Classification

Level Score Range Description
✅ Legitimate 0-20 Normal account, no red flags
🟢 Low Risk 21-40 Minor suspicious patterns
🟡 Medium Risk 41-60 Multiple concerning signals
🟠 High Risk 61-80 Strong indicators of fake activity
🔴 Critical 81-100 Almost certainly fake/bot account

📊 JSON Report Structure

```json
{
  "analysis_timestamp": "2026-04-29T15:30:45",
  "total_reactions": 500,
  "statistics": {
    "suspicious_accounts": 15,
    "suspicious_percentage": 3.0,
    "risk_distribution": {
      "LEGITIMATE": 342,
      "LOW_RISK": 98,
      "MEDIUM_RISK": 45,
      "HIGH_RISK": 12,
      "CRITICAL": 3
    },
    "average_confidence": 0.82
  },
  "suspicious_accounts": [...],
  "methodology": {
    "approach": "Hybrid ML + Rule-based scoring",
    "features_analyzed": 22
  }
}
```

⚠️ Limitations

· API Dependency: Requires valid Facebook access token
· Permission Scope: Limited to pages/content you can access
· Privacy Restrictions: Cannot access private profiles
· Rate Limits: Subject to Facebook API rate limiting
· False Positives: ML models require diverse training data for optimal accuracy

🗺️ Roadmap

· Real-time monitoring mode
· Historical trend analysis
· Browser extension for one-click analysis
· Pre-trained models for common bot patterns
· Multi-platform support (Instagram, Twitter)
· Community-contributed heuristics database

👨‍💻 Author

Emaf-png

· GitHub: @Emaf-png

📄 License

This project is licensed under the MIT License - see LICENSE file for details.

⚡ Acknowledgments

· Facebook Graph API Team
· scikit-learn community
· NLTK Project
· All contributors and testers

---

⭐ If you find Nova useful, please star this repository!

⚠️ Disclaimer: This tool is for legitimate security research and analysis purposes only. Always comply with Facebook's Terms of Service and obtain proper authorization before analyzing content.
