# 🎯 Succession Planning Engine

A fully interactive **Succession Planning POC** built in Python Streamlit, powered by HRMS data, **Korn Ferry KFALP**, and **Korn Ferry viaEdge** psychometric assessments.

---

## 🚀 Live Demo

Deploy instantly on **Streamlit Community Cloud** (free):

[![Deploy on Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)

---

## 📦 Features

| Tab | What it does |
|-----|-------------|
| 🏆 **Succession Pipeline** | Select any critical role → see top-3 ranked successors with LPS gauges, cluster bars, and pipeline comparison |
| 👤 **Employee Profile** | Full candidate deep-dive — speedometers, colour-coded sliders, KFALP + viaEdge radar charts |
| ⚖️ **Compare Employees** | Multi-employee comparison with overlaid sliders and radar overlay |
| 🌐 **Org Chart** | Interactive hierarchy tree built from org_structure.csv — hover for LPS and grade info |
| 📊 **Org Readiness** | Bench strength heatmap, LPS band distribution, 9-box summary, KPI banner |
| 🧠 **KF Assessment** | KFALP and viaEdge dimension explorer with heatmaps, histograms, and full behavioural descriptors |
| 📈 **Career Path** | Individual promotion timeline (grade + performance dual-axis), organisation-wide velocity scatter |

### Leadership Potential Score (LPS)
Configurable weighted average of 5 clusters via sidebar sliders:
- **Performance** (default 25%) — ratings, trajectory
- **KF Assessment** (default 30%) — KFALP + viaEdge blended composite
- **Career Velocity** (default 20%) — promotions per year, last-5yr momentum
- **Leadership Breadth** (default 15%) — cross-functional, international, critical projects
- **Readiness & Mobility** (default 10%) — grade gap, mobility willingness, flight risk

---

## 📁 Required CSV Files

Upload all 7 files via the sidebar:

| File | Description |
|------|-------------|
| `employees_master.csv` | 420 employees × 64 features |
| `succession_pipeline.csv` | Top-3 successors per critical role |
| `kf_kfalp_detail.csv` | KFALP scores + behavioural descriptors per employee per dimension |
| `kf_viaedge_detail.csv` | viaEdge scores + descriptors per employee per dimension |
| `kf_attribute_reference.csv` | Full KF attribute reference guide (all bands, all dimensions) |
| `promotion_history.csv` | One row per promotion event — year, grade, performance |
| `org_structure.csv` | Hierarchy for interactive org chart |

---

## 🛠 Local Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/succession-planning-engine.git
cd succession-planning-engine

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

## ☁️ Deploy to Streamlit Cloud (Free)

1. Fork this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your fork → set **Main file path** to `app.py`
4. Click **Deploy** — live in ~2 minutes
5. Upload your 7 CSV files via the sidebar

---

## 🧱 Tech Stack

| Layer | Library |
|-------|---------|
| Frontend | Streamlit 1.32+ |
| Visualisation | Plotly 5.x |
| Data | Pandas, NumPy |
| Network/Org | NetworkX |
| Fonts | Syne + DM Sans (Google Fonts) |

---

## 📊 Dataset Design

Synthetic data generated with:
- **Age − Tenure = Career Start Age always 21–23**
- **Promotions never exceed Grade − 1** (hard constraint)
- **Performance ↔ Promotions correlated** (r ≈ 0.70)
- **CTC strictly within grade band**
- **KF scores correlated with grade + performance** (r ≈ 0.65–0.72)
- All KF terminology exact (Korn Ferry KFALP and viaEdge official dimensions)

---

## 🗂 Project Structure

```
succession-planning-engine/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── data/                   # (optional) place CSV files here for local dev
```

---

## 🔐 Notes

- No data is stored — all uploads are session-only
- Safe for synthetic/demo data; for live HRMS data, deploy on private Streamlit Cloud or internal server
- LPS weights recalculate instantly on every slider change

---

*Built with Korn Ferry KFALP and viaEdge assessment frameworks.*
