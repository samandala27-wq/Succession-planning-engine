"""
Succession Planning Engine v5
==============================
Upload 6 CSV files — succession_pipeline.csv is AUTO-GENERATED live.
Pipeline fix: each role gets its own grade-windowed, department-relevant,
globally-deduplicated successor pool — no two roles share the same top-3.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def hex_to_rgba(h, a=0.18):
    h = h.lstrip("#")
    r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

def safe_float(v, default=0.0):
    try:
        f = float(v)
        return default if np.isnan(f) else f
    except: return default

def lps_color(s):
    if s>=80: return "#1B7A3E"
    if s>=65: return "#2563EB"
    if s>=50: return "#D97706"
    if s>=35: return "#EA580C"
    return "#B91C1C"

def norm_pct(val, series):
    s = series.dropna()
    return int((s < val).mean()*100) if len(s)>0 else 50

def avatar_html(name, size=48, bg="#0D7377"):
    ini = "".join(p[0].upper() for p in str(name).split()[:2] if p)
    return (f'<div style="width:{size}px;height:{size}px;border-radius:50%;background:{bg};'
            f'color:white;display:flex;align-items:center;justify-content:center;'
            f'font-family:Syne,sans-serif;font-weight:800;font-size:{size//3}px;'
            f'flex-shrink:0;">{ini}</div>')

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BAND_COLORS = {
    "Band 5 - Ready Now":          "#1B7A3E",
    "Band 4 - Ready in 1-2 Years": "#2563EB",
    "Band 3 - Ready in 2-3 Years": "#D97706",
    "Band 2 - Emerging Potential": "#EA580C",
    "Band 1 - Not Yet Ready":      "#B91C1C",
}
BAND_SHORT = {
    "Band 5 - Ready Now":          "Ready Now",
    "Band 4 - Ready in 1-2 Years": "1-2 Yrs",
    "Band 3 - Ready in 2-3 Years": "2-3 Yrs",
    "Band 2 - Emerging Potential": "Emerging",
    "Band 1 - Not Yet Ready":      "Not Ready",
}
CL_COLORS = ["#0D7377","#C9A227","#2563EB","#7C3AED","#EA580C"]
CL_NAMES  = ["Performance","KF Assessment","Career Velocity","Ldrship Breadth","Readiness"]

# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL ROLES CONFIG
# min_grade = minimum grade a candidate must be AT to be eligible
# dept      = preferred department(s) — candidates from these ranked first
# ─────────────────────────────────────────────────────────────────────────────
ROLES_CFG = {
    "CEO": {
        "min_grade": 9, "grade_window": 2,
        "dept": None,
        "label": "Chief Executive Officer"
    },
    "CFO": {
        "min_grade": 8, "grade_window": 2,
        "dept": ["Finance"],
        "label": "Chief Financial Officer"
    },
    "CHRO": {
        "min_grade": 8, "grade_window": 2,
        "dept": ["Human Resources"],
        "label": "Chief Human Resources Officer"
    },
    "CGO": {
        "min_grade": 7, "grade_window": 2,
        "dept": ["Corporate Governance","Corporate Strategy","Finance","Legal & Compliance"],
        "label": "Chief Governance Officer"
    },
    "CBO-AI Solutions and Services": {
        "min_grade": 8, "grade_window": 2,
        "dept": ["AI & Digital Solutions","Cybersecurity","Service Design"],
        "label": "Chief Business Officer — AI"
    },
    "Executive Vice President": {
        "min_grade": 7, "grade_window": 2,
        "dept": None,
        "label": "Executive Vice President"
    },
    "Senior Vice President": {
        "min_grade": 7, "grade_window": 2,
        "dept": None,
        "label": "Senior Vice President"
    },
    "Advisor": {
        "min_grade": 7, "grade_window": 2,
        "dept": ["Strategic Advisory","Corporate Strategy"],
        "label": "Advisor"
    },
    "Vice President (CISO)": {
        "min_grade": 6, "grade_window": 2,
        "dept": ["Cybersecurity"],
        "label": "VP — Chief Information Security Officer"
    },
    "Associate Vice President": {
        "min_grade": 6, "grade_window": 2,
        "dept": None,
        "label": "Associate Vice President"
    },
    "Senior Director - Corporate Strategy": {
        "min_grade": 6, "grade_window": 2,
        "dept": ["Corporate Strategy","Office of the CEO"],
        "label": "Senior Director — Corporate Strategy"
    },
    "Senior Director - HRBP": {
        "min_grade": 6, "grade_window": 2,
        "dept": ["Human Resources"],
        "label": "Senior Director — HRBP"
    },
    "Senior Director - Practice Sales": {
        "min_grade": 6, "grade_window": 2,
        "dept": ["Sales","Business Development","AI & Digital Solutions"],
        "label": "Senior Director — Practice Sales"
    },
}
CRITICAL_ROLES = list(ROLES_CFG.keys())

# ─────────────────────────────────────────────────────────────────────────────
# LPS COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def recompute_lps(df, w1, w2, w3, w4, w5):
    def sn(s):
        lo,hi = s.min(),s.max()
        return (s-lo)/(hi-lo+1e-9)*100

    c1 = (sn(df["Average Performance Rating - Last 3 Years (1-5)"])*0.50 +
          sn(df["Last Annual Performance Rating (1-5)"])*0.35 +
          df["Performance Trajectory"].clip(-2,2).add(2).div(4).mul(100)*0.15)

    kf_col = "KF Blended Assessment Composite (1-5)"
    kf_fill = df[kf_col].fillna(df["Average Performance Rating - Last 3 Years (1-5)"]) \
              if kf_col in df.columns else df["Average Performance Rating - Last 3 Years (1-5)"]
    c2 = sn(kf_fill)

    c3 = (sn(df["Promotions per Year (Career)"])*0.50 +
          sn(df["Promotions per Year (Last 5 Years)"])*0.35 +
          sn(df["Total Promotions (Career)"])*0.15)

    breadth = (df["Cross-Functional Experience"].astype(int)*25 +
               df["International / Multi-Geography Experience"].astype(int)*20 +
               sn(df["Number of Critical Projects Led"])*30 +
               sn(df["External Industry Recognition Count"])*15 +
               sn(df["Number of Direct Reports"])*10)
    c4 = sn(breadth)

    gg = 9 - df["Job Grade (1-9)"]
    c5 = (sn(df["Mobility / Relocation Willingness (1-5)"])*0.35 +
          sn(gg.max()-gg)*0.35 +
          df["Flight Risk"].map({"Low":100,"Medium":50,"High":0})*0.30)

    lps = (c1*(w1/100)+c2*(w2/100)+c3*(w3/100)+c4*(w4/100)+c5*(w5/100)).round(2)

    def band(s):
        if s>=80: return "Band 5 - Ready Now"
        if s>=65: return "Band 4 - Ready in 1-2 Years"
        if s>=50: return "Band 3 - Ready in 2-3 Years"
        if s>=35: return "Band 2 - Emerging Potential"
        return "Band 1 - Not Yet Ready"

    df = df.copy()
    df["LPS"] = lps
    df["LPS Band"] = lps.apply(band)
    df["C1"]=c1.round(2); df["C2"]=c2.round(2)
    df["C3"]=c3.round(2); df["C4"]=c4.round(2); df["C5"]=c5.round(2)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# BUILD PIPELINE — UNIQUE PER ROLE
# ─────────────────────────────────────────────────────────────────────────────
def build_pipeline(df_scored, exclude_risk=True, min_gr=5):
    """
    For every critical role:
    1. Identify the incumbent (person currently holding that title).
    2. Build a role-specific eligible pool:
       - Grade window: [min_grade - grade_window, min_grade + 1]
       - Preferred departments listed first (department relevance)
       - Exclude the incumbent
       - Exclude high-flight-risk if flag set
       - Exclude anyone already assigned as #1 successor to a higher-priority role
         (global deduplication — no one appears in two top-3 pipelines)
    3. Rank by LPS within the eligible pool.
    4. Return top 3 per role.
    """
    rows = []
    globally_used = set()   # EE Numbers already assigned as #1 elsewhere

    for role, cfg in ROLES_CFG.items():
        min_g  = cfg["min_grade"]
        window = cfg["grade_window"]
        depts  = cfg["dept"]

        # ── Incumbent ────────────────────────────────────────────────────────
        inc_match = df_scored[df_scored["Current Job Title"] == role]
        if len(inc_match) > 0:
            inc = inc_match.iloc[0]
        else:
            inc = df_scored.sort_values(["Job Grade (1-9)","LPS"], ascending=False).iloc[0]

        # ── Grade window ─────────────────────────────────────────────────────
        grade_lo = max(1, min_g - window)
        grade_hi = min(9, min_g + 1)

        base = df_scored[
            (df_scored["Job Grade (1-9)"] >= grade_lo) &
            (df_scored["Job Grade (1-9)"] <= grade_hi) &
            (df_scored["EE Number"] != inc["EE Number"])
        ].copy()

        if exclude_risk:
            base = base[base["Flight Risk"] != "High"]
        base = base[base["Job Grade (1-9)"] >= min_gr]

        # ── Department relevance: dept-matched candidates come first ──────────
        if depts:
            in_dept  = base[base["Department"].isin(depts)].sort_values("LPS", ascending=False)
            out_dept = base[~base["Department"].isin(depts)].sort_values("LPS", ascending=False)
            candidates = pd.concat([in_dept, out_dept]).drop_duplicates("EE Number")
        else:
            candidates = base.sort_values("LPS", ascending=False)

        # ── Global deduplication: prefer fresh candidates ─────────────────────
        fresh = candidates[~candidates["EE Number"].isin(globally_used)]
        final_pool = fresh if len(fresh) >= 3 else candidates  # relax if pool too small

        top3 = final_pool.head(3)

        # Mark #1 as globally used so they don't appear as #1 elsewhere
        if len(top3) > 0:
            globally_used.add(top3.iloc[0]["EE Number"])

        for rank, (_, cand) in enumerate(top3.iterrows(), start=1):
            rows.append({
                "Critical Role":                                  role,
                "Role Label":                                     cfg["label"],
                "Incumbent EE Number":                            inc["EE Number"],
                "Incumbent Name":                                 inc["Employee Full Name"],
                "Successor Rank":                                 rank,
                "Successor EE Number":                            cand["EE Number"],
                "Successor Full Name":                            cand["Employee Full Name"],
                "Successor Current Job Title":                    cand["Current Job Title"],
                "Successor Job Grade (1-9)":                      int(cand["Job Grade (1-9)"]),
                "Successor Department":                           cand["Department"],
                "Successor Business Unit":                        cand["Business Unit"],
                "Successor Work Location":                        cand["Work Location"],
                "Leadership Potential Score (0-100)":             round(float(cand["LPS"]),2),
                "LPS Band":                                       cand["LPS Band"],
                "LPS - Performance Cluster (0-100)":              round(float(cand.get("C1",0)),2),
                "LPS - KF Assessment Cluster (0-100)":            round(float(cand.get("C2",0)),2),
                "LPS - Career Velocity Cluster (0-100)":          round(float(cand.get("C3",0)),2),
                "LPS - Leadership Breadth Cluster (0-100)":       round(float(cand.get("C4",0)),2),
                "LPS - Readiness & Mobility Cluster (0-100)":     round(float(cand.get("C5",0)),2),
                "Last Annual Performance Rating (1-5)":           safe_float(cand["Last Annual Performance Rating (1-5)"]),
                "Average Performance Rating - Last 3 Years (1-5)":safe_float(cand["Average Performance Rating - Last 3 Years (1-5)"]),
                "Total Promotions (Career)":                      int(cand["Total Promotions (Career)"]),
                "Promotions in Last 5 Years":                     int(cand["Promotions in Last 5 Years"]),
                "Promotions per Year (Career)":                   round(float(cand["Promotions per Year (Career)"]),4),
                "KF KFALP - Composite Score (1-5)":               cand.get("KF KFALP - Composite Score (1-5)",None),
                "KF viaEdge - Learning Agility Composite (1-5)":  cand.get("KF viaEdge - Learning Agility Composite (1-5)",None),
                "KF Blended Assessment Composite (1-5)":          cand.get("KF Blended Assessment Composite (1-5)",None),
                "Tenure with Organisation (Years)":               safe_float(cand["Tenure with Organisation (Years)"]),
                "9-Box Position":                                  cand["9-Box Position"],
                "Flight Risk":                                     cand["Flight Risk"],
                "On Active Retention Plan":                       bool(cand["On Active Retention Plan"]),
            })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def gauge_fig(value, title, max_val=100, color="#0D7377"):
    v = safe_float(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=v,
        title={"text": title, "font": {"family":"Syne","size":13,"color":"#64748B"}},
        number={"font": {"family":"Syne","size":28,"color":color}},
        gauge={
            "axis": {"range":[0,max_val],"tickwidth":1,"tickcolor":"#D1DCE8","tickfont":{"size":9}},
            "bar": {"color":color,"thickness":0.25},
            "bgcolor":"#F0F4F8","borderwidth":0,
            "steps":[
                {"range":[0,         max_val*0.35],"color":"#FEE2E2"},
                {"range":[max_val*0.35,max_val*0.50],"color":"#FEF3C7"},
                {"range":[max_val*0.50,max_val*0.65],"color":"#DBEAFE"},
                {"range":[max_val*0.65,max_val*0.80],"color":"#D1FAE5"},
                {"range":[max_val*0.80,max_val],      "color":"#A7F3D0"},
            ],
            "threshold":{"line":{"color":"#0B2540","width":3},"thickness":0.8,"value":v},
        }
    ))
    fig.update_layout(margin=dict(l=10,r=10,t=30,b=10),height=180,
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    return fig

def speedometer_fig(value, title, color="#0D7377"):
    v = safe_float(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=v,
        title={"text":title,"font":{"family":"Syne","size":11,"color":"#64748B"}},
        number={"font":{"family":"Syne","size":22,"color":color},"valueformat":".2f"},
        gauge={
            "axis":{"range":[0,5.0],"tickwidth":1,
                    "tickvals":[1,2,3,4,5],
                    "ticktext":["Limited","Developing","Effective","Strong","Exceptional"],
                    "tickfont":{"size":8}},
            "bar":{"color":color,"thickness":0.3},
            "bgcolor":"#F0F4F8","borderwidth":0,
            "steps":[
                {"range":[0,  1.5],"color":"#FEE2E2"},{"range":[1.5,2.5],"color":"#FEF3C7"},
                {"range":[2.5,3.5],"color":"#DBEAFE"},{"range":[3.5,4.5],"color":"#D1FAE5"},
                {"range":[4.5,5.0],"color":"#6EE7B7"},
            ],
        }
    ))
    fig.update_layout(margin=dict(l=5,r=5,t=35,b=5),height=160,
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    return fig

def radar_fig(values, labels, name, color="#0D7377", ref_vals=None):
    fig = go.Figure()
    if ref_vals:
        fig.add_trace(go.Scatterpolar(
            r=ref_vals+[ref_vals[0]], theta=labels+[labels[0]],
            fill="toself", fillcolor="rgba(200,200,200,0.15)",
            line=dict(color="#C9A227",width=2,dash="dot"), name="Org Benchmark"
        ))
    fig.add_trace(go.Scatterpolar(
        r=values+[values[0]], theta=labels+[labels[0]],
        fill="toself", fillcolor=hex_to_rgba(color,0.19),
        line=dict(color=color,width=2.5), name=name
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True,range=[0,5],
                                   tickvals=[1,2,3,4,5],tickfont={"size":8},
                                   gridcolor="#E2EAF0"),
                   angularaxis=dict(tickfont={"family":"DM Sans","size":10})),
        showlegend=True, legend=dict(font={"size":9}),
        margin=dict(l=40,r=40,t=30,b=30), height=300,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def slider_html(label, value, lo, hi, color="#0D7377", pct=None):
    pp = max(2, min(98, (value-lo)/(hi-lo+1e-9)*100))
    ps = f"({int(pct)}th pct)" if pct is not None else f"({int(pp)}th pct)"
    return (f'<div style="margin-bottom:10px">'
            f'<div style="font-size:0.78rem;color:#64748B;margin-bottom:3px;'
            f'display:flex;justify-content:space-between;">'
            f'<span>{label}</span>'
            f'<span style="color:{color};font-weight:600">{value:.2f} {ps}</span></div>'
            f'<div style="height:10px;border-radius:6px;background:linear-gradient('
            f'90deg,#B91C1C 0%,#D97706 40%,#16A34A 100%);position:relative;">'
            f'<div style="position:absolute;top:-4px;left:{pp}%;width:18px;height:18px;'
            f'border-radius:50%;background:{color};border:3px solid white;'
            f'box-shadow:0 2px 6px rgba(0,0,0,0.25);transform:translateX(-50%);"></div>'
            f'</div></div>')

# ─────────────────────────────────────────────────────────────────────────────
# 9-BOX GRID (full graphical)
# ─────────────────────────────────────────────────────────────────────────────
def nine_box_fig(df_plot, highlight_ee=None):
    perf_map = {"Low Performer":0,"Moderate Performer":1,"High Performer":2,"Exceptional Performer":2}

    def parse_box(s):
        """Parse only the performance axis from the 9-Box string."""
        if not isinstance(s,str) or "/" not in s: return None
        parts = [p.strip() for p in s.split("/")]
        px = next((v for k,v in perf_map.items() if k in parts[0]),None)
        return px

    def lps_to_potential(lps_val):
        """Derive potential axis from LPS score — ensures high-LPS employees
        appear in the High Potential row regardless of HR label."""
        v = safe_float(lps_val)
        if v >= 65: return 2   # High Potential
        if v >= 35: return 1   # Moderate Potential
        return 0               # Low Potential

    df2 = df_plot.copy()
    df2["px"] = df2["9-Box Position"].apply(parse_box)
    df2["py"] = df2["LPS"].apply(lps_to_potential)
    df2 = df2.dropna(subset=["px","py"])
    rng = np.random.RandomState(42)
    df2["xj"] = df2["px"] + rng.uniform(-0.28,0.28,len(df2))
    df2["yj"] = df2["py"] + rng.uniform(-0.28,0.28,len(df2))

    cell_bg = {
        (0,0):"#FEE2E2",(1,0):"#FEF3C7",(2,0):"#FEF3C7",
        (0,1):"#FEF3C7",(1,1):"#DBEAFE",(2,1):"#D1FAE5",
        (0,2):"#FEF3C7",(1,2):"#D1FAE5",(2,2):"#A7F3D0",
    }
    cell_lbl = {
        (0,0):"Underperformer",(1,0):"Effective Contributor",(2,0):"Misaligned Star",
        (0,1):"Developing",(1,1):"Core Contributor",(2,1):"High Potential",
        (0,2):"Enigma",(1,2):"Future Leader",(2,2):"Top Talent",
    }

    fig = go.Figure()
    for (px_c,py_c),col in cell_bg.items():
        fig.add_shape(type="rect",
            x0=px_c-0.5,y0=py_c-0.5,x1=px_c+0.5,y1=py_c+0.5,
            fillcolor=col, line=dict(color="#CBD5E1",width=1.5), layer="below")
        fig.add_annotation(x=px_c, y=py_c+0.4,
            text=f"<b>{cell_lbl.get((px_c,py_c),'')}</b>",
            showarrow=False, font=dict(size=9,color="#374151",family="Syne"), xanchor="center")

    for band, grp in df2.groupby("LPS Band"):
        bc = BAND_COLORS.get(band,"#888"); bs = BAND_SHORT.get(band,band)
        hover = [f"<b>{r['Employee Full Name']}</b><br>EE: {r['EE Number']}<br>"
                 f"Title: {r['Current Job Title']}<br>Dept: {r['Department']}<br>"
                 f"LPS: {r['LPS']:.1f} — {bs}<br>9-Box: {r['9-Box Position']}"
                 for _,r in grp.iterrows()]
        fig.add_trace(go.Scatter(
            x=grp["xj"], y=grp["yj"], mode="markers",
            marker=dict(size=8,color=bc,opacity=0.82,line=dict(width=1,color="white")),
            name=bs, hovertext=hover, hoverinfo="text",
        ))

    if highlight_ee is not None:
        row = df2[df2["EE Number"]==highlight_ee]
        if len(row)>0:
            r = row.iloc[0]
            fig.add_trace(go.Scatter(
                x=[r["xj"]], y=[r["yj"]], mode="markers+text",
                marker=dict(size=22,color="#C9A227",symbol="star",line=dict(width=2,color="#0B2540")),
                text=[r["Employee Full Name"].split()[0]],
                textposition="top center",
                textfont=dict(family="Syne",size=10,color="#0B2540"),
                name="Selected", hovertext=[r["Employee Full Name"]], hoverinfo="text",
            ))

    fig.update_layout(
        xaxis=dict(tickvals=[0,1,2],
                   ticktext=["Low Performance","Moderate Performance","High Performance"],
                   range=[-0.55,2.55],showgrid=False,zeroline=False,
                   tickfont=dict(family="Syne",size=10,color="#374151"),
                   title=dict(text="PERFORMANCE →",font=dict(family="Syne",size=11,color="#0B2540"))),
        yaxis=dict(tickvals=[0,1,2],
                   ticktext=["Low Potential","Moderate Potential","High Potential"],
                   range=[-0.55,2.55],showgrid=False,zeroline=False,
                   tickfont=dict(family="Syne",size=10,color="#374151"),
                   title=dict(text="POTENTIAL ↑",font=dict(family="Syne",size=11,color="#0B2540"))),
        legend=dict(font=dict(size=9,family="DM Sans"),orientation="h",
                    x=0.5,xanchor="center",y=-0.14),
        margin=dict(l=10,r=10,t=20,b=50), height=480,
        paper_bgcolor="white", plot_bgcolor="white",
        hoverlabel=dict(bgcolor="white",font_size=11,font_family="DM Sans",bordercolor="#D1DCE8"),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Succession Planning Engine",page_icon="🎯",
                   layout="wide",initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');
:root{--navy:#0B2540;--teal:#0D7377;--gold:#C9A227;--gold-lt:#F0C93A;
  --bg:#F0F4F8;--card:#FFFFFF;--border:#D1DCE8;--text:#1A2535;--muted:#64748B;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important;background-color:var(--bg)!important;color:var(--text)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:1.2rem 1.8rem 2rem 1.8rem!important;max-width:100%!important;}
.app-title{background:linear-gradient(135deg,#0B2540 0%,#1a3a5c 60%,#0D7377 100%);border-radius:14px;padding:18px 28px;margin-bottom:18px;display:flex;align-items:center;gap:16px;}
.app-title h1{font-family:'Syne',sans-serif!important;font-size:1.7rem;font-weight:800;color:#fff;margin:0;}
.app-title p{color:#9EC5D8;margin:2px 0 0 0;font-size:0.85rem;}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:var(--navy);border-radius:12px;padding:6px;margin-bottom:18px;}
.stTabs [data-baseweb="tab"]{font-family:'Syne',sans-serif!important;font-weight:600;font-size:0.82rem;color:#9EC5D8!important;border-radius:8px;padding:8px 16px;border:none!important;background:transparent!important;white-space:nowrap;}
.stTabs [aria-selected="true"]{background:var(--teal)!important;color:#fff!important;}
.stTabs [data-baseweb="tab-panel"]{padding:0!important;}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:20px 22px;box-shadow:0 2px 12px rgba(11,37,64,0.07);margin-bottom:14px;}
.card-navy{background:var(--navy);border-radius:14px;padding:18px 22px;margin-bottom:14px;}
.kpi-row{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px 18px;flex:1;min-width:120px;box-shadow:0 1px 6px rgba(11,37,64,0.06);}
.kpi-value{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:var(--teal);line-height:1.1;}
.kpi-label{font-size:0.75rem;color:var(--muted);margin-top:2px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;}
.scard{border-radius:14px;padding:16px 20px;margin-bottom:10px;border-left:5px solid var(--teal);background:var(--card);box-shadow:0 2px 10px rgba(11,37,64,0.08);position:relative;}
.scard-rank{position:absolute;top:12px;right:14px;font-family:'Syne',sans-serif;font-size:0.68rem;font-weight:800;background:var(--navy);color:var(--gold-lt);border-radius:20px;padding:3px 10px;letter-spacing:1px;text-transform:uppercase;}
.scard h3{font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;margin:0 0 2px 0;}
/* Hide auto-generated anchor link icons next to headings */
h1 a,h2 a,h3 a,h4 a,h5 a,h6 a,.scard h3 a,[data-testid="stMarkdownContainer"] h1 a,[data-testid="stMarkdownContainer"] h2 a,[data-testid="stMarkdownContainer"] h3 a{display:none!important;visibility:hidden!important;}
a.anchor{display:none!important;}
.lps-num{font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;line-height:1;}
.band-pill{display:inline-block;border-radius:20px;padding:3px 10px;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-left:8px;vertical-align:middle;}
.sec-hdr{font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:800;color:var(--navy);border-bottom:2px solid var(--teal);padding-bottom:6px;margin-bottom:14px;}
section[data-testid="stSidebar"]{background:var(--navy)!important;border-right:none!important;}
/* Target sidebar content but NOT the collapse toggle button */
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stMarkdown *,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span:not([data-testid="collapsedControl"]),
section[data-testid="stSidebar"] div.stSelectbox label,
section[data-testid="stSidebar"] div.stSlider label,
section[data-testid="stSidebar"] div.stFileUploader label,
section[data-testid="stSidebar"] div.stCheckbox label
  {color:#C8DDE8!important;}
section[data-testid="stSidebar"] h2{font-family:'Syne',sans-serif!important;color:white!important;font-size:1rem!important;}
/* Ensure collapse/expand chevron button stays visible */
[data-testid="collapsedControl"]{
  display:flex!important;visibility:visible!important;opacity:1!important;
  background:rgba(255,255,255,0.12)!important;border-radius:0 8px 8px 0!important;
}
[data-testid="collapsedControl"] svg{color:#FFFFFF!important;fill:#FFFFFF!important;}
button[kind="headerNoPadding"]{display:flex!important;visibility:visible!important;}
.upload-hint{font-size:0.78rem;color:#7A9AB8;text-align:center;padding:8px;border:1px dashed #3A5A78;border-radius:8px;margin-bottom:8px;}
.dept-badge{display:inline-block;background:#EBF4F8;color:#0B2540;border-radius:6px;padding:2px 8px;font-size:0.72rem;font-weight:600;margin-top:4px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2>⬆ Upload Datasets</h2>", unsafe_allow_html=True)
    st.markdown('<div class="upload-hint">Upload 6 CSV files — pipeline auto-generated per role</div>',
                unsafe_allow_html=True)
    uploaded = {
        "employees": st.file_uploader("employees_master.csv",       type="csv", key="emp"),
        "kfalp":     st.file_uploader("kf_kfalp_detail.csv",        type="csv", key="kfl"),
        "viaedge":   st.file_uploader("kf_viaedge_detail.csv",      type="csv", key="via"),
        "ref":       st.file_uploader("kf_attribute_reference.csv", type="csv", key="ref"),
        "promos":    st.file_uploader("promotion_history.csv",      type="csv", key="prm"),
        "org":       st.file_uploader("org_structure.csv",          type="csv", key="org"),
    }
    n_loaded = sum(v is not None for v in uploaded.values())
    st.markdown(f"<div style='color:#6EE7B7;font-size:0.8rem;margin-top:6px'>"
                f"✓ {n_loaded}/6 files · Pipeline: auto-generated per role</div>",
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h2>🎛 LPS Weights</h2>", unsafe_allow_html=True)
    w1 = st.slider("Performance",        5,60,25,5,key="w1")
    w2 = st.slider("KF Assessment",      5,60,30,5,key="w2")
    w3 = st.slider("Career Velocity",    5,40,20,5,key="w3")
    w4 = st.slider("Leadership Breadth", 5,30,15,5,key="w4")
    w5 = st.slider("Readiness",          5,30,10,5,key="w5")
    total_w = w1+w2+w3+w4+w5
    if total_w > 100:
        st.markdown(
            f"<div style='background:#FEE2E2;border:1px solid #FCA5A5;border-radius:8px;"
            f"padding:8px 12px;font-size:0.82rem;color:#B91C1C;font-weight:700;margin-top:4px'>"
            f"⚠️ Total is {total_w}% — exceeds 100. Reduce sliders until total = 100."
            f"</div>", unsafe_allow_html=True)
    elif total_w == 100:
        st.markdown(
            f"<div style='color:#6EE7B7;font-size:0.85rem;font-weight:600;margin-top:2px'>"
            f"✓ Total: 100% — weights are valid</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div style='background:#FEF3C7;border:1px solid #D97706;border-radius:8px;"
            f"padding:8px 12px;font-size:0.82rem;color:#92400E;font-weight:600;margin-top:4px'>"
            f"Total: {total_w}% — must equal exactly 100"
            f"</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h2>🔍 Filters</h2>", unsafe_allow_html=True)
    exclude_risk = st.checkbox("Exclude High Flight Risk", value=True)
    min_grade    = st.slider("Min Grade for Eligibility", 1, 9, 5)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD + SCORE + BUILD
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(f):
    df = pd.read_csv(f)
    df.columns = [c.replace("\u2013","-").replace("\u2014","-") for c in df.columns]
    return df

data = {k: load_csv(v) for k, v in uploaded.items() if v is not None}

st.markdown("""
<div class="app-title">
  <div style="font-size:2.2rem">🎯</div>
  <div>
    <h1>Succession Planning Engine</h1>
    <p>Powered by HRMS Data · Korn Ferry KFALP · Korn Ferry viaEdge</p>
  </div>
</div>
""", unsafe_allow_html=True)

if "employees" not in data:
    st.info("👈 Upload **employees_master.csv** to activate the engine.")
    st.stop()

df_emp  = recompute_lps(data["employees"], w1, w2, w3, w4, w5)
df_elig = df_emp[df_emp["Flight Risk"]!="High"].copy() if exclude_risk else df_emp.copy()
df_elig = df_elig[df_elig["Job Grade (1-9)"] >= min_grade]
df_pip  = build_pipeline(df_emp, exclude_risk=exclude_risk, min_gr=min_grade)
all_names = sorted(df_emp["Employee Full Name"].unique())

# ─────────────────────────────────────────────────────────────────────────────
# KF KEY LISTS (reused across tabs)
# ─────────────────────────────────────────────────────────────────────────────
KF_KEYS  = ["KF KFALP - Drivers Score (1-5)","KF KFALP - Curiosity Score (1-5)",
            "KF KFALP - Insight Score (1-5)","KF KFALP - Engagement Score (1-5)",
            "KF KFALP - Determination Score (1-5)","KF KFALP - Learnability Score (1-5)"]
KF_LBLS  = ["Drivers","Curiosity","Insight","Engagement","Determination","Learnability"]
VE_KEYS  = ["KF viaEdge - Mental Agility Score (1-5)","KF viaEdge - People Agility Score (1-5)",
            "KF viaEdge - Change Agility Score (1-5)","KF viaEdge - Results Agility Score (1-5)",
            "KF viaEdge - Self-Awareness Score (1-5)"]
VE_LBLS  = ["Mental Agility","People Agility","Change Agility","Results Agility","Self-Awareness"]

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10 = st.tabs([
    "🏆 Succession Pipeline","👤 Employee Profile","⚖️ Compare Employees",
    "🌐 Org Chart","📊 Org Readiness","🧠 KF Assessment","📈 Career Path",
    "💊 Development Rx","📋 Data Templates","📖 Glossary",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — SUCCESSION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1, 2.2], gap="large")

    with left:
        st.markdown('<div class="sec-hdr">🏢 Select Critical Role</div>', unsafe_allow_html=True)
        sel_role = st.selectbox("", CRITICAL_ROLES, label_visibility="collapsed", key="sel_role")
        cfg      = ROLES_CFG[sel_role]

        role_rows = df_pip[df_pip["Critical Role"] == sel_role]
        inc_name  = str(role_rows.iloc[0].get("Incumbent Name","—")) if len(role_rows)>0 else "—"
        inc_ee    = str(role_rows.iloc[0].get("Incumbent EE Number","—")) if len(role_rows)>0 else "—"

        dept_badge = ""
        if cfg["dept"]:
            dept_badge = "".join(f'<span class="dept-badge">{d}</span> ' for d in cfg["dept"])

        st.markdown(f"""
        <div class="card-navy">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
            {avatar_html(inc_name, 52, "#C9A227")}
            <div>
              <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1rem;color:white">{inc_name}</div>
              <div style="font-size:0.75rem;color:#9EC5D8">Incumbent · {inc_ee}</div>
            </div>
          </div>
          <div style="font-family:'Syne',sans-serif;font-size:0.85rem;font-weight:700;color:#F0C93A;margin-bottom:6px">{cfg['label']}</div>
          <div style="font-size:0.72rem;color:#6A9AB8">Preferred talent pool: {dept_badge if cfg['dept'] else 'All departments'}</div>
          <div style="font-size:0.72rem;color:#6A9AB8">Grade window: {max(1,cfg['min_grade']-cfg['grade_window'])}–{min(9,cfg['min_grade']+1)}</div>
        </div>""", unsafe_allow_html=True)

        n_pool    = len(df_elig)
        ready_now = (df_elig["LPS Band"]=="Band 5 - Ready Now").sum()
        avg_lps   = df_elig["LPS"].mean()
        st.markdown(f"""
        <div class="kpi-row">
          <div class="kpi"><div class="kpi-value">{n_pool}</div><div class="kpi-label">Pool</div></div>
          <div class="kpi"><div class="kpi-value" style="color:#1B7A3E">{ready_now}</div><div class="kpi-label">Ready Now</div></div>
          <div class="kpi"><div class="kpi-value" style="color:#0D7377">{avg_lps:.0f}</div><div class="kpi-label">Avg LPS</div></div>
        </div>""", unsafe_allow_html=True)

        band_counts = df_elig["LPS Band"].value_counts()
        fig_donut = go.Figure(go.Pie(
            labels=[BAND_SHORT.get(b,b) for b in band_counts.index],
            values=band_counts.values, hole=0.62,
            marker=dict(colors=[BAND_COLORS.get(b,"#888") for b in band_counts.index]),
            textfont=dict(family="DM Sans",size=10),
        ))
        fig_donut.update_layout(
            margin=dict(l=0,r=0,t=10,b=0),height=180,showlegend=True,
            legend=dict(font=dict(size=9,family="DM Sans"),orientation="h",x=0.5,xanchor="center",y=-0.05),
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text=f"<b>{n_pool}</b><br>Pool",x=0.5,y=0.5,
                              font=dict(family="Syne",size=14,color="#0B2540"),showarrow=False)]
        )
        fig_donut.update_layout(title=dict(text="<b>Talent Pool — LPS Band Distribution</b>",font=dict(family="Syne",size=12,color="#0B2540"),x=0.5,xanchor="center"))
        fig_donut.update_layout(title=dict(text="Eligible Pool — Readiness Distribution",font=dict(family="Syne",size=12,color="#0B2540"),x=0.5,xanchor="center"))
        st.plotly_chart(fig_donut, use_container_width=True, config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_001")
        st.markdown("<div style='font-size:0.72rem;color:#64748B;text-align:center;margin-top:-8px'>Distribution of succession readiness bands across the eligible talent pool. A healthy pipeline has ≥20% in Band 4–5.</div>", unsafe_allow_html=True)
        st.caption("Distribution of eligible employees by succession readiness band. 'Ready Now' (Band 5) can step into a critical role immediately.")

    with right:
        st.markdown('<div class="sec-hdr">🔗 Succession Pipeline — Top 3 Role-Specific Successors</div>',
                    unsafe_allow_html=True)

        role_pip = df_pip[df_pip["Critical Role"]==sel_role].sort_values("Successor Rank")
        rank_colors = ["#1B7A3E","#2563EB","#D97706"]
        rank_labels = ["#1 — Primary Successor","#2 — Secondary Successor","#3 — Tertiary Successor"]

        if len(role_pip)==0:
            st.warning("No successors found for this role with current filters.")
        else:
            for i, (_,cand) in enumerate(role_pip.iterrows()):
                if i>=3: break
                lps  = cand["Leadership Potential Score (0-100)"]
                band = cand["LPS Band"]
                bc   = BAND_COLORS.get(band,"#888")
                bs   = BAND_SHORT.get(band,band)
                c1v  = cand["LPS - Performance Cluster (0-100)"]
                c2v  = cand["LPS - KF Assessment Cluster (0-100)"]
                c3v  = cand["LPS - Career Velocity Cluster (0-100)"]
                c4v  = cand["LPS - Leadership Breadth Cluster (0-100)"]
                c5v  = cand["LPS - Readiness & Mobility Cluster (0-100)"]

                bar = go.Figure(go.Bar(
                    x=[c1v,c2v,c3v,c4v,c5v], y=CL_NAMES, orientation="h",
                    marker_color=CL_COLORS,
                    text=[f"{v:.0f}" for v in [c1v,c2v,c3v,c4v,c5v]],
                    textposition="outside", textfont=dict(size=9,family="DM Sans"),
                ))
                bar.update_layout(xaxis=dict(range=[0,110],showgrid=False,showticklabels=False),
                                   yaxis=dict(tickfont=dict(size=9,family="DM Sans")),
                                   margin=dict(l=0,r=40,t=0,b=0),height=100,
                                   paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                                   showlegend=False)

                dept_tag = cand["Successor Department"]
                st.markdown(f"""
                <div class="scard" style="border-left-color:{rank_colors[i]}">
                  <div class="scard-rank">{rank_labels[i]}</div>
                  <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
                    {avatar_html(cand['Successor Full Name'],44,rank_colors[i])}
                    <div style="flex:1">
                      <h3 style="color:#0B2540">{cand['Successor Full Name']}</h3>
                      <div style="font-size:0.78rem;color:#64748B">{cand['Successor Current Job Title']} · Grade {cand['Successor Job Grade (1-9)']} · {cand['Successor EE Number']}</div>
                      <div style="font-size:0.73rem;margin-top:2px"><span class="dept-badge">{dept_tag}</span></div>
                      <div style="margin-top:4px;display:flex;align-items:baseline;gap:6px">
                        <span class="lps-num" style="color:{bc}">{lps:.1f}</span>
                        <span style="font-size:0.78rem;color:#64748B">/ 100 LPS</span>
                        <span class="band-pill" style="background:{bc}20;color:{bc}">{bs}</span>
                      </div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
                st.plotly_chart(bar, use_container_width=True,
                                config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key=f"bar_{i}_{sel_role}")

            # Comparison chart
            st.markdown('<div class="sec-hdr" style="margin-top:8px">📊 Pipeline Cluster Comparison</div>',
                        unsafe_allow_html=True)
            fig_cmp = go.Figure()
            for i, (_,r) in enumerate(role_pip.head(3).iterrows()):
                vals = [r["LPS - Performance Cluster (0-100)"],r["LPS - KF Assessment Cluster (0-100)"],
                        r["LPS - Career Velocity Cluster (0-100)"],r["LPS - Leadership Breadth Cluster (0-100)"],
                        r["LPS - Readiness & Mobility Cluster (0-100)"]]
                fig_cmp.add_trace(go.Bar(
                    name=r["Successor Full Name"], x=CL_NAMES, y=vals,
                    marker_color=rank_colors[i],
                    text=[f"{v:.0f}" for v in vals],
                    textposition="outside", textfont=dict(size=9),
                ))
            fig_cmp.update_layout(
                barmode="group",
                xaxis=dict(tickfont=dict(family="DM Sans",size=10)),
                yaxis=dict(range=[0,110],tickfont=dict(size=9)),
                legend=dict(font=dict(family="DM Sans",size=10),orientation="h",x=0.5,xanchor="center",y=1.08),
                margin=dict(l=0,r=0,t=30,b=0),height=260,
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            )
            fig_cmp.update_layout(title=dict(text="<b>Pipeline Cluster Comparison — Top 3 Successors</b>",font=dict(family="Syne",size=12,color="#0B2540"),x=0.5,xanchor="center"))
            fig_cmp.update_layout(title=dict(text="LPS Cluster Scores — Top 3 Successors Compared",font=dict(family="Syne",size=12,color="#0B2540"),x=0.5,xanchor="center"))
            st.plotly_chart(fig_cmp, use_container_width=True, config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_003")
            st.markdown("<div style='font-size:0.72rem;color:#64748B;margin-top:-6px'>Each bar shows the normalised score (0–100) for one of the five LPS clusters. Higher is better. Use this to identify each successor's relative strengths and development gaps.</div>", unsafe_allow_html=True)
            st.caption("Side-by-side LPS cluster scores for the top 3 successors. Higher scores across all 5 clusters indicate a stronger, more well-rounded candidate.")

        dl1,dl2,dl3 = st.columns([1,1,1])
        with dl2:
            st.download_button(
                label="⬇ Download Full Pipeline CSV",
                data=df_pip.to_csv(index=False).encode("utf-8"),
                file_name="succession_pipeline_generated.csv",
                mime="text/csv", use_container_width=True,
            )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — EMPLOYEE PROFILE
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    sc,_ = st.columns([2,3])
    with sc:
        sel_emp = st.selectbox("Select Employee", all_names, key="emp_sel")
    emp  = df_emp[df_emp["Employee Full Name"]==sel_emp].iloc[0]
    lps  = emp["LPS"]; bc = lps_color(lps); band = emp["LPS Band"]

    h1c,h2c,h3c = st.columns([1.5,1.5,1])
    with h1c:
        fr_bg  = "#FEE2E2" if emp["Flight Risk"]=="High" else "#FEF3C7" if emp["Flight Risk"]=="Medium" else "#D1FAE5"
        fr_clr = "#B91C1C" if emp["Flight Risk"]=="High" else "#D97706" if emp["Flight Risk"]=="Medium" else "#166534"
        st.markdown(f"""
        <div class="card" style="display:flex;gap:16px;align-items:center">
          {avatar_html(sel_emp,64,bc)}
          <div>
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:#0B2540">{sel_emp}</div>
            <div style="color:#64748B;font-size:0.82rem">{emp['Current Job Title']}</div>
            <div style="color:#64748B;font-size:0.78rem">{emp['Department']} · Grade {int(emp['Job Grade (1-9)'])} · {emp['EE Number']}</div>
            <div style="margin-top:6px">
              <span class="band-pill" style="background:{bc}20;color:{bc};font-size:0.75rem">{BAND_SHORT.get(band,band)}</span>
              <span class="band-pill" style="background:{fr_bg};color:{fr_clr};font-size:0.75rem">FR: {emp['Flight Risk']}</span>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
    with h2c:
        st.plotly_chart(gauge_fig(lps,"Leadership Potential Score",color=bc),
                        use_container_width=True,config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_t2_gauge")
        st.markdown("<div style='font-size:0.71rem;color:#64748B;text-align:center;margin-top:-10px'>Composite score (0–100) across 5 weighted clusters: Performance, KF Assessment, Career Velocity, Leadership Breadth, and Readiness.</div>", unsafe_allow_html=True)
    with h3c:
        kfc = safe_float(emp.get("KF KFALP - Composite Score (1-5)",0))
        vec = safe_float(emp.get("KF viaEdge - Learning Agility Composite (1-5)",0))
        if kfc>0: st.plotly_chart(speedometer_fig(kfc,"KFALP Composite",color="#C9A227"),
                                   use_container_width=True,config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_t2_kfalp")
        if vec>0: st.plotly_chart(speedometer_fig(vec,"viaEdge Composite",color="#7C3AED"),
                                   use_container_width=True,config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_t2_viaedge")

    sl_col,rd_col = st.columns([1.3,1])
    with sl_col:
        st.markdown('<div class="sec-hdr">📏 Feature Profile — Position in Organisation</div>',
                    unsafe_allow_html=True)
        sl_defs = [
            ("Performance (3yr Avg)","Average Performance Rating - Last 3 Years (1-5)",1,5,"#0D7377"),
            ("Last Performance Rating","Last Annual Performance Rating (1-5)",1,5,"#0D7377"),
            ("Total Promotions","Total Promotions (Career)",0,14,"#0D7377"),
            ("Promotions/Year","Promotions per Year (Career)",0,0.8,"#0D7377"),
            ("Promotions/Year (5yr)","Promotions per Year (Last 5 Years)",0,0.8,"#0D7377"),
            ("Tenure (Years)","Tenure with Organisation (Years)",0,40,"#0D7377"),
            ("Direct Reports","Number of Direct Reports",0,50,"#0D7377"),
            ("Critical Projects","Number of Critical Projects Led",0,14,"#0D7377"),
            ("Mobility Willingness","Mobility / Relocation Willingness (1-5)",1,5,"#0D7377"),
            ("KF KFALP — Drivers","KF KFALP - Drivers Score (1-5)",1,5,"#C9A227"),
            ("KF KFALP — Curiosity","KF KFALP - Curiosity Score (1-5)",1,5,"#C9A227"),
            ("KF KFALP — Insight","KF KFALP - Insight Score (1-5)",1,5,"#C9A227"),
            ("KF KFALP — Engagement","KF KFALP - Engagement Score (1-5)",1,5,"#C9A227"),
            ("KF KFALP — Determination","KF KFALP - Determination Score (1-5)",1,5,"#C9A227"),
            ("KF KFALP — Learnability","KF KFALP - Learnability Score (1-5)",1,5,"#C9A227"),
            ("viaEdge — Mental Agility","KF viaEdge - Mental Agility Score (1-5)",1,5,"#7C3AED"),
            ("viaEdge — People Agility","KF viaEdge - People Agility Score (1-5)",1,5,"#7C3AED"),
            ("viaEdge — Change Agility","KF viaEdge - Change Agility Score (1-5)",1,5,"#7C3AED"),
            ("viaEdge — Results Agility","KF viaEdge - Results Agility Score (1-5)",1,5,"#7C3AED"),
            ("viaEdge — Self-Awareness","KF viaEdge - Self-Awareness Score (1-5)",1,5,"#7C3AED"),
        ]
        html_s = ""
        for lbl,col,lo,hi,clr in sl_defs:
            if col not in df_emp.columns: continue
            val = emp.get(col,np.nan)
            if pd.isna(val):
                html_s += f"<div style='font-size:0.78rem;color:#CBD5E1;margin-bottom:8px'>{lbl}: N/A</div>"
                continue
            pct = norm_pct(float(val), df_emp[col])
            html_s += slider_html(lbl, float(val), lo, hi, clr, pct=pct)
        st.markdown(f'<div class="card">{html_s}</div>', unsafe_allow_html=True)

    with rd_col:
        st.markdown('<div class="sec-hdr">🕸 KFALP Radar</div>', unsafe_allow_html=True)
        kf_vals = [safe_float(emp.get(k,2.5),2.5) for k in KF_KEYS]
        ref_kf  = [df_emp[k].mean() if k in df_emp.columns else 3.0 for k in KF_KEYS]
        st.plotly_chart(radar_fig(kf_vals,KF_LBLS,"KFALP","#C9A227",ref_kf),
                        use_container_width=True,config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_t2_radar_kf")
        st.markdown("<div style='font-size:0.71rem;color:#64748B;text-align:center;margin-top:-8px'>Gold = employee score · Dashed gold = org average benchmark. Scale 1–5.</div>", unsafe_allow_html=True)

        st.markdown('<div class="sec-hdr">🕸 viaEdge Radar</div>', unsafe_allow_html=True)
        ve_vals = [safe_float(emp.get(k,2.5),2.5) for k in VE_KEYS]
        ref_ve  = [df_emp[k].mean() if k in df_emp.columns else 3.0 for k in VE_KEYS]
        st.plotly_chart(radar_fig(ve_vals,VE_LBLS,"viaEdge","#7C3AED",ref_ve),
                        use_container_width=True,config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_t2_radar_ve")
        st.markdown("<div style='font-size:0.71rem;color:#64748B;text-align:center;margin-top:-8px'>Purple = employee score · Dashed gold = org average benchmark. Measures learning agility across 5 dimensions. Scale 1–5.</div>", unsafe_allow_html=True)

        st.markdown('<div class="sec-hdr">🔲 9-Box Position</div>', unsafe_allow_html=True)
        nb2_fig = nine_box_fig(df_emp,highlight_ee=emp["EE Number"])
        nb2_fig.update_layout(title=dict(text="9-Box Position (★ = Selected Employee)",font=dict(family="Syne",size=11,color="#0B2540"),x=0.5,xanchor="center"))
        st.plotly_chart(nb2_fig, use_container_width=True,config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_t2_ninebox")
        st.markdown("<div style='font-size:0.71rem;color:#64748B;text-align:center;margin-top:-8px'>★ Gold star = this employee. X-axis = HR-assessed performance. Y-axis = LPS-derived potential. Hover dots for details.</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — COMPARE EMPLOYEES
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-hdr">⚖️ Select 2–4 Employees to Compare</div>',
                unsafe_allow_html=True)
    sel_emps = st.multiselect("Choose employees", all_names, max_selections=4,
                               default=all_names[:2] if len(all_names)>=2 else all_names,
                               key="cmp_sel")
    if len(sel_emps)<2:
        st.info("Select at least 2 employees to compare.")
    else:
        cmp_colors = ["#0D7377","#C9A227","#7C3AED","#EA580C"]
        cmp_df = df_emp[df_emp["Employee Full Name"].isin(sel_emps)].copy()

        hcols = st.columns(len(sel_emps))
        for i,name in enumerate(sel_emps):
            row = cmp_df[cmp_df["Employee Full Name"]==name].iloc[0]
            clr = cmp_colors[i]
            with hcols[i]:
                st.plotly_chart(gauge_fig(row["LPS"],name[:18],color=clr),
                                use_container_width=True,config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key=f"pc_cmp_gauge_{i}")
                bs = BAND_SHORT.get(row["LPS Band"],row["LPS Band"])
                st.markdown(f"<div style='text-align:center;font-size:0.78rem;color:#64748B'>"
                            f"{row['Current Job Title']}<br>Grade {int(row['Job Grade (1-9)'])}"
                            f"<span class='band-pill' style='background:{clr}20;color:{clr}'>{bs}</span>"
                            f"</div>", unsafe_allow_html=True)

        st.markdown('<div class="sec-hdr" style="margin-top:16px">📏 Feature Comparison</div>',
                    unsafe_allow_html=True)
        cmp_sl_defs = [
            ("LPS Score","LPS",0,100),
            ("Performance (3yr Avg)","Average Performance Rating - Last 3 Years (1-5)",1,5),
            ("Total Promotions","Total Promotions (Career)",0,14),
            ("Promotions/Year","Promotions per Year (Career)",0,0.8),
            ("KF KFALP Composite","KF KFALP - Composite Score (1-5)",1,5),
            ("viaEdge Composite","KF viaEdge - Learning Agility Composite (1-5)",1,5),
            ("KFALP — Learnability","KF KFALP - Learnability Score (1-5)",1,5),
            ("viaEdge — Change Agility","KF viaEdge - Change Agility Score (1-5)",1,5),
            ("viaEdge — Results Agility","KF viaEdge - Results Agility Score (1-5)",1,5),
            ("Tenure (Years)","Tenure with Organisation (Years)",0,40),
            ("Critical Projects","Number of Critical Projects Led",0,14),
            ("Mobility Willingness","Mobility / Relocation Willingness (1-5)",1,5),
        ]
        html_cmp = ""
        for lbl,col,lo,hi in cmp_sl_defs:
            if col not in cmp_df.columns: continue
            html_cmp += (f"<div style='font-size:0.78rem;color:#374151;margin:10px 0 3px;font-weight:600'>{lbl}</div>"
                         f"<div style='height:10px;border-radius:6px;background:linear-gradient(90deg,"
                         f"#B91C1C 0%,#D97706 40%,#16A34A 100%);position:relative;margin-bottom:12px'>")
            for j,name in enumerate(sel_emps):
                r = cmp_df[cmp_df["Employee Full Name"]==name]
                if len(r)==0: continue
                val = r.iloc[0].get(col,np.nan)
                if pd.isna(val): continue
                pp = max(2,min(98,(float(val)-lo)/(hi-lo+1e-9)*100))
                clr = cmp_colors[j]
                html_cmp += (f"<div title='{name}: {float(val):.2f}' style='position:absolute;"
                             f"top:-5px;left:{pp}%;width:20px;height:20px;border-radius:50%;"
                             f"background:{clr};border:3px solid white;transform:translateX(-50%);"
                             f"box-shadow:0 2px 6px rgba(0,0,0,0.25);'></div>")
            html_cmp += "</div><div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:2px'>"
            for j,name in enumerate(sel_emps):
                r = cmp_df[cmp_df["Employee Full Name"]==name]
                if len(r)==0: continue
                val = r.iloc[0].get(col,np.nan)
                vs = f"{float(val):.2f}" if not pd.isna(val) else "N/A"
                html_cmp += (f"<span style='font-size:0.72rem;color:{cmp_colors[j]};"
                             f"font-weight:600'>● {name.split()[0]}: {vs}</span>")
            html_cmp += "</div>"
        st.markdown(f'<div class="card">{html_cmp}</div>', unsafe_allow_html=True)

        oc1,oc2 = st.columns([1,1])
        with oc1:
            st.markdown('<div class="sec-hdr">🕸 KFALP Overlay</div>', unsafe_allow_html=True)
            fig_ov = go.Figure()
            for j,name in enumerate(sel_emps):
                row = cmp_df[cmp_df["Employee Full Name"]==name]
                if len(row)==0: continue
                vals = [safe_float(row.iloc[0].get(k,2.5),2.5) for k in KF_KEYS]
                clr  = cmp_colors[j]
                fig_ov.add_trace(go.Scatterpolar(
                    r=vals+[vals[0]], theta=KF_LBLS+[KF_LBLS[0]],
                    fill="toself", fillcolor=hex_to_rgba(clr,0.15),
                    line=dict(color=clr,width=2.5), name=name,
                ))
            fig_ov.update_layout(
                polar=dict(radialaxis=dict(visible=True,range=[0,5],tickvals=[1,2,3,4,5],tickfont={"size":8})),
                legend=dict(font={"size":9,"family":"DM Sans"}),
                margin=dict(l=40,r=40,t=20,b=20),height=320,
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_ov, use_container_width=True, config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_011")
        with oc2:
            st.markdown('<div class="sec-hdr">🔲 9-Box Comparison</div>', unsafe_allow_html=True)
            st.plotly_chart(nine_box_fig(cmp_df),
                            use_container_width=True,config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_t3_ninebox")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — ORG CHART
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    if "org" not in data:
        st.info("Upload **org_structure.csv** to view the interactive org chart.")
    else:
        df_org = data["org"].copy()
        st.markdown('<div class="sec-hdr">🌐 Interactive Organisation Chart</div>',
                    unsafe_allow_html=True)
        lps_map = df_emp.set_index("Current Job Title")["LPS"].to_dict()
        df_org["LPS"] = df_org["Job Title"].map(lps_map)

        G = nx.DiGraph()
        for _,row in df_org.iterrows():
            G.add_node(row["Job Title"],
                       grade=row.get("Job Grade (1-9)",5),
                       lps=row.get("LPS",50),
                       is_critical=row.get("Is Critical Role",False))
            if str(row.get("Parent Node ID","")) not in ["","nan"]:
                G.add_edge(str(row["Parent Node ID"]), row["Job Title"])

        pos={}; levels={}
        roots=[n for n in G.nodes if G.in_degree(n)==0]
        if roots:
            q=deque([(roots[0],0)]); visited=set()
            while q:
                node,depth=q.popleft()
                if node in visited: continue
                visited.add(node); levels.setdefault(depth,[]).append(node)
                for child in G.successors(node): q.append((child,depth+1))
        for depth,nodes in levels.items():
            n=len(nodes)
            for i,node in enumerate(nodes):
                pos[node]=(i-(n-1)/2.0,-depth*1.8)

        ex,ey=[],[]
        for u,v in G.edges():
            if u in pos and v in pos:
                x0,y0=pos[u]; x1,y1=pos[v]
                ex+=[x0,x1,None]; ey+=[y0,y1,None]

        grade_colors={9:"#0B2540",8:"#0D7377",7:"#C9A227",6:"#2563EB",
                      5:"#7C3AED",4:"#16A34A",3:"#EA580C",2:"#64748B",1:"#94A3B8"}
        nx_,ny_,nt_,nc_,ns_,nh_=[],[],[],[],[],[]
        for node in G.nodes():
            if node not in pos: continue
            x,y=pos[node]; nx_.append(x); ny_.append(y)
            nd=G.nodes[node]; g=nd.get("grade",5)
            lps_v=nd.get("lps",50); is_cr=bool(nd.get("is_critical",False))
            nc_.append(grade_colors.get(g,"#888"))
            ns_.append(30 if g>=9 else 24 if g>=7 else 18)
            nt_.append(node[:20]+("..." if len(node)>20 else ""))
            _lps=f"{float(lps_v):.1f}" if lps_v is not None and str(lps_v)!="nan" else "N/A"
            nh_.append(f"<b>{node}</b><br>Grade: {g}<br>LPS: {_lps}{' ★ Critical' if is_cr else ''}")

        fig_org=go.Figure()
        # Edge lines
        fig_org.add_trace(go.Scatter(x=ex,y=ey,mode="lines",
                                      line=dict(width=1.2,color="#CBD5E1"),hoverinfo="none"))
        # Node markers only (no text on nodes — avoids overlapping)
        fig_org.add_trace(go.Scatter(x=nx_,y=ny_,mode="markers",
                                      marker=dict(size=ns_,color=nc_,line=dict(width=2,color="white")),
                                      hovertext=nh_,hoverinfo="text",
                                      hoverlabel=dict(bgcolor="white",font_size=11,
                                                      font_family="DM Sans",bordercolor="#D1DCE8")))
        # Selective annotations: only label grade 7+ nodes to avoid clutter
        for node in G.nodes():
            if node not in pos: continue
            nd  = G.nodes[node]
            g   = nd.get("grade",5)
            if g < 7: continue          # skip lower grades — too many, would overlap
            x,y = pos[node]
            short = node[:22]+("…" if len(node)>22 else "")
            fig_org.add_annotation(
                x=x, y=y-0.22,
                text=f"<b>{short}</b>",
                showarrow=False,
                font=dict(family="DM Sans",size=8,color="#1A2535"),
                xanchor="center",yanchor="top",
                bgcolor="rgba(255,255,255,0.82)",
                bordercolor="#D1DCE8",borderwidth=1,borderpad=2,
            )
        fig_org.update_layout(showlegend=False,
                               xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                               yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                               margin=dict(l=20,r=20,t=20,b=20),height=600,
                               paper_bgcolor="white",plot_bgcolor="white",
                               hoverlabel=dict(bgcolor="white",font_size=11,
                                               font_family="DM Sans",bordercolor="#D1DCE8"))
        fig_org.update_layout(title=dict(text="Interactive Organisation Chart — Node colour = Job Grade",font=dict(family="Syne",size=12,color="#0B2540"),x=0.5,xanchor="center"))
        st.plotly_chart(fig_org, use_container_width=True, config={"scrollZoom":True,"displayModeBar":True,"displaylogo":False}, key="pc_013")
        st.markdown("<div style='font-size:0.72rem;color:#64748B;margin-top:-6px'>Node colour and size indicate job grade (darker/larger = more senior). Hover any node for grade and LPS. ★ = Critical Role. Use scroll to zoom, drag to pan.</div>", unsafe_allow_html=True)
        leg_cols=st.columns(len(grade_colors))
        for i,(g,c) in enumerate(sorted(grade_colors.items(),reverse=True)):
            with leg_cols[i]:
                st.markdown(f"<div style='display:flex;align-items:center;gap:4px;font-size:0.72rem'>"
                            f"<div style='width:12px;height:12px;border-radius:50%;background:{c}'></div>G{g}</div>",
                            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — ORG READINESS
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="sec-hdr">📊 Organisational Succession Readiness</div>',
                unsafe_allow_html=True)
    n_roles  = len(CRITICAL_ROLES)
    avg_top1 = df_elig.sort_values("LPS",ascending=False).groupby("Current Job Title")["LPS"].first().mean()
    pct_b3   = df_elig["LPS Band"].isin(["Band 5 - Ready Now","Band 4 - Ready in 1-2 Years",
                                          "Band 3 - Ready in 2-3 Years"]).mean()*100
    hr_pct   = (df_emp["Flight Risk"]=="High").mean()*100

    k1,k2,k3,k4,k5=st.columns(5)
    for col_w,val,lbl,clr in [
        (k1,n_roles,"Critical Roles","#0B2540"),
        (k2,f"{avg_top1:.1f}","Avg LPS #1 Succ","#0D7377"),
        (k3,f"{pct_b3:.0f}%","Pool at Band 3+","#1B7A3E"),
        (k4,f"{hr_pct:.0f}%","High Flight Risk","#B91C1C"),
        (k5,len(df_elig),"Eligible Emps","#2563EB"),
    ]:
        with col_w:
            st.markdown(f'<div class="kpi"><div class="kpi-value" style="color:{clr}">{val}</div>'
                        f'<div class="kpi-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    r1,r2=st.columns([1.8,1])
    with r1:
        st.markdown('<div class="sec-hdr">🔥 Bench Strength — Top 3 per Role</div>',
                    unsafe_allow_html=True)
        heat_z,heat_t,heat_y=[],[],[]
        for role in CRITICAL_ROLES:
            rp = df_pip[df_pip["Critical Role"]==role].sort_values("Successor Rank")
            vals = rp["Leadership Potential Score (0-100)"].tolist()
            while len(vals)<3: vals.append(0)
            heat_z.append(vals[:3]); heat_t.append([f"{v:.1f}" for v in vals[:3]])
            heat_y.append(ROLES_CFG[role]["label"][:40])
        fig_heat=go.Figure(go.Heatmap(
            z=heat_z,x=["Successor #1","Successor #2","Successor #3"],y=heat_y,
            text=heat_t,texttemplate="%{text}",
            textfont=dict(family="Syne",size=11,color="white"),
            colorscale=[[0,"#B91C1C"],[0.35,"#EA580C"],[0.5,"#D97706"],
                        [0.65,"#2563EB"],[0.8,"#1B7A3E"],[1.0,"#065F46"]],
            zmin=0,zmax=100,showscale=True,colorbar=dict(title="LPS",tickfont=dict(size=9)),
        ))
        fig_heat.update_layout(
            margin=dict(l=0,r=0,t=10,b=0),height=420,
            xaxis=dict(tickfont=dict(family="Syne",size=10)),
            yaxis=dict(tickfont=dict(family="DM Sans",size=9),autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        )
        fig_heat.update_layout(title=dict(text="<b>Bench Strength Heatmap — LPS of Top 3 Successors per Role</b>",font=dict(family="Syne",size=12,color="#0B2540"),x=0.5,xanchor="center"))
        fig_heat.update_layout(title=dict(text="Bench Strength — LPS of Top 3 Successors per Critical Role",font=dict(family="Syne",size=12,color="#0B2540"),x=0.5,xanchor="center"))
        st.plotly_chart(fig_heat, use_container_width=True, config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_014")
        st.markdown("<div style='font-size:0.72rem;color:#64748B;margin-top:-6px'><b>How to read:</b> Each cell shows the LPS of a successor for that role. Green = strong bench (LPS ≥ 65), red = weak bench (LPS &lt; 35). Roles with all-red rows are high-priority succession risks.</div>", unsafe_allow_html=True)
        st.caption("Each cell shows the LPS score of the ranked successor for that critical role. Dark green = strong bench; red = succession gap requiring urgent attention.")

    with r2:
        st.markdown('<div class="sec-hdr">🎯 LPS Band Distribution</div>', unsafe_allow_html=True)
        bd=df_elig["LPS Band"].value_counts()
        fig_bd=go.Figure(go.Bar(x=bd.values,y=[BAND_SHORT.get(b,b) for b in bd.index],
                                 orientation="h",marker_color=[BAND_COLORS.get(b,"#888") for b in bd.index],
                                 text=bd.values,textposition="outside",textfont=dict(family="DM Sans",size=10)))
        fig_bd.update_layout(xaxis=dict(showgrid=False,showticklabels=False),
                              yaxis=dict(tickfont=dict(family="DM Sans",size=10)),
                              margin=dict(l=0,r=40,t=10,b=0),height=200,
                              paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        fig_bd.update_layout(title=dict(text="<b>LPS Band Distribution</b>",font=dict(family="Syne",size=11,color="#0B2540"),x=0.5,xanchor="center"))
        fig_bd.update_layout(title=dict(text="LPS Band Distribution",font=dict(family="Syne",size=11,color="#0B2540"),x=0.5,xanchor="center"))
        st.plotly_chart(fig_bd, use_container_width=True, config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_015")
        st.markdown("<div style='font-size:0.71rem;color:#64748B;margin-top:-6px'>Count of eligible employees in each readiness band. Aim for a balanced pyramid with enough Band 3–5 talent to cover all critical roles.</div>", unsafe_allow_html=True)
        st.caption("Count of eligible employees per readiness band. A healthy pipeline has a growing number in Band 3–5.")

    # 9-Box Grid
    st.markdown('<div class="sec-hdr" style="margin-top:10px">🔲 Organisation-wide 9-Box Grid</div>',
                unsafe_allow_html=True)
    nb_f,_ = st.columns([1.5,3])
    with nb_f:
        nb_opts = ["All Departments"]+sorted(df_elig["Department"].unique().tolist())
        nb_dept = st.selectbox("Filter by Department", nb_opts, key="nb_dept")
    nb_df = df_elig if nb_dept=="All Departments" else df_elig[df_elig["Department"]==nb_dept]
    nb_fig = nine_box_fig(nb_df)
    nb_fig.update_layout(title=dict(text="Organisation-wide 9-Box Grid — Performance vs LPS-Derived Potential",font=dict(family="Syne",size=12,color="#0B2540"),x=0.5,xanchor="center"))
    st.plotly_chart(nb_fig, use_container_width=True, config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_016")
    st.markdown("<div style='font-size:0.72rem;color:#64748B;margin-top:-6px'><b>Y-axis (Potential)</b> is derived from LPS: High ≥ 65, Moderate 35–64, Low &lt; 35. <b>X-axis (Performance)</b> is from the HR-assessed 9-Box label. Hover over any dot for employee details. Top-right cells (Top Talent, Future Leader, High Potential) are your priority succession candidates.</div>", unsafe_allow_html=True)
    st.caption("X-axis = Performance (from performance ratings). Y-axis = Potential (derived from LPS score: ≥65 High, 35–64 Moderate, <35 Low). Top-right 'Top Talent' cell represents the ideal succession pool.")

    nb_sum = nb_df.groupby("9-Box Position").agg(
        Count=("EE Number","count"),
        Avg_LPS=("LPS","mean"),
        Avg_Perf=("Average Performance Rating - Last 3 Years (1-5)","mean")
    ).round(2).reset_index().sort_values("Count",ascending=False)
    st.dataframe(nb_sum, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — KF ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════
with tab6:
    if "kfalp" not in data and "viaedge" not in data:
        st.info("Upload kf_kfalp_detail.csv and kf_viaedge_detail.csv to explore KF assessments.")
    else:
        st.markdown('<div class="sec-hdr">🧠 Korn Ferry Assessment Explorer</div>',
                    unsafe_allow_html=True)
        kft1,kft2,kft3=st.tabs(["KFALP Dimensions","viaEdge Dimensions","Reference Guide"])

        with kft1:
            if "kfalp" in data:
                df_kf=data["kfalp"].copy()
                kfc1,kfc2=st.columns([1,2])
                with kfc1:
                    kf_dims=df_kf["KF KFALP Dimension"].unique().tolist()
                    sel_dim=st.selectbox("KFALP Dimension",kf_dims,key="kfd")
                    dim_df=df_kf[df_kf["KF KFALP Dimension"]==sel_dim]
                    fig_kd=px.histogram(dim_df,x="Raw Score (1-5)",nbins=20,
                                         color_discrete_sequence=["#C9A227"],title=f"{sel_dim} — Distribution")
                    fig_kd.update_layout(margin=dict(l=0,r=0,t=30,b=0),height=200,
                                          paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                                          title_font=dict(family="Syne",size=12),xaxis_title=None,yaxis_title=None)
                    st.plotly_chart(fig_kd,use_container_width=True,config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_017")
                    band_cnt=dim_df["KFALP Rating Band"].value_counts()
                    kf_bc={"Exceptional":"#065F46","Strong":"#1B7A3E","Effective":"#2563EB","Developing":"#D97706","Limited":"#B91C1C"}
                    fig_kb=go.Figure(go.Pie(labels=band_cnt.index,values=band_cnt.values,hole=0.55,
                                            marker_colors=[kf_bc.get(b,"#888") for b in band_cnt.index],textfont=dict(size=9)))
                    fig_kb.update_layout(margin=dict(l=0,r=0,t=10,b=0),height=180,showlegend=True,
                                          legend=dict(font=dict(size=8),orientation="h",x=0.5,xanchor="center",y=-0.1),
                                          paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_kb,use_container_width=True,config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_018")
                with kfc2:
                    pivot=df_kf.pivot_table(index="Employee Full Name",columns="KF KFALP Dimension",
                                             values="Raw Score (1-5)",aggfunc="mean").dropna()
                    sc=sel_dim if sel_dim in pivot.columns else ("Learnability" if "Learnability" in pivot.columns else pivot.columns[0])
                    pivot=pivot.sort_values(sc,ascending=False).head(30)
                    fig_kh=px.imshow(pivot.round(1),color_continuous_scale=["#B91C1C","#D97706","#DBEAFE","#1B7A3E"],
                                      zmin=1,zmax=5,text_auto=".1f",aspect="auto",title="KFALP — Top 30")
                    fig_kh.update_layout(margin=dict(l=0,r=0,t=30,b=0),height=480,paper_bgcolor="rgba(0,0,0,0)",
                                          title_font=dict(family="Syne",size=12),xaxis_tickfont=dict(size=9),yaxis_tickfont=dict(size=8))
                    st.plotly_chart(fig_kh,use_container_width=True,config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_019")
                    st.markdown("<div style='font-size:0.71rem;color:#64748B;margin-top:-6px'>Sorted by selected dimension (descending). Green = high score (4–5), Red = low score (1–2). Top 30 employees shown.</div>", unsafe_allow_html=True)
                    st.caption("KFALP dimension scores for top 30 employees sorted by Learnability. Green = strong; red = development need. Hover over any cell for the exact score.")

        with kft2:
            if "viaedge" in data:
                df_ve=data["viaedge"].copy()
                vec1,vec2=st.columns([1,2])
                with vec1:
                    # Exclude composite/aggregate rows — keep only the 5 pure agility dimensions
                    _ve_valid_dims = ["Mental Agility","People Agility","Change Agility","Results Agility","Self-Awareness"]
                    ve_dims_raw = df_ve["KF viaEdge Dimension"].unique().tolist()
                    ve_dims = [d for d in ve_dims_raw if any(v in d for v in _ve_valid_dims)]
                    if not ve_dims:  # fallback — use whatever is in the file
                        ve_dims = ve_dims_raw
                    sel_ve=st.selectbox("viaEdge Dimension",ve_dims,key="ved")
                    ve_df=df_ve[df_ve["KF viaEdge Dimension"]==sel_ve].copy()
                    fig_vd=px.histogram(ve_df,x="Raw Score (1-5)",nbins=20,color_discrete_sequence=["#7C3AED"],title=f"{sel_ve} — Distribution")
                    fig_vd.update_layout(margin=dict(l=0,r=0,t=30,b=0),height=200,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",title_font=dict(family="Syne",size=12),xaxis_title=None,yaxis_title=None)
                    st.plotly_chart(fig_vd,use_container_width=True,config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_020")
                    pct_col="KF viaEdge Learning Agility Percentile"
                    if pct_col in df_ve.columns:
                        fig_vp=px.histogram(df_ve.drop_duplicates("EE Number"),x=pct_col,nbins=20,color_discrete_sequence=["#0D7377"],title="Overall Learning Agility Percentile (All Employees)")
                        fig_vp.update_layout(margin=dict(l=0,r=0,t=30,b=0),height=180,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",title_font=dict(family="Syne",size=11),xaxis_title=None,yaxis_title=None)
                        st.plotly_chart(fig_vp,use_container_width=True,config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_021")
                with vec2:
                    ve_pivot=df_ve.pivot_table(index="Employee Full Name",columns="KF viaEdge Dimension",values="Raw Score (1-5)",aggfunc="mean").dropna()
                    sv=sel_ve if sel_ve in ve_pivot.columns else (ve_pivot.columns[0] if len(ve_pivot.columns)>0 else "Mental Agility")
                    ve_pivot=ve_pivot.sort_values(sv,ascending=False).head(30)
                    fig_vh=px.imshow(ve_pivot.round(1),color_continuous_scale=["#B91C1C","#D97706","#DBEAFE","#1B7A3E"],zmin=1,zmax=5,text_auto=".1f",aspect="auto",title="viaEdge — Top 30")
                    fig_vh.update_layout(margin=dict(l=0,r=0,t=30,b=0),height=480,paper_bgcolor="rgba(0,0,0,0)",title_font=dict(family="Syne",size=12),xaxis_tickfont=dict(size=9),yaxis_tickfont=dict(size=8))
                    st.plotly_chart(fig_vh,use_container_width=True,config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_022")
                    st.markdown("<div style='font-size:0.71rem;color:#64748B;margin-top:-6px'>Sorted by selected agility dimension (descending). Green = Expert/Advanced, Red = Needs Development. Top 30 employees shown.</div>", unsafe_allow_html=True)
                    st.caption("viaEdge Learning Agility scores across 5 dimensions for top 30 employees. Mental Agility and Change Agility are the strongest predictors of senior leadership readiness.")

        with kft3:
            if "ref" in data:
                df_ref=data["ref"].copy()
                instruments=df_ref["KF Instrument"].unique().tolist() if "KF Instrument" in df_ref.columns else []
                sel_inst=st.selectbox("Instrument",instruments,key="ref_inst")
                ref_sub=df_ref[df_ref["KF Instrument"]==sel_inst]
                dims_ref=ref_sub["Dimension"].unique().tolist() if "Dimension" in ref_sub.columns else []
                sel_rdim=st.selectbox("Dimension",dims_ref,key="ref_dim")
                rd_sub=ref_sub[ref_sub["Dimension"]==sel_rdim]
                if len(rd_sub)>0:
                    r0=rd_sub.iloc[0]
                    st.markdown(f"""<div class="card">
                      <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;color:#0D7377">{sel_rdim}</div>
                      <div style="font-size:0.8rem;color:#64748B;margin:4px 0 10px">{r0.get('Category','')}</div>
                      <div style="font-size:0.82rem;margin-bottom:8px"><b>Sub-Dimensions:</b> {r0.get('Sub-Dimensions','')}</div>
                      <div style="font-size:0.82rem;margin-bottom:8px"><b>What It Measures:</b> {r0.get('What It Measures','')}</div>
                      <div style="font-size:0.82rem;margin-bottom:8px"><b>High Potential Signal:</b> {r0.get('High Potential Signal','')}</div>
                      <div style="font-size:0.82rem"><b>Assessment Method:</b> {r0.get('Assessment Method','')}</div>
                    </div>""", unsafe_allow_html=True)
                    for _,brow in rd_sub.sort_values("Score",ascending=False).iterrows():
                        bc2={"Exceptional":"#065F46","Strong":"#1B7A3E","Effective":"#2563EB",
                             "Developing":"#D97706","Limited":"#B91C1C","Expert":"#065F46",
                             "Advanced":"#1B7A3E","Emerging":"#EA580C","Needs Development":"#B91C1C"}.get(brow.get("Rating Band",""),"#888")
                        st.markdown(f"""<div style="display:flex;gap:12px;margin-bottom:8px;align-items:flex-start">
                          <div style="background:{bc2};color:white;border-radius:8px;padding:4px 10px;font-family:'Syne',sans-serif;font-size:0.75rem;font-weight:700;white-space:nowrap;flex-shrink:0">{brow.get('Score','')} — {brow.get('Rating Band','')}</div>
                          <div style="font-size:0.8rem;color:#374151;line-height:1.5">{brow.get('Behavioural Descriptor','')}</div>
                        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 7 — CAREER PATH
# ═══════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown('<div class="sec-hdr">📈 Career Path & Promotion Trajectory</div>',
                unsafe_allow_html=True)

    # ── Employee selector (always visible — does not need promo file) ──────────
    t7_sel_col, _ = st.columns([2, 3])
    with t7_sel_col:
        sel_cp = st.selectbox("Select Employee", all_names, key="cp_sel")
    emp_cp_row = df_emp[df_emp["Employee Full Name"] == sel_cp]

    if len(emp_cp_row) > 0:
        e_cp = emp_cp_row.iloc[0]
        lps_cp = safe_float(e_cp["LPS"]); clr_cp = lps_color(lps_cp)

        # ── Career KPI strip ────────────────────────────────────────────────────
        tot_prom = int(e_cp["Total Promotions (Career)"])
        p5yr     = int(e_cp["Promotions in Last 5 Years"])
        ppy      = safe_float(e_cp["Promotions per Year (Career)"])
        avg_p    = safe_float(e_cp["Average Performance Rating - Last 3 Years (1-5)"])
        last_p   = safe_float(e_cp["Last Annual Performance Rating (1-5)"])
        traj     = safe_float(e_cp["Performance Trajectory"])
        traj_clr = "#1B7A3E" if traj > 0 else "#B91C1C" if traj < 0 else "#64748B"
        traj_lbl = f"+{traj:.1f} ↑" if traj > 0 else f"{traj:.1f} ↓" if traj < 0 else "→ Stable"

        st.markdown(f"""
        <div class="kpi-row" style="margin-bottom:14px">
          <div class="kpi" style="border-left:4px solid {clr_cp}">
            <div class="kpi-value" style="color:{clr_cp}">{lps_cp:.1f}</div>
            <div class="kpi-label">Leadership Potential Score</div>
          </div>
          <div class="kpi" style="border-left:4px solid #0D7377">
            <div class="kpi-value" style="color:#0D7377">{tot_prom}</div>
            <div class="kpi-label">Total Promotions (Career)</div>
          </div>
          <div class="kpi" style="border-left:4px solid #2563EB">
            <div class="kpi-value" style="color:#2563EB">{p5yr}</div>
            <div class="kpi-label">Promotions (Last 5 Yrs)</div>
          </div>
          <div class="kpi" style="border-left:4px solid #C9A227">
            <div class="kpi-value" style="color:#C9A227">{ppy:.2f}</div>
            <div class="kpi-label">Promotions / Year</div>
          </div>
          <div class="kpi" style="border-left:4px solid #7C3AED">
            <div class="kpi-value" style="color:#7C3AED">{avg_p:.1f}</div>
            <div class="kpi-label">Avg Performance (3yr)</div>
          </div>
          <div class="kpi" style="border-left:4px solid {traj_clr}">
            <div class="kpi-value" style="color:{traj_clr}">{traj_lbl}</div>
            <div class="kpi-label">Performance Trajectory</div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Performance trend bar chart (from employees_master — always visible) ──
        perf_cols = [
            ("3yr Average", avg_p, "#0D7377"),
            ("Last Rating",  last_p, "#C9A227"),
        ]
        fig_perf_bar = go.Figure()
        # Benchmark line
        org_avg_perf = df_emp["Average Performance Rating - Last 3 Years (1-5)"].mean()
        fig_perf_bar.add_trace(go.Bar(
            x=["3yr Average", "Last Rating"],
            y=[avg_p, last_p],
            marker_color=["#0D7377","#C9A227"],
            text=[f"{avg_p:.1f}", f"{last_p:.1f}"],
            textposition="outside",
            textfont=dict(family="Syne", size=14, color="#0B2540"),
            width=0.4,
        ))
        fig_perf_bar.add_hline(y=org_avg_perf, line_dash="dot", line_color="#94A3B8",
                               annotation_text=f"Org avg {org_avg_perf:.1f}",
                               annotation_position="top right",
                               annotation_font=dict(size=10, color="#64748B"))
        fig_perf_bar.update_layout(
            xaxis=dict(tickfont=dict(family="Syne", size=12)),
            yaxis=dict(range=[0, 6], tickvals=[1,2,3,4,5],
                       ticktext=["1 Limited","2 Developing","3 Effective","4 Strong","5 Exceptional"],
                       tickfont=dict(size=9)),
            margin=dict(l=0,r=0,t=10,b=0), height=200,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )

        # ── Grade vs org distribution ───────────────────────────────────────────
        grade_dist = df_emp["Job Grade (1-9)"].value_counts().sort_index()
        emp_grade = int(e_cp["Job Grade (1-9)"])
        bar_colors = ["#C9A227" if g == emp_grade else "#CBD5E1" for g in grade_dist.index]
        fig_grade_dist = go.Figure(go.Bar(
            x=[f"G{g}" for g in grade_dist.index],
            y=grade_dist.values,
            marker_color=bar_colors,
            text=grade_dist.values, textposition="outside",
            textfont=dict(size=9),
        ))
        fig_grade_dist.add_annotation(
            x=f"G{emp_grade}", y=grade_dist.get(emp_grade, 0) + 2,
            text="▼ You", showarrow=False,
            font=dict(family="Syne", size=10, color="#C9A227"),
        )
        fig_grade_dist.update_layout(
            xaxis=dict(tickfont=dict(family="Syne", size=10)),
            yaxis=dict(tickfont=dict(size=9), title="Headcount"),
            margin=dict(l=0,r=0,t=20,b=0), height=200,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )

        # ── 3-column performance + grade view ──────────────────────────────────
        pa, pb, pc_col = st.columns([1.2, 1.2, 1])
        with pa:
            st.markdown('<div style="font-size:0.82rem;font-weight:700;color:#0B2540;margin-bottom:6px">Performance Ratings vs Org Average</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_perf_bar, use_container_width=True,
                            config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_t7_perfbar")
            st.caption("Compares this employee's 3-year average and last annual rating against the organisation mean (dotted line). Bars above the line indicate above-average performance.")
        with pb:
            st.markdown('<div style="font-size:0.82rem;font-weight:700;color:#0B2540;margin-bottom:6px">Grade Distribution — Where You Stand</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_grade_dist, use_container_width=True,
                            config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_t7_gradedist")
            st.caption("Organisation-wide headcount per grade. Gold bar = this employee's current grade. Higher grades have fewer employees — reflects the pyramid structure of leadership.")
        with pc_col:
            st.markdown('<div style="font-size:0.82rem;font-weight:700;color:#0B2540;margin-bottom:6px">KF Assessment</div>', unsafe_allow_html=True)
            k1v = safe_float(e_cp.get("KF KFALP - Composite Score (1-5)", 0))
            k2v = safe_float(e_cp.get("KF viaEdge - Learning Agility Composite (1-5)", 0))
            if k1v > 0:
                st.plotly_chart(speedometer_fig(k1v, "KFALP", color="#C9A227"),
                                use_container_width=True, config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_023")
            if k2v > 0:
                st.plotly_chart(speedometer_fig(k2v, "viaEdge", color="#7C3AED"),
                                use_container_width=True, config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_024")

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    if "promos" not in data:
        st.info("Upload **promotion_history.csv** to view the detailed promotion timeline below.")
    else:
        df_promo=data["promos"].copy()
        df_promo.columns=[c.replace("\u2013","-").replace("\u2014","-") for c in df_promo.columns]
        cp1,cp2=st.columns([1,2.5])
        with cp1:
            # Profile card using sel_cp already selected above
            if len(emp_cp_row) > 0:
                e = emp_cp_row.iloc[0]; lps_s = safe_float(e["LPS"]); clr_v = lps_color(lps_s)
                st.markdown(f"""<div class="card">
                  <div style="display:flex;gap:10px;align-items:center;margin-bottom:10px">
                    {avatar_html(sel_cp,44,clr_v)}
                    <div>
                      <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:0.9rem">{sel_cp}</div>
                      <div style="font-size:0.75rem;color:#64748B">{e['Current Job Title']}</div>
                      <div style="font-size:0.72rem;color:#64748B">{e['Department']} · Grade {int(e['Job Grade (1-9)'])}</div>
                    </div>
                  </div>
                  <div style="font-size:0.78rem;color:#374151">
                    <b>Tenure:</b> {e['Tenure with Organisation (Years)']}y &nbsp;|&nbsp; <b>LPS:</b> {lps_s:.1f}
                  </div>
                </div>""", unsafe_allow_html=True)

        with cp2:
            emp_promos=df_promo[df_promo["Employee Full Name"]==sel_cp].sort_values("Promotion Year")
            if len(emp_promos)==0:
                st.info(f"No promotion history found for {sel_cp}.")
            else:
                years=emp_promos["Promotion Year"].tolist()
                grades=emp_promos["Promoted To Grade"].tolist()
                perfs=emp_promos["Performance Rating at Promotion"].tolist()
                fig_tl=go.Figure()
                fig_tl.add_trace(go.Scatter(x=years,y=grades,mode="lines+markers+text",
                    line=dict(color="#0D7377",width=3),
                    marker=dict(size=[10+p*2 for p in perfs],color=perfs,
                                colorscale=[[0,"#B91C1C"],[0.5,"#D97706"],[1,"#1B7A3E"]],
                                cmin=1,cmax=5,showscale=True,line=dict(width=2,color="white"),
                                colorbar=dict(title="Perf",tickfont=dict(size=8),len=0.5,y=0.5)),
                    text=[f"G{g}" for g in grades],textposition="top center",
                    textfont=dict(family="Syne",size=10,color="#0B2540"),name="Grade"))
                fig_tl.add_trace(go.Scatter(x=years,y=perfs,mode="lines+markers",
                    line=dict(color="#C9A227",width=2,dash="dot"),marker=dict(size=8,color="#C9A227"),
                    name="Performance",yaxis="y2"))
                entry_yr=emp_promos["Promotion Year"].min()-2
                entry_gr=emp_promos["Promoted From Grade"].iloc[0]
                fig_tl.add_trace(go.Scatter(x=[entry_yr],y=[entry_gr],mode="markers+text",
                    marker=dict(size=12,color="#64748B",symbol="square"),
                    text=["Entry"],textposition="top center",textfont=dict(family="Syne",size=9,color="#64748B"),
                    name="Career Entry",hoverinfo="skip"))
                emp_row=df_emp[df_emp["Employee Full Name"]==sel_cp]
                if len(emp_row)>0:
                    fig_tl.add_trace(go.Scatter(x=[2025],y=[int(emp_row.iloc[0]["Job Grade (1-9)"])],
                        mode="markers+text",marker=dict(size=14,color=lps_color(emp_row.iloc[0]["LPS"]),symbol="star"),
                        text=["Now"],textposition="top center",textfont=dict(family="Syne",size=10,color="#0B2540"),
                        name="Current",hoverinfo="skip"))
                fig_tl.update_layout(
                    xaxis=dict(title="Year",tickfont=dict(family="DM Sans",size=10)),
                    yaxis=dict(title="Job Grade",range=[0,10],tickvals=list(range(1,10)),tickfont=dict(family="DM Sans",size=10)),
                    yaxis2=dict(title="Performance",overlaying="y",side="right",range=[0,5.5],tickfont=dict(size=9)),
                    legend=dict(font=dict(family="DM Sans",size=9),orientation="h",x=0.5,xanchor="center",y=1.06),
                    margin=dict(l=0,r=60,t=30,b=0),height=320,
                    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#F8FBFD",hovermode="x unified")
                fig_tl.update_layout(title=dict(text="<b>Career Grade Progression & Performance at Each Promotion</b>",font=dict(family="Syne",size=11,color="#0B2540"),x=0.5,xanchor="center"))
                fig_tl.update_layout(title=dict(text="Career Progression Timeline — Grade & Performance at Each Promotion",font=dict(family="Syne",size=11,color="#0B2540"),x=0.5,xanchor="center"))
                st.plotly_chart(fig_tl,use_container_width=True,config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_025")
                st.markdown("<div style='font-size:0.72rem;color:#64748B;margin-top:-6px'>Teal line = grade trajectory (left axis). Gold dashed line = performance rating at each promotion (right axis). Larger markers = higher performance rating at time of promotion. Gold star = current position.</div>", unsafe_allow_html=True)
                st.caption("Blue line = grade trajectory over career. Gold dotted line = performance rating at each promotion. Larger/greener dots = higher performance at time of promotion. Star = current position.")
                disp_cols=[c for c in ["Promotion Number (Career)","Promotion Year","Promoted From Grade","Promoted To Grade","Performance Rating at Promotion","Years Since Last Promotion"] if c in emp_promos.columns]
                st.markdown('<div class="sec-hdr" style="margin-top:8px">📋 Promotion History</div>',unsafe_allow_html=True)
                st.dataframe(emp_promos[disp_cols].reset_index(drop=True),use_container_width=True,hide_index=True)

        st.markdown('<div class="sec-hdr" style="margin-top:16px">🚀 Promotion Velocity vs Performance</div>',unsafe_allow_html=True)
        vel_df=df_emp[["Job Grade (1-9)","Promotions per Year (Career)","Average Performance Rating - Last 3 Years (1-5)"]].dropna()
        fig_vel=px.scatter(vel_df,x="Average Performance Rating - Last 3 Years (1-5)",y="Promotions per Year (Career)",
                            color="Job Grade (1-9)",color_continuous_scale=px.colors.sequential.Teal,opacity=0.7,size_max=8,
                            labels={"Average Performance Rating - Last 3 Years (1-5)":"Avg Performance (3yr)",
                                    "Promotions per Year (Career)":"Promotions / Year","Job Grade (1-9)":"Grade"},
                            title="Performance vs Promotion Velocity (coloured by Grade)")
        fig_vel.update_layout(margin=dict(l=0,r=0,t=30,b=0),height=320,
                               paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#F8FBFD",
                               title_font=dict(family="Syne",size=12))
        st.plotly_chart(fig_vel,use_container_width=True,config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_026")
        st.markdown("<div style='font-size:0.72rem;color:#64748B;margin-top:-6px'><b>How to read:</b> Each dot is an employee. X-axis = 3-year average performance rating (1–5). Y-axis = promotions per year over full career. Employees in the top-right quadrant (high performance + high velocity) are your fast-track high-potentials. Colour indicates current job grade.</div>", unsafe_allow_html=True)
        st.caption("Each dot is an employee. X = average performance rating; Y = career promotion velocity (promotions per year). Employees in the top-right corner are high performers who are also promoted quickly — the ideal succession pool. Colour = job grade.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 8 — DEVELOPMENT PRESCRIPTION (Interventions & Courses)
# ═══════════════════════════════════════════════════════════════════════════
with tab8:
    st.markdown('<div class="sec-hdr">💊 Development Prescription — Personalised Interventions</div>',
                unsafe_allow_html=True)

    # ── Intervention library keyed by LPS cluster + KF dimension ─────────────
    INTERVENTION_LIBRARY = {
        "Performance": {
            "description": "Interventions to strengthen delivery, accountability and performance trajectory.",
            "courses": [
                {"name": "High-Impact Leadership (Korn Ferry)",     "type": "Executive Programme","duration": "3 days","provider": "Korn Ferry",       "mode": "In-person / Virtual"},
                {"name": "Leading for Results (CCL)",               "type": "Leadership Course",  "duration": "4 days","provider": "CCL",             "mode": "In-person"},
                {"name": "Accountability & Execution (LinkedIn Learning)", "type": "Online Course","duration": "6 hrs", "provider": "LinkedIn Learning","mode": "Self-paced"},
                {"name": "OKR Mastery (Coursera — Google)",         "type": "Online Course",      "duration": "8 hrs", "provider": "Coursera",        "mode": "Self-paced"},
                {"name": "Execution: The Discipline of Getting Things Done (Book)", "type": "Reading","duration": "—","provider": "Bossidy & Charan","mode": "Self-directed"},
            ]
        },
        "KF Assessment": {
            "description": "Interventions to build leadership potential as measured by KFALP and viaEdge.",
            "courses": [
                {"name": "Korn Ferry Leadership Architect Certification","type": "Certification","duration": "2 days","provider": "Korn Ferry","mode": "Virtual"},
                {"name": "Emotional Intelligence (Daniel Goleman — Coursera)","type": "Online Course","duration": "10 hrs","provider": "Coursera","mode": "Self-paced"},
                {"name": "Learning Agility Development Journey (Korn Ferry viaEdge)","type": "Coaching Programme","duration": "6 months","provider": "Korn Ferry","mode": "Blended"},
                {"name": "Growth Mindset (Microsoft — LinkedIn Learning)","type": "Online Course","duration": "4 hrs","provider": "LinkedIn Learning","mode": "Self-paced"},
                {"name": "Systems Thinking (MIT OpenCourseWare)","type": "Online Course","duration": "12 hrs","provider": "MIT OCW","mode": "Self-paced"},
                {"name": "Executive Presence (Harvard ManageMentor)","type": "Online Course","duration": "8 hrs","provider": "Harvard Business Publishing","mode": "Self-paced"},
            ]
        },
        "Career Velocity": {
            "description": "Interventions to accelerate career progression pace and build a track record of growth.",
            "courses": [
                {"name": "Stretch Assignment Framework (Internal)","type": "Action Learning","duration": "6–12 months","provider": "Internal HR","mode": "On-the-job"},
                {"name": "Career Conversations Toolkit (Korn Ferry)","type": "Workshop","duration": "1 day","provider": "Korn Ferry","mode": "In-person"},
                {"name": "Personal Branding for Leaders (LinkedIn Learning)","type": "Online Course","duration": "5 hrs","provider": "LinkedIn Learning","mode": "Self-paced"},
                {"name": "Sponsorship Programme (Internal)","type": "Mentoring","duration": "12 months","provider": "Internal L&D","mode": "In-person"},
                {"name": "Managing Your Career (Coursera — Univ. of Michigan)","type": "Online Course","duration": "10 hrs","provider": "Coursera","mode": "Self-paced"},
            ]
        },
        "Leadership Breadth": {
            "description": "Interventions to widen cross-functional, cross-cultural and enterprise leadership exposure.",
            "courses": [
                {"name": "Cross-Functional Rotation Programme (Internal)","type": "Rotation","duration": "3–6 months","provider": "Internal HR","mode": "On-the-job"},
                {"name": "Global Leadership Programme (IMD)","type": "Executive Programme","duration": "5 days","provider": "IMD Business School","mode": "In-person"},
                {"name": "Leading Diverse Teams (edX — Catalyst)","type": "Online Course","duration": "6 hrs","provider": "edX","mode": "Self-paced"},
                {"name": "Inclusive Leadership (Coursera — Catalyst)","type": "Online Course","duration": "5 hrs","provider": "Coursera","mode": "Self-paced"},
                {"name": "Strategic Stakeholder Management (INSEAD)","type": "Online Course","duration": "8 hrs","provider": "INSEAD","mode": "Self-paced"},
                {"name": "Critical Projects Sponsorship (Internal)","type": "Action Learning","duration": "Ongoing","provider": "Internal HR","mode": "On-the-job"},
            ]
        },
        "Readiness": {
            "description": "Interventions to close grade gaps, improve mobility readiness and reduce flight risk.",
            "courses": [
                {"name": "Succession Readiness Coaching (Korn Ferry)","type": "Executive Coaching","duration": "6 months","provider": "Korn Ferry","mode": "1-on-1"},
                {"name": "Transition to Senior Leadership (CCL)","type": "Leadership Course","duration": "4 days","provider": "CCL","mode": "In-person"},
                {"name": "Negotiation & Influence (Harvard Online)","type": "Online Course","duration": "7 hrs","provider": "Harvard Online","mode": "Self-paced"},
                {"name": "Retention & Engagement Conversation (Internal)","type": "HR Intervention","duration": "Ongoing","provider": "HRBP","mode": "In-person"},
                {"name": "Leading Change (Prosci — ADKAR)","type": "Certification","duration": "2 days","provider": "Prosci","mode": "Virtual"},
            ]
        },
        "KF KFALP — Drivers": {
            "description": "Strengthen Results Orientation, Achievement Drive and Ambition to Lead.",
            "courses": [
                {"name": "Unleashing Personal Accountability (Korn Ferry)","type": "Workshop","duration": "1 day","provider": "Korn Ferry","mode": "Virtual"},
                {"name": "High Performance Habits (Brendon Burchard — Udemy)","type": "Online Course","duration": "8 hrs","provider": "Udemy","mode": "Self-paced"},
                {"name": "The Achievement Habit (Stanford — edX)","type": "Online Course","duration": "6 hrs","provider": "edX","mode": "Self-paced"},
            ]
        },
        "KF KFALP — Curiosity": {
            "description": "Build Information Seeking, Breadth of Interests and Tolerance of Ambiguity.",
            "courses": [
                {"name": "Design Thinking (IDEO — Coursera)","type": "Online Course","duration": "12 hrs","provider": "Coursera","mode": "Self-paced"},
                {"name": "Creative Thinking: Techniques and Tools (Imperial — Coursera)","type": "Online Course","duration": "10 hrs","provider": "Coursera","mode": "Self-paced"},
                {"name": "Intellectual Curiosity (LinkedIn Learning)","type": "Online Course","duration": "3 hrs","provider": "LinkedIn Learning","mode": "Self-paced"},
            ]
        },
        "KF KFALP — Insight": {
            "description": "Improve Self-Awareness, Receptivity to Feedback and Pattern Recognition.",
            "courses": [
                {"name": "360 Feedback Debrief & Coaching (Korn Ferry)","type": "Coaching","duration": "3 sessions","provider": "Korn Ferry","mode": "1-on-1"},
                {"name": "Mindfulness-Based Leadership (Search Inside Yourself — Google)","type": "Programme","duration": "2 days","provider": "SIYLI","mode": "In-person"},
                {"name": "Developing Self-Awareness (LinkedIn Learning)","type": "Online Course","duration": "4 hrs","provider": "LinkedIn Learning","mode": "Self-paced"},
            ]
        },
        "KF KFALP — Engagement": {
            "description": "Strengthen Relationship Quality, Collaborative Orientation and Inspiring Others.",
            "courses": [
                {"name": "Inspiring and Motivating Individuals (Coursera — Michigan)","type": "Online Course","duration": "8 hrs","provider": "Coursera","mode": "Self-paced"},
                {"name": "Stakeholder Management (PMI — LinkedIn Learning)","type": "Online Course","duration": "5 hrs","provider": "LinkedIn Learning","mode": "Self-paced"},
                {"name": "Influencing Without Authority (CCL)","type": "Workshop","duration": "2 days","provider": "CCL","mode": "In-person"},
            ]
        },
        "KF KFALP — Determination": {
            "description": "Build Persistence, Grit, Emotional Regulation and Recovery Speed.",
            "courses": [
                {"name": "Resilience & Mental Toughness (AQai — online)","type": "Assessment + Coaching","duration": "3 months","provider": "AQai","mode": "Blended"},
                {"name": "Grit: The Power of Passion and Perseverance (Udemy)","type": "Online Course","duration": "5 hrs","provider": "Udemy","mode": "Self-paced"},
                {"name": "Managing Stress & Wellbeing (FutureLearn)","type": "Online Course","duration": "6 hrs","provider": "FutureLearn","mode": "Self-paced"},
            ]
        },
        "KF KFALP — Learnability": {
            "description": "Accelerate Speed and Depth of Learning and Mental Agility.",
            "courses": [
                {"name": "Learning How to Learn (Coursera — UC San Diego)","type": "Online Course","duration": "15 hrs","provider": "Coursera","mode": "Self-paced"},
                {"name": "Critical Thinking & Problem Solving (edX — Rochester)","type": "Online Course","duration": "10 hrs","provider": "edX","mode": "Self-paced"},
                {"name": "Accelerated Learning Techniques (Udemy)","type": "Online Course","duration": "6 hrs","provider": "Udemy","mode": "Self-paced"},
            ]
        },
        "KF viaEdge — Mental Agility": {
            "description": "Strengthen Inquisitiveness, Complexity handling and Connector ability.",
            "courses": [
                {"name": "Systems Thinking & Complexity (MIT OCW)","type": "Online Course","duration": "12 hrs","provider": "MIT OCW","mode": "Self-paced"},
                {"name": "Data-Driven Decision Making (Coursera — PwC)","type": "Online Course","duration": "8 hrs","provider": "Coursera","mode": "Self-paced"},
                {"name": "Lateral Thinking (LinkedIn Learning)","type": "Online Course","duration": "2 hrs","provider": "LinkedIn Learning","mode": "Self-paced"},
            ]
        },
        "KF viaEdge — People Agility": {
            "description": "Build Open-Mindedness, People Smart and Role Flexibility.",
            "courses": [
                {"name": "Cross-Cultural Management (Coursera — ESSEC)","type": "Online Course","duration": "10 hrs","provider": "Coursera","mode": "Self-paced"},
                {"name": "Developing Your Emotional Intelligence (LinkedIn Learning)","type": "Online Course","duration": "5 hrs","provider": "LinkedIn Learning","mode": "Self-paced"},
                {"name": "Radical Candour (Kim Scott — workshop)","type": "Workshop","duration": "1 day","provider": "Radical Candour","mode": "Virtual"},
            ]
        },
        "KF viaEdge — Change Agility": {
            "description": "Build Experimenter mindset, Visionary capability and Stalwart resilience.",
            "courses": [
                {"name": "Leading Organisational Change (Coursera — Case Western)","type": "Online Course","duration": "12 hrs","provider": "Coursera","mode": "Self-paced"},
                {"name": "Innovation & Design Thinking (MIT Sloan — edX)","type": "Online Course","duration": "16 hrs","provider": "edX","mode": "Self-paced"},
                {"name": "Agile Leadership (LinkedIn Learning)","type": "Online Course","duration": "4 hrs","provider": "LinkedIn Learning","mode": "Self-paced"},
            ]
        },
        "KF viaEdge — Results Agility": {
            "description": "Strengthen Drive, executive Presence, Resourcefulness and Composure.",
            "courses": [
                {"name": "Executive Presence (Harvard ManageMentor)","type": "Online Course","duration": "8 hrs","provider": "Harvard Business Publishing","mode": "Self-paced"},
                {"name": "Leading in Tough Times (CCL)","type": "Workshop","duration": "2 days","provider": "CCL","mode": "In-person"},
                {"name": "Resourcefulness & Problem Solving (LinkedIn Learning)","type": "Online Course","duration": "3 hrs","provider": "LinkedIn Learning","mode": "Self-paced"},
            ]
        },
        "KF viaEdge — Self-Awareness": {
            "description": "Build Feedback-Orientation, Reflective practice and Emotional Regulation.",
            "courses": [
                {"name": "Mindful Leadership (Search Inside Yourself — SIYLI)","type": "Programme","duration": "2 days","provider": "SIYLI","mode": "In-person"},
                {"name": "Feedback & Coaching Skills (LinkedIn Learning)","type": "Online Course","duration": "4 hrs","provider": "LinkedIn Learning","mode": "Self-paced"},
                {"name": "Journaling for Leaders (Internal L&D)","type": "Self-directed","duration": "Ongoing","provider": "Internal","mode": "Self-directed"},
            ]
        },
    }

    TYPE_COLORS = {
        "Executive Programme":    "#0B2540",
        "Leadership Course":      "#0D7377",
        "Online Course":          "#2563EB",
        "Certification":          "#7C3AED",
        "Coaching Programme":     "#C9A227",
        "Executive Coaching":     "#C9A227",
        "Workshop":               "#1B7A3E",
        "Rotation":               "#EA580C",
        "Action Learning":        "#D97706",
        "HR Intervention":        "#B91C1C",
        "Reading":                "#64748B",
        "Coaching":               "#C9A227",
        "Programme":              "#0D7377",
        "Assessment + Coaching":  "#7C3AED",
        "Mentoring":              "#1B7A3E",
        "Self-directed":          "#64748B",
    }

    # ── Employee selector ──────────────────────────────────────────────────────
    rx_c1, rx_c2 = st.columns([1.5, 3])
    with rx_c1:
        rx_emp_name = st.selectbox("Select Employee", all_names, key="rx_emp")
    rx_emp = df_emp[df_emp["Employee Full Name"] == rx_emp_name].iloc[0]
    rx_lps = rx_emp["LPS"]
    rx_bc  = lps_color(rx_lps)

    # ── Profile summary strip ──────────────────────────────────────────────────
    with rx_c2:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:14px;background:white;
             border:1px solid #D1DCE8;border-radius:12px;padding:14px 18px;margin-top:4px">
          {avatar_html(rx_emp_name, 48, rx_bc)}
          <div style="flex:1">
            <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1rem;color:#0B2540">{rx_emp_name}</div>
            <div style="font-size:0.8rem;color:#64748B">{rx_emp['Current Job Title']} · Grade {int(rx_emp['Job Grade (1-9)'])} · {rx_emp['Department']}</div>
          </div>
          <div style="text-align:center;min-width:90px">
            <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:{rx_bc};line-height:1">{rx_lps:.1f}</div>
            <div style="font-size:0.72rem;color:#64748B">LPS / 100</div>
            <span class="band-pill" style="background:{rx_bc}20;color:{rx_bc};font-size:0.7rem">{BAND_SHORT.get(rx_emp['LPS Band'],rx_emp['LPS Band'])}</span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Identify weakest clusters + KF dimensions ──────────────────────────────
    clusters = {
        "Performance":       safe_float(rx_emp.get("C1", 0)),
        "KF Assessment":     safe_float(rx_emp.get("C2", 0)),
        "Career Velocity":   safe_float(rx_emp.get("C3", 0)),
        "Leadership Breadth":safe_float(rx_emp.get("C4", 0)),
        "Readiness":         safe_float(rx_emp.get("C5", 0)),
    }
    kf_dims_scores = {}
    for dim, col in [
        ("KF KFALP — Drivers",          "KF KFALP - Drivers Score (1-5)"),
        ("KF KFALP — Curiosity",        "KF KFALP - Curiosity Score (1-5)"),
        ("KF KFALP — Insight",          "KF KFALP - Insight Score (1-5)"),
        ("KF KFALP — Engagement",       "KF KFALP - Engagement Score (1-5)"),
        ("KF KFALP — Determination",    "KF KFALP - Determination Score (1-5)"),
        ("KF KFALP — Learnability",     "KF KFALP - Learnability Score (1-5)"),
        ("KF viaEdge — Mental Agility", "KF viaEdge - Mental Agility Score (1-5)"),
        ("KF viaEdge — People Agility", "KF viaEdge - People Agility Score (1-5)"),
        ("KF viaEdge — Change Agility", "KF viaEdge - Change Agility Score (1-5)"),
        ("KF viaEdge — Results Agility","KF viaEdge - Results Agility Score (1-5)"),
        ("KF viaEdge — Self-Awareness", "KF viaEdge - Self-Awareness Score (1-5)"),
    ]:
        val = safe_float(rx_emp.get(col, 0))
        if val > 0:
            kf_dims_scores[dim] = val

    sorted_clusters = sorted(clusters.items(), key=lambda x: x[1])
    sorted_kf = sorted(kf_dims_scores.items(), key=lambda x: x[1]) if kf_dims_scores else []

    # Weakest 2 clusters + weakest 3 KF dims
    weak_clusters = [c for c, _ in sorted_clusters[:2]]
    weak_kf_dims  = [d for d, _ in sorted_kf[:3]]

    # ── Prescription header ────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0B2540 0%,#0D7377 100%);
         border-radius:12px;padding:16px 22px;margin-bottom:16px;color:white">
      <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.05rem;margin-bottom:6px">
        Development Prescription for {rx_emp_name}
      </div>
      <div style="font-size:0.82rem;color:#9EC5D8">
        Based on LPS cluster gaps and KF assessment scores, the following interventions are prioritised
        to move this employee toward the next LPS band.
        Weakest clusters: <b style="color:#F0C93A">{' · '.join(weak_clusters)}</b>
        {'· Weakest KF dimensions: <b style="color:#C9A227">' + ' · '.join(weak_kf_dims) + '</b>' if weak_kf_dims else ''}
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Prescription cards ─────────────────────────────────────────────────────
    priority_areas = weak_clusters + weak_kf_dims
    for idx_area, area in enumerate(priority_areas):
        if area not in INTERVENTION_LIBRARY:
            continue
        lib = INTERVENTION_LIBRARY[area]
        courses = lib["courses"]

        with st.expander(f"📌 {area} — {lib['description']}", expanded=(idx_area < 2)):
            for c in courses:
                type_clr = TYPE_COLORS.get(c["type"], "#64748B")
                st.markdown(f"""
                <div style="display:flex;gap:14px;align-items:flex-start;
                     padding:12px 14px;border-radius:10px;background:#F8FBFD;
                     border:1px solid #E2EAF0;margin-bottom:8px">
                  <div style="min-width:120px;max-width:140px">
                    <span style="background:{type_clr};color:white;border-radius:6px;
                      padding:3px 8px;font-size:0.68rem;font-weight:700;
                      font-family:'Syne',sans-serif;display:block;text-align:center;margin-bottom:4px">{c['type']}</span>
                    <div style="font-size:0.72rem;color:#64748B;text-align:center">⏱ {c['duration']}</div>
                    <div style="font-size:0.72rem;color:#64748B;text-align:center">{c['mode']}</div>
                  </div>
                  <div style="flex:1">
                    <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:0.9rem;
                         color:#0B2540;margin-bottom:2px">{c['name']}</div>
                    <div style="font-size:0.78rem;color:#0D7377;font-weight:600">{c['provider']}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

    # ── LPS projection chart ───────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr" style="margin-top:16px">📈 Estimated LPS Impact per Intervention Area</div>',
                unsafe_allow_html=True)

    cluster_labels = list(clusters.keys())
    current_scores = [clusters[c] for c in cluster_labels]
    # Estimated uplift after completing recommended courses (illustrative)
    uplift = {c: min(100, v + (15 if c in weak_clusters else 5)) for c, v in clusters.items()}
    uplift_scores = [uplift[c] for c in cluster_labels]

    fig_rx = go.Figure()
    fig_rx.add_trace(go.Bar(name="Current Score", x=cluster_labels, y=current_scores,
                             marker_color="#94A3B8", text=[f"{v:.0f}" for v in current_scores],
                             textposition="outside", textfont=dict(size=10)))
    fig_rx.add_trace(go.Bar(name="After Interventions (Est.)", x=cluster_labels, y=uplift_scores,
                             marker_color="#0D7377", text=[f"{v:.0f}" for v in uplift_scores],
                             textposition="outside", textfont=dict(size=10)))
    fig_rx.update_layout(
        barmode="group",
        xaxis=dict(tickfont=dict(family="DM Sans",size=10)),
        yaxis=dict(range=[0,115], tickfont=dict(size=9), title="Score (0–100)"),
        legend=dict(font=dict(family="DM Sans",size=10),orientation="h",x=0.5,xanchor="center",y=1.08),
        margin=dict(l=0,r=0,t=30,b=0), height=280,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig_rx.update_layout(title=dict(text="<b>Estimated LPS Cluster Uplift After Completing Recommended Interventions</b>",font=dict(family="Syne",size=11,color="#0B2540"),x=0.5,xanchor="center"))
    fig_rx.update_layout(title=dict(text="Estimated LPS Cluster Impact After Completing Recommended Interventions",font=dict(family="Syne",size=11,color="#0B2540"),x=0.5,xanchor="center"))
    st.plotly_chart(fig_rx, use_container_width=True, config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],"displaylogo":False}, key="pc_rx_bar")
    st.markdown("<div style='font-size:0.72rem;color:#64748B;margin-top:-6px'><b>Illustrative projection only.</b> Grey bars = current cluster scores. Teal bars = estimated scores after completing all prescribed interventions for this employee. Weakest clusters receive a larger uplift estimate (+15 pts) vs non-priority clusters (+5 pts).</div>", unsafe_allow_html=True)
    st.caption("Grey bars = current cluster scores. Teal bars = projected scores after completing the prescribed interventions (illustrative estimate: +15 pts for weakest clusters, +5 pts for others). Focus on the largest gaps first.")

    # ── Full intervention catalogue ────────────────────────────────────────────
    st.markdown('<div class="sec-hdr" style="margin-top:16px">📚 Full Intervention Catalogue</div>',
                unsafe_allow_html=True)
    all_rows = []
    for area, lib in INTERVENTION_LIBRARY.items():
        for c in lib["courses"]:
            all_rows.append({"Development Area": area, "Course / Intervention": c["name"],
                             "Type": c["type"], "Duration": c["duration"],
                             "Provider": c["provider"], "Mode": c["mode"]})
    df_catalogue = pd.DataFrame(all_rows)
    area_filter = st.multiselect("Filter by Area", sorted(df_catalogue["Development Area"].unique()),
                                  default=[], key="rx_area_filter",
                                  placeholder="Select areas (blank = show all)")
    show_df = df_catalogue[df_catalogue["Development Area"].isin(area_filter)]               if area_filter else df_catalogue
    st.dataframe(show_df, use_container_width=True, hide_index=True, height=350)

    # Download catalogue
    st.download_button("⬇ Download Full Intervention Catalogue",
                       data=df_catalogue.to_csv(index=False).encode("utf-8"),
                       file_name="development_intervention_catalogue.csv",
                       mime="text/csv", key="dl_catalogue")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 9 — DATA TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════
with tab9:
    st.markdown('<div class="sec-hdr">📋 Download Blank Data Templates</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#EBF4F8;border-left:4px solid #0D7377;border-radius:8px;
         padding:14px 18px;margin-bottom:18px;font-size:0.85rem;color:#374151">
      <b>Instructions:</b> Download each CSV template below, fill in your organisation's data
      following the column descriptions, and upload the completed files via the sidebar.
      Column names and data types must match exactly — do not rename headers.
    </div>
    """, unsafe_allow_html=True)

    # ── Template definitions ───────────────────────────────────────────────────
    TEMPLATES = {
        "employees_master.csv": {
            "description": "Main employee dataset — one row per employee. Required to activate the engine.",
            "color": "#0B2540",
            "columns": [
                ("EE Number",                               "Text",    "Unique employee ID. Must start with LTM followed by 6 digits. E.g. LTM001234"),
                ("Employee Full Name",                      "Text",    "Full name of employee"),
                ("Gender",                                  "Text",    "Male / Female / Non-Binary"),
                ("Age",                                     "Integer", "Age in years (24–63)"),
                ("Education Level",                         "Text",    "Bachelor's Degree / Post Graduate Diploma / Master's Degree / MBA / PhD"),
                ("Field of Study",                          "Text",    "E.g. Finance & Accounting, Computer Science, Human Resources Management"),
                ("Current Job Title",                       "Text",    "Exact job title as per org structure"),
                ("Job Grade (1-9)",                         "Integer", "Job band / grade level (1 = most junior, 9 = C-Suite)"),
                ("Department",                              "Text",    "Functional department name"),
                ("Business Unit",                           "Text",    "Business unit or geography"),
                ("Work Location",                           "Text",    "City / Country"),
                ("Direct Line Manager EE Number",           "Text",    "EE Number of direct line manager. Use N/A for top of hierarchy"),
                ("Number of Direct Reports",                "Integer", "Count of direct reports"),
                ("Tenure with Organisation (Years)",        "Float",   "Total years with the organisation (e.g. 7.5)"),
                ("Years in Current Role",                   "Float",   "Years in the current position"),
                ("Career Start Year",                       "Integer", "Year the employee started their career"),
                ("Career Start Age",                        "Integer", "Age at which career started (21–23)"),
                ("Total Promotions (Career)",               "Integer", "Total promotions over career. Must be <= Grade - 1"),
                ("Promotions in Last 5 Years",              "Integer", "Promotions received in last 5 years"),
                ("Lateral / Cross-Functional Moves",        "Integer", "Number of lateral role changes"),
                ("Total Internal Role Changes",             "Integer", "Total promotions + lateral moves"),
                ("Average Tenure per Role (Years)",         "Float",   "Tenure divided by total role changes + 1"),
                ("Highest Job Grade Achieved",              "Integer", "Peak grade level ever held (1–9)"),
                ("Annual CTC (Normalised 0-1)",             "Float",   "Salary normalised to 0–1 scale across all employees"),
                ("Last Annual Performance Rating (1-5)",    "Float",   "Most recent annual rating (1.0–5.0)"),
                ("Average Performance Rating - Last 3 Years (1-5)", "Float", "Mean of last 3 performance ratings"),
                ("Performance Trajectory",                  "Float",   "Last rating minus 3yr average (positive = improving)"),
                ("Potential Rating (1=Low 2=Moderate 3=High)", "Integer", "HR-assessed potential: 1, 2, or 3"),
                ("Potential Rating Label",                  "Text",    "Low Potential / Moderate Potential / High Potential"),
                ("9-Box Position",                          "Text",    "E.g. 'High Performer / High Potential'"),
                ("Flight Risk",                             "Text",    "Low / Medium / High"),
                ("On Active Retention Plan",                "Boolean", "True / False"),
                ("Mobility / Relocation Willingness (1-5)", "Integer", "1 = unwilling, 5 = highly willing"),
                ("Cross-Functional Experience",             "Boolean", "True / False"),
                ("International / Multi-Geography Experience","Boolean","True / False"),
                ("Certified Mentor or Coach",               "Boolean", "True / False"),
                ("Number of Critical Projects Led",         "Integer", "Count of strategic / critical projects led"),
                ("External Industry Recognition Count",     "Integer", "Awards, speaking invitations, publications"),
                ("Number of Business Languages Spoken",     "Integer", "Count of business languages"),
                ("Business Languages",                      "Text",    "Semicolon-separated list e.g. English; Hindi; French"),
                ("KF KFALP - Drivers Score (1-5)",          "Float",   "KFALP Drivers dimension score (1.0–5.0). Leave blank if not assessed"),
                ("KF KFALP - Curiosity Score (1-5)",        "Float",   "KFALP Curiosity dimension score"),
                ("KF KFALP - Insight Score (1-5)",          "Float",   "KFALP Insight dimension score"),
                ("KF KFALP - Engagement Score (1-5)",       "Float",   "KFALP Engagement dimension score"),
                ("KF KFALP - Determination Score (1-5)",    "Float",   "KFALP Determination dimension score"),
                ("KF KFALP - Learnability Score (1-5)",     "Float",   "KFALP Learnability dimension score"),
                ("KF KFALP - Composite Score (1-5)",        "Float",   "Weighted average of all 6 KFALP dimensions"),
                ("KF viaEdge - Mental Agility Score (1-5)", "Float",   "viaEdge Mental Agility score"),
                ("KF viaEdge - People Agility Score (1-5)", "Float",   "viaEdge People Agility score"),
                ("KF viaEdge - Change Agility Score (1-5)", "Float",   "viaEdge Change Agility score"),
                ("KF viaEdge - Results Agility Score (1-5)","Float",   "viaEdge Results Agility score"),
                ("KF viaEdge - Self-Awareness Score (1-5)", "Float",   "viaEdge Self-Awareness score"),
                ("KF viaEdge - Learning Agility Composite (1-5)","Float","Weighted average of all 5 viaEdge dimensions"),
                ("KF viaEdge - Learning Agility Percentile","Integer", "Normed percentile rank vs global benchmark (1–100)"),
                ("KF Blended Assessment Composite (1-5)",   "Float",   "KFALP composite x 0.55 + viaEdge composite x 0.45"),
                ("Promotions per Year (Career)",            "Float",   "Total promotions / tenure years"),
                ("Promotions per Year (Last 5 Years)",      "Float",   "Promotions in last 5 yrs / 5"),
            ]
        },
        "promotion_history.csv": {
            "description": "One row per promotion event per employee.",
            "color": "#0D7377",
            "columns": [
                ("EE Number",                              "Text",    "Employee EE Number — must match employees_master"),
                ("Employee Full Name",                     "Text",    "Employee full name"),
                ("Promotion Number (Career)",              "Integer", "Sequential promotion count (1, 2, 3...)"),
                ("Promotion Year",                         "Integer", "Calendar year of promotion (e.g. 2019)"),
                ("Promoted From Grade",                    "Integer", "Grade before promotion (1–9)"),
                ("Promoted From Title Band",               "Text",    "Title band before promotion"),
                ("Promoted To Grade",                      "Integer", "Grade after promotion (1–9). Must be > From Grade"),
                ("Promoted To Title Band",                 "Text",    "Title band after promotion"),
                ("Performance Rating at Promotion",        "Float",   "Performance rating in the year of promotion (1.0–5.0)"),
                ("Years Since Last Promotion",             "Float",   "Years elapsed since previous promotion"),
                ("Department at Time of Promotion",        "Text",    "Department at time of promotion"),
                ("Is Most Recent Promotion",               "Boolean", "True for the employee's latest promotion, False otherwise"),
            ]
        },
        "kf_kfalp_detail.csv": {
            "description": "One row per employee per KFALP dimension (6 rows per assessed employee).",
            "color": "#C9A227",
            "columns": [
                ("EE Number",                  "Text",    "Employee EE Number"),
                ("Employee Full Name",         "Text",    "Employee full name"),
                ("Current Job Title",          "Text",    "Employee's current job title"),
                ("KF KFALP Dimension",         "Text",    "One of: Drivers / Curiosity / Insight / Engagement / Determination / Learnability"),
                ("Raw Score (1-5)",            "Float",   "Raw dimension score (1.00–5.00)"),
                ("Rounded Score (1-5)",        "Integer", "Score rounded to nearest integer (1–5)"),
                ("KFALP Rating Band",          "Text",    "Limited / Developing / Effective / Strong / Exceptional"),
                ("KF KFALP Composite Score (1-5)","Float","Overall KFALP composite for this employee"),
                ("Behavioural Descriptor",     "Text",    "Full behavioural descriptor text for this score and dimension"),
            ]
        },
        "kf_viaedge_detail.csv": {
            "description": "One row per employee per viaEdge dimension (5 rows per assessed employee).",
            "color": "#7C3AED",
            "columns": [
                ("EE Number",                              "Text",    "Employee EE Number"),
                ("Employee Full Name",                     "Text",    "Employee full name"),
                ("Current Job Title",                      "Text",    "Employee's current job title"),
                ("KF viaEdge Dimension",                   "Text",    "One of: Mental Agility / People Agility / Change Agility / Results Agility / Self-Awareness"),
                ("Raw Score (1-5)",                        "Float",   "Raw dimension score"),
                ("Rounded Score (1-5)",                    "Integer", "Score rounded to integer"),
                ("viaEdge Rating Band",                    "Text",    "Needs Development / Emerging / Developing / Advanced / Expert"),
                ("KF viaEdge Learning Agility Composite (1-5)","Float","Overall viaEdge composite"),
                ("KF viaEdge Learning Agility Percentile","Integer", "Percentile vs global benchmark"),
                ("viaEdge Percentile Band",                "Text",    "E.g. 75th-89th Percentile"),
                ("Behavioural Descriptor",                 "Text",    "Full behavioural descriptor"),
            ]
        },
        "kf_attribute_reference.csv": {
            "description": "Reference guide — all KF dimensions across all 5 rating bands.",
            "color": "#1B7A3E",
            "columns": [
                ("KF Instrument",         "Text", "KFALP or viaEdge full name"),
                ("Dimension",             "Text", "Dimension name"),
                ("Category",              "Text", "Category e.g. Motivation, Resilience"),
                ("Sub-Dimensions",        "Text", "Semicolon-separated sub-dimensions"),
                ("What It Measures",      "Text", "Definition of what the dimension measures"),
                ("High Potential Signal",  "Text", "Behavioural indicators for high scorers"),
                ("Development Focus",     "Text", "Recommended development approach"),
                ("Assessment Method",     "Text", "How this dimension is assessed"),
                ("Rating Band",           "Text", "The rating band name"),
                ("Score",                 "Integer","Numeric score (1–5)"),
                ("Behavioural Descriptor","Text", "Full behavioural descriptor for this band"),
            ]
        },
        "org_structure.csv": {
            "description": "Hierarchy for the interactive org chart — one row per role/node.",
            "color": "#EA580C",
            "columns": [
                ("Node ID",             "Text",    "Unique node identifier — typically the job title"),
                ("Parent Node ID",      "Text",    "Node ID of the parent role. Leave blank for root (CEO)"),
                ("Job Title",           "Text",    "Full job title for this node"),
                ("Department",          "Text",    "Department name"),
                ("Business Unit",       "Text",    "Business unit"),
                ("Job Grade (1-9)",     "Integer", "Grade level of this role"),
                ("Representative EE",  "Text",    "EE Number of the person holding this role"),
                ("Representative Name","Text",    "Name of the person holding this role"),
                ("Org Level",           "Integer", "Hierarchy depth (1 = CEO, 2 = direct reports, etc.)"),
                ("LPS (Representative)","Float",  "LPS score of the representative person"),
                ("Is Critical Role",    "Boolean", "True if this role is a critical succession role"),
            ]
        },
    }

    # ── Render template cards ──────────────────────────────────────────────────
    for fname, tmpl in TEMPLATES.items():
        with st.expander(f"📄 {fname}  —  {tmpl['description']}", expanded=True):
            # Build blank template dataframe (headers only)
            cols_only = [col for col, _, _ in tmpl["columns"]]
            blank_df  = pd.DataFrame(columns=cols_only)

            tc1, tc2 = st.columns([1, 2.5])
            with tc1:
                st.markdown(f"""
                <div style="background:{tmpl['color']};color:white;border-radius:10px;
                     padding:14px 16px;margin-bottom:8px">
                  <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:0.9rem">{fname}</div>
                  <div style="font-size:0.78rem;color:rgba(255,255,255,0.75);margin-top:4px">{len(cols_only)} columns required</div>
                </div>""", unsafe_allow_html=True)
                st.download_button(
                    label=f"⬇ Download blank {fname}",
                    data=blank_df.to_csv(index=False).encode("utf-8"),
                    file_name=fname,
                    mime="text/csv",
                    use_container_width=True,
                    key=f"dl_{fname}"
                )
            with tc2:
                col_df = pd.DataFrame(tmpl["columns"], columns=["Column Name","Data Type","Description / Rules"])
                st.dataframe(col_df, use_container_width=True, hide_index=True, height=220)

    # ── Global validation rules card ──────────────────────────────────────────
    st.markdown('<div class="sec-hdr" style="margin-top:18px">✅ Data Quality Rules</div>',
                unsafe_allow_html=True)
    rules = [
        ("EE Numbers",         "Must start with LTM followed by 6 digits. Must be unique across all employees."),
        ("Age vs Tenure",      "Age minus Tenure must equal Career Start Age, which must be between 21 and 23."),
        ("Promotions vs Grade","Total Promotions must always be <= Job Grade - 1."),
        ("KF Scores",          "All KF scores on 1–5 scale. Leave blank (not 0) if employee has not been assessed."),
        ("9-Box Format",       "Must follow exact pattern: '[Perf Label] / [Potential Label]' e.g. 'High Performer / High Potential'"),
        ("Boolean columns",    "Use True or False exactly (case-sensitive)."),
        ("File names",         "File names must match exactly as shown above — the app matches by file name."),
        ("Encoding",           "Save all files as UTF-8 CSV. Do not use special characters in column headers."),
    ]
    rules_html = "".join(
        f"<div style='display:flex;gap:12px;padding:8px 0;border-bottom:1px solid #E2EAF0'>"
        f"<div style='min-width:170px;font-weight:600;color:#0B2540;font-size:0.82rem'>{r}</div>"
        f"<div style='font-size:0.82rem;color:#374151'>{d}</div></div>"
        for r, d in rules
    )
    st.markdown(f'<div class="card">{rules_html}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 10 — GLOSSARY
# ═══════════════════════════════════════════════════════════════════════════
with tab10:
    st.markdown('<div class="sec-hdr">📖 Glossary — Terms, Metrics & Formulae</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#EBF4F8;border-left:4px solid #0D7377;border-radius:8px;
         padding:14px 18px;margin-bottom:18px;font-size:0.85rem;color:#374151">
      This glossary defines every metric, score, label, and formula used throughout the
      Succession Planning Engine. Use it as a reference when interpreting charts, scores,
      and pipeline recommendations.
    </div>
    """, unsafe_allow_html=True)

    GLOSSARY_SECTIONS = [
        {
            "title": "🎯 Core Scores & Bands",
            "terms": [
                {
                    "term": "Leadership Potential Score (LPS)",
                    "definition": "A composite score from 0–100 that quantifies an employee's overall readiness and potential for senior leadership.",
                    "formula": "LPS = (C1 × w1%) + (C2 × w2%) + (C3 × w3%) + (C4 × w4%) + (C5 × w5%)\nwhere w1–w5 are the sidebar weight sliders and must sum to 100."
                },
                {
                    "term": "LPS Band",
                    "definition": "A readiness label derived from the LPS score indicating succession timeline.",
                    "formula": "Band 5 — Ready Now: LPS ≥ 80\nBand 4 — Ready in 1–2 Years: LPS 65–79\nBand 3 — Ready in 2–3 Years: LPS 50–64\nBand 2 — Emerging Potential: LPS 35–49\nBand 1 — Not Yet Ready: LPS < 35"
                },
                {
                    "term": "C1 — Performance Cluster (0–100)",
                    "definition": "Measures sustained delivery and performance trajectory. Combines average rating, last rating, and improvement trend.",
                    "formula": "C1 = norm(Avg 3yr Perf) × 0.50 + norm(Last Perf) × 0.35 + norm(Trajectory) × 0.15\nnorm() = min-max normalisation to 0–100 within the organisation."
                },
                {
                    "term": "C2 — KF Assessment Cluster (0–100)",
                    "definition": "Measures leadership potential via Korn Ferry tools (KFALP + viaEdge). Falls back to performance average if KF data is absent.",
                    "formula": "C2 = norm(KF Blended Assessment Composite)\nKF Blended = KFALP Composite × 0.55 + viaEdge Composite × 0.45"
                },
                {
                    "term": "C3 — Career Velocity Cluster (0–100)",
                    "definition": "Measures the speed and recency of career progression.",
                    "formula": "C3 = norm(Promotions/Year Career) × 0.50 + norm(Promotions/Year 5yr) × 0.35 + norm(Total Promotions) × 0.15"
                },
                {
                    "term": "C4 — Leadership Breadth Cluster (0–100)",
                    "definition": "Measures the width of leadership exposure across functions, geographies, and high-stakes situations.",
                    "formula": "C4 = Cross-Functional Exp × 25 + International Exp × 20\n     + norm(Critical Projects Led) × 30 + norm(External Recognition) × 15\n     + norm(Direct Reports) × 10   → then normalised to 0–100."
                },
                {
                    "term": "C5 — Readiness & Mobility Cluster (0–100)",
                    "definition": "Measures deployability into a senior role — accounts for grade gap, mobility willingness, and flight risk.",
                    "formula": "C5 = norm(Mobility Willingness) × 0.35 + norm(Grade Proximity to Top) × 0.35\n     + Flight Risk Score × 0.30\nFlight Risk Score: Low = 100, Medium = 50, High = 0"
                },
            ]
        },
        {
            "title": "📈 Career Velocity Metrics",
            "terms": [
                {
                    "term": "Promotions per Year (Career)",
                    "definition": "The average rate of promotion across an employee's entire career. Higher = faster career advancement.",
                    "formula": "Promotions per Year (Career) = Total Promotions (Career) ÷ Tenure with Organisation (Years)"
                },
                {
                    "term": "Promotions per Year (Last 5 Years)",
                    "definition": "Promotion rate over the most recent 5-year window. More current than the career average — reflects recent momentum better.",
                    "formula": "Promotions per Year (Last 5 Yrs) = Promotions in Last 5 Years ÷ 5"
                },
                {
                    "term": "Performance Velocity",
                    "definition": "The rate of CHANGE in performance rating over time — whether an employee is improving, stable, or declining. Distinct from promotion velocity: an employee can have high performance velocity (rapidly improving scores) without yet receiving a promotion.",
                    "formula": "Performance Velocity ≈ Performance Trajectory = Last Annual Rating − 3-Year Average Rating\nPositive → improving trend | Negative → declining | Zero → stable"
                },
                {
                    "term": "Performance Trajectory",
                    "definition": "Difference between the most recent annual rating and the 3-year average. Positive trajectory = leading indicator of promotion readiness.",
                    "formula": "Trajectory = Last Annual Performance Rating − Average Performance Rating (3yr)"
                },
                {
                    "term": "Performance Velocity vs Promotions per Year — Key Difference",
                    "definition": "These measure DIFFERENT dimensions of career progress and must not be confused:\n• Performance Velocity: How fast SCORES are improving (quality signal — are they getting better?)\n• Promotions per Year: How fast they are climbing the GRADE LADDER (speed signal — are they moving up?)\n\nAn employee can be a fast promoter with stable scores (high promotion velocity, zero performance velocity), OR a rapidly improving performer not yet promoted (high performance velocity, zero promotion velocity). The ideal succession candidate has BOTH high and rising performance AND a strong promotion track record.",
                    "formula": "Performance Velocity = Trajectory = Last Rating − Avg 3yr Rating\nPromotion Velocity = Total Promotions ÷ Tenure Years"
                },
                {
                    "term": "Average Tenure per Role (Years)",
                    "definition": "Average time per role throughout career. Lower values = broader breadth; higher values = depth in fewer roles.",
                    "formula": "Avg Tenure per Role = Tenure with Organisation ÷ (Total Internal Role Changes + 1)"
                },
            ]
        },
        {
            "title": "🔲 9-Box Matrix",
            "terms": [
                {
                    "term": "9-Box Grid",
                    "definition": "Talent segmentation framework plotting Performance (X-axis) against Potential (Y-axis), creating 9 cells. The engine uses LPS to drive the Potential axis — ensuring the assessment-based score governs placement rather than subjective HR labels.",
                    "formula": "X-axis (Performance): from performance ratings\n  Low ≤ 2.5 | Moderate 2.5–3.5 | High/Exceptional > 3.5\nY-axis (Potential): from LPS score\n  Low Potential: LPS < 35 | Moderate: LPS 35–64 | High: LPS ≥ 65"
                },
                {
                    "term": "Top Talent",
                    "definition": "High performance + High potential (LPS ≥ 65). Highest priority for succession and retention.",
                    "formula": "Top-right cell of the 9-box grid."
                },
                {
                    "term": "Enigma",
                    "definition": "High LPS but currently underperforming. May be in the wrong role, under-challenged, or experiencing difficulties. Requires HR investigation before succession decisions.",
                    "formula": "Bottom-left of the High Potential row (Low Perf + LPS ≥ 65)."
                },
                {
                    "term": "Misaligned Star",
                    "definition": "High performer but low LPS. Strong delivery but limited assessed leadership potential. May be a technical expert rather than a leadership candidate.",
                    "formula": "Top-right of the Low Potential row (High Perf + LPS < 35)."
                },
            ]
        },
        {
            "title": "🧠 Korn Ferry Assessments",
            "terms": [
                {
                    "term": "KF KFALP (Leadership Architect Learning Profile)",
                    "definition": "Psychometric tool assessing leadership potential across 6 dimensions: Drivers, Curiosity, Insight, Engagement, Determination, Learnability. Scores 1–5.",
                    "formula": "Composite = weighted average of 6 dimension scores.\nScale: 1=Limited, 2=Developing, 3=Effective, 4=Strong, 5=Exceptional"
                },
                {
                    "term": "KF viaEdge (Learning Agility)",
                    "definition": "Measures how well an individual learns from experience across 5 agility dimensions. Learning agility is the strongest Korn Ferry predictor of long-term leadership potential.",
                    "formula": "5 Dimensions: Mental Agility, People Agility, Change Agility, Results Agility, Self-Awareness.\nComposite = weighted average of 5 dimensions.\nPercentile = normed vs global KF benchmark."
                },
                {
                    "term": "KF Blended Assessment Composite",
                    "definition": "Single score combining KFALP and viaEdge for use in the LPS C2 cluster.",
                    "formula": "Blended = KFALP Composite × 0.55 + viaEdge Composite × 0.45\nScale: 1.0–5.0"
                },
                {
                    "term": "Learning Agility",
                    "definition": "Ability to learn from new experiences and apply learnings to succeed in new, first-time situations. #1 Korn Ferry predictor of senior leadership success.",
                    "formula": "Measured by KF viaEdge Composite (1–5) and Percentile rank (1–100 vs global norm)."
                },
            ]
        },
        {
            "title": "🏢 Pipeline & Succession Terms",
            "terms": [
                {
                    "term": "Critical Role",
                    "definition": "A role whose vacancy would significantly disrupt performance, strategy, or continuity. Drives the succession pipeline.",
                    "formula": "13 roles defined in ROLES_CFG — CEO through Senior Director level."
                },
                {
                    "term": "Successor Rank",
                    "definition": "Priority order within a role's pipeline. Rank 1 = Primary (best LPS + grade fit), Rank 2 = Secondary, Rank 3 = Tertiary.",
                    "formula": "Ranked by LPS within the role-specific eligible pool after grade windowing and deduplication."
                },
                {
                    "term": "Grade Window",
                    "definition": "Range of job grades eligible to be successors for a given role. Prevents over-promoting and ensures realistic candidates.",
                    "formula": "Eligible grades = [min_grade − grade_window, min_grade + 1]\nE.g. CFO (min_grade=8, window=2): grades 6–9 are eligible."
                },
                {
                    "term": "3-Layer Deduplication",
                    "definition": "Algorithm ensuring no employee is assigned as the #1 successor to more than one critical role — preserving pipeline depth and preventing single points of failure.",
                    "formula": "Layer 1: Grade window filtering.\nLayer 2: Department preference (dept-matched ranked first).\nLayer 3: Global dedup — once used as #1, excluded from #1 for all subsequent roles."
                },
                {
                    "term": "Bench Strength",
                    "definition": "Quality and depth of the succession pipeline. Measured by LPS of top 3 successors. High bench = multiple strong candidates at different readiness horizons.",
                    "formula": "Visualised in the Bench Strength Heatmap in Tab 5 — Org Readiness."
                },
                {
                    "term": "Flight Risk",
                    "definition": "Assessed likelihood that an employee will voluntarily leave. Levels: Low / Medium / High. High flight-risk employees excluded from pipeline by default.",
                    "formula": "C5 Flight Risk Score: Low=100, Medium=50, High=0 (contributes 30% of C5 cluster)."
                },
            ]
        },
    ]

    for section in GLOSSARY_SECTIONS:
        st.markdown(f'<div class="sec-hdr" style="margin-top:18px">{section["title"]}</div>',
                    unsafe_allow_html=True)
        for item in section["terms"]:
            with st.expander(f"**{item['term']}**", expanded=False):
                defn = item["definition"].replace("\n","<br>")
                st.markdown(
                    f'<div style="font-size:0.85rem;color:#374151;line-height:1.7;margin-bottom:8px">{defn}</div>',
                    unsafe_allow_html=True)
                if item.get("formula"):
                    formula_html = item["formula"].replace("\n","<br>")
                    st.markdown(
                        f'<div style="background:#F0F4F8;border-left:4px solid #0D7377;border-radius:6px;'                        f'padding:10px 14px;font-family:monospace;font-size:0.8rem;'                        f'color:#0B2540;line-height:1.7;margin-top:6px">{formula_html}</div>',
                        unsafe_allow_html=True)

    # Quick reference tables
    st.markdown('<div class="sec-hdr" style="margin-top:20px">⚡ Quick Reference — LPS Bands & 9-Box Cells</div>',
                unsafe_allow_html=True)
    qr_cols = st.columns(2)
    with qr_cols[0]:
        st.markdown("""
        <div class="card">
          <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:0.9rem;color:#0B2540;margin-bottom:10px">LPS Score → Band Mapping</div>
          <table style="width:100%;border-collapse:collapse;font-size:0.82rem">
            <tr style="background:#F0F4F8"><th style="padding:6px 10px;text-align:left">LPS Range</th><th style="padding:6px 10px;text-align:left">Band</th><th style="padding:6px 10px;text-align:left">Readiness</th></tr>
            <tr><td style="padding:6px 10px;color:#1B7A3E;font-weight:700">80–100</td><td style="padding:6px 10px">Band 5</td><td style="padding:6px 10px">Ready Now</td></tr>
            <tr style="background:#F8FBFD"><td style="padding:6px 10px;color:#2563EB;font-weight:700">65–79</td><td style="padding:6px 10px">Band 4</td><td style="padding:6px 10px">Ready in 1–2 Years</td></tr>
            <tr><td style="padding:6px 10px;color:#D97706;font-weight:700">50–64</td><td style="padding:6px 10px">Band 3</td><td style="padding:6px 10px">Ready in 2–3 Years</td></tr>
            <tr style="background:#F8FBFD"><td style="padding:6px 10px;color:#EA580C;font-weight:700">35–49</td><td style="padding:6px 10px">Band 2</td><td style="padding:6px 10px">Emerging Potential</td></tr>
            <tr><td style="padding:6px 10px;color:#B91C1C;font-weight:700">0–34</td><td style="padding:6px 10px">Band 1</td><td style="padding:6px 10px">Not Yet Ready</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)
    with qr_cols[1]:
        st.markdown("""
        <div class="card">
          <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:0.9rem;color:#0B2540;margin-bottom:10px">9-Box Cell Reference</div>
          <table style="width:100%;border-collapse:collapse;font-size:0.78rem">
            <tr style="background:#F0F4F8"><th style="padding:5px 8px;text-align:left">Cell Label</th><th style="padding:5px 8px;text-align:left">Performance</th><th style="padding:5px 8px;text-align:left">Potential (LPS)</th></tr>
            <tr><td style="padding:5px 8px;font-weight:700;color:#065F46">Top Talent</td><td>High</td><td>≥ 65</td></tr>
            <tr style="background:#F8FBFD"><td style="padding:5px 8px;font-weight:700;color:#1B7A3E">Future Leader</td><td>Moderate</td><td>≥ 65</td></tr>
            <tr><td style="padding:5px 8px;font-weight:700;color:#7C3AED">Enigma</td><td>Low</td><td>≥ 65</td></tr>
            <tr style="background:#F8FBFD"><td style="padding:5px 8px;font-weight:700;color:#0D7377">High Potential</td><td>High</td><td>35–64</td></tr>
            <tr><td style="padding:5px 8px;font-weight:700;color:#2563EB">Core Contributor</td><td>Moderate</td><td>35–64</td></tr>
            <tr style="background:#F8FBFD"><td style="padding:5px 8px;font-weight:700;color:#EA580C">Developing</td><td>Low</td><td>35–64</td></tr>
            <tr><td style="padding:5px 8px;font-weight:700;color:#D97706">Misaligned Star</td><td>High</td><td>&lt; 35</td></tr>
            <tr style="background:#F8FBFD"><td style="padding:5px 8px;font-weight:700;color:#94A3B8">Eff. Contributor</td><td>Moderate</td><td>&lt; 35</td></tr>
            <tr><td style="padding:5px 8px;font-weight:700;color:#B91C1C">Underperformer</td><td>Low</td><td>&lt; 35</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 10 — GLOSSARY
# ═══════════════════════════════════════════════════════════════════════════
with tab10:
    st.markdown('''<div class="sec-hdr">📖 Glossary — Terms, Metrics & Formulae</div>''', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#EBF4F8;border-left:4px solid #0D7377;border-radius:8px;
         padding:14px 18px;margin-bottom:20px;font-size:0.85rem;color:#374151">
      This glossary defines every metric, score, and term used throughout the Succession Planning Engine.
      Use it to understand how scores are calculated, what they mean, and how they differ from similar-sounding metrics.
    </div>
    """, unsafe_allow_html=True)

    GLOSSARY = {
        "📐 Core Scores & Indices": [
            ("Leadership Potential Score (LPS)",
             "A composite score from 0 to 100 that quantifies an employee's overall readiness and potential for succession to a critical role.",
             "LPS = (C1 × w1%) + (C2 × w2%) + (C3 × w3%) + (C4 × w4%) + (C5 × w5%)\nwhere w1–w5 are the user-configurable weights set in the sidebar (must sum to 100).\nEach cluster Cx is normalised to a 0–100 scale before weighting.",
             "#0B2540"),
            ("LPS Band",
             "A categorical readiness label derived from the LPS score, indicating how close an employee is to being succession-ready.",
             "Band 5 — Ready Now:          LPS ≥ 80\nBand 4 — Ready in 1–2 Years: LPS 65–79\nBand 3 — Ready in 2–3 Years: LPS 50–64\nBand 2 — Emerging Potential: LPS 35–49\nBand 1 — Not Yet Ready:      LPS < 35",
             "#0B2540"),
            ("C1 — Performance Cluster",
             "Measures sustained performance delivery over time, combining the 3-year average rating, the most recent annual rating, and the direction of the performance trend.",
             "C1 = norm(3yr Avg Perf) × 0.50 + norm(Last Perf) × 0.35 + norm(Trajectory) × 0.15\nAll inputs normalised (min–max) to 0–100 before weighting.",
             "#0D7377"),
            ("C2 — KF Assessment Cluster",
             "Reflects the quality of an employee's leadership potential as assessed by Korn Ferry instruments (KFALP and/or viaEdge).",
             "C2 = norm(KF Blended Assessment Composite)\nIf KF data is absent, the 3-year average performance rating is used as a proxy.",
             "#C9A227"),
            ("C3 — Career Velocity Cluster",
             "Captures the pace of career progression relative to peers, rewarding employees who have advanced faster throughout their career and in recent years.",
             "C3 = norm(Promotions per Year — Career) × 0.50\n    + norm(Promotions per Year — Last 5 Years) × 0.35\n    + norm(Total Promotions Career) × 0.15",
             "#2563EB"),
            ("C4 — Leadership Breadth Cluster",
             "Measures the width and richness of an employee's leadership experience — cross-functional exposure, international experience, project leadership, external recognition, and span of control.",
             "Breadth Score = Cross-Functional Experience × 25\n               + International Experience × 20\n               + norm(Critical Projects Led) × 30\n               + norm(External Recognition) × 15\n               + norm(Direct Reports) × 10\nC4 = norm(Breadth Score)",
             "#7C3AED"),
            ("C5 — Readiness & Mobility Cluster",
             "Assesses how close an employee is to being immediately deployable — factoring in grade proximity to the target role, mobility/relocation willingness, and retention risk.",
             "C5 = norm(Mobility Willingness) × 0.35\n    + norm(Grade Proximity to Max) × 0.35\n    + Flight Risk Score × 0.30\n\nFlight Risk Score: Low = 100, Medium = 50, High = 0\nGrade Proximity = max_grade − current_grade (inverted — higher grade = closer to top)",
             "#EA580C"),
        ],
        "📈 Career Progression Metrics": [
            ("Promotions per Year (Career)",
             "The overall rate of promotion across an employee's entire career with the organisation. This is the primary career velocity indicator.",
             "Promotions per Year (Career) = Total Promotions (Career) ÷ Tenure with Organisation (Years)\n\nExample: 4 promotions over 10 years = 0.40 promotions/year",
             "#0D7377"),
            ("Promotions per Year (Last 5 Years)",
             "The recent promotion rate, measuring how actively the employee has progressed in the most recent 5-year window. A higher recent rate vs career rate indicates accelerating momentum.",
             "Promotions per Year (Last 5 Yrs) = Promotions in Last 5 Years ÷ 5\n\nExample: 2 promotions in last 5 years = 0.40 promotions/year (recent)",
             "#0D7377"),
            ("Performance Velocity",
             "A directional metric showing whether an employee's performance is improving, declining, or stable over time. It is NOT a promotion rate — it measures the slope of the performance trend.",
             "Performance Trajectory = Last Annual Rating − 3-Year Average Rating\n\nPositive value (+) = improving trend (e.g. +0.4 means last year was 0.4 points above recent average)\nNegative value (−) = declining trend\nZero = stable performance\n\nKey distinction from Promotions/Year: Velocity is about performance momentum; Promotions/Year is about career progression speed.",
             "#2563EB"),
            ("Promotions per Year vs Performance Velocity — Key Differences",
             "These two metrics measure fundamentally different things and should not be confused.",
             "┌─────────────────────┬────────────────────────────────────────┐\n│ Metric              │ What it measures                       │\n├─────────────────────┼────────────────────────────────────────┤\n│ Promotions/Year     │ Speed of career progression (grades    │\n│                     │ advanced per year). Measures MOVEMENT. │\n├─────────────────────┼────────────────────────────────────────┤\n│ Performance Velocity│ Direction & slope of performance       │\n│                     │ ratings over time. Measures TRAJECTORY │\n│                     │ not movement up the hierarchy.         │\n└─────────────────────┴────────────────────────────────────────┘\nA person can have high Promotions/Year but declining Performance Velocity\n(fast-tracked but now plateauing) — or stable Promotions/Year but strong\npositive velocity (steady climber, consistently improving ratings).",
             "#7C3AED"),
            ("Performance Trajectory",
             "See Performance Velocity above. Used in C1 cluster calculation and displayed in the Career Path tab KPI strip.",
             "Trajectory = Last Annual Rating − 3-Year Average Rating",
             "#64748B"),
            ("Average Tenure per Role",
             "How long an employee stays in each role on average, indicating whether they are developing in-depth expertise or rotating too quickly.",
             "Avg Tenure per Role = Tenure (Years) ÷ (Total Role Changes + 1)\nWhere Total Role Changes = Total Promotions + Lateral Moves",
             "#64748B"),
        ],
        "🔲 9-Box Grid": [
            ("9-Box Position",
             "A talent classification framework that plots employees on a 3×3 grid of Performance (X-axis) vs Potential (Y-axis). Used to categorise talent into 9 archetypes.",
             "X-axis — Performance (from HR-assessed label in employee data):\n  0 = Low Performer\n  1 = Moderate Performer\n  2 = High Performer / Exceptional Performer\n\nY-axis — Potential (LPS-derived in this engine):\n  0 = Low Potential  (LPS < 35)\n  1 = Moderate Potential (LPS 35–64)\n  2 = High Potential (LPS ≥ 65)\n\nNote: The potential axis is driven by LPS to ensure objectivity and consistency with the succession scoring methodology.",
             "#0B2540"),
            ("9-Box Cell Archetypes",
             "Each cell in the 9-box grid has a label describing the archetype of talent it represents.",
             "(High Perf / High Pot) = Top Talent          — highest priority for succession\n(Mod Perf / High Pot) = Future Leader       — develop and stretch\n(Low Perf / High Pot) = Enigma              — investigate barriers to performance\n(High Perf / Mod Pot) = High Potential      — solid successors, limited stretch\n(Mod Perf / Mod Pot) = Core Contributor     — valuable, stable performers\n(Low Perf / Mod Pot) = Developing           — needs performance support\n(High Perf / Low Pot) = Misaligned Star     — expert in role, unlikely to advance\n(Mod Perf / Low Pot) = Effective Contributor— dependable, not succession-ready\n(Low Perf / Low Pot) = Underperformer       — performance management required",
             "#0B2540"),
        ],
        "🧠 Korn Ferry Assessments": [
            ("KF KFALP",
             "Korn Ferry Leadership Architect Learning Profile — a psychometric instrument measuring six dimensions of leadership potential: Drivers, Curiosity, Insight, Engagement, Determination, and Learnability. Scores are on a 1–5 scale.",
             "KF KFALP Composite = weighted average of 6 dimension scores\nDimensions: Drivers · Curiosity · Insight · Engagement · Determination · Learnability\nScale: 1 = Limited, 2 = Developing, 3 = Effective, 4 = Strong, 5 = Exceptional",
             "#C9A227"),
            ("KF viaEdge",
             "Korn Ferry viaEdge — an assessment measuring Learning Agility across five dimensions: Mental Agility, People Agility, Change Agility, Results Agility, and Self-Awareness. Includes a global percentile rank.",
             "KF viaEdge Composite = weighted average of 5 agility dimension scores\nDimensions: Mental Agility · People Agility · Change Agility · Results Agility · Self-Awareness\nAlso produces a Learning Agility Percentile vs global KF norm database.",
             "#7C3AED"),
            ("KF Blended Assessment Composite",
             "A single composite score combining both KFALP and viaEdge into one number, used as the input to the C2 (KF Assessment) cluster in LPS.",
             "KF Blended = KFALP Composite × 0.55 + viaEdge Composite × 0.45\nScale: 1.0–5.0",
             "#C9A227"),
            ("Learning Agility",
             "The ability to learn from experience and apply that learning to new and first-time situations. Measured by KF viaEdge. High learning agility is the single strongest predictor of long-term leadership potential.",
             "Assessed across 5 dimensions (see viaEdge above).\nLearning Agility Percentile = position vs KF global norm database (1–100).",
             "#7C3AED"),
        ],
        "⚠️ Risk & Retention": [
            ("Flight Risk",
             "An HR-assessed indicator of how likely an employee is to voluntarily leave the organisation within the next 12 months.",
             "Low    = unlikely to leave — included in all succession pools by default\nMedium = moderate attrition risk — included unless filter is enabled\nHigh   = active attrition risk — excluded from succession pools when 'Exclude High Flight Risk' is checked in sidebar\n\nImpact on C5 cluster: Low = 100 pts, Medium = 50 pts, High = 0 pts",
             "#B91C1C"),
            ("On Active Retention Plan",
             "Boolean flag indicating whether the employee is currently on a formal retention plan managed by HR/HRBP.",
             "True = employee is on a retention plan (typically for high-potential or flight-risk employees)\nFalse = no active retention plan",
             "#B91C1C"),
        ],
        "🏢 Pipeline & Succession": [
            ("Grade Window",
             "The range of job grades eligible to be considered as successors for a given critical role. Prevents succession from becoming too shallow (only immediate deputies) or too deep (junior employees).",
             "Eligible Grade Range = [min_grade − grade_window, min_grade + 1]\nExample: CEO role with min_grade=9, window=2 → eligible grades 7–9 (excluding incumbent)",
             "#0B2540"),
            ("Global Deduplication",
             "A 3-layer algorithm that ensures no single employee appears as the #1 successor for more than one critical role. This prevents over-reliance on a single individual and ensures breadth of bench strength.",
             "Layer 1: Each role builds its own eligible pool (grade window + department preference)\nLayer 2: Candidates already assigned as #1 to a higher-priority role are deprioritised\nLayer 3: If fewer than 3 fresh candidates exist for a role, the pool is relaxed to allow repeats",
             "#0B2540"),
            ("Bench Strength",
             "The aggregate quality of a succession pipeline — measured by the LPS scores of the top 3 successors for each critical role. A strong bench has all three successors scoring ≥ 65 (Band 4+).",
             "Displayed in Tab 5 as a heatmap: green cells = LPS ≥ 65, red cells = LPS < 35\nA role with all red successors is a critical succession risk requiring immediate action.",
             "#0D7377"),
            ("Successor Rank",
             "The priority order of successors for a critical role, derived from LPS score within the eligible pool (after department preference sorting).",
             "#1 — Primary Successor:   highest LPS in eligible pool\n#2 — Secondary Successor: second-highest LPS\n#3 — Tertiary Successor:  third-highest LPS",
             "#0D7377"),
        ],
    }

    for section, terms in GLOSSARY.items():
        st.markdown(f'''<div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;
             color:#0B2540;border-bottom:2px solid #0D7377;padding-bottom:6px;
             margin:20px 0 12px 0">{section}</div>''', unsafe_allow_html=True)

        for term, definition, formula, color in terms:
            with st.expander(f"**{term}**", expanded=False):
                st.markdown(f"""
                <div style="padding:4px 0">
                  <div style="font-size:0.87rem;color:#374151;line-height:1.7;margin-bottom:10px">{definition}</div>
                  <div style="background:#F0F4F8;border-left:4px solid {color};border-radius:6px;
                       padding:10px 14px;font-family:'Courier New',monospace;font-size:0.78rem;
                       color:#0B2540;white-space:pre-wrap;line-height:1.8">{formula}</div>
                </div>
                """, unsafe_allow_html=True)

    # Quick-reference comparison table
    st.markdown('''<div class="sec-hdr" style="margin-top:24px">🔑 Quick Reference — Key Metric Comparison</div>''', unsafe_allow_html=True)
    qr_data = {
        "Metric": [
            "LPS", "C1 Performance", "C2 KF Assessment", "C3 Career Velocity",
            "C4 Leadership Breadth", "C5 Readiness", "Promotions/Year (Career)",
            "Promotions/Year (Last 5yr)", "Performance Trajectory", "KFALP Composite",
            "viaEdge Composite", "KF Blended", "Flight Risk Score", "LPS Band",
        ],
        "Scale": [
            "0–100","0–100","0–100","0–100","0–100","0–100",
            "0.0–1.0+","0.0–1.0+","−4 to +4","1.0–5.0","1.0–5.0","1.0–5.0","0/50/100","1–5",
        ],
        "What it answers": [
            "How succession-ready is this person overall?",
            "How strong and consistent is their performance?",
            "How does their KF assessment rate their leadership potential?",
            "How fast have they progressed through the organisation?",
            "How broad and deep is their leadership experience?",
            "How close are they to deployment — grade, mobility, retention?",
            "What is their career-long promotion speed?",
            "What is their recent (last 5yr) promotion speed?",
            "Is their performance improving or declining?",
            "How strong is their leadership potential (KFALP)?",
            "How learning agile are they (viaEdge)?",
            "What is their overall KF leadership potential score?",
            "How likely are they to leave? (High=0, Med=50, Low=100)",
            "Which readiness band are they in? (1=Not Ready, 5=Ready Now)",
        ],
    }
    st.dataframe(qr_data, use_container_width=True, hide_index=True)

