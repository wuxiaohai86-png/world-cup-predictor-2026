"""
World Cup 2026 AI Predictor — 21st.dev inspired Streamlit Web App
Design language: glass-morphism + glow orbs + gradient text + dark theme
"""
import sys, io, os, json, warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="2026 世界杯 AI 预测",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════
# IMPORT PREDICTION ENGINE (silently)
# ═══════════════════════════════════════════════════════════════════════════

old_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w', encoding='utf-8')
from predict_match import predict_match
from group_projection import predict_group, load_tournament
from models import score_probability_matrix, calculate_poisson_lambdas
from data import load_match_data, calculate_team_stats
sys.stdout.close()
sys.stdout = old_stdout

# ═══════════════════════════════════════════════════════════════════════════
# 21ST-INSPIRED DESIGN SYSTEM (CSS)
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ── CSS Variables (shadcn-inspired palette) ── */
    :root {
        --background: #09090b;
        --foreground: #fafafa;
        --muted: #27272a;
        --muted-foreground: #a1a1aa;
        --primary: #f97316;
        --primary-foreground: #fff7ed;
        --accent: #8b5cf6;
        --secondary: #06b6d4;
        --border: #27272a;
        --card: #18181b;
        --ring: #f97316;
        --radius: 0.75rem;
    }

    /* ── Global Reset ── */
    .stApp {
        background: var(--background) !important;
    }
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 960px;
    }

    /* ── Typography ── */
    h1, h2, h3, h4, h5, h6 {
        color: var(--foreground) !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }

    /* ── Animated Glow Orbs (Canvas-free CSS) ── */
    @keyframes orbFloat1 {
        0%, 100% { transform: translate(0, 0) scale(1); }
        33% { transform: translate(30px, -20px) scale(1.05); }
        66% { transform: translate(-20px, 10px) scale(0.95); }
    }
    @keyframes orbFloat2 {
        0%, 100% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(-40px, 30px) scale(1.08); }
    }
    @keyframes orbFloat3 {
        0%, 100% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(25px, -15px) scale(0.92); }
    }
    @keyframes orbPulse {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1.0; }
    }

    .glow-orbs {
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }
    .glow-orb {
        position: absolute;
        border-radius: 50%;
        filter: blur(120px);
        will-change: transform;
    }
    .glow-orb-1 {
        width: 520px; height: 520px;
        top: -10%; left: 50%;
        transform: translateX(-50%);
        background: rgba(249, 115, 22, 0.06);
        animation: orbFloat1 20s ease-in-out infinite, orbPulse 8s ease-in-out infinite;
    }
    .glow-orb-2 {
        width: 360px; height: 360px;
        bottom: 10%; right: -5%;
        background: rgba(139, 92, 246, 0.05);
        animation: orbFloat2 25s ease-in-out infinite;
    }
    .glow-orb-3 {
        width: 400px; height: 400px;
        top: 50%; left: 20%;
        background: rgba(6, 182, 212, 0.04);
        animation: orbFloat3 22s ease-in-out infinite;
    }

    /* ── Hero Header ── */
    .hero {
        position: relative;
        z-index: 1;
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 1.2rem;
        border-radius: 9999px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(24, 24, 27, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        color: var(--muted-foreground);
        margin-bottom: 1.5rem;
    }
    .hero-badge .dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #22c55e;
        animation: orbPulse 2s ease-in-out infinite;
    }
    .hero-title {
        font-size: clamp(2rem, 6vw, 3.5rem);
        font-weight: 700 !important;
        line-height: 1.15;
        margin: 0 0 0.75rem;
        letter-spacing: -0.03em !important;
    }
    .hero-title .gradient-text {
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 50%, var(--secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: var(--muted-foreground);
        max-width: 560px;
        margin: 0 auto 1.5rem;
        line-height: 1.6;
    }

    /* ── Glass Card ── */
    .glass-card {
        position: relative;
        z-index: 1;
        background: rgba(24, 24, 27, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        margin-bottom: 1rem;
    }
    .glass-card.highlight {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.1), rgba(139, 92, 246, 0.08));
        border: 1px solid rgba(249, 115, 22, 0.2);
    }

    /* ── Prediction Result Card ── */
    .result-card {
        position: relative;
        z-index: 1;
        text-align: center;
        padding: 2rem 1.5rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.12), rgba(139, 92, 246, 0.1), rgba(6, 182, 212, 0.06));
        border: 1px solid rgba(249, 115, 22, 0.2);
        margin: 1rem 0;
    }
    .result-card .winner-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.25em;
        color: var(--muted-foreground);
        margin-bottom: 0.5rem;
    }
    .result-card .winner-team {
        font-size: 2rem;
        font-weight: 700;
        color: var(--foreground);
        margin: 0.25rem 0;
    }
    .result-card .confidence {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary), var(--accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
        margin: 0.5rem 0;
    }
    .result-card .confidence-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        color: var(--muted-foreground);
    }

    /* ── Probability Pill ── */
    .prob-pills {
        display: flex;
        gap: 0.75rem;
        justify-content: center;
        flex-wrap: wrap;
        margin: 1.25rem 0;
    }
    .prob-pill {
        flex: 1;
        min-width: 90px;
        max-width: 160px;
        text-align: center;
        padding: 1rem 0.75rem;
        border-radius: 16px;
        background: rgba(24, 24, 27, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .prob-pill .pill-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--foreground);
    }
    .prob-pill .pill-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: var(--muted-foreground);
        margin-top: 0.25rem;
    }
    .prob-pill.winner {
        border-color: rgba(249, 115, 22, 0.4);
        background: rgba(249, 115, 22, 0.1);
    }

    /* ── Scoreline List ── */
    .score-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 1rem;
        border-radius: 10px;
        background: rgba(24, 24, 27, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.04);
        margin: 0.3rem 0;
        font-size: 0.95rem;
        color: var(--foreground);
    }
    .score-row.top-pick {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.15), rgba(139, 92, 246, 0.1));
        border: 1px solid rgba(249, 115, 22, 0.25);
        font-weight: 600;
    }
    .score-row .score-num {
        font-weight: 700;
        font-size: 1.1rem;
        font-variant-numeric: tabular-nums;
    }
    .score-row .score-prob {
        font-weight: 600;
        color: var(--primary);
    }

    /* ── Betting Pills ── */
    .bet-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: 0.75rem;
        margin: 1rem 0;
    }
    .bet-item {
        text-align: center;
        padding: 0.75rem;
        border-radius: 12px;
        background: rgba(24, 24, 27, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .bet-item .bet-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--foreground);
    }
    .bet-item .bet-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--muted-foreground);
        margin-top: 0.25rem;
    }

    /* ── Tab Override ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.6rem 1.2rem;
        border-radius: 10px 10px 0 0;
        font-size: 0.9rem;
        font-weight: 500;
        color: var(--muted-foreground) !important;
        background: transparent !important;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--foreground) !important;
        background: rgba(24, 24, 27, 0.5) !important;
        border-bottom: 2px solid var(--primary) !important;
    }

    /* ── Button ── */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, var(--primary), #ea580c) !important;
        color: white !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 0.75rem 2rem !important;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 30px rgba(249, 115, 22, 0.3);
    }

    /* ── Select Box ── */
    .stSelectbox [data-baseweb="select"] {
        background: rgba(24, 24, 27, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }

    /* ── Dataframe Table ── */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    .stDataFrame table {
        background: rgba(24, 24, 27, 0.5) !important;
    }

    /* ── Progress Bar ── */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary), var(--accent)) !important;
        border-radius: 9999px !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetricValue"] {
        font-weight: 700 !important;
        color: var(--foreground) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--muted-foreground) !important;
    }

    /* ── Footer ── */
    .site-footer {
        position: relative;
        z-index: 1;
        text-align: center;
        padding: 2rem 1rem;
        color: var(--muted-foreground);
        font-size: 0.75rem;
        opacity: 0.6;
    }

    /* ── Mobile ── */
    @media (max-width: 640px) {
        .result-card .confidence { font-size: 2.5rem; }
        .result-card .winner-team { font-size: 1.5rem; }
        .glow-orb-1 { width: 300px; height: 300px; }
        .glow-orb-2 { width: 200px; height: 200px; }
        .glow-orb-3 { width: 240px; height: 240px; }
        .prob-pill .pill-value { font-size: 1.3rem; }
    }
</style>

<!-- Glow Orbs (injected outside Streamlit container for fixed positioning) -->
<div class="glow-orbs">
    <div class="glow-orb glow-orb-1"></div>
    <div class="glow-orb glow-orb-2"></div>
    <div class="glow-orb glow-orb-3"></div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data
def get_teams():
    t = load_tournament()
    all_teams = set()
    for teams in t['groups'].values():
        all_teams.update(teams)
    return sorted(all_teams)

@st.cache_data
def get_groups():
    return load_tournament()['groups']

TEAMS = get_teams()
GROUPS = get_groups()

# ═══════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <div class="hero-badge">
        <span class="dot"></span> AI 实时预测 · 964 场历史数据训练
    </div>
    <h1 class="hero-title">
        世界杯 <span class="gradient-text">AI 预测器</span>
    </h1>
    <p class="hero-subtitle">
        随机森林 + 逻辑回归 + Poisson 分布 · 18维特征 · 胜平负 · 精确比分 · 冷门指数 · 小组推演
    </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["🔮 单场预测", "📊 小组推演", "ℹ️ 模型说明"])

# ── TAB 1: Single Match ──
with tab1:
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        home_team = st.selectbox(
            "🏠 主队",
            TEAMS,
            index=TEAMS.index('Mexico') if 'Mexico' in TEAMS else 0,
            label_visibility="collapsed",
            placeholder="选择主队",
        )
    with col2:
        st.markdown(
            "<div style='text-align:center;padding-top:0.5rem;font-size:1.5rem;color:#52525b;'>vs</div>",
            unsafe_allow_html=True)
    with col3:
        away_team = st.selectbox(
            "✈️ 客队",
            TEAMS,
            index=TEAMS.index('South Africa') if 'South Africa' in TEAMS else 1,
            label_visibility="collapsed",
            placeholder="选择客队",
        )

    if st.button("🔮 开始预测", use_container_width=True):
        if home_team == away_team:
            st.error("不能选同一支球队！")
        else:
            with st.spinner("AI 正在分析中..."):
                result = predict_match(home_team, away_team, 2026)
                winner = result['ensemble_winner']
                ens = result['ensemble']
                exp_home, exp_away = result['expected_score']
                top_scores = result['top_scorelines']
                upset = result['upset']

                max_prob = max(ens.values())
                winner_emoji = {'home': '🏠', 'away': '✈️', 'draw': '🤝'}
                winner_label = {
                    'home': f'{winner_emoji["home"]} {home_team}',
                    'away': f'{winner_emoji["away"]} {away_team}',
                    'draw': '🤝 平局',
                }

                # ── Result Card ──
                st.markdown(f"""
                <div class="result-card">
                    <div class="winner-label">AI 预测胜出</div>
                    <div class="winner-team">{winner_label[winner]}</div>
                    <div class="confidence">{max_prob*100:.0f}%</div>
                    <div class="confidence-label">置信度</div>
                </div>
                """, unsafe_allow_html=True)

                # ── Probability Pills ──
                pill_order = ['home', 'draw', 'away']
                pill_labels = {
                    'home': ('🏠', home_team),
                    'draw': ('🤝', '平局'),
                    'away': ('✈️', away_team),
                }
                st.markdown('<div class="prob-pills">', unsafe_allow_html=True)
                for key in pill_order:
                    p = ens.get(key, 0)
                    is_win = (key == winner)
                    win_class = 'winner' if is_win else ''
                    emoji, lbl = pill_labels[key]
                    st.markdown(f"""
                    <div class="prob-pill {win_class}">
                        <div class="pill-value">{p*100:.1f}%</div>
                        <div class="pill-label">{emoji} {lbl}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # ── Expected Score ──
                st.markdown(f"##### ⚽ 期望进球：{home_team} **{exp_home:.2f}** — **{exp_away:.2f}** {away_team}")

                # ── Top Scorelines ──
                for rank, ((h, a), prob) in enumerate(top_scores[:5]):
                    cls = 'top-pick' if rank == 0 else ''
                    medal = ['🥇','🥈','🥉','',''][rank]
                    st.markdown(f"""
                    <div class="score-row {cls}">
                        <span>{medal} {home_team} <span class="score-num">{h}</span> — <span class="score-num">{a}</span> {away_team}</span>
                        <span class="score-prob">{prob*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Betting Grid ──
                _df = load_match_data()
                _ts = calculate_team_stats(_df, 2022)
                hl, al = calculate_poisson_lambdas(home_team, away_team, _ts, _df, 2026)
                sp, _ = score_probability_matrix(hl, al)
                over25 = sum(p for (h, a), p in sp.items() if h + a > 2.5)
                over15 = sum(p for (h, a), p in sp.items() if h + a > 1.5)
                btts = sum(p for (h, a), p in sp.items() if h > 0 and a > 0)

                st.markdown(f"""
                <div class="bet-grid">
                    <div class="bet-item"><div class="bet-value">{over15*100:.0f}%</div><div class="bet-label">大 1.5</div></div>
                    <div class="bet-item"><div class="bet-value">{over25*100:.0f}%</div><div class="bet-label">大 2.5</div></div>
                    <div class="bet-item"><div class="bet-value">{btts*100:.0f}%</div><div class="bet-label">双方进球</div></div>
                    <div class="bet-item"><div class="bet-value">{exp_home + exp_away:.1f}</div><div class="bet-label">总进球</div></div>
                </div>
                """, unsafe_allow_html=True)

                # ── Upset Mini Card ──
                st.markdown(f"""
                <div class="glass-card">
                    <span style="text-transform:uppercase;letter-spacing:0.2em;font-size:0.7rem;color:var(--muted-foreground);">⚡ 冷门指数</span><br>
                    <span style="font-size:1.2rem;font-weight:700;">{upset['shock_index']:.0f}<span style="font-size:0.7rem;color:var(--muted-foreground);">/100</span></span>
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    冷门概率 <b>{upset['upset_probability']*100:.0f}%</b>
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    热门方 <b>{upset['favorite']}</b>
                </div>
                """, unsafe_allow_html=True)

# ── TAB 2: Group Projection ──
with tab2:
    st.markdown("##### 📊 小组积分推演")
    st.caption("AI 模拟全部 6 场比赛")

    group_name = st.selectbox("选择小组", sorted(GROUPS.keys()), key='group_tab2')

    if st.button("🔮 推演积分榜", use_container_width=True):
        with st.spinner("正在模拟..."):
            gr = predict_group(group_name, GROUPS[group_name], silent=True)

            # Standings
            table_data = []
            for i, (team, stats) in enumerate(gr['standings'], 1):
                icon = '✅' if i <= 2 else ('🟡' if i == 3 else '❌')
                table_data.append({
                    '': icon, '#': i, '球队': team,
                    '分': stats['pts'], '进': stats['gf'], '失': stats['ga'], '净': f"{stats['gd']:+d}",
                })
            st.dataframe(pd.DataFrame(table_data).set_index('#'), use_container_width=True)
            st.caption("✅ 晋级 | 🟡 可能以小组第三晋级 | ❌ 出局")

            # Matches
            st.markdown("##### 📋 模拟比分")
            for m in gr['matches']:
                w = {'home': '🏠', 'away': '✈️', 'draw': '🤝'}.get(m['winner'], '')
                st.markdown(
                    f"**{m['home']}**  {m['score']}  **{m['away']}**  "
                    f"({w} 主{m['home_prob']*100:.0f}% 平{m['draw_prob']*100:.0f}% 客{m['away_prob']*100:.0f}%)"
                )

# ── TAB 3: About ──
with tab3:
    st.markdown("""
    <div class="glass-card" style="line-height:1.8;">
    <h3>⚙️ 技术架构</h3>
    <table style="width:100%;font-size:0.9rem;">
    <tr><td style="color:var(--muted-foreground);width:140px;">Poisson 模型</td><td>球队攻/防强度 → 期望进球 → 比分概率矩阵</td></tr>
    <tr><td style="color:var(--muted-foreground);">随机森林</td><td>18维特征 · 100棵树 · 76% 全量准确率</td></tr>
    <tr><td style="color:var(--muted-foreground);">逻辑回归</td><td>标准化特征 · 73% 全量准确率</td></tr>
    <tr><td style="color:var(--muted-foreground);">校准层</td><td>实力差距平局压制 · Poisson感知 · 东道主揭幕战加成</td></tr>
    <tr><td style="color:var(--muted-foreground);">数据源</td><td>964场世界杯 (1930-2022) · FIFA排名 · 预选赛成绩</td></tr>
    <tr><td style="color:var(--muted-foreground);">小组推演</td><td>48队 · 12组 · 洲际强度因子 · 预选赛表现加成</td></tr>
    </table>

    <h3 style="margin-top:1.5rem;">⚠️ 免责声明</h3>
    <p style="color:var(--muted-foreground);font-size:0.85rem;">
    预测基于历史统计数据，不包含当前阵容、伤病、临场状态。<br>
    <b>仅供娱乐参考，请勿用于赌博。</b>
    </p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="site-footer">
    2026 FIFA World Cup AI Predictor · ML Models · 仅供娱乐
</div>
""", unsafe_allow_html=True)
