"""
World Cup 2026 AI Predictor — Streamlit Web App
Mobile-first design for consumer users. Zero setup required.
"""
import sys
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import json

# Page config — must be first Streamlit call
st.set_page_config(
    page_title="2026世界杯AI预测",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Import prediction engine (suppress its prints)
import io, os
old_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w', encoding='utf-8')
from predict_match import predict_match
from group_projection import predict_group, load_tournament
sys.stdout.close()
sys.stdout = old_stdout

# ═══════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data
def get_teams():
    """Get list of all 48 qualified teams."""
    t = load_tournament()
    all_teams = []
    for group_name, teams in sorted(t['groups'].items()):
        for team in teams:
            all_teams.append(team)
    return sorted(set(all_teams))

@st.cache_data
def get_groups():
    """Get group assignments."""
    t = load_tournament()
    return t['groups']

TEAMS = get_teams()
GROUPS = get_groups()

# ═══════════════════════════════════════════════════════════════════════════
# STYLING
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #1a237e, #0d47a1);
        border-radius: 16px;
        margin-bottom: 1rem;
        color: white;
    }
    .main-header h1 {
        font-size: 2.2rem;
        margin: 0;
        color: white !important;
    }
    .main-header p {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0.3rem 0 0 0;
        color: white !important;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-card h2 {
        color: white !important;
        font-size: 1.8rem;
        margin: 0;
    }
    .prob-row {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .prob-item {
        text-align: center;
        padding: 0.5rem;
        border-radius: 12px;
        background: rgba(255,255,255,0.1);
        min-width: 80px;
    }
    .prob-item .value {
        font-size: 1.8rem;
        font-weight: bold;
    }
    .prob-item .label {
        font-size: 0.8rem;
        opacity: 0.8;
    }
    .scoreline-card {
        background: #263238;
        border-radius: 12px;
        padding: 0.6rem 1rem;
        color: white;
        margin: 0.3rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .scoreline-card.top {
        background: linear-gradient(135deg, #ff6f00, #ff8f00);
        font-weight: bold;
    }
    .info-box {
        background: #37474f;
        border-radius: 12px;
        padding: 1rem;
        color: #eceff1;
        margin: 0.5rem 0;
    }
    .info-box h4 {
        color: #ffab00 !important;
        margin: 0 0 0.5rem 0;
    }
    .standings-table td {
        padding: 0.3rem 0.5rem;
    }
    .footer {
        text-align: center;
        color: #78909c;
        font-size: 0.75rem;
        margin-top: 2rem;
        padding: 1rem;
    }
    /* Mobile tweaks */
    @media (max-width: 600px) {
        .main-header h1 { font-size: 1.5rem; }
        .prediction-card h2 { font-size: 1.4rem; }
        .prob-item .value { font-size: 1.3rem; }
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #ff6f00, #ff8f00);
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        border: none;
        border-radius: 12px;
        padding: 0.8rem;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff8f00, #ffa000);
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>⚽ 2026 世界杯 AI 预测器</h1>
    <p>机器学习模型 · 964场历史数据训练 · 18维特征 · Poisson 比分引擎</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TABS: Single Match | Group Projection | All Groups
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["🔮 单场预测", "📊 小组推演", "ℹ️ 使用说明"])

# ── TAB 1: Single Match Prediction ──
with tab1:
    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        home_team = st.selectbox(
            "🏠 主队 (Home)",
            TEAMS,
            index=TEAMS.index('Mexico') if 'Mexico' in TEAMS else 0,
            key='home'
        )

    with col2:
        st.markdown("<div style='text-align:center;padding-top:2rem;font-size:2rem;'>⚔️</div>", unsafe_allow_html=True)

    with col3:
        away_team = st.selectbox(
            "✈️ 客队 (Away)",
            TEAMS,
            index=TEAMS.index('South Africa') if 'South Africa' in TEAMS else 1,
            key='away'
        )

    predict_btn = st.button("🔮 开始预测", use_container_width=True)

    if predict_btn:
        if home_team == away_team:
            st.error("不能选同一支球队！")
        else:
            with st.spinner(f"AI 正在分析 {home_team} vs {away_team}..."):

                # Run prediction
                result = predict_match(home_team, away_team, 2026)

                # ── Winner card ──
                winner = result['ensemble_winner']
                winner_label = {'home': f'🏆 {home_team} 胜', 'away': f'🏆 {away_team} 胜', 'draw': '🤝 平局'}
                winner_emoji = {'home': '🏠', 'away': '✈️', 'draw': '🤝'}

                ens = result['ensemble']
                max_prob = max(ens.values())
                max_label = winner_label[winner]

                st.markdown(f"""
                <div class="prediction-card">
                    <p style="margin:0;opacity:0.8;">AI 预测结果</p>
                    <h2>{max_label}</h2>
                    <p style="font-size:3rem;margin:0.3rem 0;">{max_prob*100:.0f}%</p>
                    <p style="opacity:0.8;">置信度</p>
                </div>
                """, unsafe_allow_html=True)

                # ── Probability bars ──
                st.markdown("##### 📊 胜平负概率")
                c1, c2, c3 = st.columns(3)
                labels = {
                    'home': (f'🏠 {home_team}', ens.get('home', 0)),
                    'draw': ('🤝 平局', ens.get('draw', 0)),
                    'away': (f'✈️ {away_team}', ens.get('away', 0)),
                }
                for i, (k, (lbl, prob)) in enumerate(labels.items()):
                    col = [c1, c2, c3][i]
                    with col:
                        st.metric(lbl, f"{prob*100:.1f}%")
                        st.progress(prob)

                # ── Score prediction ──
                exp_home, exp_away = result['expected_score']
                top_scores = result['top_scorelines']

                st.markdown("##### ⚽ 比分预测")
                st.markdown(f"**期望进球：** {home_team} **{exp_home:.2f}** — **{exp_away:.2f}** {away_team}")

                # Top 5 scorelines
                st.markdown("**最可能比分：**")
                for rank, ((h, a), prob) in enumerate(top_scores[:5]):
                    is_top = (rank == 0)
                    bg = "background: linear-gradient(135deg, #e65100, #ff8f00);" if is_top else "background: #37474f;"
                    st.markdown(f"""
                    <div class="scoreline-card" style="{bg}">
                        <span>{'🥇' if rank == 0 else '🥈' if rank == 1 else '🥉' if rank == 2 else '  '} {home_team} <b>{h}</b> — <b>{a}</b> {away_team}</span>
                        <span style="font-weight:bold;">{prob*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Betting insights ──
                st.markdown("##### 💰 投注参考")
                # Calculate from score_probs
                from models import score_probability_matrix, calculate_poisson_lambdas
                import numpy as np
                from data import load_match_data, calculate_team_stats
                df = load_match_data()
                ts = calculate_team_stats(df, 2022)
                hl, al = calculate_poisson_lambdas(home_team, away_team, ts, df, 2026)
                sp, _ = score_probability_matrix(hl, al)

                over25 = sum(p for (h, a), p in sp.items() if h + a > 2.5)
                over15 = sum(p for (h, a), p in sp.items() if h + a > 1.5)
                btts = sum(p for (h, a), p in sp.items() if h > 0 and a > 0)

                bc1, bc2, bc3, bc4 = st.columns(4)
                bc1.metric("超过 1.5 球", f"{over15*100:.0f}%")
                bc2.metric("超过 2.5 球", f"{over25*100:.0f}%")
                bc3.metric("双方进球", f"{btts*100:.0f}%")
                bc4.metric("期望总进球", f"{exp_home + exp_away:.1f}")

                # ── Upset analysis ──
                upset = result['upset']
                st.markdown("##### ⚡ 冷门分析")
                uc1, uc2, uc3 = st.columns(3)
                uc1.metric("热门方", upset['favorite'])
                uc2.metric("冷门概率", f"{upset['upset_probability']*100:.1f}%")
                uc3.metric("冲击指数", f"{upset['shock_index']:.0f}/100")

# ── TAB 2: Group Projection ──
with tab2:
    st.markdown("##### 📊 小组赛积分推演")
    st.markdown("*AI 模拟全部 6 场比赛，基于 Poisson 期望进球模型*")

    group_name = st.selectbox(
        "选择小组",
        sorted(GROUPS.keys()),
        key='group_select'
    )

    if st.button("🔮 推演小组积分", use_container_width=True):
        with st.spinner(f"正在模拟 Group {group_name} 全部 6 场比赛..."):
            result = predict_group(group_name, GROUPS[group_name], silent=True)

            # Standings table
            st.markdown(f"### Group {group_name} — 积分榜")
            st.markdown(f"*🏟️ 模拟 6 场比赛后的积分情况*")

            table_data = []
            for i, (team, stats) in enumerate(result['standings'], 1):
                pos_icon = '✅' if i <= 2 else ('🟡' if i == 3 else '❌')
                table_data.append({
                    '': pos_icon,
                    '#': i,
                    '球队': team,
                    '积分': stats['pts'],
                    '进球': stats['gf'],
                    '失球': stats['ga'],
                    '净胜球': f"{stats['gd']:+d}",
                })

            st.dataframe(
                pd.DataFrame(table_data).set_index('#'),
                use_container_width=True,
                column_config={
                    '': st.column_config.Column(width='small'),
                    '球队': st.column_config.Column(width='medium'),
                }
            )
            st.caption('✅ 出线 | 🟡 可能以小组第三晋级 | ❌ 淘汰')

            # Match results
            st.markdown("##### 📋 模拟比分")
            for m in result['matches']:
                emoji = {'home': '🏠', 'away': '✈️', 'draw': '🤝'}
                w = emoji.get(m['winner'], '')
                st.markdown(
                    f"**{m['home']}** {m['score']} **{m['away']}**  "
                    f"({w} | 主{m['home_prob']*100:.0f}% 平{m['draw_prob']*100:.0f}% 客{m['away_prob']*100:.0f}%)"
                )

# ── TAB 3: Info ──
with tab3:
    st.markdown("""
    ### 📖 使用说明

    **这是什么？**

    一个基于机器学习的 2026 年 FIFA 世界杯比赛预测工具。模型用 1930-2022 年全部 964 场世界杯比赛训练，支持所有 48 支参赛队伍。

    **预测内容：**
    - 🏆 胜平负概率（三模型集成）
    - ⚽ 精确比分预测（Poisson 分布）
    - 💰 大小球、双方进球概率
    - ⚡ 冷门概率 & 冲击指数

    **模型说明：**
    - **Poisson 评分模型**：基于球队进攻/防守强度计算期望进球 → 最准确
    - **随机森林**：76% 准确率（历史数据）
    - **逻辑回归**：73% 准确率（历史数据）
    - 校准：实力差距平局压制 + 东道主加成

    **⚠️ 免责声明：**
    - 预测基于历史统计数据，不代表真实比赛结果
    - 不包含当前阵容、伤病、临场状态等信息
    - **仅供娱乐参考，请勿用于赌博**

    **技术栈：** Python · Scikit-learn · Poisson Distribution · Streamlit
    """)

# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer">
    ⚽ 2026 World Cup AI Predictor | 数据来源: FIFA 1930-2022<br>
    ML Models: Random Forest + Logistic Regression + Poisson Ensemble<br>
    仅供娱乐 | 不构成投注建议
</div>
""", unsafe_allow_html=True)
