# -*- coding: utf-8 -*-
# 한글
# -*- coding: utf-8 -*-
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from indicators import (
    add_technical_indicators, detect_rsi_divergence, analyze_btc_4year_cycle,
    analyze_volume_exhaustion, find_support_resistance, analyze_ma_and_momentum
)
from neely_rules import NeelyRuleEngine
from fractal_analyzer import FractalMapper
from trade_manager import FuturesTradeManager, SignalJournalManager
from quant_pro import (
    OKXDerivativesAnalyzer, NeelyTimePriceCluster,
    MultiScenarioEngine, NeelyStrategyBacktester
)

st.set_page_config(page_title="OKX Neely Master AI v6.6", page_icon="🧠", layout="wide")

st.title("🧠 OKX 선물 글렌 닐리 마스터 AI & 퀀트 검증 플랫폼 (v6.6)")
st.markdown("구조적 지지·저항/피보나치/EMA 동적 SL·TP · 실전 시그널 기록기 & 모의투자 자동 채점 · CVD/OI 수급 · 로그 스케일 닐리 파동")

if 'analysis_active' not in st.session_state:
    st.session_state['analysis_active'] = False

@st.cache_data(ttl=20)
def fetch_okx_ohlcv_deep(symbol: str, timeframe: str, target_bars: int = 300):
    exchange = ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    all_rows = []
    after_ts = None

    while len(all_rows) < target_bars:
        batch_limit = min(300, target_bars - len(all_rows))
        params = {'after': after_ts} if after_ts is not None else {}
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=batch_limit, params=params)
        if not ohlcv:
            break
        all_rows = ohlcv + all_rows
        after_ts = ohlcv[0][0]
        if len(ohlcv) < batch_limit:
            break

    df = pd.DataFrame(all_rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') + pd.Timedelta(hours=9)
    df = add_technical_indicators(df)
    return df

@st.cache_data(ttl=15)
def get_cached_okx_positioning(symbol: str):
    return OKXDerivativesAnalyzer().fetch_market_positioning(symbol)

st.sidebar.header("⚙️ 닐리 & 퀀트 엔진 설정 (v6.6)")
symbol = st.sidebar.text_input("거래 종목 (OKX Swap)", "BTC/USDT")
wave_view_choice = st.sidebar.radio(
    "🧭 파동 카운팅 관점 선택",
    [
        "관점 A: 5파동 (A) 완료 + 복합조정 (B)파 이후 (C)파 전개",
        "관점 B: 표준 단일 파동 (A)-(B)-(C) 카운팅"
    ],
    index=0
)
wave_view_mode = "view_complex_b" if wave_view_choice.startswith("관점 A") else "view_standard_a"

min_swing = st.sidebar.slider("파동 민감도 기준 (ATR 동적 가중치)", 0.004, 0.025, 0.010, 0.001)
active_tf = st.sidebar.selectbox("메인 매매 타점 & 백테스트 기준", ["15m", "1h", "1d", "1w"], index=1)

st.sidebar.subheader("🎨 이중 계층 차트 레이어 토글")
visible_candles = st.sidebar.slider("일봉·분봉 기본 확대 캔들 수", 100, 420, 280, 20)
use_log_chart = st.sidebar.checkbox("주봉·일봉 로그 스케일(Log) 비율 & 차트 적용", value=True)
show_macro_waves = st.sidebar.checkbox("대파동 뼈대 ((1)~(5) / (A)~(C)) 표시", value=True)
show_sub_waves = st.sidebar.checkbox("내부 세부파동 (1-1~5 / A-1~5 / B-1~3) 표시", value=True)
show_projection = st.sidebar.checkbox("진행 중인 파동 예상 경로 점선 표시", value=True)
show_clusters = st.sidebar.checkbox("닐리 가격-시간 클러스터 존 표시", value=True)
show_ema = st.sidebar.checkbox("EMA 이동평균선 (20·50·100·200) 표시", value=True)

st.sidebar.subheader("💰 선물 리스크 & 수급 설정")
use_okx_deriv = st.sidebar.checkbox("OKX 선물 실시간 수급·CVD·유동성 연동", value=True)
capital_usdt = st.sidebar.number_input("운용 시드머니 (USDT)", min_value=10.0, value=1000.0, step=100.0)
risk_pct = st.sidebar.slider("1회 매매 최대 허용 손실률 (%)", 0.5, 10.0, 2.0, 0.5)

if st.sidebar.button("🚀 전체 타임프레임 프랙탈 · 수급 · 백테스트 실행", type="primary"):
    st.session_state['analysis_active'] = True

if st.session_state['analysis_active']:
    fractal_mapper = FractalMapper()
    trade_mgr = FuturesTradeManager()
    journal_mgr = SignalJournalManager("signal_journal.csv")
    cluster_engine = NeelyTimePriceCluster()
    scenario_engine = MultiScenarioEngine()
    backtester = NeelyStrategyBacktester()

    tf_map = {'15m': '15분봉', '1h': '1시간봉', '1d': '일봉', '1w': '주봉'}
    fetch_targets = {'15m': 600, '1h': 900, '1d': 1350, '1w': 300}
    detail_tf_map = {'1w': '1d', '1d': '1h', '1h': '15m', '15m': None}

    raw_dfs = {}
    all_tf_data = {}
    sens_ratio = min_swing / 0.010

    dual_specs = {
        '15m': (0.014 * sens_ratio, 0.005 * sens_ratio),
        '1h':  (0.032 * sens_ratio, 0.011 * sens_ratio),
        '1d':  (0.110 * sens_ratio, 0.035 * sens_ratio),
        '1w':  (0.240 * sens_ratio, 0.070 * sens_ratio)
    }

    with st.spinner("OKX 실시간 선물 CVD·OI·유동성, 크로스 타임프레임 파동 및 구조적 타점 연산 중..."):
        okx_pos = get_cached_okx_positioning(symbol) if use_okx_deriv else None

        for tf_code in ['15m', '1h', '1d', '1w']:
            raw_dfs[tf_code] = fetch_okx_ohlcv_deep(symbol, tf_code, target_bars=fetch_targets[tf_code])

        for tf_code, tf_label in tf_map.items():
            m_pct, s_pct = dual_specs[tf_code]
            is_log_tf = use_log_chart and (tf_code in ['1w', '1d'])
            tf_engine = NeelyRuleEngine(min_swing_pct=s_pct, use_log_scale=is_log_tf)
            df = raw_dfs[tf_code]

            if tf_code == '1w':
                zoom_cnt = min(len(df), 205)
            elif tf_code == '1d':
                zoom_cnt = min(len(df), visible_candles)
            else:
                zoom_cnt = min(len(df), min(visible_candles, 180))

            cycle_df = df.iloc[-zoom_cnt:].copy()

            sub_tf_code = detail_tf_map[tf_code]
            detail_df = raw_dfs[sub_tf_code] if sub_tf_code else None

            dual_data = tf_engine.extract_dual_degree_waves(
                df, cycle_df, macro_swing_pct=m_pct, sub_swing_pct=s_pct,
                detail_df=detail_df, wave_view_mode=wave_view_mode
            )
            pivots, waves = tf_engine.extract_monowaves(cycle_df, swing_pct=s_pct)
            cluster_info = cluster_engine.analyze_clusters(cycle_df, dual_data)

            synced_cands = dual_data.get('synced_audits', [])
            extra_cands = tf_engine.evaluate_all_patterns(waves, cycle_df)
            candidates = synced_cands + [c for c in extra_cands if c['category'] not in [x['category'] for x in synced_cands]]
            top_pat = candidates[0] if candidates else None

            all_tf_data[tf_code] = {
                'label': tf_label,
                'df': df,
                'zoom_cnt': zoom_cnt,
                'pivots': pivots,
                'waves': waves,
                'candidates': candidates,
                'top_pattern': top_pat,
                'dual': dual_data,
                'clusters': cluster_info,
                'is_log': is_log_tf
            }

    latest_dt = all_tf_data['1d']['df']['timestamp'].iloc[-1]
    cycle_info = analyze_btc_4year_cycle(latest_dt)

    # 1. 상단 프랙탈 좌표 & 4년 사이클 인지
    st.subheader("🌐 이중 계층 프랙탈 좌표 & 닐리 파동 추론 · 4년 사이클 인지")
    col_a, col_b = st.columns([1.35, 1])
    with col_a:
        fractal_report = fractal_mapper.synthesize_fractal_context(all_tf_data, cycle_info, okx_pos)
        st.info(fractal_report['sub_completion_alert'])
        for line in fractal_report['narrative_lines']:
            st.markdown(f"- {line}")
    with col_b:
        st.warning(
            f"⏳ **비트코인 4년 사이클 레이더 (참고용 거시 맥락)**\n\n"
            f"- **현재 위상**: `{cycle_info['phase']}`\n"
            f"- **4차 반감기(2024.04.20) 경과**: **{cycle_info['elapsed_days']}일** ({cycle_info['months_post']:.1f}개월 / 주기 진행률 **{cycle_info['progress_pct']:.1f}%**)\n"
            f"- **해석**: {cycle_info['note']}"
        )

    # 2. OKX 실시간 선물 수급 · CVD · OI 4분면 · 유동성 대시보드
    if okx_pos:
        st.subheader(f"📡 [{symbol}] OKX 무기한 선물 실시간 수급 · CVD · OI 4분면 · 유동성 풀")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("현재 펀딩비 (8h)", f"{okx_pos['funding_rate_pct']:+.4f}%", f"다음 예상 {okx_pos['next_funding_pct']:+.4f}%")
        oi_disp = f"${okx_pos['open_interest_usdt']/1e9:.2f}B" if okx_pos['open_interest_usdt'] >= 1e9 else f"${okx_pos['open_interest_usdt']:,.0f}"
        m2.metric("미결제약정 (OI)", oi_disp, f"24H {okx_pos.get('oi_change_24h_pct', 0.0):+.2f}%")
        m3.metric("롱/숏 계정 비율", f"{okx_pos['long_short_ratio']:.2f}")
        m4.metric("테이커 매수/매도비", f"{okx_pos['taker_buy_sell_ratio']:.2f}", f"24H CVD {okx_pos.get('cvd_24h', 0.0):+,.0f}")
        m5.metric("수급 심리 점수", f"{okx_pos['sentiment_score']:+d}점", okx_pos['bias_label'].split('(')[0].strip())

        q1, q2, q3 = st.columns(3)
        with q1:
            st.info(f"**🧭 OI + 가격 4분면 국면**\n\n**{okx_pos.get('oi_regime_title', '분석 중')}**\n\n{okx_pos.get('oi_regime_desc', '')}")
        with q2:
            st.info(f"**🌊 CVD 누적 델타 & 다이버전스**\n\n**{okx_pos.get('cvd_div_label', '분석 중')}**\n\n최근 1H 순델타: `{okx_pos.get('net_delta_1h', 0.0):+,.1f}` | 24H 누적: `{okx_pos.get('cvd_24h', 0.0):+,.1f}`")
        with q3:
            st.info(f"**🧲 유동성 풀(BSL/SSL) & 스윕**\n\n{okx_pos.get('sweep_status', '분석 중')}")

        for d_line in okx_pos['details']:
            st.markdown(f"- {d_line}")

        cvd_df = okx_pos.get('cvd_df')
        if cvd_df is not None and not cvd_df.empty:
            with st.expander("📊 최근 24시간 실시간 CVD(누적 거래량 델타) 흐름 차트 보기", expanded=False):
                fig_cvd = go.Figure()
                bar_colors = ['#26a69a' if d >= 0 else '#ef5350' for d in cvd_df['delta']]
                fig_cvd.add_trace(go.Bar(
                    x=cvd_df['time'], y=cvd_df['delta'],
                    name='시간당 순델타 (Buy - Sell)', marker_color=bar_colors, opacity=0.55
                ))
                fig_cvd.add_trace(go.Scatter(
                    x=cvd_df['time'], y=cvd_df['cvd'],
                    name='24H 누적 CVD 선', mode='lines+markers',
                    line=dict(color='#00e5ff', width=2.4), yaxis='y2'
                ))
                fig_cvd.update_layout(
                    title="OKX 무기한 선물 24시간 테이커 순델타 & 누적 CVD 추이",
                    height=290, margin=dict(t=45, b=30, l=45, r=45),
                    template="plotly_dark",
                    yaxis=dict(title="시간당 델타"),
                    yaxis2=dict(title="누적 CVD", overlaying="y", side="right"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_cvd, use_container_width=True)

    st.divider()

    # 3. 다중 시나리오 & 무효화 트리거 (주봉 대파동 기준점 동기화)
    main_data = all_tf_data[active_tf]
    scenarios = scenario_engine.build_scenarios(
        main_data['df'], main_data['dual'], cycle_info, okx_pos,
        weekly_dual=all_tf_data['1w']['dual']
    )
    if scenarios:
        st.subheader(f"🔀 [{tf_map[active_tf]}] 다중 파동 시나리오 & 명확한 무효화 트리거 (주봉 대파동 동기화)")
        sc_cols = st.columns(len(scenarios))
        for s_idx, sc in enumerate(scenarios):
            with sc_cols[s_idx]:
                if s_idx == 0:
                    st.info(
                        f"**{sc['rank_tag']}**\n\n"
                        f"### {sc['title']}\n"
                        f"- **파동 논리**: {sc['logic']}\n"
                        f"- **목표 구간**: {sc['target_zone']}\n"
                        f"- **🚨 무효화 트리거**: {sc['invalidation_trigger']}\n"
                        f"- **대응 전략**: {sc['action_plan']}"
                    )
                else:
                    st.warning(
                        f"**{sc['rank_tag']}**\n\n"
                        f"### {sc['title']}\n"
                        f"- **파동 논리**: {sc['logic']}\n"
                        f"- **목표 구간**: {sc['target_zone']}\n"
                        f"- **🚨 무효화 트리거**: {sc['invalidation_trigger']}\n"
                        f"- **대응 전략**: {sc['action_plan']}"
                    )

    st.divider()

    # 4. OKX 선물 AI 트레이딩 시그널 & 구조적(피벗·피보나치·EMA) SL/TP 포지션 사이징
    df_main = main_data['df']
    pivots_main = main_data['pivots']
    waves_main = main_data['waves']
    top_pat_main = main_data['top_pattern']
    current_price = df_main['close'].iloc[-1]
    atr_val = df_main['atr'].iloc[-1]

    div_info = detect_rsi_divergence(df_main, pivots_main)
    vol_info = analyze_volume_exhaustion(df_main, waves_main)
    sr_info = find_support_resistance(pivots_main, current_price)
    ma_info = analyze_ma_and_momentum(df_main, 'LONG' if (top_pat_main and top_pat_main['direction'] == 'DOWN') else 'SHORT')
    trade_plan = trade_mgr.generate_trade_plan(
        current_price, atr_val, top_pat_main, df_main['rsi'].iloc[-1],
        div_info, vol_info, sr_info, ma_info, cycle_info, capital_usdt, risk_pct,
        active_tf=active_tf, dual_data=main_data['dual'], df=df_main, pivots=pivots_main, okx_pos=okx_pos
    )

    st.subheader(f"🎯 [{tf_map[active_tf]}] OKX 선물 AI 구조적 지정가 시그널 & 포지션 사이징")
    if trade_plan['signal'] in ['LONG', 'SHORT']:
        sig_color = "🟢 LONG (매수 대기)" if trade_plan['signal'] == 'LONG' else "🔴 SHORT (매도 대기)"
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("추천 포지션", sig_color, f"신뢰도 {trade_plan['confidence']}점")
        c2.metric("권장 지정가 (평균 Entry)", f"${trade_plan['entry_price']:,.1f}", trade_plan['entry_delta_label'])
        c3.metric("구조적 무효화 손절 (SL)", f"${trade_plan['stop_loss']:,.1f}", f"-{trade_plan['sl_pct']:.2f}%")
        c4.metric(f"2차 목표가 (TP2 - {trade_plan['rr_ratio']:.2f}R)", f"${trade_plan['tp2']:,.1f}", trade_plan['tp2_basis'][:18])
        c5.metric("동적 손익비 (R:R)", f"1 : {trade_plan['rr_ratio']:.2f}", f"TP1 {trade_plan['rr1']:.2f}R")

        st.success(
            f"📌 **분할 지정가 오더북 가이드 (현재가 `${trade_plan['current_price']:,.1f}` 기준)**: "
            f"1차 지정가(50%) **`${trade_plan['entry_1']:,.1f}`** ➔ 2차 지정가(50%) **`${trade_plan['entry_2']:,.1f}`** "
            f"(평균 단가 **`${trade_plan['entry_price']:,.1f}`**)\n\n"
            f"💼 **시드 ${capital_usdt:,.0f} (손실한도 {risk_pct}% = ${trade_plan['max_loss_usdt']:,.1f}) 기준 권장 세팅**: "
            f"총 진입 규모 **`{trade_plan['position_size_btc']:.4f} BTC`** (`${trade_plan['position_value_usdt']:,.0f}` 상당) | "
            f"권장 격리 레버리지 **`{trade_plan['safe_leverage']}x`**"
        )
        st.markdown(
            f"**진입 근거:** {trade_plan['setup_type']} | "
            f"**구조적 분할 익절:** TP1 `${trade_plan['tp1']:,.1f}` (`{trade_plan['rr1']:.2f}R`) ➔ "
            f"TP2 `${trade_plan['tp2']:,.1f}` (`{trade_plan['rr_ratio']:.2f}R`) ➔ "
            f"TP3 `${trade_plan['tp3']:,.1f}` (`{trade_plan['rr3']:.2f}R`)"
        )
        for chk in trade_plan['checks']:
            st.markdown(f"- {chk}")
    else:
        st.warning(f"⏸️ {trade_plan['signal']}: {trade_plan['reason']}")

    # 4-B. 실전 시그널 기록기 & 모의투자 확률 검증기 (signal_journal.csv)
    st.subheader("📑 실전 시그널 기록기 & 모의투자 확률 검증기 (Paper Trading Simulator)")
    j_col1, j_col2, j_col3 = st.columns([1.2, 1.3, 0.9])
    now_kst_str = (pd.Timestamp.utcnow() + pd.Timedelta(hours=9)).strftime('%Y-%m-%d %H:%M')

    with j_col1:
        if trade_plan['signal'] in ['LONG', 'SHORT']:
            if st.button("📥 현재 시그널 모의 기록 (Paper Entry)", use_container_width=True):
                saved_item = journal_mgr.record_signal(symbol, active_tf, trade_plan, risk_pct, now_kst_str)
                st.toast(f"✅ 시그널 [{saved_item['id']}] 기록 완료! ({saved_item['direction']} @ ${saved_item['avg_entry']:,.1f})")
        else:
            st.button("📥 현재 시그널 모의 기록 (대기 중)", disabled=True, use_container_width=True)

    with j_col2:
        if st.button("🔄 OKX 실시간 캔들 대조 & 자동 채점 실행", type="primary", use_container_width=True):
            _, st_summary = journal_mgr.evaluate_all_with_okx()
            st.toast(f"🔄 자동 채점 완료! 총 {st_summary['total']}건 중 체결 {st_summary['filled']}건 반영됨")

    with j_col3:
        if st.button("🗑️ 기록 전체 초기화", use_container_width=True):
            journal_mgr.clear_journal()
            st.toast("🗑️ 시그널 기록이 초기화되었습니다.")

    journal_df = journal_mgr.load_journal()
    if not journal_df.empty:
        j_stats = journal_mgr.summarize_stats(journal_df)
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("누적 기록 시그널", f"{j_stats['total']}건", f"지정가 체결 {j_stats['filled']}건")
        s2.metric("확정 종료 건수", f"{j_stats['closed']}건")
        s3.metric("실전 모의 승률", f"{j_stats['win_rate']}%")
        s4.metric("평균 실현/평가 R배수", f"{j_stats['avg_r']:+.2f} R")
        s5.metric("누적 모의 계좌 손익", f"{j_stats['total_pnl']:+.2f}%", f"1회 리스크 {risk_pct}% 기준")

        disp_cols = [
            'id', 'recorded_at', 'symbol', 'timeframe', 'direction', 'confidence',
            'ref_price', 'entry_1', 'entry_2', 'avg_entry', 'stop_loss', 'tp1', 'tp2',
            'status', 'fill_stage', 'realized_r', 'account_pnl_pct'
        ]
        st.dataframe(journal_df[disp_cols].iloc[::-1], use_container_width=True, height=250)
    else:
        st.caption("💡 마음에 드는 타점이 나올 때마다 **[📥 현재 시그널 모의 기록]**을 눌러 쌓아둔 뒤, 시간이 지난 후 **[🔄 OKX 실시간 캔들 대조 & 자동 채점 실행]**을 누르면 실제 체결 여부와 승률·손익비가 자동 계산됩니다.")

    st.divider()

    # 5. 타임프레임별 차트 & 닐리 규칙 & 가격-시간 클러스터 탭
    tabs = st.tabs(["주봉 (일봉 기반 세부파동)", "일봉 (1시간봉 기반 세부파동)", "1시간봉 (15분봉 기반)", "15분봉 세부"])
    for idx, tf_code in enumerate(['1w', '1d', '1h', '15m']):
        with tabs[idx]:
            tdata = all_tf_data[tf_code]
            df_t = tdata['df']
            zoom_cnt = tdata['zoom_cnt']
            dual = tdata['dual']
            cands = tdata['candidates']
            diag = dual.get('correction_diagnosis')
            c_info = tdata['clusters']
            is_log_t = tdata['is_log']

            if diag:
                st.info(f"📐 **[{tdata['label']} 조정파 규칙 추론 — {diag['pattern_title']}]**: {diag['reason']}")

            col_left, col_right = st.columns([2.35, 1])

            with col_left:
                fig = go.Figure(data=[go.Candlestick(
                    x=df_t['timestamp'], open=df_t['open'], high=df_t['high'], low=df_t['low'], close=df_t['close'],
                    name='캔들',
                    increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
                )])

                if show_ema:
                    ema_styles = [
                        ('ema_20', 'EMA20', '#ffd54f', 1.1),
                        ('ema_50', 'EMA50', '#ff8a65', 1.3),
                        ('ema_100', 'EMA100', '#4fc3f7', 1.5),
                        ('ema_200', 'EMA200', '#ba68c8', 1.8)
                    ]
                    for col_name, label_name, color_hex, width_val in ema_styles:
                        fig.add_trace(go.Scatter(
                            x=df_t['timestamp'], y=df_t[col_name],
                            mode='lines', name=label_name,
                            line=dict(color=color_hex, width=width_val),
                            opacity=0.6
                        ))

                sub_pivots = dual.get('sub_pivots', [])
                if show_sub_waves and sub_pivots:
                    sub_texts = []
                    sub_pos = []
                    for p in sub_pivots:
                        lbl = p['label']
                        if not lbl:
                            sub_texts.append('')
                            sub_pos.append('bottom center')
                        elif show_macro_waves and p.get('is_shared_macro'):
                            if p['type'] == 'HIGH':
                                sub_texts.append(f"{lbl}<br> <br> ")
                                sub_pos.append('top center')
                            else:
                                sub_texts.append(f" <br> <br>{lbl}")
                                sub_pos.append('bottom center')
                        else:
                            sub_texts.append(lbl)
                            sub_pos.append('top center' if p['type'] == 'HIGH' else 'bottom center')

                    fig.add_trace(go.Scatter(
                        x=[p['time'] for p in sub_pivots],
                        y=[p['price'] for p in sub_pivots],
                        mode='lines+markers+text',
                        name='세부파동 (전체 카운팅)',
                        line=dict(color='rgba(255, 255, 255, 0.85)', width=1.6),
                        marker=dict(size=5, color='#80d8ff'),
                        text=sub_texts,
                        textfont=dict(size=11, color='#80d8ff'),
                        textposition=sub_pos
                    ))

                m_pivots = dual.get('macro_pivots', [])
                m_waves = dual.get('macro_waves', [])
                if show_macro_waves and m_pivots and m_waves:
                    m_labels = ["<b>(0)</b>"] + [
                        f"<b>({w['local_wave_id']} 진행중)</b>" if w.get('is_active') else f"<b>({w['local_wave_id']})</b>"
                        for w in m_waves
                    ]
                    m_pos = ['top center' if p['type'] == 'HIGH' else 'bottom center' for p in m_pivots]
                    m_hover = [f"사이클 시작 (0): ${m_pivots[0]['price']:,.1f}"] + [
                        f"대파동: ({w['local_wave_id']}) {'[진행 중]' if w.get('is_active') else '[완료]'}<br>"
                        f"내부 구조: {w['subdivision_rule']}파동 전개 ({w['structure_label']})<br>"
                        f"소속 패턴: {w['pattern_tag']}<br>가격: ${w['end_price']:,.1f}<br>시각: {w['end_time']}"
                        for w in m_waves
                    ]
                    fig.add_trace(go.Scatter(
                        x=[p['time'] for p in m_pivots],
                        y=[p['price'] for p in m_pivots],
                        mode='lines+markers+text',
                        name='대파동 (1)~(5)/(A)~(C)',
                        line=dict(color='#ffd600', width=3.0),
                        marker=dict(size=10, color='#ffd600', line=dict(width=1.5, color='#000000')),
                        text=m_labels,
                        textfont=dict(size=14, color='#ffd600'),
                        textposition=m_pos,
                        hovertext=m_hover,
                        hoverinfo='text'
                    ))

                proj = dual.get('projection')
                if show_projection and proj:
                    proj_pos = ['bottom center', 'top center', 'bottom center', 'top center', 'middle right'][:len(proj['prices'])]
                    fig.add_trace(go.Scatter(
                        x=proj['times'],
                        y=proj['prices'],
                        mode='lines+markers+text',
                        name='예상 세부경로',
                        line=dict(color='#29b6f6', width=2.2, dash='dash'),
                        marker=dict(size=6, color='#29b6f6'),
                        text=proj['labels'],
                        textfont=dict(size=11, color='#4fc3f7'),
                        textposition=proj_pos
                    ))

                p_box = c_info.get('primary_box')
                if show_clusters and p_box:
                    fig.add_shape(
                        type="rect",
                        x0=p_box['x0'], x1=p_box['x1'],
                        y0=p_box['y0'], y1=p_box['y1'],
                        fillcolor="rgba(171, 71, 188, 0.22)",
                        line=dict(color="#ce93d8", width=1.5, dash="dot")
                    )
                    fig.add_annotation(
                        x=p_box['x1'], y=p_box['y1'],
                        text=p_box['label'],
                        showarrow=False, xanchor="right", yanchor="bottom",
                        font=dict(size=10, color="#f3e5f5"),
                        bgcolor="rgba(106, 27, 154, 0.65)"
                    )

                if tf_code == active_tf and trade_plan['signal'] in ['LONG', 'SHORT']:
                    fig.add_hline(y=trade_plan['entry_price'], line_dash="dot", line_color="#00e5ff", annotation_text=f"Entry 지정가 ${trade_plan['entry_price']:,.0f}")
                    fig.add_hline(y=trade_plan['stop_loss'], line_dash="dash", line_color="#ff5252", annotation_text=f"SL ${trade_plan['stop_loss']:,.0f}")
                    fig.add_hline(y=trade_plan['tp1'], line_dash="dot", line_color="#a5d6a7", annotation_text=f"TP1 ${trade_plan['tp1']:,.0f} ({trade_plan['rr1']:.1f}R)")
                    fig.add_hline(y=trade_plan['tp2'], line_dash="dash", line_color="#00e676", annotation_text=f"TP2 ${trade_plan['tp2']:,.0f} ({trade_plan['rr_ratio']:.1f}R)")

                dt_step = df_t['timestamp'].iloc[-1] - df_t['timestamp'].iloc[-2] if len(df_t) >= 2 else pd.Timedelta(hours=1)
                x_start = df_t['timestamp'].iloc[-zoom_cnt]
                x_end = df_t['timestamp'].iloc[-1] + dt_step * (46 if tf_code == '1w' else 38)
                if show_clusters and p_box:
                    x_end = max(x_end, pd.Timestamp(p_box['x1']) + dt_step * 6)

                vis_df = df_t.iloc[-zoom_cnt:]
                y_min = vis_df['low'].min()
                y_max = vis_df['high'].max()
                if show_projection and proj:
                    y_min = min(y_min, min(proj['prices']))
                    y_max = max(y_max, max(proj['prices']))
                if show_clusters and p_box:
                    y_min = min(y_min, p_box['y0'])
                    y_max = max(y_max, p_box['y1'])

                scale_badge = " [Log Scale]" if is_log_t else ""
                if is_log_t:
                    y_axis_cfg = dict(
                        type="log",
                        range=[np.log10(max(y_min * 0.90, 1.0)), np.log10(y_max * 1.12)]
                    )
                else:
                    y_pad = (y_max - y_min) * 0.14
                    y_axis_cfg = dict(
                        type="linear",
                        range=[y_min - y_pad, y_max + y_pad]
                    )

                fig.update_layout(
                    title=dict(text=f"{symbol} {tdata['label']}{scale_badge} 크로스 타임프레임 파동 & 클러스터 분석", x=0.01, y=0.97, font=dict(size=16)),
                    xaxis_title="한국 시간 (KST)", yaxis_title="가격 (USDT)",
                    xaxis_range=[x_start, x_end],
                    yaxis=y_axis_cfg,
                    margin=dict(t=85, b=50, l=50, r=90),
                    height=740,
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font=dict(size=11))
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                st.markdown("### ⚖️ 닐리 규칙 검증 및 점수화")
                if cands:
                    for rank, c in enumerate(cands[:3], 1):
                        with st.expander(f"#{rank} {c['pattern_name']} — [{c['score']}점]", expanded=True):
                            st.write(f"**진행 상태:** {c['stage_status']}")
                            for a in c['audits']:
                                st.markdown(f"- {a}")
                else:
                    st.info("현재 구간에서 기준 점수를 통과한 패턴 후보가 없습니다.")

                st.markdown("### ⏳ 닐리 시간 비율 & 가격 클러스터")
                for tr_line in c_info.get('time_ratios', []):
                    st.markdown(f"- {tr_line}")
                p_clusters = c_info.get('price_clusters', [])
                if p_clusters:
                    for cl in p_clusters[:2]:
                        with st.expander(f"🎯 피보나치 클러스터 ${cl['low']:,.0f} ~ ${cl['high']:,.0f} ({cl['confluence_count']}중첩)", expanded=True):
                            for r_txt in cl['reasons']:
                                st.markdown(f"- {r_txt}")
                else:
                    st.caption("현재 구간 내 중첩 클러스터 탐색 중")

            if m_waves:
                st.markdown("#### 📋 대파동 사이클 및 내부 세부파동 구조 요약표")
                rows = []
                for mw in m_waves:
                    w_id = mw['local_wave_id']
                    sub_r = mw['subdivision_rule']
                    if sub_r == '5':
                        sub_seq_txt = f"{w_id}-1 ~ {w_id}-5 (5파동 충격)"
                    elif sub_r == 'c3':
                        sub_seq_txt = f"{w_id}-1 ~ {w_id}-3 (확장형 플랫 / 복합 W-X-Y)"
                    else:
                        sub_seq_txt = f"{w_id}-1 ~ {w_id}-3 (3파동 조정)"
                    if mw.get('is_active'):
                        sub_seq_txt += " [진행 중]"
                    rows.append({
                        '대파동': f"({w_id}) {'진행중' if mw.get('is_active') else '완료'}",
                        '소속 패턴': mw['pattern_tag'],
                        '내부 세부 구조': sub_seq_txt,
                        '방향': mw['direction'],
                        '시작 (KST)': str(mw['start_time'])[:10],
                        '종료 (KST)': str(mw['end_time'])[:10],
                        '시작가': f"${mw['start_price']:,.1f}",
                        '종료가': f"${mw['end_price']:,.1f}"
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.divider()

    # 6. 과거 차트 백테스팅 & 승률/손익비 검증기
    st.subheader(f"🧪 [{tf_map[active_tf]}] 과거 차트 워크포워드 백테스팅 & 승률/손익비 검증기 (OKX 수수료 반영)")
    bt_res = backtester.run_backtest(df_main, risk_pct=risk_pct, rr_target=2.4)
    if bt_res.get('total_trades', 0) > 0:
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("총 검증 시그널", f"{bt_res['total_trades']}회", f"{bt_res['win_count']}승 {bt_res['loss_count']}패")
        b2.metric("실전 승률 (Win Rate)", f"{bt_res['win_rate']}%")
        b3.metric("평균 기대 R배수", f"{bt_res['avg_rr']:+.2f} R", f"PF {bt_res['profit_factor']}")
        b4.metric("누적 계좌 수익률", f"{bt_res['total_return_pct']:+.2f}%", f"1회 리스크 {risk_pct}% 기준")
        b5.metric("최대 낙폭 (MDD)", f"-{bt_res['max_drawdown_pct']:.2f}%")

        bt_col1, bt_col2 = st.columns([1.2, 1.3])
        with bt_col1:
            eq_df = bt_res['equity_curve']
            fig_eq = go.Figure(data=[go.Scatter(
                x=eq_df['time'], y=eq_df['equity'],
                mode='lines', name='누적 자산 곡선 (초기=100)',
                line=dict(color='#00e676', width=2.3),
                fill='tozeroy', fillcolor='rgba(0, 230, 118, 0.12)'
            )])
            fig_eq.update_layout(
                title="누적 자산 성장 곡선 (초기 시드 = 100)",
                height=320, margin=dict(t=45, b=30, l=40, r=20),
                template="plotly_dark"
            )
            st.plotly_chart(fig_eq, use_container_width=True)
        with bt_col2:
            st.markdown("##### 📜 최근 10건 시뮬레이션 매매 검증 로그")
            st.dataframe(bt_res['trades_df'].tail(10).iloc[::-1], use_container_width=True, height=285)
    else:
        st.info("선택한 타임프레임 구간 내 백테스팅 표본이 부족합니다. 1시간봉 또는 일봉을 선택해 검증해 보세요.")
else:
    st.info("👈 왼쪽 사이드바에서 **[전체 타임프레임 프랙탈 · 수급 · 백테스트 실행]** 버튼을 눌러주세요.")