# -*- coding: utf-8 -*-
# 한글
# -*- coding: utf-8 -*-
import requests
import ccxt
import numpy as np
import pandas as pd


class OKXDerivativesAnalyzer:
    """
    OKX 무기한 선물(Swap) 실시간 수급·포지셔닝 심화 분석기 (v6.5.2)
    """

    def fetch_market_positioning(self, symbol: str = "BTC/USDT") -> dict:
        base_ccy = symbol.split('/')[0].upper()
        swap_symbol = f"{base_ccy}/USDT:USDT"

        funding_pct = 0.0
        next_funding_pct = 0.0
        oi_usdt = 0.0
        ls_ratio = 1.0
        taker_ratio = 1.0
        live_count = 0

        try:
            ex = ccxt.okx({'enableRateLimit': True, 'timeout': 5000, 'options': {'defaultType': 'swap'}})
            fr = ex.fetch_funding_rate(swap_symbol)
            if fr and fr.get('fundingRate') is not None:
                funding_pct = float(fr['fundingRate']) * 100.0
                live_count += 1
            if fr and fr.get('nextFundingRate') is not None:
                next_funding_pct = float(fr['nextFundingRate']) * 100.0
        except Exception:
            pass

        try:
            ex = ccxt.okx({'enableRateLimit': True, 'timeout': 5000, 'options': {'defaultType': 'swap'}})
            oi = ex.fetch_open_interest(swap_symbol)
            if oi and oi.get('openInterestValue') is not None:
                oi_usdt = float(oi['openInterestValue'])
                live_count += 1
            elif oi and oi.get('openInterestAmount') is not None:
                oi_usdt = float(oi['openInterestAmount'])
                live_count += 1
        except Exception:
            pass

        df_1h = None
        try:
            ex = ccxt.okx({'enableRateLimit': True, 'timeout': 5000, 'options': {'defaultType': 'swap'}})
            ohlcv = ex.fetch_ohlcv(swap_symbol, timeframe='1h', limit=48)
            if ohlcv and len(ohlcv) >= 12:
                df_1h = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], unit='ms') + pd.Timedelta(hours=9)
                live_count += 1
        except Exception:
            pass

        headers = {'User-Agent': 'Mozilla/5.0'}

        try:
            r_ls = requests.get(
                f"https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?ccy={base_ccy}&period=1H",
                headers=headers, timeout=4
            ).json()
            if r_ls.get('code') == '0' and r_ls.get('data'):
                ls_ratio = float(r_ls['data'][0][1])
                live_count += 1
        except Exception:
            pass

        cvd_records = []
        net_delta_1h = 0.0
        cvd_24h = 0.0
        try:
            r_tk = requests.get(
                f"https://www.okx.com/api/v5/rubik/stat/taker-volume?ccy={base_ccy}&instType=CONTRACTS&period=1H",
                headers=headers, timeout=4
            ).json()
            if r_tk.get('code') == '0' and r_tk.get('data'):
                raw_tk = r_tk['data'][:24][::-1]
                cum_val = 0.0
                for row in raw_tk:
                    ts_dt = pd.to_datetime(int(row[0]), unit='ms') + pd.Timedelta(hours=9)
                    s_vol = float(row[1])
                    b_vol = float(row[2])
                    delta = b_vol - s_vol
                    cum_val += delta
                    cvd_records.append({
                        'time': ts_dt,
                        'buy_vol': b_vol,
                        'sell_vol': s_vol,
                        'delta': delta,
                        'cvd': cum_val
                    })
                last_s = float(r_tk['data'][0][1])
                last_b = float(r_tk['data'][0][2])
                if last_s > 0:
                    taker_ratio = last_b / last_s
                net_delta_1h = last_b - last_s
                cvd_24h = cum_val
                live_count += 1
        except Exception:
            pass

        if not cvd_records and df_1h is not None:
            cum_val = 0.0
            for _, r in df_1h.tail(24).iterrows():
                rng = max(r['high'] - r['low'], 1e-8)
                buy_ratio = (r['close'] - r['low']) / rng
                b_vol = r['volume'] * buy_ratio
                s_vol = r['volume'] * (1.0 - buy_ratio)
                delta = b_vol - s_vol
                cum_val += delta
                cvd_records.append({
                    'time': r['timestamp'],
                    'buy_vol': b_vol,
                    'sell_vol': s_vol,
                    'delta': delta,
                    'cvd': cum_val
                })
            if cvd_records:
                net_delta_1h = cvd_records[-1]['delta']
                cvd_24h = cum_val
                if cvd_records[-1]['sell_vol'] > 0:
                    taker_ratio = cvd_records[-1]['buy_vol'] / cvd_records[-1]['sell_vol']

        oi_change_4h_pct = 0.0
        oi_change_24h_pct = 0.0
        try:
            r_oi = requests.get(
                f"https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume?ccy={base_ccy}&period=1H",
                headers=headers, timeout=4
            ).json()
            if r_oi.get('code') == '0' and r_oi.get('data') and len(r_oi['data']) >= 5:
                oi_now = float(r_oi['data'][0][1])
                oi_4h = float(r_oi['data'][4][1])
                oi_24h = float(r_oi['data'][min(23, len(r_oi['data']) - 1)][1])
                if oi_4h > 0:
                    oi_change_4h_pct = ((oi_now - oi_4h) / oi_4h) * 100.0
                if oi_24h > 0:
                    oi_change_24h_pct = ((oi_now - oi_24h) / oi_24h) * 100.0
                if oi_usdt <= 0:
                    oi_usdt = oi_now
                live_count += 1
        except Exception:
            pass

        price_chg_4h = 0.0
        price_chg_24h = 0.0
        if df_1h is not None and len(df_1h) >= 24:
            c_now = float(df_1h['close'].iloc[-1])
            c_4h = float(df_1h['close'].iloc[-5])
            c_24h = float(df_1h['close'].iloc[-24])
            price_chg_4h = ((c_now - c_4h) / max(c_4h, 1e-8)) * 100.0
            price_chg_24h = ((c_now - c_24h) / max(c_24h, 1e-8)) * 100.0

        oi_regime_bias = 0
        if price_chg_4h >= 0.25 and oi_change_4h_pct >= 0.30:
            oi_regime_title = "🟢 신규 롱 유입 (Long Build-up)"
            oi_regime_desc = f"4H 가격 `{price_chg_4h:+.2f}%` + OI `{oi_change_4h_pct:+.2f}%` 동반 상승 (강세 추세 강화)"
            oi_regime_bias = 15
        elif price_chg_4h >= 0.25 and oi_change_4h_pct <= -0.30:
            oi_regime_title = "🟡 숏 커버링 / 스퀴즈 (Short Covering)"
            oi_regime_desc = f"4H 가격 `{price_chg_4h:+.2f}%` 반등 중 OI `{oi_change_4h_pct:+.2f}%` 감소 (기존 숏 손절 소화)"
            oi_regime_bias = 5
        elif price_chg_4h <= -0.25 and oi_change_4h_pct >= 0.30:
            oi_regime_title = "🔴 신규 숏 공격 유입 (Short Build-up)"
            oi_regime_desc = f"4H 가격 `{price_chg_4h:+.2f}%` 하락 중 OI `{oi_change_4h_pct:+.2f}%` 급증 (신규 숏 베팅 주도)"
            oi_regime_bias = -15
        elif price_chg_4h <= -0.25 and oi_change_4h_pct <= -0.30:
            oi_regime_title = "🔵 롱 강제청산 / 투매 소화 (Long Liquidation)"
            oi_regime_desc = f"4H 가격 `{price_chg_4h:+.2f}%` 하락과 OI `{oi_change_4h_pct:+.2f}%` 동반 감소 (롱 청산 마무리 → 바닥 변곡 임박)"
            oi_regime_bias = 8
        else:
            oi_regime_title = "⚪ 미결제약정 보합 (포지션 탐색)"
            oi_regime_desc = f"4H 가격 `{price_chg_4h:+.2f}%` / OI `{oi_change_4h_pct:+.2f}%` (방향성 에너지 응축 구간)"
            oi_regime_bias = 0

        cvd_div_type = "NONE"
        cvd_div_label = "특이 수급 다이버전스 없음"
        cvd_div_score = 0
        if df_1h is not None and len(df_1h) >= 24 and len(cvd_records) >= 16:
            p_slice = df_1h['close'].tail(len(cvd_records)).values
            cvd_vals = np.array([x['cvd'] for x in cvd_records])
            mid = len(p_slice) // 2

            p_min1, p_min2 = float(np.min(p_slice[:mid])), float(np.min(p_slice[mid:]))
            p_max1, p_max2 = float(np.max(p_slice[:mid])), float(np.max(p_slice[mid:]))
            c_min1, c_min2 = float(np.min(cvd_vals[:mid])), float(np.min(cvd_vals[mid:]))
            c_max1, c_max2 = float(np.max(cvd_vals[:mid])), float(np.max(cvd_vals[mid:]))

            if p_min2 < p_min1 * 0.998 and c_min2 > c_min1:
                cvd_div_type = "BULLISH_ABSORPTION"
                cvd_div_label = "🟢 강세 CVD 흡수 다이버전스 (가격 신저점 vs 지정가 매수 흡수)"
                cvd_div_score = 22
            elif p_max2 > p_max1 * 1.002 and c_max2 < c_max1:
                cvd_div_type = "BEARISH_EXHAUSTION"
                cvd_div_label = "🔴 약세 CVD 고갈 다이버전스 (가격 신고점 vs 시장가 매수세 이탈)"
                cvd_div_score = -22
            elif cvd_24h > 0 and price_chg_24h > 0:
                cvd_div_type = "BULLISH_CONFIRMED"
                cvd_div_label = "🟢 24H CVD 순매수 추세 일치 (진성 매수세 동반)"
                cvd_div_score = 10
            elif cvd_24h < 0 and price_chg_24h < 0:
                cvd_div_type = "BEARISH_CONFIRMED"
                cvd_div_label = "🔴 24H CVD 순매도 추세 일치 (진성 매도세 동반)"
                cvd_div_score = -10

        bsl_price = 0.0
        ssl_price = 0.0
        sweep_status = "유동성 풀 대기 구간"
        sweep_type = "NONE"
        sweep_score = 0

        if df_1h is not None and len(df_1h) >= 24:
            hist_part = df_1h.iloc[:-3]
            recent_3 = df_1h.iloc[-3:]
            curr_close = float(df_1h['close'].iloc[-1])

            bsl_price = float(hist_part['high'].max())
            ssl_price = float(hist_part['low'].min())

            swept_low = (recent_3['low'].min() < ssl_price) and (curr_close > ssl_price)
            swept_high = (recent_3['high'].max() > bsl_price) and (curr_close < bsl_price)

            if swept_low:
                sweep_type = "SWEEP_LOW"
                sweep_status = f"🟢 하방 유동성 스윕(SSL `${ssl_price:,.1f}` 헌팅 후 복귀 → 반등 유리)"
                sweep_score = 12
            elif swept_high:
                sweep_type = "SWEEP_HIGH"
                sweep_status = f"🔴 상방 유동성 스윕(BSL `${bsl_price:,.1f}` 헌팅 후 이탈 → 하락 유리)"
                sweep_score = -12
            else:
                dist_ssl = ((curr_close - ssl_price) / max(curr_close, 1e-8)) * 100.0
                dist_bsl = ((bsl_price - curr_close) / max(curr_close, 1e-8)) * 100.0
                sweep_status = f"상단 BSL `${bsl_price:,.1f}` (+{dist_bsl:.2f}%) | 하단 SSL `${ssl_price:,.1f}` (-{dist_ssl:.2f}%)"

        score = oi_regime_bias + cvd_div_score + sweep_score
        details = []

        if funding_pct <= -0.01:
            score += 20
            details.append(f"🟢 **펀딩비** `{funding_pct:+.4f}%` (음수 펀딩비 → 숏 과밀, 숏 스퀴즈 우호)")
        elif funding_pct >= 0.03:
            score -= 20
            details.append(f"🔴 **펀딩비** `{funding_pct:+.4f}%` (과열 양수 펀딩비 → 롱 청산 압력 주의)")
        else:
            details.append(f"⚪ **펀딩비** `{funding_pct:+.4f}%` (중립 안정권)")

        if ls_ratio >= 1.65:
            score -= 15
            details.append(f"🔴 **롱/숏 계정 비율** `{ls_ratio:.2f}` (개인 롱 쏠림 과다 → 하방 스윕 경계)")
        elif ls_ratio <= 0.85:
            score += 15
            details.append(f"🟢 **롱/숏 계정 비율** `{ls_ratio:.2f}` (숏 계정 우세 → 반등 연료 축적)")
        else:
            details.append(f"⚪ **롱/숏 계정 비율** `{ls_ratio:.2f}` (균형 구간)")

        if taker_ratio >= 1.12:
            score += 15
            details.append(f"🟢 **테이커 체결비** `{taker_ratio:.2f}` (1H 순델타 `{net_delta_1h:+,.1f}` — 시장가 매수 유입)")
        elif taker_ratio <= 0.88:
            score -= 15
            details.append(f"🔴 **테이커 체결비** `{taker_ratio:.2f}` (1H 순델타 `{net_delta_1h:+,.1f}` — 시장가 매도 우세)")
        else:
            details.append(f"⚪ **테이커 체결비** `{taker_ratio:.2f}` (1H 순델타 `{net_delta_1h:+,.1f}` — 보합)")

        details.append(f"📊 **OI + 가격 4분면**: {oi_regime_title} — {oi_regime_desc}")
        details.append(f"🌊 **CVD 누적 델타(24H)**: `{cvd_24h:+,.1f}` — {cvd_div_label}")
        details.append(f"🧲 **유동성 풀 & 스윕**: {sweep_status}")

        if score >= 20:
            bias_label = "상방 수급 우세 (CVD 흡수 / 숏 스퀴즈)"
        elif score <= -20:
            bias_label = "하방 수급 우세 (신규 숏 유입 / 매도 우세)"
        else:
            bias_label = "수급 중립 (방향성 탐색 구간)"

        return {
            'is_live': live_count > 0,
            'funding_rate_pct': funding_pct,
            'next_funding_pct': next_funding_pct,
            'open_interest_usdt': oi_usdt,
            'oi_change_4h_pct': oi_change_4h_pct,
            'oi_change_24h_pct': oi_change_24h_pct,
            'oi_regime_title': oi_regime_title,
            'oi_regime_desc': oi_regime_desc,
            'long_short_ratio': ls_ratio,
            'taker_buy_sell_ratio': taker_ratio,
            'net_delta_1h': net_delta_1h,
            'cvd_24h': cvd_24h,
            'cvd_div_type': cvd_div_type,
            'cvd_div_label': cvd_div_label,
            'cvd_df': pd.DataFrame(cvd_records) if cvd_records else pd.DataFrame(),
            'bsl_price': bsl_price,
            'ssl_price': ssl_price,
            'sweep_type': sweep_type,
            'sweep_status': sweep_status,
            'sentiment_score': max(-100, min(100, score)),
            'bias_label': bias_label,
            'details': details
        }


class NeelyTimePriceCluster:
    """
    글렌 닐리(Glenn Neely) 시간 비율(Time Symmetry) & 가격-시간 클러스터 분석기
    """

    def analyze_clusters(self, df: pd.DataFrame, dual_data: dict) -> dict:
        m_waves = dual_data.get('macro_waves', [])
        if len(m_waves) < 3 or len(df) < 10:
            return {'time_ratios': [], 'price_clusters': [], 'primary_box': None}

        dt_step = pd.Timestamp(df['timestamp'].iloc[-1]) - pd.Timestamp(df['timestamp'].iloc[-2])
        imp_waves = [w for w in m_waves if w['local_wave_id'] in ['1', '2', '3', '4', '5']]
        corr_waves = [w for w in m_waves if w['local_wave_id'] in ['A', 'B', 'C']]

        time_ratios = []
        target_t_start = None
        target_t_end = None

        if len(imp_waves) >= 4:
            t2 = imp_waves[1]['duration_bars']
            t4 = imp_waves[3]['duration_bars']
            alt_ok = abs(t2 - t4) >= max(2, int(min(t2, t4) * 0.25))
            time_ratios.append(
                f"⏱️ **2파 vs 4파 시간 교대**: 2파 `{t2}봉` vs 4파 `{t4}봉` → "
                f"{'✅ 시간 교대 법칙 충족' if alt_ok else '⚠️ 시간 유사 (복합 시간 연장 가능)'}"
            )

        if len(corr_waves) >= 2:
            wa, wb = corr_waves[0], corr_waves[1]
            t_a = max(1, wa['duration_bars'])
            t_b = max(1, wb['duration_bars'])
            ratio_ba = t_b / t_a
            b_start_time = pd.Timestamp(wb['end_time'])

            t_c_0618 = b_start_time + dt_step * int(round(t_a * 0.618))
            t_c_1000 = b_start_time + dt_step * int(round(t_a * 1.000))
            t_c_1618 = b_start_time + dt_step * int(round(t_a * 1.618))

            target_t_start = min(t_c_0618, t_c_1000)
            target_t_end = max(t_c_1000, t_c_1618)

            time_ratios.append(
                f"⏱️ **(A)파 vs (B)파 시간 비율**: (A)파 `{t_a}봉` / (B)파 `{t_b}봉` (`B/A = {ratio_ba:.2f}배`)"
            )
            time_ratios.append(
                f"📅 **(C)파 닐리 시간 대칭 목표창**: `0.618배` **{str(t_c_0618)[:10]}** ~ "
                f"`1.0배(등배)` **{str(t_c_1000)[:10]}** ~ `1.618배` **{str(t_c_1618)[:10]}**"
            )

        fib_levels = []
        if len(imp_waves) == 5:
            p0 = imp_waves[0]['start_price']
            p5 = imp_waves[4]['end_price']
            span = p5 - p0
            for r in [0.382, 0.500, 0.618, 0.786]:
                lvl = p5 - span * r
                fib_levels.append((lvl, f"대파동 (0)→(5) {r*100:.1f}% 되돌림 (${lvl:,.0f})"))
            fib_levels.append((imp_waves[3]['end_price'], f"대파동 (4)파 저점 지지 (${imp_waves[3]['end_price']:,.0f})"))

        if len(corr_waves) >= 2:
            wa, wb = corr_waves[0], corr_waves[1]
            a_move = wa['end_price'] - wa['start_price']
            for ext in [0.618, 1.000, 1.272, 1.618]:
                lvl = wb['end_price'] + a_move * ext
                if lvl > 0:
                    fib_levels.append((lvl, f"(C)파 = (A)파 × {ext:.3f} 확장 (${lvl:,.0f})"))

        fib_levels.sort(key=lambda x: x[0])
        clusters = []
        used = [False] * len(fib_levels)

        for i in range(len(fib_levels)):
            if used[i]:
                continue
            group = [fib_levels[i]]
            used[i] = True
            for j in range(i + 1, len(fib_levels)):
                if not used[j] and abs(fib_levels[j][0] - fib_levels[i][0]) / max(fib_levels[i][0], 1e-8) <= 0.032:
                    group.append(fib_levels[j])
                    used[j] = True
            if len(group) >= 2:
                prices = [g[0] for g in group]
                clusters.append({
                    'low': min(prices),
                    'high': max(prices),
                    'mid': float(np.mean(prices)),
                    'confluence_count': len(group),
                    'reasons': [g[1] for g in group]
                })

        clusters.sort(key=lambda x: x['confluence_count'], reverse=True)

        primary_box = None
        if clusters and target_t_start is not None and target_t_end is not None:
            top_c = clusters[0]
            pad_y = max(top_c['mid'] * 0.015, (top_c['high'] - top_c['low']) * 0.25)
            primary_box = {
                'x0': target_t_start,
                'x1': target_t_end,
                'y0': top_c['low'] - pad_y,
                'y1': top_c['high'] + pad_y,
                'label': f"닐리 가격-시간 클러스터 (${top_c['low']:,.0f}~${top_c['high']:,.0f})"
            }

        return {
            'time_ratios': time_ratios,
            'price_clusters': clusters,
            'primary_box': primary_box
        }


class MultiScenarioEngine:
    """
    주봉 대파동(Macro) 기준점과 현재가 위치를 동기화한 다중 파동 시나리오 엔진
    """

    def build_scenarios(
        self,
        df: pd.DataFrame,
        dual_data: dict,
        cycle_info: dict = None,
        okx_pos: dict = None,
        weekly_dual: dict = None
    ) -> list:
        # 주봉 대파동 데이터가 있으면 상위 사이클 (A)-(B)-(C) 기준점으로 우선 사용
        ref_dual = weekly_dual if (weekly_dual and len(weekly_dual.get('macro_waves', [])) >= 2) else dual_data
        m_waves = ref_dual.get('macro_waves', [])
        curr_price = float(df['close'].iloc[-1])

        if len(m_waves) < 2:
            return []

        corr_waves = [w for w in m_waves if w['local_wave_id'] in ['A', 'B', 'C']]
        imp_waves = [w for w in m_waves if w['local_wave_id'] in ['1', '2', '3', '4', '5']]
        scenarios = []

        if len(corr_waves) >= 2:
            wa = corr_waves[0]
            wb = corr_waves[1]
            wc = corr_waves[2] if len(corr_waves) == 3 else None

            proj = ref_dual.get('projection')
            c_target = proj['target_price'] if proj else (wc['end_price'] * 0.90 if wc else wb['end_price'] - wa['price_change'] * 0.85)
            is_down_corr = (wa['direction'] == 'DOWN')

            if is_down_corr:
                a_low = wa['end_price']
                b_high = wb['end_price']
                reclaim_mid = min(b_high * 0.98, (b_high + curr_price) * 0.5)

                # 현재가가 아직 주봉 (A)파 저점 위에서 (C)파 목표 클러스터를 향해 진행 중인 상태
                if curr_price > a_low:
                    scenarios.append({
                        'rank_tag': "관점 A (메인 — 확률 65%): 지그재그 (C)파 5파동 하방 전개",
                        'title': f"5-3-5 지그재그 — `(A)` 저점(`${a_low:,.0f}`) 테스트 후 클러스터(`${c_target:,.0f}`) 목표",
                        'bias': 'BEARISH_TO_TARGET',
                        'logic': (
                            f"대파동 `(5)` 완성 이후 **`${a_low:,.0f}`에서 5파동 `(A)`파가 완료**되고 **`${b_high:,.0f}`에서 `(B)`파 반등이 종료**되어, "
                            f"현재 **`(C)`파(`C-1~C-5`)**가 진행 중입니다. 정상 지그재그 비율상 직전 `(A)`파 저점(`${a_low:,.0f}`) 이탈 후 "
                            f"`${c_target:,.0f}` 클러스터 권역까지 하방 여력이 열려 있습니다."
                        ),
                        'target_zone': f"1차 `(A)`저점 **`${a_low:,.0f}`** 테스트 ➔ 2차 클러스터 **`${c_target:,.0f}`**",
                        'invalidation_trigger': f"**`${b_high:,.0f}`** (`(B)`파 고점 상향 돌파 시 기존 `(C)`파 하락 카운팅 무효화)",
                        'action_plan': f"`${b_high:,.0f}` 하회 유지 시 단기 반등(C-4) 구간 분할 숏, `${a_low:,.0f}~${c_target:,.0f}` 클러스터 도달 후 반전 롱 준비"
                    })

                    scenarios.append({
                        'rank_tag': "관점 B (대안 — 확률 35%): 단축형(Truncated) (C)파 & W-X-Y 조기 마감",
                        'title': f"`(A)`파 저점(`${a_low:,.0f}`) 지지 방어 후 신규 상승 파동 전환",
                        'bias': 'BULLISH_REVERSAL',
                        'logic': (
                            f"현재가(`${curr_price:,.0f}`) 부근에서 매수세가 유입되며 `(C)`파가 직전 `(A)`파 저점(`${a_low:,.0f}`)을 깨지 않고 "
                            f"단축형(Truncated) 또는 `W-X-Y` 복합조정으로 조기 마감한 뒤 새로운 상승 파동으로 전환하는 관점입니다."
                        ),
                        'target_zone': f"1차 **`${reclaim_mid:,.0f}`** 돌파 ➔ 2차 **`${b_high:,.0f}`** (`(B)`파 고점 재도전)",
                        'invalidation_trigger': f"**`${a_low:,.0f}`** (주봉 `(A)`파 저점 지지 이탈 시 조기 바닥론 즉시 폐기)",
                        'action_plan': f"`${a_low:,.0f}` 손절 기준 설정 후 단기 눌림목 분할 롱 탐색, `${reclaim_mid:,.0f}` 안착 시 추세 추종"
                    })
                # 현재가가 이미 주봉 (A)파 저점 아래로 내려와 클러스터에 도달한 상태
                else:
                    cluster_floor = min(c_target, curr_price * 0.94)
                    rebound_1 = curr_price + (b_high - curr_price) * 0.382
                    rebound_2 = curr_price + (b_high - curr_price) * 0.618

                    scenarios.append({
                        'rank_tag': "관점 A (메인 — 확률 65%): 지그재그 (C-5) 클러스터 종점 완성 & 반전 롱",
                        'title': f"5-3-5 지그재그 막바지 — `${c_target:,.0f}` 클러스터 지지 후 상방 전환",
                        'bias': 'BOTTOM_REVERSAL_LONG',
                        'logic': (
                            f"`${a_low:,.0f}`에서 `(A)`파가 완료되고 `${b_high:,.0f}`에서 `(B)`파가 마무리된 뒤, "
                            f"현재 **`(C)`파가 `(A)`파 저점을 하향 돌파해 목표 클러스터(`${c_target:,.0f}`) 권역에 진입**했습니다. "
                            f"`C-5` 마지막 파동 소화 후 중기 상승 전환이 유력합니다."
                        ),
                        'target_zone': f"클러스터 바닥 **`${c_target:,.0f}`** 지지 후 1차 반등 **`${rebound_1:,.0f}`** ➔ 2차 **`${rebound_2:,.0f}`**",
                        'invalidation_trigger': f"**`${cluster_floor:,.0f}`** 하향 이탈 시 단기 바닥 반전 시나리오 무효화",
                        'action_plan': f"클러스터 구간 분할 롱 지정가 대기 후 손절가 `${cluster_floor:,.0f}` 엄수"
                    })

                    scenarios.append({
                        'rank_tag': "관점 B (대안 — 확률 35%): (C)파 1.618배 확장 투매 진행",
                        'title': "클러스터 지지 실패 및 (C)파 파동 연장(Extension) 하락",
                        'bias': 'EXTENDED_BEAR',
                        'logic': (
                            f"현재가(`${curr_price:,.0f}`) 부근 클러스터에서 매수 방어가 실패할 경우, "
                            f"`(C)`파가 `(A)`파 길이의 `1.382~1.618배` 구간까지 한 단계 더 투매가 출현하는 시나리오입니다."
                        ),
                        'target_zone': f"1차 **`${cluster_floor:,.0f}`** 이탈 ➔ 2차 확장 목표 **`${cluster_floor * 0.90:,.0f}`**",
                        'invalidation_trigger': f"**`${rebound_1:,.0f}`** 상향 돌파 시 추가 급락론 폐기",
                        'action_plan': f"`${cluster_floor:,.0f}` 이탈 확인 전까지 추격 숏 자제"
                    })
            else:
                scenarios.append({
                    'rank_tag': "메인 시나리오 (확률 65%)",
                    'title': "상방 조정 (C)파 전개",
                    'bias': 'BULLISH',
                    'logic': "상방 A-B-C 파동이 진행 중입니다.",
                    'target_zone': f"`${c_target:,.0f}`",
                    'invalidation_trigger': f"**`${wb['end_price']:,.0f}`** 이탈 시 무효화",
                    'action_plan': "눌림목 분할 롱 대응"
                })
        elif len(imp_waves) >= 3:
            w_last = imp_waves[-1]
            scenarios.append({
                'rank_tag': "메인 시나리오 (확률 65%)",
                'title': f"충격파 ({w_last['local_wave_id']})파 추세 연장",
                'bias': 'TREND_FOLLOW',
                'logic': "상승/하락 충격파 내부 소파동이 유지되고 있으며 1-4파 비겹침 규칙이 유효합니다.",
                'target_zone': f"`${w_last['end_price'] * 1.08:,.0f}`",
                'invalidation_trigger': f"**`${w_last['start_price']:,.0f}`** 이탈 시 추세 연장 무효화",
                'action_plan': "추세 방향 눌림목 분할 진입"
            })

        return scenarios


class NeelyStrategyBacktester:
    """
    과거 차트 워크포워드(Walk-Forward) 백테스팅 & 승률/손익비 검증기
    """

    def run_backtest(self, df: pd.DataFrame, risk_pct: float = 2.0, rr_target: float = 2.4) -> dict:
        if df is None or len(df) < 80:
            return {'total_trades': 0}

        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        rsi = df['rsi'].values
        ema20 = df['ema_20'].values
        ema50 = df['ema_50'].values
        atr = df['atr'].values
        times = df['timestamp'].values

        trades = []
        equity = 100.0
        peak_equity = 100.0
        max_dd = 0.0
        equity_curve = [{'time': times[50], 'equity': 100.0}]

        in_pos = False
        pos_dir = None
        entry_p = 0.0
        sl_p = 0.0
        tp_p = 0.0
        entry_time = None
        cooldown = 0

        for i in range(50, len(df) - 1):
            if cooldown > 0:
                cooldown -= 1

            if in_pos:
                hit_sl = False
                hit_tp = False
                if pos_dir == 'LONG':
                    if lows[i] <= sl_p:
                        hit_sl = True
                    elif highs[i] >= tp_p:
                        hit_tp = True
                else:
                    if highs[i] >= sl_p:
                        hit_sl = True
                    elif lows[i] <= tp_p:
                        hit_tp = True

                if hit_sl or hit_tp:
                    r_mult = -1.05 if hit_sl else (rr_target - 0.05)
                    pnl_pct = risk_pct * r_mult
                    equity *= (1.0 + pnl_pct / 100.0)
                    peak_equity = max(peak_equity, equity)
                    dd = (peak_equity - equity) / peak_equity * 100.0
                    max_dd = max(max_dd, dd)

                    trades.append({
                        '진입시각': str(entry_time)[:16],
                        '청산시각': str(times[i])[:16],
                        '방향': pos_dir,
                        '진입가': round(entry_p, 1),
                        '손절가': round(sl_p, 1),
                        '목표가': round(tp_p, 1),
                        '결과': '익절 (TP)' if hit_tp else '손절 (SL)',
                        'R배수': round(r_mult, 2),
                        '계좌손익(%)': round(pnl_pct, 2)
                    })
                    equity_curve.append({'time': times[i], 'equity': round(equity, 2)})
                    in_pos = False
                    cooldown = 3
                continue

            if cooldown > 0 or np.isnan(atr[i]) or atr[i] <= 0:
                continue

            win_low = np.min(lows[i - 20:i])
            win_high = np.max(highs[i - 20:i])
            prev_rsi_min = np.min(rsi[i - 20:i - 2])
            prev_rsi_max = np.max(rsi[i - 20:i - 2])

            long_cond = (
                (lows[i - 1] <= win_low * 1.003 and rsi[i - 1] > prev_rsi_min + 2.0 and rsi[i] < 46 and closes[i] > ema20[i]) or
                (ema20[i - 1] <= ema50[i - 1] and ema20[i] > ema50[i] and rsi[i] < 62)
            )
            short_cond = (
                (highs[i - 1] >= win_high * 0.997 and rsi[i - 1] < prev_rsi_max - 2.0 and rsi[i] > 54 and closes[i] < ema20[i]) or
                (ema20[i - 1] >= ema50[i - 1] and ema20[i] < ema50[i] and rsi[i] > 38)
            )

            if long_cond:
                in_pos = True
                pos_dir = 'LONG'
                entry_p = closes[i]
                risk_dist = max(atr[i] * 1.5, entry_p * 0.004)
                sl_p = entry_p - risk_dist
                tp_p = entry_p + risk_dist * rr_target
                entry_time = times[i]
            elif short_cond:
                in_pos = True
                pos_dir = 'SHORT'
                entry_p = closes[i]
                risk_dist = max(atr[i] * 1.5, entry_p * 0.004)
                sl_p = entry_p + risk_dist
                tp_p = entry_p - risk_dist * rr_target
                entry_time = times[i]

        if not trades:
            return {'total_trades': 0}

        wins = [t for t in trades if t['R배수'] > 0]
        losses = [t for t in trades if t['R배수'] <= 0]
        win_rate = len(wins) / len(trades) * 100.0
        gross_profit = sum(t['R배수'] for t in wins)
        gross_loss = abs(sum(t['R배수'] for t in losses))
        profit_factor = gross_profit / max(gross_loss, 0.01)
        avg_r = float(np.mean([t['R배수'] for t in trades]))
        total_ret = equity - 100.0

        return {
            'total_trades': len(trades),
            'win_count': len(wins),
            'loss_count': len(losses),
            'win_rate': round(win_rate, 1),
            'avg_rr': round(avg_r, 2),
            'profit_factor': round(profit_factor, 2),
            'total_return_pct': round(total_ret, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'equity_curve': pd.DataFrame(equity_curve),
            'trades_df': pd.DataFrame(trades)
        }