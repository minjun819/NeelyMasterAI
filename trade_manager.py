# -*- coding: utf-8 -*-
import os
import time
import uuid
import ccxt
import numpy as np
import pandas as pd


class FuturesTradeManager:
    """
    OKX 선물 AI 트레이딩 시그널 & 구조적(피벗·피보나치·FVG·EMA·유동성·HTF·ADX) SL/TP 엔진 (v7.2)
    """

    _htf_cache = {}

    def _fetch_htf_regime(self, active_tf: str, current_price: float, df: pd.DataFrame = None) -> dict:
        tf_map = {
            '15m': ('1h', '1시간봉'),
            '1h': ('4h', '4시간봉'),
            '4h': ('1d', '일봉'),
            '1d': ('1w', '주봉'),
        }
        htf_tf, htf_label = tf_map.get(active_tf, ('4h', '4시간봉'))
        bucket = int(time.time() // 180)
        cache_key = (htf_tf, bucket)

        if cache_key in self._htf_cache:
            return self._htf_cache[cache_key]

        result = {
            'htf_tf': htf_tf,
            'htf_label': htf_label,
            'regime': 'NEUTRAL',
            'summary': f"{htf_label} 추세 데이터 분석중",
            'ema50': None,
            'ema200': None,
            'adx': None,
        }

        try:
            ex = ccxt.okx({'enableRateLimit': True, 'timeout': 5000, 'options': {'defaultType': 'swap'}})
            raw = ex.fetch_ohlcv('BTC/USDT:USDT', timeframe=htf_tf, limit=210)
            if raw and len(raw) >= 60:
                hdf = pd.DataFrame(raw, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                close = hdf['close'].astype(float)
                high = hdf['high'].astype(float)
                low = hdf['low'].astype(float)

                ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
                ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
                h_close = float(close.iloc[-1])

                prev_c = close.shift(1)
                tr = pd.concat([(high - low), (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
                up_m = high.diff()
                dn_m = -low.diff()
                p_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
                m_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
                tr14 = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean().replace(0, np.nan)
                p_di = 100.0 * pd.Series(p_dm).ewm(alpha=1.0 / 14.0, adjust=False).mean() / tr14
                m_di = 100.0 * pd.Series(m_dm).ewm(alpha=1.0 / 14.0, adjust=False).mean() / tr14
                dx = 100.0 * (p_di - m_di).abs() / (p_di + m_di).replace(0, np.nan)
                h_adx = float(dx.ewm(alpha=1.0 / 14.0, adjust=False).mean().fillna(20.0).iloc[-1])

                if h_close > ema50 and ema50 >= ema200:
                    regime = 'BULL'
                    summary = f"{htf_label} 상승추세 (현재가 > EMA50 `${ema50:,.1f}` > EMA200 `${ema200:,.1f}`, ADX `{h_adx:.1f}`)"
                elif h_close < ema50 and ema50 <= ema200:
                    regime = 'BEAR'
                    summary = f"{htf_label} 하락추세 (현재가 < EMA50 `${ema50:,.1f}` < EMA200 `${ema200:,.1f}`, ADX `{h_adx:.1f}`)"
                else:
                    regime = 'NEUTRAL'
                    summary = f"{htf_label} 전환/수렴 구간 (EMA50 `${ema50:,.1f}` / EMA200 `${ema200:,.1f}`, ADX `{h_adx:.1f}`)"

                result.update({
                    'regime': regime,
                    'summary': summary,
                    'ema50': ema50,
                    'ema200': ema200,
                    'adx': h_adx,
                })
                self._htf_cache[cache_key] = result
                return result
        except Exception:
            pass

        # API 조회 실패 시 현재 차트 EMA200 기반 폴백
        if df is not None and 'ema_200' in df.columns and not pd.isna(df['ema_200'].iloc[-1]):
            e200 = float(df['ema_200'].iloc[-1])
            if current_price > e200:
                result['regime'] = 'BULL'
                result['summary'] = f"장기 이평선(EMA200 `${e200:,.1f}`) 상단 상승 우위"
            else:
                result['regime'] = 'BEAR'
                result['summary'] = f"장기 이평선(EMA200 `${e200:,.1f}`) 하단 하락 우위"
            result['ema200'] = e200

        self._htf_cache[cache_key] = result
        return result

    def _compute_adx_and_fvg(self, df: pd.DataFrame, atr_val: float) -> dict:
        adx_val = 22.0
        bull_fvg = None
        bear_fvg = None

        if df is None or len(df) < 20:
            return {'adx': adx_val, 'bull_fvg': bull_fvg, 'bear_fvg': bear_fvg}

        try:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)
            open_p = df['open'].astype(float) if 'open' in df.columns else close.shift(1).fillna(close)

            prev_close = close.shift(1)
            tr = pd.concat(
                [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
                axis=1
            ).max(axis=1)
            up_move = high.diff()
            down_move = -low.diff()
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

            tr14 = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean().replace(0, np.nan)
            plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / 14.0, adjust=False).mean() / tr14
            minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / 14.0, adjust=False).mean() / tr14
            dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
            adx_series = dx.ewm(alpha=1.0 / 14.0, adjust=False).mean()
            if not pd.isna(adx_series.iloc[-1]):
                adx_val = float(adx_series.iloc[-1])

            start_idx = max(2, len(df) - 40)
            for i in range(start_idx, len(df)):
                c_low = float(low.iloc[i])
                c_high = float(high.iloc[i])
                p2_high = float(high.iloc[i - 2])
                p2_low = float(low.iloc[i - 2])
                m_close = float(close.iloc[i - 1])
                m_open = float(open_p.iloc[i - 1])

                if c_low > p2_high and m_close > m_open and (c_low - p2_high) > atr_val * 0.25:
                    sub_low = float(low.iloc[i + 1:].min()) if i + 1 < len(df) else c_low
                    if sub_low > p2_high:
                        bull_fvg = (p2_high, c_low)

                if c_high < p2_low and m_close < m_open and (p2_low - c_high) > atr_val * 0.25:
                    sub_high = float(high.iloc[i + 1:].max()) if i + 1 < len(df) else c_high
                    if sub_high < p2_low:
                        bear_fvg = (c_high, p2_low)
        except Exception:
            pass

        return {'adx': adx_val, 'bull_fvg': bull_fvg, 'bear_fvg': bear_fvg}

    def _find_structural_sl(
        self,
        direction: str,
        current_price: float,
        atr_val: float,
        max_sl_pct: float,
        df: pd.DataFrame = None,
        pivots: list = None,
        dual_data: dict = None,
        okx_pos: dict = None
    ) -> tuple:
        candidates = []
        buffer = max(atr_val * 0.30, current_price * 0.0012)

        if pivots:
            for p in pivots[-15:]:
                pv = float(p.get('price', 0.0)) if isinstance(p, dict) else 0.0
                if pv <= 0:
                    continue
                if direction == 'LONG' and pv < current_price:
                    candidates.append((pv - buffer, f"직전 스윙 저점(${pv:,.1f}) 하단 무효화"))
                elif direction == 'SHORT' and pv > current_price:
                    candidates.append((pv + buffer, f"직전 스윙 고점(${pv:,.1f}) 상단 무효화"))

        if dual_data:
            for w_key in ['sub_waves', 'macro_waves']:
                w_list = dual_data.get(w_key, [])
                if w_list:
                    last_w = w_list[-1]
                    sp = float(last_w.get('start_price', 0.0))
                    if sp <= 0:
                        continue
                    w_lbl = (
                        last_w.get('local_wave_id')
                        or last_w.get('label')
                        or last_w.get('wave_id')
                        or ('세부' if w_key == 'sub_waves' else '대파동')
                    )
                    if direction == 'LONG' and sp < current_price:
                        candidates.append((sp - buffer, f"({w_lbl})파 시작점(${sp:,.1f}) 이탈"))
                    elif direction == 'SHORT' and sp > current_price:
                        candidates.append((sp + buffer, f"({w_lbl})파 시작점(${sp:,.1f}) 돌파"))

        if df is not None and len(df) > 0:
            for col, label in [('ema_50', 'EMA50'), ('ema_100', 'EMA100'), ('ema_200', 'EMA200')]:
                if col in df.columns and not pd.isna(df[col].iloc[-1]):
                    ev = float(df[col].iloc[-1])
                    if direction == 'LONG' and ev < current_price:
                        candidates.append((ev - buffer, f"{label}(${ev:,.1f}) 지지 이탈"))
                    elif direction == 'SHORT' and ev > current_price:
                        candidates.append((ev + buffer, f"{label}(${ev:,.1f}) 저항 돌파"))

        if okx_pos:
            ssl_p = float(okx_pos.get('ssl_price', 0.0))
            bsl_p = float(okx_pos.get('bsl_price', 0.0))
            if direction == 'LONG' and 0 < ssl_p < current_price:
                candidates.append((ssl_p - buffer, f"OKX 하단 유동성 풀 SSL(${ssl_p:,.1f}) 이탈"))
            elif direction == 'SHORT' and bsl_p > current_price:
                candidates.append((bsl_p + buffer, f"OKX 상단 유동성 풀 BSL(${bsl_p:,.1f}) 돌파"))

        min_dist = max(atr_val * 0.90, current_price * 0.004)
        max_dist = current_price * max_sl_pct

        valid = []
        if direction == 'LONG':
            lo_bound = current_price - max_dist
            hi_bound = current_price - min_dist
            for lvl, reason in candidates:
                if lo_bound <= lvl <= hi_bound:
                    valid.append((lvl, reason))
            valid.sort(key=lambda x: x[0], reverse=True)
            if valid:
                return valid[0]
            fallback = max(lo_bound, current_price - atr_val * 1.65)
            return fallback, f"타임프레임 최대 손절 상한({max_sl_pct * 100:.1f}%) 구조적 손절"
        else:
            lo_bound = current_price + min_dist
            hi_bound = current_price + max_dist
            for lvl, reason in candidates:
                if lo_bound <= lvl <= hi_bound:
                    valid.append((lvl, reason))
            valid.sort(key=lambda x: x[0])
            if valid:
                return valid[0]
            fallback = min(hi_bound, current_price + atr_val * 1.65)
            return fallback, f"타임프레임 최대 손절 상한({max_sl_pct * 100:.1f}%) 구조적 손절"

    def _find_smart_limit_entries(
        self,
        direction: str,
        current_price: float,
        atr_val: float,
        structural_sl: float,
        rsi_val: float,
        df: pd.DataFrame = None,
        pivots: list = None,
        dual_data: dict = None,
        okx_pos: dict = None,
        fvg_info: dict = None
    ) -> dict:
        candidates = []

        if fvg_info:
            if direction == 'LONG' and fvg_info.get('bull_fvg'):
                f_bot, f_top = fvg_info['bull_fvg']
                if f_top < current_price:
                    candidates.append((f_top, "상승 FVG 매물대 상단"))
                if f_bot < current_price:
                    candidates.append(((f_bot + f_top) * 0.5, "상승 FVG 매물대 중단"))
            elif direction == 'SHORT' and fvg_info.get('bear_fvg'):
                f_bot, f_top = fvg_info['bear_fvg']
                if f_bot > current_price:
                    candidates.append((f_bot, "하락 FVG 매물대 하단"))
                if f_top > current_price:
                    candidates.append(((f_bot + f_top) * 0.5, "하락 FVG 매물대 중단"))

        if dual_data:
            sub_waves = dual_data.get('sub_waves', [])
            if sub_waves:
                last_sw = sub_waves[-1]
                ep_sw = float(last_sw.get('end_price', 0.0))
                sp_sw = float(last_sw.get('start_price', 0.0))
                sw_span = abs(ep_sw - sp_sw)
                if sw_span > atr_val * 0.4:
                    for fib in [0.382, 0.500, 0.618]:
                        if direction == 'LONG':
                            candidates.append((current_price - sw_span * fib, f"세부파동 {fib:.3f} 되돌림"))
                        else:
                            candidates.append((current_price + sw_span * fib, f"세부파동 {fib:.3f} 반등"))

            proj = dual_data.get('projection')
            if proj and len(proj.get('prices', [])) >= 2:
                p_next = float(proj['prices'][1])
                if direction == 'LONG' and p_next < current_price:
                    candidates.append((p_next, "예상 세부경로 단기 눌림목"))
                elif direction == 'SHORT' and p_next > current_price:
                    candidates.append((p_next, "예상 세부경로 단기 반등 고점"))

        if df is not None and len(df) > 0:
            for col, label in [('ema_20', 'EMA20'), ('ema_50', 'EMA50')]:
                if col in df.columns and not pd.isna(df[col].iloc[-1]):
                    ema_v = float(df[col].iloc[-1])
                    if direction == 'LONG' and ema_v < current_price:
                        candidates.append((ema_v, f"{label} 동적 지지"))
                    elif direction == 'SHORT' and ema_v > current_price:
                        candidates.append((ema_v, f"{label} 동적 저항"))

        if pivots:
            for p in pivots[-12:]:
                pv = float(p.get('price', 0.0)) if isinstance(p, dict) else 0.0
                if pv <= 0:
                    continue
                if direction == 'LONG' and pv < current_price:
                    candidates.append((pv, "직전 스윙 피벗 지지"))
                elif direction == 'SHORT' and pv > current_price:
                    candidates.append((pv, "직전 스윙 피벗 저항"))

        if okx_pos:
            ssl_p = float(okx_pos.get('ssl_price', 0.0))
            bsl_p = float(okx_pos.get('bsl_price', 0.0))
            if direction == 'LONG' and 0 < ssl_p < current_price:
                candidates.append((ssl_p, "OKX 하단 유동성 풀(SSL) 스윕 반등 구간"))
            elif direction == 'SHORT' and bsl_p > current_price:
                candidates.append((bsl_p, "OKX 상단 유동성 풀(BSL) 스윕 저항 구간"))

        valid_cands = []
        if direction == 'LONG':
            min_bound = structural_sl + atr_val * 0.40
            max_bound = current_price - atr_val * 0.12
            for lvl, reason in candidates:
                if min_bound <= lvl <= max_bound:
                    valid_cands.append((lvl, reason))
            valid_cands.sort(key=lambda x: x[0], reverse=True)

            if rsi_val <= 30.0:
                e1 = current_price - atr_val * 0.15
                e2 = max(min_bound, current_price - atr_val * 0.45)
                basis = "RSI 과매도 구간 (소폭 눌림 분할 지정가)"
            elif len(valid_cands) >= 2:
                e1, r1 = valid_cands[0]
                deep_cands = [x for x in valid_cands[1:] if (e1 - x[0]) >= atr_val * 0.22]
                if deep_cands:
                    e2, r2 = deep_cands[0]
                    basis = f"{r1} + {r2}"
                else:
                    e2 = max(min_bound, e1 - atr_val * 0.45)
                    basis = f"{r1} + ATR 심화 눌림"
            elif len(valid_cands) == 1:
                e1, r1 = valid_cands[0]
                e2 = max(min_bound, e1 - atr_val * 0.45)
                basis = f"{r1} + ATR 2차 눌림"
            else:
                e1 = max(min_bound, current_price - atr_val * 0.35)
                e2 = max(min_bound, current_price - atr_val * 0.75)
                basis = "ATR 변동성 기반 1·2차 눌림목 대기"

            avg_entry = (e1 + e2) * 0.5
            diff_pct = ((avg_entry - current_price) / current_price) * 100.0
            delta_label = f"현재가 대비 {diff_pct:.2f}% 눌림 대기"
        else:
            min_bound = current_price + atr_val * 0.12
            max_bound = structural_sl - atr_val * 0.40
            for lvl, reason in candidates:
                if min_bound <= lvl <= max_bound:
                    valid_cands.append((lvl, reason))
            valid_cands.sort(key=lambda x: x[0])

            if rsi_val >= 70.0:
                e1 = current_price + atr_val * 0.15
                e2 = min(max_bound, current_price + atr_val * 0.45)
                basis = "RSI 과매수 구간 (소폭 반등 분할 지정가)"
            elif len(valid_cands) >= 2:
                e1, r1 = valid_cands[0]
                high_cands = [x for x in valid_cands[1:] if (x[0] - e1) >= atr_val * 0.22]
                if high_cands:
                    e2, r2 = high_cands[0]
                    basis = f"{r1} + {r2}"
                else:
                    e2 = min(max_bound, e1 + atr_val * 0.45)
                    basis = f"{r1} + ATR 추가 반등"
            elif len(valid_cands) == 1:
                e1, r1 = valid_cands[0]
                e2 = min(max_bound, e1 + atr_val * 0.45)
                basis = f"{r1} + ATR 2차 반등"
            else:
                e1 = min(max_bound, current_price + atr_val * 0.35)
                e2 = min(max_bound, current_price + atr_val * 0.75)
                basis = "ATR 변동성 기반 1·2차 반등 되돌림 대기"

            avg_entry = (e1 + e2) * 0.5
            diff_pct = ((avg_entry - current_price) / current_price) * 100.0
            delta_label = f"현재가 대비 +{diff_pct:.2f}% 반등 대기"

        return {
            'entry_1': e1,
            'entry_2': e2,
            'avg_entry': avg_entry,
            'delta_label': delta_label,
            'basis': basis
        }

    def _find_structural_tps(
        self,
        direction: str,
        entry_price: float,
        actual_risk: float,
        atr_val: float,
        rsi_val: float,
        df: pd.DataFrame = None,
        pivots: list = None,
        dual_data: dict = None,
        okx_pos: dict = None
    ) -> dict:
        raw_targets = []

        if pivots:
            for p in pivots:
                pv = float(p.get('price', 0.0)) if isinstance(p, dict) else 0.0
                if pv <= 0:
                    continue
                if direction == 'LONG' and pv > entry_price:
                    raw_targets.append((pv, "직전 스윙 고점 저항"))
                elif direction == 'SHORT' and pv < entry_price:
                    raw_targets.append((pv, "직전 스윙 저점 지지"))

        if df is not None and len(df) > 0:
            for col, label in [('ema_20', 'EMA20'), ('ema_50', 'EMA50'), ('ema_100', 'EMA100'), ('ema_200', 'EMA200')]:
                if col in df.columns and not pd.isna(df[col].iloc[-1]):
                    ev = float(df[col].iloc[-1])
                    if direction == 'LONG' and ev > entry_price:
                        raw_targets.append((ev, f"{label} 이동평균 저항"))
                    elif direction == 'SHORT' and ev < entry_price:
                        raw_targets.append((ev, f"{label} 이동평균 지지"))

        if dual_data:
            for w_key, prefix in [('sub_waves', '세부파동'), ('macro_waves', '대파동')]:
                w_list = dual_data.get(w_key, [])
                if w_list:
                    last_w = w_list[-1]
                    ep_w = float(last_w.get('end_price', 0.0))
                    sp_w = float(last_w.get('start_price', 0.0))
                    span = abs(ep_w - sp_w)
                    if span > atr_val * 0.5:
                        for fib in [0.382, 0.500, 0.618, 0.786, 1.000]:
                            if direction == 'LONG':
                                raw_targets.append((entry_price + span * fib, f"{prefix} 피보나치 {fib:.3f} 목표"))
                            else:
                                raw_targets.append((max(entry_price * 0.35, entry_price - span * fib), f"{prefix} 피보나치 {fib:.3f} 목표"))
                        for ext in [1.272, 1.618]:
                            if direction == 'LONG':
                                raw_targets.append((entry_price + span * ext, f"{prefix} 피보나치 {ext:.3f} 확장"))
                            else:
                                raw_targets.append((max(entry_price * 0.30, entry_price - span * ext), f"{prefix} 피보나치 {ext:.3f} 확장"))

            proj = dual_data.get('projection')
            if proj and proj.get('target_price'):
                pt = float(proj['target_price'])
                if (direction == 'LONG' and pt > entry_price) or (direction == 'SHORT' and pt < entry_price):
                    raw_targets.append((pt, "닐리 예상 파동 종점"))

        if okx_pos:
            bsl_p = float(okx_pos.get('bsl_price', 0.0))
            ssl_p = float(okx_pos.get('ssl_price', 0.0))
            if direction == 'LONG' and bsl_p > entry_price:
                raw_targets.append((bsl_p, "OKX 상단 유동성 풀(BSL)"))
            elif direction == 'SHORT' and 0 < ssl_p < entry_price:
                raw_targets.append((ssl_p, "OKX 하단 유동성 풀(SSL)"))

        if direction == 'LONG':
            raw_targets.sort(key=lambda x: x[0])
        else:
            raw_targets.sort(key=lambda x: x[0], reverse=True)

        merged = []
        tol = max(atr_val * 0.30, entry_price * 0.002)
        for lvl, reason in raw_targets:
            r_mult = abs(lvl - entry_price) / actual_risk
            if r_mult < 0.90:
                continue
            if merged and abs(lvl - merged[-1]['price']) <= tol:
                if reason not in merged[-1]['reasons']:
                    merged[-1]['reasons'].append(reason)
                merged[-1]['price'] = (merged[-1]['price'] + lvl) * 0.5
                merged[-1]['rr'] = abs(merged[-1]['price'] - entry_price) / actual_risk
            else:
                merged.append({'price': lvl, 'rr': r_mult, 'reasons': [reason]})

        rsi_stretched = (direction == 'LONG' and rsi_val >= 62.0) or (direction == 'SHORT' and rsi_val <= 38.0)
        tp1_min_r, tp1_max_r = (0.95, 1.70) if rsi_stretched else (1.15, 1.95)
        tp2_min_r, tp2_max_r = (1.65, 2.90) if rsi_stretched else (1.85, 3.30)

        tp1_cand = next((m for m in merged if tp1_min_r <= m['rr'] <= tp1_max_r), None)
        if tp1_cand:
            tp1 = tp1_cand['price']
            tp1_basis = " + ".join(tp1_cand['reasons'][:2])
        else:
            def_r1 = 1.35 if rsi_stretched else 1.50
            tp1 = entry_price + actual_risk * def_r1 if direction == 'LONG' else max(entry_price * 0.45, entry_price - actual_risk * def_r1)
            tp1_basis = f"ATR/R:R 균형 1차 목표 ({def_r1:.2f}R)"
        rr1 = abs(tp1 - entry_price) / actual_risk

        tp2_cand = next(
            (m for m in merged if tp2_min_r <= m['rr'] <= tp2_max_r and m['rr'] >= rr1 + 0.40),
            None
        )
        if tp2_cand:
            tp2 = tp2_cand['price']
            tp2_basis = " + ".join(tp2_cand['reasons'][:2])
        else:
            def_r2 = max(rr1 + 0.70, 2.00 if rsi_stretched else 2.30)
            tp2 = entry_price + actual_risk * def_r2 if direction == 'LONG' else max(entry_price * 0.40, entry_price - actual_risk * def_r2)
            tp2_basis = f"추세 확장 2차 목표 ({def_r2:.2f}R)"
        rr2 = abs(tp2 - entry_price) / actual_risk

        tp3_cand = next(
            (m for m in merged if m['rr'] >= max(rr2 + 0.60, 2.75) and m['rr'] <= 5.50),
            None
        )
        if tp3_cand:
            tp3 = tp3_cand['price']
            tp3_basis = " + ".join(tp3_cand['reasons'][:2])
        else:
            def_r3 = rr2 + 1.05
            tp3 = entry_price + actual_risk * def_r3 if direction == 'LONG' else max(entry_price * 0.35, entry_price - actual_risk * def_r3)
            tp3_basis = f"피보나치 확장 최종 목표 ({def_r3:.2f}R)"
        rr3 = abs(tp3 - entry_price) / actual_risk

        return {
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'rr1': rr1,
            'rr2': rr2,
            'rr3': rr3,
            'tp1_basis': tp1_basis,
            'tp2_basis': tp2_basis,
            'tp3_basis': tp3_basis
        }

    def generate_trade_plan(
        self,
        current_price: float,
        atr_val: float,
        top_pattern: dict,
        rsi_val: float,
        div_info: dict,
        vol_info: dict,
        sr_info: dict,
        ma_info: dict,
        cycle_info: dict,
        capital_usdt: float,
        risk_pct: float,
        active_tf: str = "1h",
        dual_data: dict = None,
        df: pd.DataFrame = None,
        pivots: list = None,
        okx_pos: dict = None
    ) -> dict:
        if current_price <= 0 or atr_val <= 0:
            return {'signal': 'WAIT', 'reason': '실시간 가격 또는 ATR 데이터 집계 대기 중입니다.'}

        max_sl_cap_map = {'15m': 0.012, '1h': 0.020, '4h': 0.035, '1d': 0.065, '1w': 0.120}
        max_sl_pct = max_sl_cap_map.get(active_tf, 0.025)

        direction = None
        setup_type = ""
        confidence = 68
        checks = []

        m_waves = dual_data.get('macro_waves', []) if dual_data else []
        active_mw = m_waves[-1] if m_waves else None

        if active_mw and active_mw.get('is_active'):
            w_id = active_mw.get('local_wave_id', '진행')
            w_dir = active_mw.get('direction', 'DOWN')
            if w_dir == 'DOWN':
                if rsi_val <= 34 or (div_info and div_info.get('type') == 'BULLISH'):
                    direction = 'LONG'
                    setup_type = f"({w_id})파 과매도/다이버전스 반전 매수 셋업"
                    confidence += 10
                else:
                    direction = 'SHORT'
                    setup_type = f"({w_id})파 하방 파동 추세 추종 및 반등 매도 셋업"
                    confidence += 12
            else:
                if rsi_val >= 68 or (div_info and div_info.get('type') == 'BEARISH'):
                    direction = 'SHORT'
                    setup_type = f"({w_id})파 과매수/다이버전스 고점 반전 매도 셋업"
                    confidence += 10
                else:
                    direction = 'LONG'
                    setup_type = f"({w_id})파 상방 파동 추세 추종 눌림목 매수 셋업"
                    confidence += 12
        elif top_pattern:
            if top_pattern['category'] == 'IMPULSE':
                direction = 'SHORT' if top_pattern['direction'] == 'UP' else 'LONG'
                setup_type = f"{top_pattern['pattern_name']} 이후 되돌림 타점"
            else:
                direction = 'LONG' if top_pattern['direction'] == 'DOWN' else 'SHORT'
                setup_type = f"{top_pattern['pattern_name']} 종료 반전 타점"
            confidence = max(confidence, int(top_pattern.get('score', 75)) - 10)
        else:
            direction = 'LONG' if rsi_val < 50 else 'SHORT'
            setup_type = "파동 수렴 구간 기본 방향성 시나리오"

        # v7.2 실제 상위 타임프레임(4H/1D) 실시간 추세 조회 및 대추세/역추세 파동 분류
        htf_info = self._fetch_htf_regime(active_tf, current_price, df)
        htf_regime = htf_info['regime']
        htf_label = htf_info['htf_label']
        is_counter_trend = False

        if (direction == 'LONG' and htf_regime == 'BULL') or (direction == 'SHORT' and htf_regime == 'BEAR'):
            confidence += 8
            wave_role_tag = "🌊 대추세 순방향 파동"
            checks.append(f"🌊 **상위 {htf_label} 추세 순방향 일치 (+8점)**: {htf_info['summary']}")
        elif (direction == 'LONG' and htf_regime == 'BEAR') or (direction == 'SHORT' and htf_regime == 'BULL'):
            confidence -= 6
            is_counter_trend = True
            wave_role_tag = "⚡ 역추세 단기 반등/조정 파동"
            checks.append(f"⚡ **역추세 단기 반등/조정 파동 경고 (-6점)**: {htf_info['summary']} — 상위 추세와 반대 방향이므로 **비중 50% 축소 및 TP1에서 70% 우선 익절 후 본절(BE) 이동** 권장")
        else:
            wave_role_tag = "⚖️ 상위 추세 전환/수렴 파동"
            checks.append(f"⚖️ **상위 {htf_label} 전환/수렴 구간**: {htf_info['summary']}")

        fvg_adx = self._compute_adx_and_fvg(df, atr_val)
        adx_val = fvg_adx['adx']

        if ma_info:
            checks.append(f"📊 **이평선 & 모멘텀**: {ma_info.get('summary', '추세 정렬 확인')}")
            if ma_info.get('aligned'):
                confidence += 6

        if cycle_info:
            checks.append(f"⏳ **4년 반감기 사이클**: `{cycle_info['phase']}` — {cycle_info['note']}")
            if (direction == 'SHORT' and '조정' in cycle_info['phase']) or (direction == 'LONG' and '상승' in cycle_info['phase']):
                confidence += 4

        has_div = bool(div_info and div_info.get('type') != 'NONE')
        if has_div:
            checks.append(f"📈 **RSI 다이버전스 (+7점)**: `{div_info['type']}` 감지 (현재 RSI `{rsi_val:.1f}`)")
            confidence += 7
        else:
            checks.append(f"📉 **RSI 상태**: 현재 RSI `{rsi_val:.1f}`")

        if (direction == 'LONG' and fvg_adx['bull_fvg']) or (direction == 'SHORT' and fvg_adx['bear_fvg']):
            confidence += 5
            fvg_zone = fvg_adx['bull_fvg'] if direction == 'LONG' else fvg_adx['bear_fvg']
            checks.append(f"🧲 **미체결 FVG 불균형 매물대 (+5점)**: `${fvg_zone[0]:,.1f} ~ ${fvg_zone[1]:,.1f}`")

        has_sweep = False
        if okx_pos:
            cvd_t = okx_pos.get('cvd_div_type', 'NONE')
            if (direction == 'LONG' and cvd_t in ['BULLISH_ABSORPTION', 'BULLISH_CONFIRMED']) or \
               (direction == 'SHORT' and cvd_t in ['BEARISH_EXHAUSTION', 'BEARISH_CONFIRMED']):
                confidence += 7
                checks.append(f"✅ **OKX CVD 수급 일치 (+7점)**: {okx_pos.get('cvd_div_label')}")
            elif (direction == 'LONG' and cvd_t in ['BEARISH_EXHAUSTION', 'BEARISH_CONFIRMED']) or \
                 (direction == 'SHORT' and cvd_t in ['BULLISH_ABSORPTION', 'BULLISH_CONFIRMED']):
                confidence -= 6
                checks.append(f"⚠️ **OKX CVD 역방향 주의 (-6점)**: {okx_pos.get('cvd_div_label')}")

            sw_t = okx_pos.get('sweep_type', 'NONE')
            if (direction == 'LONG' and sw_t == 'SWEEP_LOW') or (direction == 'SHORT' and sw_t == 'SWEEP_HIGH'):
                has_sweep = True
                confidence += 7
                checks.append(f"✅ **유동성 스윕 트리거 포착 (+7점)**: {okx_pos.get('sweep_status')}")

        if adx_val < 18.0 and not (has_div or has_sweep):
            confidence -= 8
            checks.append(f"⏸️ **ADX 횡보장 주의 (-8점)**: 현재 ADX `{adx_val:.1f}` (추세 강도 약함)")
        else:
            checks.append(f"⚡ **ADX 추세 강도**: `{adx_val:.1f}`")

        if vol_info and vol_info.get('summary'):
            checks.append(f"🔊 **거래량 검증**: {vol_info['summary']}")

        if sr_info and sr_info.get('nearest_level'):
            checks.append(f"🧱 **주요 지지/저항 컨플루언스**: `${sr_info['nearest_level']:,.1f}` 인접")

        confidence = min(98, max(50, confidence))
        is_wait_zone = confidence < 75
        status_prefix = "⏸️ [관망 권고 · 75점 미만]" if is_wait_zone else "✅ [진입 유효 · 75점 이상]"
        display_setup = f"{status_prefix} {wave_role_tag} | {setup_type}"

        # 1. 차트 구조 기반 손절가(Structural SL) 산출
        structural_sl, sl_basis = self._find_structural_sl(
            direction=direction,
            current_price=current_price,
            atr_val=atr_val,
            max_sl_pct=max_sl_pct,
            df=df,
            pivots=pivots,
            dual_data=dual_data,
            okx_pos=okx_pos
        )

        # 2. 구조적 지지/저항 + FVG 기반 1·2차 분할 지정가 산출
        limit_info = self._find_smart_limit_entries(
            direction=direction,
            current_price=current_price,
            atr_val=atr_val,
            structural_sl=structural_sl,
            rsi_val=rsi_val,
            df=df,
            pivots=pivots,
            dual_data=dual_data,
            okx_pos=okx_pos,
            fvg_info=fvg_adx
        )
        entry_price = limit_info['avg_entry']
        entry_1 = limit_info['entry_1']
        entry_2 = limit_info['entry_2']

        if direction == 'LONG':
            stop_loss = min(structural_sl, entry_price - max(atr_val * 0.55, entry_price * 0.0025))
            stop_loss = max(entry_price * (1.0 - max_sl_pct), stop_loss)
            actual_risk = max(entry_price - stop_loss, 1e-8)
        else:
            stop_loss = max(structural_sl, entry_price + max(atr_val * 0.55, entry_price * 0.0025))
            stop_loss = min(entry_price * (1.0 + max_sl_pct), stop_loss)
            actual_risk = max(stop_loss - entry_price, 1e-8)

        # 3. 동적 목표가(TP1/TP2/TP3) 산출
        tp_info = self._find_structural_tps(
            direction=direction,
            entry_price=entry_price,
            actual_risk=actual_risk,
            atr_val=atr_val,
            rsi_val=rsi_val,
            df=df,
            pivots=pivots,
            dual_data=dual_data,
            okx_pos=okx_pos
        )
        tp1 = tp_info['tp1']
        tp2 = tp_info['tp2']
        tp3 = tp_info['tp3']
        rr_ratio = tp_info['rr2']
        sl_pct = (actual_risk / entry_price) * 100.0

        checks.insert(
            0,
            f"🎯 **스마트 분할 지정가 근거**: `{limit_info['basis']}` "
            f"(1차 `${entry_1:,.1f}` / 2차 `${entry_2:,.1f}` ➔ 평균 단가 `${entry_price:,.1f}`)"
        )
        checks.insert(1, f"🛡️ **구조적 손절(SL) 근거**: `${stop_loss:,.1f}` (`{sl_basis}`)")
        checks.insert(
            2,
            f"🏁 **구조적 목표가(TP) 근거**: "
            f"TP1 `${tp1:,.1f}` (`{tp_info['tp1_basis']}`, `{tp_info['rr1']:.2f}R`) | "
            f"TP2 `${tp2:,.1f}` (`{tp_info['tp2_basis']}`, `{rr_ratio:.2f}R`) | "
            f"TP3 `${tp3:,.1f}` (`{tp_info['tp3_basis']}`, `{tp_info['rr3']:.2f}R`)"
        )
        if is_wait_zone:
            checks.insert(
                0,
                f"⏸️ **v7.2 관망 권고 알림 (현재 {confidence}점 / 기준 75점)**: "
                f"횡보장(ADX {adx_val:.1f}) 또는 상위 추세 충돌 구간이므로 보수적 관망을 권장하며, 아래 지정가·포지션 수치는 조건 충족 시 참고용 타점입니다."
            )

        # 역추세 파동일 경우 리스크 비중 자동 50% 보수적 산출
        effective_risk_pct = risk_pct * 0.5 if is_counter_trend else risk_pct
        max_loss_usdt = capital_usdt * (effective_risk_pct / 100.0)
        position_size_btc = max_loss_usdt / actual_risk
        position_value_usdt = position_size_btc * entry_price
        raw_lev = int(max(1, min(20, (80.0 / max(sl_pct, 0.5)) * 0.5)))

        return {
            'signal': direction,
            'confidence': confidence,
            'setup_type': display_setup,
            'current_price': current_price,
            'entry_1': entry_1,
            'entry_2': entry_2,
            'entry_price': entry_price,
            'entry_delta_label': limit_info['delta_label'],
            'entry_basis': limit_info['basis'],
            'stop_loss': stop_loss,
            'sl_pct': sl_pct,
            'sl_basis': sl_basis,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'tp1_basis': tp_info['tp1_basis'],
            'tp2_basis': tp_info['tp2_basis'],
            'tp3_basis': tp_info['tp3_basis'],
            'rr1': tp_info['rr1'],
            'rr_ratio': rr_ratio,
            'rr3': tp_info['rr3'],
            'max_loss_usdt': max_loss_usdt,
            'position_size_btc': position_size_btc,
            'position_value_usdt': position_value_usdt,
            'safe_leverage': raw_lev,
            'checks': checks
        }


class SignalJournalManager:
    """
    로컬 CSV 기반 시그널 기록기 & OKX 실시간 시세 대조 동적 R:R 자동 채점 시뮬레이터
    """

    COLUMNS = [
        'id', 'recorded_at', 'symbol', 'timeframe', 'direction', 'setup_type',
        'confidence', 'risk_pct', 'ref_price', 'entry_1', 'entry_2', 'avg_entry',
        'stop_loss', 'tp1', 'tp2', 'status', 'fill_stage', 'realized_r',
        'account_pnl_pct', 'evaluated_at'
    ]

    def __init__(self, csv_path: str = "signal_journal.csv"):
        self.csv_path = csv_path

    def load_journal(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            return pd.DataFrame(columns=self.COLUMNS)
        try:
            df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
            for col in self.COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            return df[self.COLUMNS]
        except Exception:
            return pd.DataFrame(columns=self.COLUMNS)

    def save_journal(self, df: pd.DataFrame):
        df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')

    def record_signal(self, symbol: str, timeframe: str, trade_plan: dict, risk_pct: float, recorded_time_str: str) -> dict:
        df = self.load_journal()
        new_row = {
            'id': uuid.uuid4().hex[:6].upper(),
            'recorded_at': recorded_time_str,
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': trade_plan['signal'],
            'setup_type': trade_plan['setup_type'],
            'confidence': int(trade_plan['confidence']),
            'risk_pct': float(risk_pct),
            'ref_price': round(float(trade_plan['current_price']), 2),
            'entry_1': round(float(trade_plan['entry_1']), 2),
            'entry_2': round(float(trade_plan['entry_2']), 2),
            'avg_entry': round(float(trade_plan['entry_price']), 2),
            'stop_loss': round(float(trade_plan['stop_loss']), 2),
            'tp1': round(float(trade_plan['tp1']), 2),
            'tp2': round(float(trade_plan['tp2']), 2),
            'status': '미체결 대기',
            'fill_stage': '0% (대기)',
            'realized_r': 0.0,
            'account_pnl_pct': 0.0,
            'evaluated_at': recorded_time_str
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        self.save_journal(df)
        return new_row

    def evaluate_all_with_okx(self) -> tuple:
        df = self.load_journal()
        if df.empty:
            return df, {'total': 0}

        ex = ccxt.okx({'enableRateLimit': True, 'timeout': 6000, 'options': {'defaultType': 'swap'}})
        now_kst = (pd.Timestamp.utcnow() + pd.Timedelta(hours=9)).strftime('%Y-%m-%d %H:%M')
        candle_cache = {}

        for idx, row in df.iterrows():
            status_now = str(row['status'])
            if status_now in ['익절 성공 (TP2)', '익절 후 본절 (TP1+BE)', '손절 (SL)']:
                continue

            sym = str(row['symbol'])
            tf = str(row['timeframe'])
            rec_dt = pd.to_datetime(row['recorded_at'])
            since_ms = int((rec_dt - pd.Timedelta(hours=9)).timestamp() * 1000)

            cache_key = (sym, tf, since_ms)
            if cache_key not in candle_cache:
                try:
                    raw = ex.fetch_ohlcv(sym, timeframe=tf, since=since_ms, limit=300)
                    c_df = pd.DataFrame(raw, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                    c_df['dt'] = pd.to_datetime(c_df['ts'], unit='ms') + pd.Timedelta(hours=9)
                    candle_cache[cache_key] = c_df
                except Exception:
                    candle_cache[cache_key] = pd.DataFrame()

            c_df = candle_cache[cache_key]
            if c_df.empty:
                continue

            sub_c = c_df[c_df['dt'] >= rec_dt].reset_index(drop=True)
            if sub_c.empty:
                continue

            direction = str(row['direction'])
            e1 = float(row['entry_1'])
            e2 = float(row['entry_2'])
            avg_e = float(row['avg_entry'])
            sl = float(row['stop_loss'])
            tp1 = float(row['tp1'])
            tp2 = float(row['tp2'])
            r_pct = float(row['risk_pct']) if pd.notna(row['risk_pct']) else 2.0

            filled_1 = False
            filled_2 = False
            hit_tp1 = False
            final_status = '미체결 대기'
            realized_r = 0.0

            for _, bar in sub_c.iterrows():
                hi = float(bar['high'])
                lo = float(bar['low'])

                if direction == 'LONG':
                    if not filled_1 and lo <= e1:
                        filled_1 = True
                    if not filled_2 and lo <= e2:
                        filled_2 = True

                    if filled_1:
                        active_ep = avg_e if filled_2 else e1
                        active_sl = active_ep if hit_tp1 else sl
                        fill_weight = 1.0 if filled_2 else 0.5
                        risk_unit = max(abs(active_ep - sl), 1e-8)
                        rr1 = abs(tp1 - active_ep) / risk_unit
                        rr2 = abs(tp2 - active_ep) / risk_unit

                        if lo <= active_sl:
                            if hit_tp1:
                                final_status = '익절 후 본절 (TP1+BE)'
                                realized_r = round((0.5 * rr1) * fill_weight, 2)
                            else:
                                final_status = '손절 (SL)'
                                realized_r = round(-1.00 * fill_weight, 2)
                            break

                        if not hit_tp1 and hi >= tp1:
                            hit_tp1 = True

                        if hi >= tp2:
                            final_status = '익절 성공 (TP2)'
                            realized_r = round((0.5 * rr1 + 0.5 * rr2) * fill_weight, 2)
                            break
                else:
                    if not filled_1 and hi >= e1:
                        filled_1 = True
                    if not filled_2 and hi >= e2:
                        filled_2 = True

                    if filled_1:
                        active_ep = avg_e if filled_2 else e1
                        active_sl = active_ep if hit_tp1 else sl
                        fill_weight = 1.0 if filled_2 else 0.5
                        risk_unit = max(abs(sl - active_ep), 1e-8)
                        rr1 = abs(active_ep - tp1) / risk_unit
                        rr2 = abs(active_ep - tp2) / risk_unit

                        if hi >= active_sl:
                            if hit_tp1:
                                final_status = '익절 후 본절 (TP1+BE)'
                                realized_r = round((0.5 * rr1) * fill_weight, 2)
                            else:
                                final_status = '손절 (SL)'
                                realized_r = round(-1.00 * fill_weight, 2)
                            break

                        if not hit_tp1 and lo <= tp1:
                            hit_tp1 = True

                        if lo <= tp2:
                            final_status = '익절 성공 (TP2)'
                            realized_r = round((0.5 * rr1 + 0.5 * rr2) * fill_weight, 2)
                            break

            if final_status == '미체결 대기' and filled_1:
                last_close = float(sub_c['close'].iloc[-1])
                active_ep = avg_e if filled_2 else e1
                fill_weight = 1.0 if filled_2 else 0.5
                risk_unit = max(abs(active_ep - sl), 1e-8)
                rr1 = abs(tp1 - active_ep) / risk_unit

                if direction == 'LONG':
                    open_r = (last_close - active_ep) / risk_unit
                else:
                    open_r = (active_ep - last_close) / risk_unit

                if hit_tp1:
                    realized_r = round((0.5 * rr1 + 0.5 * open_r) * fill_weight, 2)
                    final_status = '진행중 (TP1 확보)'
                else:
                    realized_r = round(open_r * fill_weight, 2)
                    final_status = '진행중 (1·2차 체결)' if filled_2 else '진행중 (1차 체결)'

            fill_txt = '100% (1·2차)' if filled_2 else ('50% (1차)' if filled_1 else '0% (대기)')
            df.at[idx, 'status'] = final_status
            df.at[idx, 'fill_stage'] = fill_txt
            df.at[idx, 'realized_r'] = realized_r
            df.at[idx, 'account_pnl_pct'] = round(realized_r * r_pct, 2)
            df.at[idx, 'evaluated_at'] = now_kst

        self.save_journal(df)
        return df, self.summarize_stats(df)

    def summarize_stats(self, df: pd.DataFrame) -> dict:
        if df is None or df.empty:
            return {'total': 0, 'filled': 0, 'closed': 0, 'win_rate': 0.0, 'avg_r': 0.0, 'total_pnl': 0.0}

        filled_df = df[df['status'] != '미체결 대기']
        closed_df = df[df['status'].isin(['익절 성공 (TP2)', '익절 후 본절 (TP1+BE)', '손절 (SL)'])]
        wins = closed_df[closed_df['realized_r'] > 0]

        win_rate = (len(wins) / len(closed_df) * 100.0) if len(closed_df) > 0 else 0.0
        avg_r = float(filled_df['realized_r'].mean()) if len(filled_df) > 0 else 0.0
        total_pnl = float(filled_df['account_pnl_pct'].sum()) if len(filled_df) > 0 else 0.0

        return {
            'total': len(df),
            'filled': len(filled_df),
            'closed': len(closed_df),
            'win_rate': round(win_rate, 1),
            'avg_r': round(avg_r, 2),
            'total_pnl': round(total_pnl, 2)
        }

    def clear_journal(self):
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)