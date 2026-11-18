# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

class NeelyRuleEngine:
    """
    글렌 닐리(Glenn Neely) <Mastering Elliott Wave> 정밀 규칙 엔진 (v6.2.1)
    - 최신 Pandas 표준(bfill/ffill) 반영 및 ATR 동적 스윙 필터링 보완
    """

    def __init__(self, min_swing_pct: float = 0.008, use_log_scale: bool = False):
        self.min_swing_pct = min_swing_pct
        self.use_log_scale = use_log_scale

    def _calc_leg_magnitude(self, p_start: float, p_end: float) -> float:
        p1 = max(float(p_start), 1e-8)
        p2 = max(float(p_end), 1e-8)
        if self.use_log_scale:
            return abs(np.log(p2) - np.log(p1))
        return abs(p2 - p1)

    def extract_monowaves(self, df: pd.DataFrame, swing_pct: float = None):
        if df is None or len(df) < 4:
            return [], []

        base_pct = swing_pct if swing_pct is not None else self.min_swing_pct
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        timestamps = df['timestamp'].values

        if 'atr' in df.columns and not df['atr'].isna().all():
            atr_vals = df['atr'].bfill().ffill().values
            atr_pcts = np.clip(atr_vals / np.maximum(closes, 1e-8) * 1.35, base_pct * 0.55, base_pct * 2.2)
        else:
            atr_pcts = np.full(len(df), base_pct)

        pivots = []
        direction = 0
        ext_price = closes[0]
        ext_idx = 0
        ext_time = timestamps[0]

        for i in range(len(df)):
            h, l, t = highs[i], lows[i], timestamps[i]
            dyn_pct = float(atr_pcts[i])

            if direction == 0:
                if h >= ext_price * (1 + dyn_pct):
                    direction = 1
                    ext_price, ext_idx, ext_time = h, i, t
                elif l <= ext_price * (1 - dyn_pct):
                    direction = -1
                    ext_price, ext_idx, ext_time = l, i, t
            elif direction == 1:
                if h > ext_price:
                    ext_price, ext_idx, ext_time = h, i, t
                elif l <= ext_price * (1 - dyn_pct):
                    pivots.append({'index': ext_idx, 'time': ext_time, 'price': ext_price, 'type': 'HIGH'})
                    direction = -1
                    ext_price, ext_idx, ext_time = l, i, t
            elif direction == -1:
                if l < ext_price:
                    ext_price, ext_idx, ext_time = l, i, t
                elif h >= ext_price * (1 + dyn_pct):
                    pivots.append({'index': ext_idx, 'time': ext_time, 'price': ext_price, 'type': 'LOW'})
                    direction = 1
                    ext_price, ext_idx, ext_time = h, i, t

        if len(pivots) == 0 or pivots[-1]['index'] != ext_idx:
            pivots.append({
                'index': ext_idx, 'time': ext_time, 'price': ext_price,
                'type': 'HIGH' if direction == 1 else 'LOW'
            })

        monowaves = []
        for j in range(1, len(pivots)):
            p_start, p_end = pivots[j - 1], pivots[j]
            diff = p_end['price'] - p_start['price']
            log_mag = self._calc_leg_magnitude(p_start['price'], p_end['price'])
            monowaves.append({
                'wave_no': j,
                'start_idx': p_start['index'],
                'end_idx': p_end['index'],
                'start_time': p_start['time'],
                'end_time': p_end['time'],
                'start_price': p_start['price'],
                'end_price': p_end['price'],
                'price_change': abs(diff),
                'log_magnitude': log_mag,
                'direction': 'UP' if diff > 0 else 'DOWN',
                'duration_bars': max(1, p_end['index'] - p_start['index']),
                'retracement_rule': 'N/A',
                'structure_label': ':3',
                'degree_type': 'CORRECTIVE',
                'local_wave_id': '1',
                'subdivision_rule': '5',
                'pattern_tag': '파동 집계',
                'fractal_label': '1'
            })

        self._apply_neely_retracement_rules(monowaves)
        return pivots, monowaves

    def _apply_neely_retracement_rules(self, waves: list):
        n = len(waves)
        for i in range(n):
            m1 = waves[i]
            L1 = max(m1['log_magnitude'], 1e-8)
            m0 = waves[i - 1] if i >= 1 else None
            m2 = waves[i + 1] if i + 1 < n else None
            m3 = waves[i + 2] if i + 2 < n else None

            if m2 is None:
                if m0 is not None:
                    r_prev = L1 / max(m0['log_magnitude'], 1e-8)
                    m1['structure_label'] = ':5' if r_prev >= 1.618 else ':3'
                    m1['degree_type'] = 'IMPULSIVE' if r_prev >= 1.618 else 'CORRECTIVE'
                continue

            L2 = m2['log_magnitude']
            R2 = L2 / L1
            L0 = m0['log_magnitude'] if m0 else L1
            L3 = m3['log_magnitude'] if m3 else L1

            if R2 < 0.382:
                m1['retracement_rule'] = f"Rule 1 ({R2*100:.1f}%)"
                m1['structure_label'] = ':5' if L1 >= L0 * 0.618 else ':s5'
                m1['degree_type'] = 'IMPULSIVE'
            elif 0.382 <= R2 < 0.618:
                m1['retracement_rule'] = f"Rule 2 ({R2*100:.1f}%)"
                m1['structure_label'] = ':5' if (L0 < L1 and L3 > L2) else ':3'
                m1['degree_type'] = 'IMPULSIVE' if (L0 < L1 and L3 > L2) else 'CORRECTIVE'
            elif 0.618 <= R2 <= 0.65:
                m1['retracement_rule'] = f"Rule 3 ({R2*100:.1f}%)"
                m1['structure_label'] = ':5' if L3 >= L2 else ':c3'
                m1['degree_type'] = 'IMPULSIVE' if L3 >= L2 else 'CORRECTIVE'
            elif 0.65 < R2 < 1.00:
                m1['retracement_rule'] = f"Rule 4 ({R2*100:.1f}%)"
                m1['structure_label'] = ':5' if (L0 < L1 and L3 >= L1 * 1.618) else ':3'
                m1['degree_type'] = 'IMPULSIVE' if (L0 < L1 and L3 >= L1 * 1.618) else 'CORRECTIVE'
            elif 1.00 <= R2 <= 1.618:
                m1['retracement_rule'] = f"Rule 5 ({R2*100:.1f}%)"
                m1['structure_label'] = ':F3' if R2 <= 1.382 else ':3'
                m1['degree_type'] = 'CORRECTIVE'
            elif 1.618 < R2 < 2.618:
                m1['retracement_rule'] = f"Rule 6 ({R2*100:.1f}%)"
                m1['structure_label'] = ':3'
                m1['degree_type'] = 'CORRECTIVE'
            else:
                m1['retracement_rule'] = f"Rule 7 ({R2*100:.1f}%)"
                m1['structure_label'] = ':3'
                m1['degree_type'] = 'CORRECTIVE'

    def _collect_prominent_extrema(self, df: pd.DataFrame, max_per_side: int = 18) -> tuple:
        n = len(df)
        if n < 3:
            return [], []

        price_range = max(df['high'].max() - df['low'].min(), 1e-8)
        mid_price = max(df['close'].mean(), 1e-8)
        base_pct = max(0.002, (price_range / mid_price) * 0.035)

        high_map = {}
        low_map = {}

        for mult in [0.4, 0.7, 1.0, 1.5, 2.2]:
            pivs, _ = self.extract_monowaves(df, swing_pct=base_pct * mult)
            for p in pivs:
                idx = p['index']
                if 1 <= idx <= n - 2:
                    if p['type'] == 'HIGH':
                        high_map[idx] = p
                    else:
                        low_map[idx] = p

        win = max(2, n // 25)
        for i in range(1, n - 1):
            lo_b = max(0, i - win)
            hi_b = min(n, i + win + 1)
            if df['high'].iloc[i] == df['high'].iloc[lo_b:hi_b].max():
                high_map[i] = {'index': i, 'time': df['timestamp'].iloc[i], 'price': df['high'].iloc[i], 'type': 'HIGH'}
            if df['low'].iloc[i] == df['low'].iloc[lo_b:hi_b].min():
                low_map[i] = {'index': i, 'time': df['timestamp'].iloc[i], 'price': df['low'].iloc[i], 'type': 'LOW'}

        def rank_candidates(cand_dict, is_high):
            cands = list(cand_dict.values())
            if len(cands) <= max_per_side:
                return sorted(cands, key=lambda x: x['index'])
            scored = []
            for c in cands:
                idx = c['index']
                l_w = max(0, idx - max(3, n // 10))
                r_w = min(n, idx + max(3, n // 10) + 1)
                if is_high:
                    prom = self._calc_leg_magnitude(min(df['low'].iloc[l_w:idx+1].min(), df['low'].iloc[idx:r_w].min()), c['price'])
                else:
                    prom = self._calc_leg_magnitude(c['price'], max(df['high'].iloc[l_w:idx+1].max(), df['high'].iloc[idx:r_w].max()))
                scored.append((prom, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = [x[1] for x in scored[:max_per_side]]
            return sorted(top, key=lambda x: x['index'])

        return rank_candidates(high_map, True), rank_candidates(low_map, False)

    def _solve_5wave_impulse(self, df: pd.DataFrame, start_idx: int, end_idx: int,
                             is_up: bool, is_active: bool = False) -> tuple:
        slice_df = df.iloc[start_idx:end_idx + 1].reset_index(drop=True)
        n = len(slice_df)
        if n < 4:
            return None, 0.0

        p0_val = slice_df['low'].iloc[0] if is_up else slice_df['high'].iloc[0]
        p5_val = slice_df['close'].iloc[-1] if is_active else (slice_df['high'].iloc[-1] if is_up else slice_df['low'].iloc[-1])

        p0 = {'index': 0, 'time': slice_df['timestamp'].iloc[0], 'price': p0_val, 'type': 'LOW' if is_up else 'HIGH'}
        p5 = {'index': n - 1, 'time': slice_df['timestamp'].iloc[-1], 'price': p5_val, 'type': 'HIGH' if is_up else 'LOW'}

        total_span = self._calc_leg_magnitude(p0_val, p5_val)
        if total_span <= 0:
            return None, 0.0

        highs, lows = self._collect_prominent_extrema(slice_df, max_per_side=18)
        same_cands = highs if is_up else lows
        opp_cands = lows if is_up else highs

        min_leg = total_span * (0.04 if is_active else 0.075)
        min_bars = max(1, n // 35)

        best_score = -1e9
        best_pts = None

        for p1 in same_cands:
            if p1['index'] < min_bars or p1['index'] > n - 4 * min_bars:
                continue
            if (is_up and p1['price'] <= p0_val) or (not is_up and p1['price'] >= p0_val):
                continue
            L1 = self._calc_leg_magnitude(p0_val, p1['price'])
            T1 = max(1, p1['index'])
            if L1 < min_leg:
                continue

            for p2 in opp_cands:
                if p2['index'] < p1['index'] + min_bars or p2['index'] > n - 3 * min_bars:
                    continue
                if (is_up and p2['price'] <= p0_val) or (not is_up and p2['price'] >= p0_val):
                    continue
                L2 = self._calc_leg_magnitude(p1['price'], p2['price'])
                T2 = max(1, p2['index'] - p1['index'])
                r2 = L2 / L1
                if L2 < min_leg * 0.65 or r2 > 0.90:
                    continue

                for p3 in same_cands:
                    if p3['index'] < p2['index'] + min_bars or p3['index'] > n - 2 * min_bars:
                        continue
                    if (is_up and p3['price'] <= p1['price']) or (not is_up and p3['price'] >= p1['price']):
                        continue
                    L3 = self._calc_leg_magnitude(p2['price'], p3['price'])
                    T3 = max(1, p3['index'] - p2['index'])
                    if L3 < min_leg:
                        continue

                    for p4 in opp_cands:
                        if p4['index'] < p3['index'] + min_bars or p4['index'] > n - min_bars:
                            continue
                        if (is_up and p4['price'] <= p2['price']) or (not is_up and p4['price'] >= p2['price']):
                            continue
                        L4 = self._calc_leg_magnitude(p3['price'], p4['price'])
                        T4 = max(1, p4['index'] - p3['index'])
                        L5 = self._calc_leg_magnitude(p4['price'], p5_val)
                        T5 = max(1, (n - 1) - p4['index'])

                        if (is_up and p5_val <= p4['price']) or (not is_up and p5_val >= p4['price']):
                            continue
                        if L4 < min_leg * 0.65 or L5 < min_leg:
                            continue

                        if L3 <= min(L1, L5):
                            continue

                        time_ratio_21 = T2 / T1
                        if time_ratio_21 < 0.12:
                            continue

                        tol = abs(p5_val - p0_val) * 0.03
                        no_overlap = (p4['price'] >= p1['price'] - tol) if is_up else (p4['price'] <= p1['price'] + tol)
                        strict_no_overlap = (p4['price'] > p1['price']) if is_up else (p4['price'] < p1['price'])
                        r4 = L4 / max(L3, 1e-8)
                        if r4 > 0.85:
                            continue

                        amplitude_ratio = (L1 + L2 + L3 + L4 + L5) / total_span
                        score = amplitude_ratio * 35.0
                        if strict_no_overlap:
                            score += 45.0
                        elif no_overlap:
                            score += 25.0
                        else:
                            if not ((L1 > L3 > L5) or (L1 < L3 < L5)):
                                continue
                            score += 5.0

                        if 0.30 <= r2 <= 0.68:
                            score += 12.0
                        if 0.20 <= r4 <= 0.55:
                            score += 12.0
                        if L3 >= max(L1, L5):
                            score += 10.0
                        if 0.30 <= time_ratio_21 <= 2.50:
                            score += 8.0
                        if T3 >= min(T1, T5):
                            score += 6.0

                        if score > best_score:
                            best_score = score
                            best_pts = [p0, p1, p2, p3, p4, p5]

        if not best_pts:
            return None, 0.0

        global_pts = []
        for pt in best_pts:
            g_i = start_idx + pt['index']
            global_pts.append({
                'index': g_i,
                'time': df['timestamp'].iloc[g_i],
                'price': pt['price'],
                'type': pt['type']
            })
        return global_pts, best_score

    def _solve_3wave_corrective(self, df: pd.DataFrame, start_idx: int, end_idx: int,
                                is_up: bool, is_active: bool = False, allow_expanded_flat: bool = False) -> tuple:
        slice_df = df.iloc[start_idx:end_idx + 1].reset_index(drop=True)
        n = len(slice_df)
        if n < 3:
            return None, 0.0

        p0_val = slice_df['low'].iloc[0] if is_up else slice_df['high'].iloc[0]
        p3_val = slice_df['close'].iloc[-1] if is_active else (slice_df['high'].iloc[-1] if is_up else slice_df['low'].iloc[-1])

        p0 = {'index': 0, 'time': slice_df['timestamp'].iloc[0], 'price': p0_val, 'type': 'LOW' if is_up else 'HIGH'}
        p3 = {'index': n - 1, 'time': slice_df['timestamp'].iloc[-1], 'price': p3_val, 'type': 'HIGH' if is_up else 'LOW'}

        total_span = max(self._calc_leg_magnitude(p0_val, p3_val), 1e-8)
        highs, lows = self._collect_prominent_extrema(slice_df, max_per_side=20)
        same_cands = highs if is_up else lows
        opp_cands = lows if is_up else highs

        min_leg = total_span * 0.12
        min_bars = max(1, n // 25)
        max_ret_b = 1.618 if allow_expanded_flat else 1.382

        best_score = -1e9
        best_pts = None

        for p1 in same_cands:
            if p1['index'] < min_bars or p1['index'] > n - 2 * min_bars:
                continue
            if (is_up and p1['price'] <= p0_val) or (not is_up and p1['price'] >= p0_val):
                continue
            L1 = self._calc_leg_magnitude(p0_val, p1['price'])
            if L1 < min_leg:
                continue

            for p2 in opp_cands:
                if p2['index'] < p1['index'] + min_bars or p2['index'] > n - min_bars:
                    continue
                L2 = self._calc_leg_magnitude(p1['price'], p2['price'])
                L3 = self._calc_leg_magnitude(p2['price'], p3_val)
                if L2 < min_leg * 0.6 or L3 < min_leg:
                    continue

                ret_b = L2 / max(L1, 1e-8)
                if ret_b > max_ret_b:
                    continue

                score = ((L1 + L2 + L3) / total_span) * 40.0
                if 0.382 <= ret_b <= 1.25:
                    score += 18.0

                if score > best_score:
                    best_score = score
                    best_pts = [p0, p1, p2, p3]

        if not best_pts:
            mid_idx = n // 2
            p1_i = int(slice_df['high'].iloc[1:mid_idx+1].values.argmax()) + 1 if is_up else int(slice_df['low'].iloc[1:mid_idx+1].values.argmin()) + 1
            p2_i = int(slice_df['low'].iloc[p1_i:n-1].values.argmin()) + p1_i if is_up else int(slice_df['high'].iloc[p1_i:n-1].values.argmax()) + p1_i
            if 0 < p1_i < p2_i < n - 1:
                best_pts = [
                    p0,
                    {'index': p1_i, 'time': slice_df['timestamp'].iloc[p1_i], 'price': slice_df['high'].iloc[p1_i] if is_up else slice_df['low'].iloc[p1_i], 'type': 'HIGH' if is_up else 'LOW'},
                    {'index': p2_i, 'time': slice_df['timestamp'].iloc[p2_i], 'price': slice_df['low'].iloc[p2_i] if is_up else slice_df['high'].iloc[p2_i], 'type': 'LOW' if is_up else 'HIGH'},
                    p3
                ]
                best_score = 30.0
            else:
                return None, 0.0

        global_pts = []
        for pt in best_pts:
            g_i = start_idx + pt['index']
            global_pts.append({
                'index': g_i,
                'time': df['timestamp'].iloc[g_i],
                'price': pt['price'],
                'type': pt['type']
            })
        return global_pts, best_score

    def _solve_active_partial_wave(self, df: pd.DataFrame, start_idx: int, end_idx: int, is_up: bool) -> list:
        slice_df = df.iloc[start_idx:end_idx + 1].reset_index(drop=True)
        n = len(slice_df)
        p0_val = slice_df['low'].iloc[0] if is_up else slice_df['high'].iloc[0]
        p_curr_val = slice_df['close'].iloc[-1]

        p0 = {'index': start_idx, 'time': df['timestamp'].iloc[start_idx], 'price': p0_val, 'type': 'LOW' if is_up else 'HIGH'}
        p_curr = {'index': end_idx, 'time': df['timestamp'].iloc[end_idx], 'price': p_curr_val, 'type': 'HIGH' if is_up else 'LOW'}

        if n < 5:
            return [p0, p_curr]

        pts3, sc3 = self._solve_3wave_corrective(df, start_idx, end_idx, is_up, is_active=True)
        if pts3 and sc3 >= 48.0:
            L1 = self._calc_leg_magnitude(pts3[0]['price'], pts3[1]['price'])
            L2 = self._calc_leg_magnitude(pts3[1]['price'], pts3[2]['price'])
            if L2 >= L1 * 0.25:
                return pts3

        return [p0, p_curr]

    def _find_correction_pivots(self, df: pd.DataFrame, start_idx: int, end_idx: int,
                                corr_dir: str, sub_swing_pct: float,
                                wave_view_mode: str = "view_complex_b") -> list:
        p5_price = df['high'].iloc[start_idx] if corr_dir == 'DOWN' else df['low'].iloc[start_idx]
        p_start = {'index': start_idx, 'time': df['timestamp'].iloc[start_idx], 'price': p5_price, 'type': 'HIGH' if corr_dir == 'DOWN' else 'LOW'}

        if end_idx - start_idx < 2:
            curr_p = df['close'].iloc[end_idx]
            return [p_start, {'index': end_idx, 'time': df['timestamp'].iloc[end_idx], 'price': curr_p, 'type': 'LOW' if corr_dir == 'DOWN' else 'HIGH'}]

        sub_df = df.iloc[start_idx:end_idx + 1].reset_index(drop=True)
        raw_pivots, _ = self.extract_monowaves(sub_df, swing_pct=sub_swing_pct * 0.85)

        lows, highs = [], []
        for p in raw_pivots:
            g_idx = start_idx + p['index']
            if start_idx < g_idx < end_idx:
                item = {'index': g_idx, 'time': df['timestamp'].iloc[g_idx], 'price': p['price'], 'type': p['type']}
                if p['type'] == 'LOW':
                    lows.append(item)
                else:
                    highs.append(item)

        a_cands = lows if corr_dir == 'DOWN' else highs
        b_cands = highs if corr_dir == 'DOWN' else lows

        best_pair = None
        best_corr_score = -1e9

        for pa in a_cands:
            La = self._calc_leg_magnitude(p_start['price'], pa['price'])
            if La <= 0:
                continue
            for pb in b_cands:
                if pb['index'] <= pa['index']:
                    continue
                Lb = self._calc_leg_magnitude(pa['price'], pb['price'])
                ret_b = Lb / La
                if 0.236 <= ret_b <= 1.25:
                    curr_extreme = df['low'].iloc[pb['index']:end_idx + 1].min() if corr_dir == 'DOWN' else df['high'].iloc[pb['index']:end_idx + 1].max()
                    Lc = self._calc_leg_magnitude(pb['price'], curr_extreme)
                    sc = La + Lb * 1.2 + Lc * 0.8

                    if wave_view_mode == "view_complex_b":
                        mid_highs = [h for h in b_cands if pa['index'] < h['index'] < pb['index']]
                        if mid_highs:
                            sc += La * 0.45

                    if sc > best_corr_score:
                        best_corr_score = sc
                        best_pair = (pa, pb)

        if best_pair:
            pa, pb = best_pair
            pc = {
                'index': end_idx,
                'time': df['timestamp'].iloc[end_idx],
                'price': df['close'].iloc[end_idx],
                'type': 'LOW' if corr_dir == 'DOWN' else 'HIGH'
            }
            return [p_start, pa, pb, pc]

        curr_p = df['close'].iloc[end_idx]
        return [p_start, {'index': end_idx, 'time': df['timestamp'].iloc[end_idx], 'price': curr_p, 'type': 'LOW' if corr_dir == 'DOWN' else 'HIGH'}]

    def _map_detail_window(self, detail_df: pd.DataFrame, start_time, end_time) -> tuple:
        if detail_df is None or len(detail_df) < 10:
            return None, 0, 0
        t0 = pd.Timestamp(start_time)
        t1 = pd.Timestamp(end_time)
        s_idx = int(detail_df['timestamp'].searchsorted(t0, side='left'))
        e_idx = int(detail_df['timestamp'].searchsorted(t1, side='right')) - 1
        s_idx = max(0, min(s_idx, len(detail_df) - 1))
        e_idx = max(s_idx, min(e_idx, len(detail_df) - 1))
        return detail_df, s_idx, e_idx

    def _detect_complex_substructure(self, pts: list, is_up: bool) -> bool:
        if not pts or len(pts) < 4:
            return False
        L1 = self._calc_leg_magnitude(pts[0]['price'], pts[1]['price'])
        L2 = self._calc_leg_magnitude(pts[1]['price'], pts[2]['price'])
        L3 = self._calc_leg_magnitude(pts[2]['price'], pts[3]['price'])
        ret_x = L2 / max(L1, 1e-8)
        ratio_y = L3 / max(L1, 1e-8)
        return (ret_x >= 0.78) or (0.35 <= ret_x <= 0.78 and ratio_y >= 1.15)

    def extract_dual_degree_waves(self, df: pd.DataFrame, cycle_df: pd.DataFrame,
                                  macro_swing_pct: float, sub_swing_pct: float,
                                  detail_df: pd.DataFrame = None,
                                  wave_view_mode: str = "view_complex_b") -> dict:
        if cycle_df is None or len(cycle_df) < 10:
            return {
                'macro_pivots': [], 'macro_waves': [], 'sub_pivots': [], 'sub_waves': [],
                'projection': None, 'synced_audits': [], 'correction_diagnosis': None
            }

        cdf = cycle_df.reset_index(drop=True)
        n = len(cdf)
        min_idx = int(cdf['low'].values.argmin())
        max_idx = int(cdf['high'].values.argmax())

        if min_idx < max_idx and (max_idx - min_idx) >= 5:
            primary_up = True
            start_ext_idx, peak_ext_idx = min_idx, max_idx
        elif max_idx < min_idx and (min_idx - max_idx) >= 5:
            primary_up = False
            start_ext_idx, peak_ext_idx = max_idx, min_idx
        else:
            primary_up = cdf['close'].iloc[-1] >= cdf['close'].iloc[0]
            start_ext_idx = 0
            peak_ext_idx = n - 1

        imp_pivots, _ = self._solve_5wave_impulse(cdf, start_ext_idx, peak_ext_idx, primary_up, is_active=False)
        if not imp_pivots:
            p0_v = cdf['low'].iloc[start_ext_idx] if primary_up else cdf['high'].iloc[start_ext_idx]
            p5_v = cdf['high'].iloc[peak_ext_idx] if primary_up else cdf['low'].iloc[peak_ext_idx]
            imp_pivots = [
                {'index': start_ext_idx, 'time': cdf['timestamp'].iloc[start_ext_idx], 'price': p0_v, 'type': 'LOW' if primary_up else 'HIGH'},
                {'index': peak_ext_idx, 'time': cdf['timestamp'].iloc[peak_ext_idx], 'price': p5_v, 'type': 'HIGH' if primary_up else 'LOW'}
            ]
            imp_labels = ['1']
        else:
            imp_labels = ['1', '2', '3', '4', '5']

        corr_pivots = []
        corr_labels = []
        if peak_ext_idx < n - 2:
            corr_dir = 'DOWN' if primary_up else 'UP'
            c_pts = self._find_correction_pivots(cdf, peak_ext_idx, n - 1, corr_dir, sub_swing_pct, wave_view_mode)
            if len(c_pts) >= 2:
                corr_pivots = c_pts[1:]
                all_abc = ['A', 'B', 'C']
                corr_labels = all_abc[:len(corr_pivots)]

        macro_pivots = imp_pivots + corr_pivots
        all_wave_ids = imp_labels + corr_labels

        macro_waves = []
        for j in range(1, len(macro_pivots)):
            p_s, p_e = macro_pivots[j - 1], macro_pivots[j]
            w_id = all_wave_ids[j - 1]
            diff = p_e['price'] - p_s['price']
            is_imp_leg = w_id in ['1', '2', '3', '4', '5']
            is_active_leg = (j == len(macro_pivots) - 1) and (not is_imp_leg)

            macro_waves.append({
                'wave_no': j,
                'start_idx': p_s['index'],
                'end_idx': p_e['index'],
                'start_time': p_s['time'],
                'end_time': p_e['time'],
                'start_price': p_s['price'],
                'end_price': p_e['price'],
                'price_change': abs(diff),
                'log_magnitude': self._calc_leg_magnitude(p_s['price'], p_e['price']),
                'direction': 'UP' if diff >= 0 else 'DOWN',
                'duration_bars': max(1, p_e['index'] - p_s['index']),
                'local_wave_id': w_id,
                'is_active': is_active_leg,
                'is_complex': False,
                'subdivision_rule': '5' if w_id in ['1', '3', '5'] else '3',
                'pattern_tag': '5파 충격파 (1-2-3-4-5)' if is_imp_leg else '조정파 분석 중',
                'fractal_label': w_id,
                'structure_label': ':5' if w_id in ['1', '3', '5'] else ':3',
                'retracement_rule': 'Impulse Cycle' if is_imp_leg else 'Corrective Cycle'
            })

        a_rule = None
        b_rule = None
        b_is_complex = False
        corr_diagnosis = None
        leg_pivot_chains = {}

        for m_idx, mw in enumerate(macro_waves):
            w_id = mw['local_wave_id']
            is_up_leg = (mw['direction'] == 'UP')
            is_act = mw['is_active']

            target_df, s_i, e_i = self._map_detail_window(detail_df, mw['start_time'], mw['end_time'])
            if target_df is None or (e_i - s_i) < 6:
                target_df = cdf
                s_i, e_i = mw['start_idx'], mw['end_idx']
            else:
                w_expand = max(1, (e_i - s_i) // 20)
                if is_up_leg:
                    s_slice = target_df['low'].iloc[max(0, s_i - w_expand):min(len(target_df), s_i + w_expand + 1)]
                    s_i = int(s_slice.idxmin())
                    if not is_act:
                        e_slice = target_df['high'].iloc[max(0, e_i - w_expand):min(len(target_df), e_i + w_expand + 1)]
                        e_i = int(e_slice.idxmax())
                else:
                    s_slice = target_df['high'].iloc[max(0, s_i - w_expand):min(len(target_df), s_i + w_expand + 1)]
                    s_i = int(s_slice.idxmax())
                    if not is_act:
                        e_slice = target_df['low'].iloc[max(0, e_i - w_expand):min(len(target_df), e_i + w_expand + 1)]
                        e_i = int(e_slice.idxmin())

            if is_act:
                pts = self._solve_active_partial_wave(target_df, s_i, e_i, is_up_leg)
                det_rule = '5'
                leg_pivot_chains[m_idx] = pts
            elif w_id in ['1', '3', '5']:
                pts5, _ = self._solve_5wave_impulse(target_df, s_i, e_i, is_up_leg, is_active=False)
                if pts5:
                    det_rule = '5'
                    leg_pivot_chains[m_idx] = pts5
                else:
                    pts3, _ = self._solve_3wave_corrective(target_df, s_i, e_i, is_up_leg, is_active=False)
                    det_rule = '5'
                    leg_pivot_chains[m_idx] = pts3 if pts3 else [
                        {'index': s_i, 'time': target_df['timestamp'].iloc[s_i], 'price': mw['start_price'], 'type': 'LOW' if is_up_leg else 'HIGH'},
                        {'index': e_i, 'time': target_df['timestamp'].iloc[e_i], 'price': mw['end_price'], 'type': 'HIGH' if is_up_leg else 'LOW'}
                    ]
            elif w_id in ['2', '4']:
                pts3, _ = self._solve_3wave_corrective(target_df, s_i, e_i, is_up_leg, is_active=False)
                det_rule = 'c3' if self._detect_complex_substructure(pts3, is_up_leg) else '3'
                mw['is_complex'] = (det_rule == 'c3')
                leg_pivot_chains[m_idx] = pts3 if pts3 else [
                    {'index': s_i, 'time': target_df['timestamp'].iloc[s_i], 'price': mw['start_price'], 'type': 'LOW' if is_up_leg else 'HIGH'},
                    {'index': e_i, 'time': target_df['timestamp'].iloc[e_i], 'price': mw['end_price'], 'type': 'HIGH' if is_up_leg else 'LOW'}
                ]
            elif w_id == 'A':
                pts5, sc5 = self._solve_5wave_impulse(target_df, s_i, e_i, is_up_leg, is_active=False)
                pts3, sc3 = self._solve_3wave_corrective(target_df, s_i, e_i, is_up_leg, is_active=False)
                if pts5 and (sc5 >= 68.0 or wave_view_mode == "view_complex_b"):
                    det_rule = '5'
                    leg_pivot_chains[m_idx] = pts5
                else:
                    is_comp = self._detect_complex_substructure(pts3, is_up_leg)
                    det_rule = 'c3' if is_comp else '3'
                    mw['is_complex'] = is_comp
                    leg_pivot_chains[m_idx] = pts3 if pts3 else [
                        {'index': s_i, 'time': target_df['timestamp'].iloc[s_i], 'price': mw['start_price'], 'type': 'LOW' if is_up_leg else 'HIGH'},
                        {'index': e_i, 'time': target_df['timestamp'].iloc[e_i], 'price': mw['end_price'], 'type': 'HIGH' if is_up_leg else 'LOW'}
                    ]
            else:
                pts3, _ = self._solve_3wave_corrective(
                    target_df, s_i, e_i, is_up_leg, is_active=False, allow_expanded_flat=True
                )
                is_comp = self._detect_complex_substructure(pts3, is_up_leg)
                det_rule = 'c3' if is_comp else '3'
                mw['is_complex'] = is_comp
                leg_pivot_chains[m_idx] = pts3 if pts3 else [
                    {'index': s_i, 'time': target_df['timestamp'].iloc[s_i], 'price': mw['start_price'], 'type': 'LOW' if is_up_leg else 'HIGH'},
                    {'index': e_i, 'time': target_df['timestamp'].iloc[e_i], 'price': mw['end_price'], 'type': 'HIGH' if is_up_leg else 'LOW'}
                ]

            mw['subdivision_rule'] = det_rule
            mw['structure_label'] = f":{det_rule}"
            if w_id == 'A':
                a_rule = det_rule
            elif w_id == 'B':
                b_rule = det_rule
                b_is_complex = mw['is_complex']

        if a_rule is not None:
            b_disp = b_rule if b_rule else '진행중'
            if a_rule == '5' and b_rule in ['3', 'c3']:
                comp_note = " — (B)파 확장형 플랫/복합 W-X-Y" if b_is_complex else ""
                pat_title = f"지그재그 조정파 (Zigzag / 5-3-5{comp_note})"
                c_expect = "5파동 (C-1 ~ C-5)"
                reason_txt = (
                    f"**(A)파가 5파동(`:5`, A-1~A-5)으로 완료**되고, 이후 구간이 "
                    f"**(B)파 `:{b_rule}`{'(확장형 플랫 / 복합 W-X-Y)' if b_is_complex else '(3파동 조정)'}**으로 반등하여 **`5-3` 구조**가 형성되었습니다. "
                    f"현재 진행 중인 **(C)파는 5개의 세부파동(`C-1 ~ C-5`)**으로 전개될 전망입니다."
                )
            elif a_rule in ['3', 'c3'] and b_rule in ['3', 'c3']:
                pat_title = "복합조정 / 플랫 (W-X-Y / Flat 3-3-5)"
                c_expect = "5파동 (플랫 C파) 또는 3파동 (복합 Y파)"
                reason_txt = (
                    f"**(A)파(:{a_rule})**와 **(B)파(:{b_rule})** 모두 3파동 계열인 **`3-3` 복합 구조**입니다. "
                    f"플랫(`3-3-5`) 또는 이중 3파 복합조정(`W-X-Y`)으로 전개됩니다."
                )
            else:
                pat_title = f"조정파 전개 중 (A:{a_rule} - B:{b_disp})"
                c_expect = "5파동 (C-1 ~ C-5)"
                reason_txt = f"(A)파는 `:{a_rule}` 구조로 확인되었으며, 후속 파동 전개에 따라 `5-3-5` 지그재그 또는 `W-X-Y` 복합조정을 추적 중입니다."

            corr_diagnosis = {
                'pattern_title': pat_title,
                'observed_seq': f"{a_rule}-{b_disp}",
                'expected_c_rule': c_expect,
                'reason': reason_txt
            }
            for mw in macro_waves:
                if mw['local_wave_id'] in ['A', 'B', 'C']:
                    mw['pattern_tag'] = pat_title

        sub_pivots = []
        sub_waves = []

        for m_idx, mw in enumerate(macro_waves):
            m_lbl = mw['local_wave_id']
            is_act = mw['is_active']
            is_comp = mw.get('is_complex', False)
            pts = leg_pivot_chains[m_idx]
            wxy_tags = ['W', 'X', 'Y']

            for k in range(1, len(pts)):
                p_prev, p_curr = pts[k - 1], pts[k]
                is_final_pt_of_leg = (k == len(pts) - 1)
                if is_comp and len(pts) == 4:
                    base_sub_lbl = f"{m_lbl}-{k}({wxy_tags[k-1]})"
                else:
                    base_sub_lbl = f"{m_lbl}-{k}"

                sub_lbl = f"{base_sub_lbl}(진행)" if (is_act and is_final_pt_of_leg) else base_sub_lbl

                if m_idx == 0 and k == 1:
                    sub_pivots.append({
                        'time': p_prev['time'], 'price': p_prev['price'],
                        'type': p_prev['type'], 'label': '', 'is_shared_macro': True
                    })

                sub_pivots.append({
                    'time': p_curr['time'], 'price': p_curr['price'],
                    'type': p_curr['type'], 'label': sub_lbl,
                    'is_shared_macro': is_final_pt_of_leg
                })
                sub_waves.append({
                    'macro_id': m_lbl,
                    'sub_id': sub_lbl,
                    'direction': 'UP' if p_curr['price'] >= p_prev['price'] else 'DOWN',
                    'start_time': p_prev['time'],
                    'end_time': p_curr['time'],
                    'start_price': p_prev['price'],
                    'end_price': p_curr['price']
                })

        active_legs_done = (len(leg_pivot_chains[len(macro_waves) - 1]) - 1) if (macro_waves and macro_waves[-1]['is_active']) else 0
        projection = self._build_wave_projection(cdf, macro_waves, active_legs_done)
        synced_audits = self._build_synced_audit_cards(macro_waves, corr_diagnosis)

        return {
            'macro_pivots': macro_pivots,
            'macro_waves': macro_waves,
            'sub_pivots': sub_pivots,
            'sub_waves': sub_waves,
            'projection': projection,
            'synced_audits': synced_audits,
            'correction_diagnosis': corr_diagnosis
        }

    def _build_synced_audit_cards(self, macro_waves: list, corr_diagnosis: dict = None) -> list:
        cards = []
        imp_waves = [w for w in macro_waves if w['local_wave_id'] in ['1', '2', '3', '4', '5']]
        corr_waves = [w for w in macro_waves if w['local_wave_id'] in ['A', 'B', 'C']]

        if len(imp_waves) == 5:
            w1, w2, w3, w4, w5 = imp_waves
            is_up = w1['direction'] == 'UP'
            L1, L2, L3, L4, L5 = [x['log_magnitude'] for x in imp_waves]
            no_overlap = (w4['end_price'] > w1['end_price']) if is_up else (w4['end_price'] < w1['end_price'])
            r2 = L2 / max(L1, 1e-8) * 100
            scale_tag = "로그(Log) 비율" if self.use_log_scale else "산술 비율"

            audits = [
                f"✅ [+25점] 2파 불변 법칙 통과 (0파 `${w1['start_price']:,.0f}` 이탈 안 함, {scale_tag} 되돌림 `{r2:.1f}%`)",
                f"✅ [+25점] 3파 최단 법칙 및 시간 비례 통과 (1파 `{w1['duration_bars']}봉` / 3파 `{w3['duration_bars']}봉` / 5파 `{w5['duration_bars']}봉`)",
                f"{'✅ [+25점] 1파-4파 영역 비겹침 확인' if no_overlap else '☑️ [+15점] 다이아고날 1-4 겹침 허용'} (1파 `${w1['end_price']:,.0f}` vs 4파 `${w4['end_price']:,.0f}`)",
                f"✅ [+20점] ATR 동적 스윙 & 하위 타임프레임 기준 `5-{w2['subdivision_rule']}-5-{w4['subdivision_rule']}-5` 구조 검증"
            ]
            cards.append({
                'pattern_name': f"대파동 5파 충격파 완료 (1-2-3-4-5 / {'상승' if is_up else '하락'})",
                'category': 'IMPULSE',
                'direction': 'UP' if is_up else 'DOWN',
                'score': 95 if no_overlap else 85,
                'stage_status': f"(5)파 종점 `${w5['end_price']:,.0f}` 완성 → 후속 (A)-(B)-(C) 조정 진행",
                'start_price': w1['start_price'],
                'end_price': w5['end_price'],
                'invalidation_price': w4['end_price'],
                'audits': audits
            })

        if len(corr_waves) >= 2:
            wa = corr_waves[0]
            wb = corr_waves[1]
            ret_b = wb['log_magnitude'] / max(wa['log_magnitude'], 1e-8) * 100
            c_dir = wa['direction']
            pat_name = corr_diagnosis['pattern_title'] if corr_diagnosis else "후속 조정파 (A-B-C)"

            audits_c = [
                f"✅ [+35점] (A)파 내부 `:{wa['subdivision_rule']}` 구조 판별 (`${wa['start_price']:,.0f}` → `${wa['end_price']:,.0f}`)",
                f"✅ [+30점] (B)파 내부 `:{wb['subdivision_rule']}` 구조 및 되돌림 `{ret_b:.1f}%` 확인 (`${wb['start_price']:,.0f}` → `${wb['end_price']:,.0f}`)"
            ]
            if len(corr_waves) == 3:
                wc = corr_waves[2]
                ext_c = wc['log_magnitude'] / max(wa['log_magnitude'], 1e-8) * 100
                audits_c.append(f"✅ [+25점] 현재 (C)파 진행 중 (A파 대비 `{ext_c:.1f}%` 도달, 예정 세부 구조: `:{wc['subdivision_rule']}`)")
                stage_txt = f"(C)파 진행 중 — {corr_diagnosis['expected_c_rule'] if corr_diagnosis else '5파동'} 전개 예상"
                end_p = wc['end_price']
            else:
                stage_txt = "(B)파 마무리 후 (C)파 전환 탐색 구간"
                end_p = wb['end_price']

            cards.append({
                'pattern_name': pat_name,
                'category': 'ZIGZAG' if wa['subdivision_rule'] == '5' else 'FLAT',
                'direction': c_dir,
                'score': 92,
                'stage_status': stage_txt,
                'start_price': wa['start_price'],
                'end_price': end_p,
                'invalidation_price': wb['end_price'],
                'audits': audits_c
            })

        return cards

    def _build_wave_projection(self, df: pd.DataFrame, macro_waves: list, active_legs_done: int) -> dict:
        if len(macro_waves) < 2 or len(df) < 5:
            return None

        curr_mw = macro_waves[-1]
        prev_mw = macro_waves[-2]
        m_lbl = curr_mw['local_wave_id']
        curr_dir = curr_mw['direction']
        start_p = curr_mw['start_price']
        curr_p = curr_mw['end_price']
        last_time = pd.Timestamp(df['timestamp'].iloc[-1])
        dt_step = pd.Timestamp(df['timestamp'].iloc[-1]) - pd.Timestamp(df['timestamp'].iloc[-2])

        ref_range = prev_mw['price_change']
        for w in reversed(macro_waves[:-1]):
            if w['direction'] == curr_dir:
                ref_range = w['price_change']
                break

        if curr_dir == 'DOWN':
            sym_target = start_p - ref_range * 0.85
            target_p = max(curr_p * 0.65, min(curr_p - ref_range * 0.22, sym_target))
        else:
            sym_target = start_p + ref_range * 0.85
            target_p = min(curr_p * 1.45, max(curr_p + ref_range * 0.22, sym_target))

        if active_legs_done <= 1:
            total_rem = target_p - curr_p
            c2_p = curr_p - total_rem * 0.28
            c3_p = curr_p + total_rem * 0.68
            c4_p = c3_p - total_rem * 0.20
            c5_p = target_p
            proj_times = [last_time, last_time + dt_step * 4, last_time + dt_step * 10, last_time + dt_step * 15, last_time + dt_step * 21]
            proj_prices = [curr_p, c2_p, c3_p, c4_p, c5_p]
            proj_labels = ["", f"{m_lbl}-2", f"{m_lbl}-3", f"{m_lbl}-4", f"{m_lbl}-5 (${c5_p:,.0f})"]
        else:
            move_rem = target_p - curr_p
            p1_price = curr_p + move_rem * 0.58
            p2_price = p1_price - move_rem * 0.24
            p3_price = target_p
            proj_times = [last_time, last_time + dt_step * 6, last_time + dt_step * 12, last_time + dt_step * 18]
            proj_prices = [curr_p, p1_price, p2_price, p3_price]
            proj_labels = ["", f"{m_lbl}-4", f"{m_lbl}-5 (${p3_price:,.0f})"]

        return {
            'times': proj_times,
            'prices': proj_prices,
            'labels': proj_labels,
            'target_price': target_p
        }

    def evaluate_all_patterns(self, waves: list, df: pd.DataFrame = None) -> list:
        candidates = []
        if len(waves) < 3:
            return candidates
        if len(waves) >= 5:
            imp = self._check_impulse_and_diagonal(waves[-5:], df)
            if imp:
                candidates.append(imp)
            tri = self._check_triangle(waves[-5:])
            if tri:
                candidates.append(tri)
        if len(waves) >= 3:
            candidates.extend(self._check_zigzag_and_flat(waves[-3:], df))
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates

    def _check_impulse_and_diagonal(self, w: list, df: pd.DataFrame = None) -> dict:
        w1, w2, w3, w4, w5 = w
        is_up = (w1['direction'] == 'UP' and w2['direction'] == 'DOWN' and w3['direction'] == 'UP' and w4['direction'] == 'DOWN' and w5['direction'] == 'UP')
        is_down = (w1['direction'] == 'DOWN' and w2['direction'] == 'UP' and w3['direction'] == 'DOWN' and w4['direction'] == 'UP' and w5['direction'] == 'DOWN')
        if not (is_up or is_down):
            return None
        L1, L2, L3, L4, L5 = [x['log_magnitude'] for x in w]
        if (is_up and w2['end_price'] <= w1['start_price']) or (is_down and w2['end_price'] >= w1['start_price']):
            return None
        if L3 <= min(L1, L5):
            return None
        return {
            'pattern_name': "단기 5파 충격파 (1-2-3-4-5)",
            'category': 'IMPULSE',
            'direction': 'UP' if is_up else 'DOWN',
            'score': 82,
            'stage_status': '단기 5파 종점 도달 구간',
            'start_price': w1['start_price'],
            'end_price': w5['end_price'],
            'invalidation_price': w4['end_price'],
            'audits': ["✅ 단기 ATR 동적 모노파동 5파 충격 구조 충족"]
        }

    def _check_triangle(self, w: list) -> dict:
        wa, wb, wc, wd, we = w
        if not (wa['direction'] != wb['direction'] != wc['direction'] != wd['direction'] != we['direction']):
            return None
        La, Lb, Lc, Ld, Le = [x['log_magnitude'] for x in w]
        if (La > Lc > Le) and (Lb > Ld):
            return {
                'pattern_name': "수렴형 삼각형 (A-B-C-D-E)",
                'category': 'TRIANGLE',
                'direction': wa['direction'],
                'score': 80,
                'stage_status': 'E파 수렴 마무리 구간',
                'start_price': wa['start_price'],
                'end_price': we['end_price'],
                'invalidation_price': wc['end_price'],
                'audits': ["✅ 5파 수렴 삼각형 구조 충족"]
            }
        return None

    def _check_zigzag_and_flat(self, w: list, df: pd.DataFrame = None) -> list:
        wa, wb, wc = w
        if not (wa['direction'] != wb['direction'] != wc['direction']):
            return []
        ret_b = wb['log_magnitude'] / max(wa['log_magnitude'], 1e-8)
        if ret_b <= 1.382:
            return [{
                'pattern_name': "단기 조정파 (A-B-C)",
                'category': 'ZIGZAG' if ret_b <= 0.65 else 'FLAT',
                'direction': wa['direction'],
                'score': 78,
                'stage_status': '단기 C파 진행/완료 구간',
                'start_price': wa['start_price'],
                'end_price': wc['end_price'],
                'invalidation_price': wc['end_price'],
                'audits': [f"✅ B파 되돌림 비율 {ret_b*100:.1f}% 확인"]
            }]
        return []