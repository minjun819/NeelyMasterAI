# -*- coding: utf-8 -*-
# 한글
import numpy as np
import pandas as pd

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['rsi'] = calculate_rsi(df['close'], 14)

    for span in [20, 50, 100, 200]:
        df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean().bfill()
    return df

def analyze_btc_4year_cycle(current_dt: pd.Timestamp) -> dict:
    """
    비트코인 약 4년(약 1,460일) 반감기 사이클 위상 인지 모듈
    - 4차 반감기: 2024-04-20 / 5차 반감기 예상: 2028-04-19
    - 절대 법칙이 아닌 거시적 시장 리듬(강세 확장기 -> 대형 ABC 조정기 -> 바닥 축적기) 인지용
    """
    halving_4th = pd.Timestamp("2024-04-20")
    cycle_days = 1460.0
    elapsed_days = max(0.0, (pd.Timestamp(current_dt) - halving_4th).total_seconds() / 86400.0)
    progress_pct = (elapsed_days % cycle_days) / cycle_days * 100.0
    months_post = elapsed_days / 30.4375

    if months_post <= 18.5:
        phase = "1단계: 반감기 후 강세 확장기 (대파동 1~5파 상승 우위 구간)"
        bias = "BULLISH"
        note = "역사적 4년 리듬상 반감기 이후 약 18개월간 거대 상승 충격파가 전개되는 구간입니다."
    elif months_post <= 33.0:
        phase = "2단계: 고점 이후 대형 조정기 (대파동 A-B-C 하락/복합조정 구간)"
        bias = "BEARISH_CORRECTIVE"
        note = "고점 형성 후 약 1년간 대형 A-B-C 조정이 진행되는 구간으로, 반등 시 C파(또는 Y/Z파) 추가 하락 압력을 인지해야 합니다."
    else:
        phase = "3단계: 조정 마무리 및 차기 반감기 선반영 축적기 (C파 종결 및 신규 사이클 준비)"
        bias = "ACCUMULATION"
        note = "대형 하락 조정이 성숙 단계에 접어들어 거시적 바닥 다지기 및 신규 상승 사이클 전환을 탐색하는 구간입니다."

    return {
        'elapsed_days': int(elapsed_days),
        'months_post': months_post,
        'progress_pct': progress_pct,
        'phase': phase,
        'bias': bias,
        'note': note
    }

def analyze_ma_and_momentum(df: pd.DataFrame, position: str) -> dict:
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    price = last['close']
    e20, e50, e100, e200 = last['ema_20'], last['ema_50'], last['ema_100'], last['ema_200']

    if e20 > e50 > e100 > e200:
        alignment = "완전 정배열 (강한 상승 추세)"
    elif e20 < e50 < e100 < e200:
        alignment = "완전 역배열 (강한 하락 추세)"
    elif e20 > e50 and price > e200:
        alignment = "단기 상승 전환 (200 EMA 상단)"
    elif e20 < e50 and price < e200:
        alignment = "단기 하락 전환 (200 EMA 하단)"
    else:
        alignment = "이평선 수렴/혼조 구간"

    bonus = 0
    notes = [f"이평선 배열: `{alignment}` (EMA20 `${e20:,.0f}` / EMA50 `${e50:,.0f}` / EMA200 `${e200:,.0f}`)"]

    for name, val in [('EMA 20', e20), ('EMA 50', e50), ('EMA 100', e100), ('EMA 200', e200)]:
        if abs(price - val) / price <= 0.01:
            bonus += 4
            notes.append(f"가격이 주요 동적 지지/저항선 **{name}(`${val:,.0f}`)**에 밀착")
            break

    hist_curr = last['macd_hist']
    hist_prev = prev['macd_hist']
    if position == 'LONG' and hist_curr > hist_prev:
        bonus += 4
        notes.append("MACD 히스토그램 상승 반전 (매수 모멘텀 유입)")
    elif position == 'SHORT' and hist_curr < hist_prev:
        bonus += 4
        notes.append("MACD 히스토그램 하락 반전 (매도 모멘텀 유입)")

    return {
        'alignment': alignment,
        'bonus': bonus,
        'desc': " | ".join(notes),
        'ema_200': e200
    }

def detect_rsi_divergence(df: pd.DataFrame, pivots: list) -> dict:
    result = {'type': 'NONE', 'detected': False, 'desc': '특이 다이버전스 없음'}
    if df is None or 'rsi' not in df.columns or len(pivots) < 4:
        return result

    highs = [p for p in pivots if p['type'] == 'HIGH']
    lows = [p for p in pivots if p['type'] == 'LOW']

    if len(highs) >= 2:
        h1, h2 = highs[-2], highs[-1]
        if h2['index'] >= len(df) - 18:
            rsi1 = df['rsi'].iloc[h1['index']]
            rsi2 = df['rsi'].iloc[h2['index']]
            if h2['price'] > h1['price'] and rsi2 < rsi1:
                return {
                    'type': 'BEARISH',
                    'detected': True,
                    'desc': f"하락 다이버전스 포착 (가격 상승 ${h1['price']:,.0f}→${h2['price']:,.0f}, RSI 하락 {rsi1:.1f}→{rsi2:.1f})"
                }

    if len(lows) >= 2:
        l1, l2 = lows[-2], lows[-1]
        if l2['index'] >= len(df) - 18:
            rsi1 = df['rsi'].iloc[l1['index']]
            rsi2 = df['rsi'].iloc[l2['index']]
            if l2['price'] < l1['price'] and rsi2 > rsi1:
                return {
                    'type': 'BULLISH',
                    'detected': True,
                    'desc': f"상승 다이버전스 포착 (가격 하락 ${l1['price']:,.0f}→${l2['price']:,.0f}, RSI 상승 {rsi1:.1f}→{rsi2:.1f})"
                }

    return result

def analyze_volume_exhaustion(df: pd.DataFrame, monowaves: list) -> dict:
    if df is None or len(monowaves) < 3:
        return {'exhausted': False, 'desc': '거래량 데이터 부족'}

    w_prev = monowaves[-3]
    w_curr = monowaves[-1]
    vol_prev = df['volume'].iloc[w_prev['start_idx']:w_prev['end_idx']+1].mean()
    vol_curr = df['volume'].iloc[w_curr['start_idx']:w_curr['end_idx']+1].mean()

    if vol_prev > 0 and vol_curr < vol_prev * 0.8:
        ratio = (vol_curr / vol_prev) * 100
        return {
            'exhausted': True,
            'desc': f"거래량 소멸 확인 (직전 동방향 파동 대비 {ratio:.1f}%로 감소 → 변곡 임박)"
        }
    return {'exhausted': False, 'desc': '거래량 유지 중 (추세 힘 잔존)'}

def find_support_resistance(pivots: list, current_price: float, tolerance: float = 0.006) -> dict:
    if not pivots:
        return {'nearest_support': None, 'nearest_resistance': None, 'all_levels': []}

    prices = sorted([p['price'] for p in pivots])
    clusters = []
    for p in prices:
        placed = False
        for c in clusters:
            if abs(p - c['level']) / c['level'] <= tolerance:
                c['level'] = (c['level'] * c['count'] + p) / (c['count'] + 1)
                c['count'] += 1
                placed = True
                break
        if not placed:
            clusters.append({'level': p, 'count': 1})

    supports = [c['level'] for c in clusters if c['level'] < current_price]
    resistances = [c['level'] for c in clusters if c['level'] > current_price]

    return {
        'nearest_support': max(supports) if supports else None,
        'nearest_resistance': min(resistances) if resistances else None,
        'all_levels': clusters
    }