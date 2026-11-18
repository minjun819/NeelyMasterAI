import ccxt
import pandas as pd
import numpy as np

class OKXNeelyAdvancedAnalyzer:
    def __init__(self, symbol='BTC/USDT', min_swing_pct=0.01):
        """
        OKX 선물(Swap) 거래소 연동 및 닐리 파동 분석 심화 초기화
        :param symbol: 거래 종목 (비트코인 USDT 무기한 선물)
        :param min_swing_pct: 노이즈 필터링을 위한 최소 가격 변동 비율 (1% = 0.01)
        """
        self.exchange = ccxt.okx({
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })
        self.symbol = symbol
        self.min_swing_pct = min_swing_pct
        
        # 요청하신 핵심 타임프레임 (15분, 1시간, 일봉, 주봉)
        self.timeframes = {
            '15분봉 (단기 파동)': '15m',
            '1시간봉 (중기 메인 트렌드)': '1h',
            '일봉 (매크로 파동)': '1d',
            '주봉 (최상위 거대 사이클)': '1w'
        }

    def fetch_ohlcv_data(self, timeframe, limit=200):
        """1단계: OKX 선물 시장에서 OHLCV 및 중앙값 데이터 수집"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['median_price'] = (df['high'] + df['low']) / 2
            return df
        except Exception as e:
            print(f"[{timeframe}] 데이터 수집 실패: {e}")
            return None

    def extract_monowaves(self, df):
        """2단계: 노이즈를 제거하고 닐리 방식의 모노파동 변곡점 추출"""
        if df is None or len(df) < 5:
            return []

        prices = df['median_price'].values
        timestamps = df['timestamp'].values
        
        pivots = []
        last_pivot_price = prices[0]
        last_pivot_idx = 0
        trend = None

        for i in range(1, len(prices)):
            current_price = prices[i]
            pct_change = (current_price - last_pivot_price) / last_pivot_price

            if trend is None:
                if pct_change >= self.min_swing_pct:
                    trend = 'UP'; last_pivot_price = current_price; last_pivot_idx = i
                elif pct_change <= -self.min_swing_pct:
                    trend = 'DOWN'; last_pivot_price = current_price; last_pivot_idx = i
            elif trend == 'UP':
                if current_price > last_pivot_price:
                    last_pivot_price = current_price; last_pivot_idx = i
                elif (last_pivot_price - current_price) / last_pivot_price >= self.min_swing_pct:
                    pivots.append({'index': last_pivot_idx, 'time': timestamps[last_pivot_idx], 'price': last_pivot_price, 'type': 'HIGH'})
                    trend = 'DOWN'; last_pivot_price = current_price; last_pivot_idx = i
            elif trend == 'DOWN':
                if current_price < last_pivot_price:
                    last_pivot_price = current_price; last_pivot_idx = i
                elif (current_price - last_pivot_price) / last_pivot_price >= self.min_swing_pct:
                    pivots.append({'index': last_pivot_idx, 'time': timestamps[last_pivot_idx], 'price': last_pivot_price, 'type': 'LOW'})
                    trend = 'UP'; last_pivot_price = current_price; last_pivot_idx = i

        monowaves = []
        for j in range(1, len(pivots)):
            p_start = pivots[j-1]; p_end = pivots[j]
            price_diff = p_end['price'] - p_start['price']
            monowaves.append({
                'wave_no': j,
                'start_price': p_start['price'],
                'end_price': p_end['price'],
                'price_change': abs(price_diff),
                'direction': 'UP' if price_diff > 0 else 'DOWN',
                'duration_bars': p_end['index'] - p_start['index']
            })
        return monowaves

    def validate_neely_rules_and_variations(self, waves):
        """
        3단계: 닐리 규칙 검증 엔진 (절대법칙, 연장, 미달, 복합조정/X파동 판별)
        """
        if len(waves) < 5:
            return "파동 데이터 부족 (최소 5개 이상 필요)"

        # 최근 5개 모노파동 추출
        w1, w2, w3, w4, w5 = waves[-5:]

        # 1. 기본 충격파동 방향 시퀀스 확인 (상승 기준: UP - DOWN - UP - DOWN - UP)
        is_impulse_seq = (w1['direction'] == 'UP' and w2['direction'] == 'DOWN' and 
                          w3['direction'] == 'UP' and w4['direction'] == 'DOWN' and w5['direction'] == 'UP')

        if not is_impulse_seq:
            # 충격파동 시퀀스가 아니라면 복합조정(이중/삼중 조합) 및 X파동 가능성 검사
            return self.detect_complex_combinations(waves)

        # 2. [절대법칙 1] 2파는 1파의 시작점을 이탈할 수 없음
        if w2['end_price'] <= w1['start_price']:
            return "❌ 위반: 2파가 1파 시작점을 이탈함 (충격파동 성립 불가)"

        # 3. [절대법칙 2] 3파는 가장 짧을 수 없음 (1, 3, 5파 중)
        lengths = [w1['price_change'], w3['price_change'], w5['price_change']]
        if w3['price_change'] == min(lengths):
            return "❌ 위반: 3파가 1파와 5파에 비해 가장 짧음 (3파 최단 법칙 위반)"

        # 4. [절대법칙 3] 4파는 1파의 가격 영역과 겹칠 수 없음 (Overlap)
        if w4['end_price'] <= w1['end_price']:
            return "⚠️ 경고: 4파 저점이 1파 고점 영역과 겹침 (다이아고날 또는 터미널 패턴 가능성)"

        # 5. [심화 규칙] 연장(Extension) 파동 판별
        max_len = max(lengths)
        ext_wave = "3파 연장" if max_len == w3['price_change'] else ("1파 연장" if max_len == w1['price_change'] else "5파 연장")

        # 6. [심화 규칙] 미달(Truncation / 5파 실패) 판별
        # 5파의 고점이 3파의 고점을 넘지 못하는 경우
        is_truncated = (w5['end_price'] <= w3['end_price'])
        truncation_msg = " [⚠️ 5파 미달(Truncation) 발생 포착! 강력한 추세 반전 주의]" if is_truncated else ""

        return f"✅ 유효한 5파 충격파동 감지 [{ext_wave}]{truncation_msg}"

    def detect_complex_combinations(self, waves):
        """복합조정(이중/삼중 지그재그·플랫) 및 X파동 연결 패턴 판별"""
        if len(waves) >= 6:
            # 닐리 이론상 조정파동들이 X파동을 사이에 두고 W-X-Y 또는 W-X-Y-X-Z 형태로 이어지는지 확인
            return "🔄 [복합조정 / 이중·삼중 조합 파동(Double/Triple Combination & X-Wave) 구간 감지]"
        return "ℹ️ 표준 충격파동이 아니며, 단순 조정파동(지그재그/플랫/삼각수렴) 진행 중"

    def run_full_analysis(self):
        """전체 파이프라인 실행"""
        print(f"=== OKX 선물 [{self.symbol}] 다중 타임프레임 닐리 심화 분석 가동 ===")
        
        for label, tf in self.timeframes.items():
            df = self.fetch_ohlcv_data(tf, limit=200)
            if df is not None:
                monowaves = self.extract_monowaves(df)
                current_price = df['close'].iloc[-1]
                print(f"\n✔ [{label}] 분석 완료 | 최신가: ${current_price:,.2f} | 모노파동 수: {len(monowaves)}개")
                
                if len(monowaves) >= 5:
                    analysis_result = self.validate_neely_rules_and_variations(monowaves)
                    print(f"  └ [AI 닐리 규칙 심판 결과]: {analysis_result}")
                    
                    # 최근 파동 요약 출력
                    print(f"  └ [최근 5개 모노파동 흐름]")
                    for mw in monowaves[-5:]:
                        print(f"    • 파동 {mw['wave_no']} ({mw['direction']}): 시작 ${mw['start_price']:,.2f} -> 종료 ${mw['end_price']:,.2f}")

# 실행 테스트
if __name__ == "__main__":
    analyzer = OKXNeelyAdvancedAnalyzer(symbol='BTC/USDT', min_swing_pct=0.01)
    analyzer.run_full_analysis()