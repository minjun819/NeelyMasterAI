# -*- coding: utf-8 -*-
# 한글
# -*- coding: utf-8 -*-

class FractalMapper:
    """
    이중 계층 프랙탈 매핑 & 다중 시나리오·수급 통합 브리핑 엔진 (v6.0)
    """

    def synthesize_fractal_context(self, all_tf_data: dict, cycle_info: dict = None, okx_pos: dict = None) -> dict:
        def get_active_info(tdata):
            dual = tdata.get('dual', {})
            mw = dual.get('macro_waves', [])
            sw = dual.get('sub_waves', [])
            diag = dual.get('correction_diagnosis')
            pat_str = diag['pattern_title'] if diag else (mw[-1]['pattern_tag'] if mw else '')

            if mw and sw:
                active_tag = "진행 중" if mw[-1].get('is_active') else "완료"
                return f"({mw[-1]['local_wave_id']})파 {active_tag} → [{sw[-1]['sub_id']}]", pat_str
            if mw:
                return f"({mw[-1]['local_wave_id']})파", pat_str
            return '집계 중', ''

        w1_lbl, w1_tag = get_active_info(all_tf_data.get('1w', {}))
        d1_lbl, d1_tag = get_active_info(all_tf_data.get('1d', {}))
        h1_lbl, h1_tag = get_active_info(all_tf_data.get('1h', {}))
        m15_lbl, m15_tag = get_active_info(all_tf_data.get('15m', {}))

        narrative = [
            f"🌐 **[주봉 거대 사이클]**: 현재 **`{w1_lbl}`** — `{w1_tag}`",
            f"📅 **[일봉 매크로 파동]**: 현재 **`{d1_lbl}`** — `{d1_tag}`",
            f"⏱️ **[1시간봉 중기 파동]**: 현재 **`{h1_lbl}`** — `{h1_tag}`",
            f"🔬 **[15분봉 세부 파동]**: 현재 **`{m15_lbl}`** — `{m15_tag}`"
        ]

        w1_diag = all_tf_data.get('1w', {}).get('dual', {}).get('correction_diagnosis')
        if w1_diag:
            narrative.append(f"🧩 **[주봉 닐리 조정파 추론 (`{w1_diag['observed_seq']}`)]**: {w1_diag['reason']}")

        if okx_pos and okx_pos.get('is_live'):
            narrative.append(
                f"📡 **[OKX 선물 실시간 수급]**: `{okx_pos['bias_label']}` "
                f"(펀딩비 `{okx_pos['funding_rate_pct']:+.4f}%` | 롱/숏비 `{okx_pos['long_short_ratio']:.2f}` | 테이커비 `{okx_pos['taker_buy_sell_ratio']:.2f}`)"
            )

        if cycle_info:
            narrative.append(
                f"⏳ **[비트코인 4년 반감기 사이클]**: 2024.04 반감기 후 **{cycle_info['elapsed_days']}일({cycle_info['months_post']:.1f}개월, {cycle_info['progress_pct']:.1f}%)** 경과 — `{cycle_info['phase']}`"
            )

        alert_msg = (
            f"🎯 **다중 타임프레임 파동 좌표**: 주봉 **`{w1_lbl}`** ➔ 일봉 **`{d1_lbl}`** ➔ "
            f"1시간봉 **`{h1_lbl}`** ➔ 15분봉 **`{m15_lbl}`**"
        )

        return {
            'narrative_lines': narrative,
            'sub_completion_alert': alert_msg
        }