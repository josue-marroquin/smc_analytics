import argparse
from datetime import timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from smc_liquidity import (
    build_dealing_range_table,
    build_fvg_table,
    build_immediate_liquidity_probability_table,
    build_market_structure_table,
    build_smc_liquidity_table,
    select_latest_dealing_range,
    select_recent_fvg_levels,
    select_recent_liquidity_levels,
    select_recent_structure_events,
)


BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1/klines"
UTC_MINUS_5 = timezone(timedelta(hours=-5))
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1h"
DISPLAY_LEVELS_PER_GROUP = 1
DISPLAY_FVGS_PER_SIDE = 2
DISPLAY_STRUCTURE_EVENTS_PER_SIDE = 3
DISPLAY_CANDLE_WINDOW = 200
RECENT_CANDLE_LOOKBACK = 200 #120
MIN_FVG_GAP_PCT = 0.0
STRUCTURE_MIN_BREAK_PCT = 0.05
STRUCTURE_MIN_BODY_RATIO = 0.5
STRUCTURE_MIN_CLOSE_POSITION = 0.6
STRUCTURE_VOLUME_WINDOW = 20
STRUCTURE_MIN_VOLUME_RATIO = 1.2
STRUCTURE_REQUIRE_VOLUME_CONFIRMATION = False
DEALING_RANGE_EQ_BAND_PCT = 5.0


def get_klines(symbol=DEFAULT_SYMBOL, interval=DEFAULT_INTERVAL, limit=300):
    """Funcion que descarga klines de Binance Futures, tipa columnas numericas y normaliza el tiempo a UTC-5."""
    url = f"{BINANCE_FUTURES_URL}?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(
        data,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "qav",
            "trades",
            "tbbav",
            "tbqav",
            "ignore",
        ],
    )

    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        df[column] = df[column].astype(float)

    # Binance entrega timestamps absolutos (UTC). Aqui los convertimos a UTC-5 fijo.
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(
        UTC_MINUS_5
    )
    return df


def get_liquidity_table(
    symbol=DEFAULT_SYMBOL,
    interval=DEFAULT_INTERVAL,
    limit=300,
    swing_window=2,
    major_window=5,
    tolerance_pct=0.001,
):
    """Funcion de conveniencia que devuelve una tabla de liquidez SMC lista para consumo por simbolo e intervalo."""
    df = get_klines(symbol=symbol, interval=interval, limit=limit)
    liquidity = build_smc_liquidity_table(
        df,
        swing_window=swing_window,
        major_window=major_window,
        tolerance_pct=tolerance_pct,
    )

    if liquidity.empty:
        return liquidity

    liquidity.insert(0, "symbol", symbol)
    liquidity.insert(1, "interval", interval)
    return liquidity


def _filter_recent_rows(
    source_df,
    latest_candle_index,
    candle_lookback=RECENT_CANDLE_LOOKBACK,
):
    """Funcion auxiliar que limita una tabla a las filas mas recientes usando el indice de vela para enfoque operativo."""
    if source_df.empty or latest_candle_index is None:
        return source_df.copy()

    min_candle_index = max(0, latest_candle_index - candle_lookback)
    recent_rows = source_df.loc[source_df["candle_index"] >= min_candle_index]
    if recent_rows.empty:
        return source_df.copy()

    return recent_rows.copy()


def _prepare_display_summary(liquidity_df, latest_candle_index=None):
    """Funcion auxiliar que prepara la vista resumida de liquidez aplicando filtro temporal y orden por cercania al precio."""
    if liquidity_df.empty:
        return liquidity_df.copy()

    recent_liquidity = _filter_recent_rows(
        liquidity_df,
        latest_candle_index=latest_candle_index,
    )
    summary = select_recent_liquidity_levels(
        recent_liquidity,
        levels_per_group=DISPLAY_LEVELS_PER_GROUP,
        prefer_unswept=True,
    ).copy()
    summary["_distance_sort"] = summary["distance_from_close_pct"].abs()
    summary = summary.sort_values(
        by=["_distance_sort", "distance_from_close_pct"],
        ascending=[True, True],
    ).drop(columns="_distance_sort")
    return summary


def _prepare_fvg_summary(fvg_df, latest_candle_index=None):
    """Funcion auxiliar que prepara la vista resumida de FVG recientes priorizando zonas no llenadas y cercanas al precio."""
    if fvg_df.empty:
        return fvg_df.copy()

    recent_fvgs = _filter_recent_rows(
        fvg_df,
        latest_candle_index=latest_candle_index,
    )
    summary = select_recent_fvg_levels(
        recent_fvgs,
        zones_per_side=DISPLAY_FVGS_PER_SIDE,
        prefer_unfilled=True,
    ).copy()
    summary["_distance_sort"] = summary["distance_from_close_pct"].abs()
    summary = summary.sort_values(
        by=["_distance_sort", "distance_from_close_pct"],
        ascending=[True, True],
    ).drop(columns="_distance_sort")
    return summary


def _prepare_structure_summary(structure_df, latest_candle_index=None):
    """Funcion auxiliar que prepara eventos estructurales recientes para tabla y grafica, ordenados por relevancia."""
    if structure_df.empty:
        return structure_df.copy()

    recent_structure = _filter_recent_rows(
        structure_df,
        latest_candle_index=latest_candle_index,
    )
    summary = select_recent_structure_events(
        recent_structure,
        events_per_side=DISPLAY_STRUCTURE_EVENTS_PER_SIDE,
    ).copy()
    summary["_distance_sort"] = summary["distance_from_close_pct"].abs()
    summary = summary.sort_values(
        by=["_distance_sort", "distance_from_close_pct"],
        ascending=[True, True],
    ).drop(columns="_distance_sort")
    return summary


def _prepare_dealing_range_summary(dealing_range_df):
    """Funcion auxiliar que reduce la tabla de dealing range al contexto mas reciente para reporte y overlay."""
    if dealing_range_df.empty:
        return dealing_range_df.copy()

    summary = select_latest_dealing_range(
        dealing_range_df,
        rows=1,
    ).copy()
    return summary


def _prepare_probability_summary(probability_df):
    """Funcion auxiliar que ordena la tabla de probabilidades para resaltar el objetivo de liquidez mas probable."""
    if probability_df.empty:
        return probability_df.copy()

    return probability_df.sort_values("probability_pct", ascending=False).reset_index(drop=True)


def print_liquidity_summary(liquidity_df, rows=20, latest_candle_index=None):
    """Funcion que imprime la tabla resumida de liquidez con columnas clave para lectura rapida de niveles activos."""
    if liquidity_df.empty:
        print("No se detectaron niveles de liquidez con la configuracion actual.")
        return

    display_columns = [
        "symbol",
        "interval",
        "liquidity_time",
        "liquidity_level",
        "liquidity_side",
        "liquidity_rank",
        "equal_level_touches",
        "swept",
        "swept_time",
        "distance_from_close_pct",
    ]

    summary = _prepare_display_summary(
        liquidity_df,
        latest_candle_index=latest_candle_index,
    )[display_columns].copy()
    summary["liquidity_level"] = summary["liquidity_level"].round(2)
    summary["distance_from_close_pct"] = summary["distance_from_close_pct"].round(3)
    print(summary.head(rows).to_string(index=False))


def print_fvg_summary(fvg_df, rows=20, latest_candle_index=None):
    """Funcion que imprime la tabla resumida de FVG con estado de llenado, tamano de gap y distancia al precio actual."""
    if fvg_df.empty:
        print("No se detectaron Fair Value Gaps con la configuracion actual.")
        return

    display_columns = [
        "symbol",
        "interval",
        "fvg_time",
        "fvg_side",
        "fvg_low",
        "fvg_high",
        "gap_size",
        "gap_size_pct",
        "filled",
        "filled_time",
        "distance_from_close_pct",
    ]

    summary = _prepare_fvg_summary(
        fvg_df,
        latest_candle_index=latest_candle_index,
    )[display_columns].copy()
    summary["fvg_low"] = summary["fvg_low"].round(2)
    summary["fvg_high"] = summary["fvg_high"].round(2)
    summary["gap_size"] = summary["gap_size"].round(2)
    summary["gap_size_pct"] = summary["gap_size_pct"].round(3)
    summary["distance_from_close_pct"] = summary["distance_from_close_pct"].round(3)
    print(summary.head(rows).to_string(index=False))


def print_structure_summary(structure_df, rows=20, latest_candle_index=None):
    """Funcion que imprime eventos BOS/CHoCH/MSS con metrica de ruptura y confirmacion de volumen."""
    if structure_df.empty:
        print("No se detectaron eventos de market structure con la configuracion actual.")
        return

    display_columns = [
        "symbol",
        "interval",
        "event_time",
        "event_side",
        "event_kind",
        "is_mss",
        "event_price",
        "broken_level",
        "protected_level",
        "break_buffer_pct",
        "volume_ratio",
        "volume_zscore",
        "volume_confirmed",
        "trend_after",
        "distance_from_close_pct",
    ]

    summary = _prepare_structure_summary(
        structure_df,
        latest_candle_index=latest_candle_index,
    )[display_columns].copy()
    summary["event_price"] = summary["event_price"].round(2)
    summary["broken_level"] = summary["broken_level"].round(2)
    summary["protected_level"] = summary["protected_level"].round(2)
    summary["break_buffer_pct"] = summary["break_buffer_pct"].round(3)
    summary["volume_ratio"] = summary["volume_ratio"].round(3)
    summary["volume_zscore"] = summary["volume_zscore"].round(3)
    summary["distance_from_close_pct"] = summary["distance_from_close_pct"].round(3)
    print(summary.head(rows).to_string(index=False))


def print_dealing_range_summary(dealing_range_df):
    """Funcion que imprime el dealing range vigente y la posicion del precio dentro de premium/discount/equilibrium."""
    if dealing_range_df.empty:
        print("No se detecto dealing range activo con la configuracion actual.")
        return

    display_columns = [
        "symbol",
        "interval",
        "range_time",
        "range_high",
        "range_low",
        "range_midpoint",
        "current_price",
        "price_position_pct",
        "zone",
        "premium_pct",
        "discount_pct",
        "distance_from_eq_pct",
    ]

    summary = _prepare_dealing_range_summary(dealing_range_df)[display_columns].copy()
    summary["range_high"] = summary["range_high"].round(2)
    summary["range_low"] = summary["range_low"].round(2)
    summary["range_midpoint"] = summary["range_midpoint"].round(2)
    summary["current_price"] = summary["current_price"].round(2)
    summary["price_position_pct"] = summary["price_position_pct"].round(2)
    summary["premium_pct"] = summary["premium_pct"].round(2)
    summary["discount_pct"] = summary["discount_pct"].round(2)
    summary["distance_from_eq_pct"] = summary["distance_from_eq_pct"].round(3)
    print(summary.to_string(index=False))


def print_probability_summary(probability_df):
    """Funcion que imprime la probabilidad relativa de tomar la liquidez inmediata superior o inferior."""
    if probability_df.empty:
        print("No se pudo estimar probabilidad de liquidez inmediata con la configuracion actual.")
        return

    display_columns = [
        "target_side",
        "target_level",
        "target_rank",
        "target_distance_pct",
        "structure_bias",
        "dealing_range_bias",
        "fvg_bias",
        "distance_score",
        "liquidity_score",
        "raw_score",
        "probability_pct",
    ]

    summary = _prepare_probability_summary(probability_df)[display_columns].copy()
    for column in [
        "target_level",
        "target_distance_pct",
        "structure_bias",
        "dealing_range_bias",
        "fvg_bias",
        "distance_score",
        "liquidity_score",
        "raw_score",
        "probability_pct",
    ]:
        summary[column] = summary[column].round(3)

    print(summary.to_string(index=False))


def _format_liquidity_label(level_row):
    """Funcion auxiliar que formatea la etiqueta de liquidez para mostrar tipo de nivel y precio exacto."""
    base_label = level_row["liquidity_label"].replace("_", " ")
    return f"{base_label} [{level_row['liquidity_level']:.2f}]"


def _format_fvg_label(fvg_row):
    """Funcion auxiliar que formatea la etiqueta de FVG mostrando lado y rango de precios del gap."""
    side_label = fvg_row["fvg_label"].replace("_", " ")
    return f"{side_label} [{fvg_row['fvg_low']:.2f} - {fvg_row['fvg_high']:.2f}]"


def _format_structure_label(structure_row):
    """Funcion auxiliar que formatea etiqueta estructural incluyendo tipo de evento, MSS y validacion por volumen."""
    side_label = structure_row["event_side"]
    kind_label = structure_row["event_kind"]
    suffix = "/MSS" if bool(structure_row["is_mss"]) else ""
    volume_suffix = " +VOL" if bool(structure_row["volume_confirmed"]) else ""
    return (
        f"{side_label} {kind_label}{suffix}{volume_suffix} "
        f"[{structure_row['event_price']:.2f}]"
    )


def _format_dealing_range_label(range_row):
    """Funcion auxiliar que formatea una etiqueta compacta del dealing range activo y su zona actual."""
    return (
        f"{range_row['zone']} | DR "
        f"[{range_row['range_low']:.2f} - {range_row['range_high']:.2f}]"
    )


def _format_probability_label(probability_row):
    """Funcion auxiliar que compacta la lectura de probabilidad del siguiente objetivo de liquidez."""
    side_label = "buy-side" if probability_row["target_side"] == "buy" else "sell-side"
    return (
        f"next {side_label} {probability_row['probability_pct']:.1f}% "
        f"[{probability_row['target_level']:.2f}]"
    )


def _get_visible_window_start(df, candle_window=DISPLAY_CANDLE_WINDOW):
    """Funcion auxiliar que define el indice inicial de la ventana visual del PNG sin alterar el analisis global."""
    if df.empty:
        return 0

    return max(0, len(df) - candle_window)


def _resolve_label_y(target_y, used_y_levels, min_gap, min_y, max_y):
    """Funcion auxiliar que resuelve colisiones verticales entre etiquetas desplazandolas dentro de limites del grafico."""
    if not used_y_levels:
        used_y_levels.append(target_y)
        return target_y

    candidate = target_y
    attempts = 0
    while attempts < 80:
        has_overlap = any(abs(candidate - used_y) < min_gap for used_y in used_y_levels)
        if not has_overlap:
            used_y_levels.append(candidate)
            return candidate

        step = ((attempts // 2) + 1) * min_gap
        direction = 1 if attempts % 2 == 0 else -1
        candidate = target_y + (direction * step)
        candidate = max(min_y, min(max_y, candidate))
        attempts += 1

    used_y_levels.append(candidate)
    return candidate


def _annotate_right_label(
    ax,
    x_anchor,
    y_anchor,
    label_text,
    color,
    used_y_levels,
    y_gap,
    min_y,
    max_y,
    x_offset,
):
    """Funcion auxiliar que dibuja anotaciones laterales con flecha y separacion automatica para reducir traslape."""
    label_y = _resolve_label_y(
        target_y=y_anchor,
        used_y_levels=used_y_levels,
        min_gap=y_gap,
        min_y=min_y,
        max_y=max_y,
    )

    ax.annotate(
        label_text,
        xy=(x_anchor, y_anchor),
        xytext=(x_anchor + x_offset, label_y),
        textcoords="data",
        color=color,
        fontsize=8,
        va="center",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.15",
            "fc": "#ffffff",
            "ec": color,
            "alpha": 0.55,
            "lw": 0.6,
        },
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "lw": 0.8,
            "alpha": 0.9,
            "shrinkA": 2,
            "shrinkB": 2,
        },
        clip_on=False,
    )


def _annotate_structure_label(
    ax,
    x_anchor,
    y_anchor,
    label_text,
    color,
    event_side,
    label_index,
):
    """Funcion auxiliar que anota eventos de estructura con flecha diagonal y offset alternado para legibilidad."""
    base_direction = 1 if event_side == "bullish" else -1
    if label_index % 2 == 1:
        base_direction *= -1

    y_offset_points = 16 * base_direction
    x_offset_points = 12

    ax.annotate(
        label_text,
        xy=(x_anchor, y_anchor),
        xytext=(x_offset_points, y_offset_points),
        textcoords="offset points",
        color=color,
        fontsize=8,
        va="center",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.15",
            "fc": "#ffffff",
            "ec": color,
            "alpha": 0.55,
            "lw": 0.6,
        },
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "lw": 0.8,
            "alpha": 0.9,
            "shrinkA": 2,
            "shrinkB": 2,
        },
        clip_on=False,
    )


def plot_liquidity_chart(
    df,
    liquidity_df,
    fvg_df,
    structure_df,
    dealing_range_df,
    probability_df,
    symbol,
    interval,
    rows=20,
    chart_path=None,
    latest_candle_index=None,
):
    """Funcion principal de visualizacion que renderiza velas, liquidez, FVG, estructura, dealing range y probabilidad en un solo PNG."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("matplotlib no esta instalado; no se pudo generar la grafica.")
        return

    if df.empty:
        print("No hay datos de klines para graficar.")
        return

    visible_start = _get_visible_window_start(df)
    visible_df = df.iloc[visible_start:].reset_index(drop=True)
    x_values = list(range(len(visible_df)))
    label_x_anchor = len(visible_df) - 1
    label_x_offset = max(6, int(len(visible_df) * 0.08))
    price_min = float(visible_df["low"].min())
    price_max = float(visible_df["high"].max())
    price_span = max(price_max - price_min, 1e-6)
    right_label_min_y = price_min - (price_span * 0.10)
    right_label_max_y = price_max + (price_span * 0.10)
    right_label_gap = price_span * 0.025
    right_label_used_y = []

    candle_colors = [
        "#1f9d55" if close_price >= open_price else "#d64545"
        for open_price, close_price in zip(visible_df["open"], visible_df["close"])
    ]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.vlines(
        x_values,
        visible_df["low"],
        visible_df["high"],
        color=candle_colors,
        linewidth=1,
        alpha=0.8,
    )

    body_bottom = visible_df[["open", "close"]].min(axis=1)
    body_height = (visible_df["close"] - visible_df["open"]).abs().clip(lower=0.01)
    ax.bar(
        x_values,
        body_height,
        bottom=body_bottom,
        width=0.6,
        color=candle_colors,
        edgecolor=candle_colors,
        alpha=0.7,
    )
    ax.plot(
        x_values,
        visible_df["close"],
        color="#2b6cb0",
        linewidth=1.2,
        alpha=0.9,
        label="close",
    )

    if not dealing_range_df.empty:
        range_row = _prepare_dealing_range_summary(dealing_range_df).iloc[0]
        range_high = float(range_row["range_high"])
        range_low = float(range_row["range_low"])
        range_mid = float(range_row["range_midpoint"])

        ax.axhspan(range_mid, range_high, color="#c53030", alpha=0.06)
        ax.axhspan(range_low, range_mid, color="#2f855a", alpha=0.06)
        ax.hlines(
            y=range_high,
            xmin=0,
            xmax=len(visible_df) - 1,
            colors="#c53030",
            linestyles="-",
            linewidth=1.0,
            alpha=0.9,
        )
        ax.hlines(
            y=range_mid,
            xmin=0,
            xmax=len(visible_df) - 1,
            colors="#4a5568",
            linestyles="--",
            linewidth=1.0,
            alpha=0.9,
        )
        ax.hlines(
            y=range_low,
            xmin=0,
            xmax=len(visible_df) - 1,
            colors="#2f855a",
            linestyles="-",
            linewidth=1.0,
            alpha=0.9,
        )
        _annotate_right_label(
            ax=ax,
            x_anchor=label_x_anchor,
            y_anchor=range_high,
            label_text=f"DR High [{range_high:.2f}]",
            color="#c53030",
            used_y_levels=right_label_used_y,
            y_gap=right_label_gap,
            min_y=right_label_min_y,
            max_y=right_label_max_y,
            x_offset=label_x_offset,
        )
        _annotate_right_label(
            ax=ax,
            x_anchor=label_x_anchor,
            y_anchor=range_mid,
            label_text=f"EQ [{range_mid:.2f}]",
            color="#4a5568",
            used_y_levels=right_label_used_y,
            y_gap=right_label_gap,
            min_y=right_label_min_y,
            max_y=right_label_max_y,
            x_offset=label_x_offset,
        )
        _annotate_right_label(
            ax=ax,
            x_anchor=label_x_anchor,
            y_anchor=range_low,
            label_text=f"DR Low [{range_low:.2f}]",
            color="#2f855a",
            used_y_levels=right_label_used_y,
            y_gap=right_label_gap,
            min_y=right_label_min_y,
            max_y=right_label_max_y,
            x_offset=label_x_offset,
        )
        ax.text(
            0,
            range_mid,
            _format_dealing_range_label(range_row),
            color="#2d3748",
            fontsize=8,
            ha="left",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.15",
                "fc": "#ffffff",
                "ec": "#2d3748",
                "alpha": 0.45,
                "lw": 0.5,
            },
        )

    if not structure_df.empty:
        structure_to_plot = _prepare_structure_summary(
            structure_df,
            latest_candle_index=latest_candle_index,
        ).head(rows)
        for label_index, (_, structure_row) in enumerate(structure_to_plot.iterrows()):
            marker_index = int(structure_row["candle_index"])
            if marker_index < visible_start:
                continue

            marker_color = "#2f855a" if structure_row["event_side"] == "bullish" else "#c53030"
            marker_style = "^" if structure_row["event_side"] == "bullish" else "v"
            marker_size = 75 if bool(structure_row["is_mss"]) else 55
            marker_alpha = 1.0 if bool(structure_row["volume_confirmed"]) else 0.55
            visible_marker_index = marker_index - visible_start
            marker_price = float(structure_row["event_price"])
            ax.scatter(
                visible_marker_index,
                marker_price,
                color=marker_color,
                marker=marker_style,
                s=marker_size,
                alpha=marker_alpha,
                zorder=5,
            )
            _annotate_structure_label(
                ax=ax,
                x_anchor=visible_marker_index,
                y_anchor=marker_price,
                label_text=_format_structure_label(structure_row),
                color=marker_color,
                event_side=structure_row["event_side"],
                label_index=label_index,
            )

    if not fvg_df.empty:
        fvgs_to_plot = _prepare_fvg_summary(
            fvg_df,
            latest_candle_index=latest_candle_index,
        ).head(rows)
        for _, fvg_row in fvgs_to_plot.iterrows():
            start_index = max(0, int(fvg_row["candle_index"]) - visible_start)
            zone_width = max(1, len(visible_df) - start_index - 1)
            zone_height = max(0.01, float(fvg_row["fvg_high"]) - float(fvg_row["fvg_low"]))
            zone_color = "#38a169" if fvg_row["fvg_side"] == "bullish" else "#c05621"
            zone_alpha = 0.12 if not bool(fvg_row["filled"]) else 0.07
            zone = Rectangle(
                (start_index, float(fvg_row["fvg_low"])),
                zone_width,
                zone_height,
                linewidth=1.0,
                edgecolor=zone_color,
                facecolor=zone_color,
                alpha=zone_alpha,
            )
            ax.add_patch(zone)
            _annotate_right_label(
                ax=ax,
                x_anchor=label_x_anchor,
                y_anchor=float(fvg_row["fvg_midpoint"]),
                label_text=_format_fvg_label(fvg_row),
                color=zone_color,
                used_y_levels=right_label_used_y,
                y_gap=right_label_gap,
                min_y=right_label_min_y,
                max_y=right_label_max_y,
                x_offset=label_x_offset,
            )

    if not liquidity_df.empty:
        levels_to_plot = _prepare_display_summary(
            liquidity_df,
            latest_candle_index=latest_candle_index,
        ).head(rows)
        for _, level_row in levels_to_plot.iterrows():
            start_index = max(0, int(level_row["candle_index"]) - visible_start)
            line_color = "#d69e2e" if level_row["liquidity_side"] == "buy" else "#319795"
            line_style = "-" if level_row["liquidity_rank"] == "major" else "--"
            ax.hlines(
                y=level_row["liquidity_level"],
                xmin=start_index,
                xmax=len(visible_df) - 1,
                colors=line_color,
                linestyles=line_style,
                linewidth=1.3,
                alpha=0.9,
            )
            _annotate_right_label(
                ax=ax,
                x_anchor=label_x_anchor,
                y_anchor=float(level_row["liquidity_level"]),
                label_text=_format_liquidity_label(level_row),
                color=line_color,
                used_y_levels=right_label_used_y,
                y_gap=right_label_gap,
                min_y=right_label_min_y,
                max_y=right_label_max_y,
                x_offset=label_x_offset,
            )

    if not probability_df.empty:
        probability_to_plot = _prepare_probability_summary(probability_df)
        for _, probability_row in probability_to_plot.iterrows():
            target_level = float(probability_row["target_level"])
            probability_color = "#d69e2e" if probability_row["target_side"] == "buy" else "#319795"
            _annotate_right_label(
                ax=ax,
                x_anchor=label_x_anchor,
                y_anchor=target_level,
                label_text=_format_probability_label(probability_row),
                color=probability_color,
                used_y_levels=right_label_used_y,
                y_gap=right_label_gap,
                min_y=right_label_min_y,
                max_y=right_label_max_y,
                x_offset=label_x_offset + 6,
            )

        best_probability = probability_to_plot.iloc[0]
        bias_text = (
            f"Immediate liquidity bias: {best_probability['target_side']} "
            f"{best_probability['probability_pct']:.1f}%"
        )
        ax.text(
            0.01,
            0.98,
            bias_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#1a202c",
            bbox={
                "boxstyle": "round,pad=0.2",
                "fc": "#ffffff",
                "ec": "#1a202c",
                "alpha": 0.65,
                "lw": 0.7,
            },
        )

    tick_step = max(1, len(visible_df) // 10)
    tick_positions = x_values[::tick_step]
    tick_labels = [
        visible_df["time"].iloc[idx].strftime("%m-%d %H:%M") for idx in tick_positions
    ]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_xlim(-1, label_x_anchor + label_x_offset + 2)
    ax.set_title(
        f"{symbol} {interval} | Last {len(visible_df)} Klines + SMC Liquidity + FVG + DR + Probability"
    )
    ax.set_xlabel("Time (UTC-5)")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2)
    ax.margins(x=0.02)
    plt.tight_layout()

    if chart_path is None:
        chart_path = f"smc_liquidity_chart_{symbol}_{interval}.png"

    output_path = Path(chart_path)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafica guardada en: {output_path}")


def parse_args():
    """Funcion que define y parsea argumentos CLI para controlar simbolo, intervalo, sensibilidad y salida de grafica."""
    parser = argparse.ArgumentParser(
        description="Detecta puntos basicos de liquidez SMC sobre klines de Binance."
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--swing-window", type=int, default=2)
    parser.add_argument("--major-window", type=int, default=5)
    parser.add_argument("--tolerance-pct", type=float, default=0.001)
    parser.add_argument(
        "--rows",
        type=int,
        default=20,
        help="Numero de filas finales a imprimir.",
    )
    parser.add_argument(
        "--chart-path",
        default=None,
        help="Ruta del archivo PNG donde se guardara la grafica.",
    )
    return parser.parse_args()


def main():
    """Funcion principal que ejecuta todo el flujo: descarga datos, calcula tablas SMC, imprime resumenes y genera grafica."""
    args = parse_args()
    df = get_klines(
        symbol=args.symbol,
        interval=args.interval,
        limit=args.limit,
    )
    liquidity = build_smc_liquidity_table(
        df,
        swing_window=args.swing_window,
        major_window=args.major_window,
        tolerance_pct=args.tolerance_pct,
    )
    fvg = build_fvg_table(
        df,
        min_gap_pct=MIN_FVG_GAP_PCT,
    )
    structure = build_market_structure_table(
        df,
        swing_window=args.swing_window,
        min_break_pct=STRUCTURE_MIN_BREAK_PCT,
        min_body_ratio=STRUCTURE_MIN_BODY_RATIO,
        min_close_position=STRUCTURE_MIN_CLOSE_POSITION,
        volume_window=STRUCTURE_VOLUME_WINDOW,
        min_volume_ratio=STRUCTURE_MIN_VOLUME_RATIO,
        require_volume_confirmation=STRUCTURE_REQUIRE_VOLUME_CONFIRMATION,
    )
    dealing_range = build_dealing_range_table(
        df,
        swing_window=args.swing_window,
        equilibrium_band_pct=DEALING_RANGE_EQ_BAND_PCT,
    )
    probability = build_immediate_liquidity_probability_table(
        df,
        liquidity_df=liquidity,
        structure_df=structure,
        fvg_df=fvg,
        dealing_range_df=dealing_range,
        candle_lookback=RECENT_CANDLE_LOOKBACK,
    )
    if not liquidity.empty:
        liquidity.insert(0, "symbol", args.symbol)
        liquidity.insert(1, "interval", args.interval)
    if not fvg.empty:
        fvg.insert(0, "symbol", args.symbol)
        fvg.insert(1, "interval", args.interval)
    if not structure.empty:
        structure.insert(0, "symbol", args.symbol)
        structure.insert(1, "interval", args.interval)
    if not dealing_range.empty:
        dealing_range.insert(0, "symbol", args.symbol)
        dealing_range.insert(1, "interval", args.interval)

    latest_candle_index = len(df) - 1
    print("\nImmediate Liquidity Probability")
    print_probability_summary(probability)
    print("\nDealing Range (Premium/Discount)")
    print_dealing_range_summary(dealing_range)
    print("\nMarket Structure")
    print_structure_summary(
        structure,
        rows=args.rows,
        latest_candle_index=latest_candle_index,
    )
    print("\nLiquidity Levels")
    print_liquidity_summary(
        liquidity,
        rows=args.rows,
        latest_candle_index=latest_candle_index,
    )
    print("\nFair Value Gaps")
    print_fvg_summary(
        fvg,
        rows=args.rows,
        latest_candle_index=latest_candle_index,
    )
    plot_liquidity_chart(
        df,
        liquidity,
        fvg,
        structure,
        dealing_range,
        probability,
        symbol=args.symbol,
        interval=args.interval,
        rows=args.rows,
        chart_path=args.chart_path,
        latest_candle_index=latest_candle_index,
    )


if __name__ == "__main__":
    main()
