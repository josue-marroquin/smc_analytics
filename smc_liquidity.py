import numpy as np
import pandas as pd


LIQUIDITY_COLUMNS = [
    "candle_index",
    "liquidity_time",
    "liquidity_level",
    "liquidity_side",
    "liquidity_rank",
    "liquidity_label",
    "reference_high",
    "reference_low",
    "equal_level_touches",
    "swept",
    "swept_time",
    "distance_from_close_pct",
]
FVG_COLUMNS = [
    "candle_index",
    "fvg_time",
    "fvg_side",
    "fvg_low",
    "fvg_high",
    "fvg_midpoint",
    "gap_size",
    "gap_size_pct",
    "filled",
    "filled_time",
    "distance_from_close_pct",
    "fvg_label",
]
MARKET_STRUCTURE_COLUMNS = [
    "candle_index",
    "event_time",
    "event_price",
    "event_side",
    "event_kind",
    "is_mss",
    "broken_level",
    "broken_liquidity_time",
    "protected_level",
    "protected_level_time",
    "break_buffer_pct",
    "volume",
    "volume_sma",
    "volume_ratio",
    "volume_zscore",
    "volume_confirmed",
    "trend_after",
    "distance_from_close_pct",
    "structure_label",
]
DEALING_RANGE_COLUMNS = [
    "candle_index",
    "range_time",
    "range_high",
    "range_low",
    "range_midpoint",
    "current_price",
    "price_position_pct",
    "zone",
    "premium_pct",
    "discount_pct",
    "range_size",
    "range_size_pct",
    "range_high_time",
    "range_low_time",
    "distance_from_eq_pct",
    "dealing_range_label",
]
LIQUIDITY_PROBABILITY_COLUMNS = [
    "target_side",
    "target_level",
    "target_rank",
    "target_time",
    "target_distance_pct",
    "current_price",
    "structure_bias",
    "dealing_range_bias",
    "fvg_bias",
    "distance_score",
    "liquidity_score",
    "raw_score",
    "probability_pct",
    "target_label",
]

LIQUIDITY_SIDE_ORDER = ("buy", "sell")
LIQUIDITY_RANK_ORDER = ("major", "minor")
FVG_SIDE_ORDER = ("bullish", "bearish")
STRUCTURE_SIDE_ORDER = ("bullish", "bearish")


def _empty_liquidity_table():
    """Funcion auxiliar que crea una tabla vacia con el esquema esperado para niveles de liquidez."""
    return pd.DataFrame(columns=LIQUIDITY_COLUMNS)


def _empty_fvg_table():
    """Funcion auxiliar que crea una tabla vacia con el esquema esperado para Fair Value Gaps."""
    return pd.DataFrame(columns=FVG_COLUMNS)


def _empty_market_structure_table():
    """Funcion auxiliar que crea una tabla vacia con el esquema esperado para eventos de estructura."""
    return pd.DataFrame(columns=MARKET_STRUCTURE_COLUMNS)


def _empty_dealing_range_table():
    """Funcion auxiliar que crea una tabla vacia con el esquema esperado para dealing range y premium/discount."""
    return pd.DataFrame(columns=DEALING_RANGE_COLUMNS)


def _empty_liquidity_probability_table():
    """Funcion auxiliar que crea una tabla vacia con el esquema esperado para probabilidades de tomar liquidez inmediata."""
    return pd.DataFrame(columns=LIQUIDITY_PROBABILITY_COLUMNS)


def _is_swing_high(highs, index, left, right):
    """Funcion auxiliar que valida si un punto es swing high comparando su maximo contra velas vecinas izquierda/derecha."""
    if index < left or index + right >= len(highs):
        return False

    current = highs[index]
    return np.all(current > highs[index - left:index]) and np.all(
        current >= highs[index + 1:index + 1 + right]
    )


def _is_swing_low(lows, index, left, right):
    """Funcion auxiliar que valida si un punto es swing low comparando su minimo contra velas vecinas izquierda/derecha."""
    if index < left or index + right >= len(lows):
        return False

    current = lows[index]
    return np.all(current < lows[index - left:index]) and np.all(
        current <= lows[index + 1:index + 1 + right]
    )


def find_swing_points(df, swing_window=2):
    """Funcion que etiqueta cada vela con banderas de swing high/swing low para construir estructura del mercado."""
    if df.empty or len(df) < (swing_window * 2) + 1:
        result = df.copy()
        result["swing_high"] = False
        result["swing_low"] = False
        return result

    result = df.copy()
    highs = result["high"].to_numpy()
    lows = result["low"].to_numpy()

    result["swing_high"] = [
        _is_swing_high(highs, idx, swing_window, swing_window)
        for idx in range(len(result))
    ]
    result["swing_low"] = [
        _is_swing_low(lows, idx, swing_window, swing_window)
        for idx in range(len(result))
    ]
    return result


def _detect_sweep(df, row_position, side, level):
    """Funcion auxiliar que detecta si un nivel de liquidez fue barrido por velas futuras y devuelve su primer timestamp."""
    future = df.iloc[row_position + 1 :]
    if future.empty:
        return False, pd.NaT

    if side == "buy":
        sweeps = future.loc[future["high"] > level, "time"]
    else:
        sweeps = future.loc[future["low"] < level, "time"]

    if sweeps.empty:
        return False, pd.NaT

    return True, sweeps.iloc[0]


def _touch_count(previous_levels, current_level, tolerance_pct):
    """Funcion auxiliar que cuenta toques en niveles similares para estimar densidad de liquidez en un rango de tolerancia."""
    if not previous_levels:
        return 1

    tolerance = current_level * tolerance_pct
    similar_levels = [
        level for level in previous_levels if abs(level - current_level) <= tolerance
    ]
    return 1 + len(similar_levels)


def _detect_fvg_fill(df, row_position, side, lower_bound, upper_bound):
    """Funcion auxiliar que verifica si un FVG fue llenado totalmente despues de su creacion y retorna el primer llenado."""
    future = df.iloc[row_position + 1 :]
    if future.empty:
        return False, pd.NaT

    if side == "bullish":
        fills = future.loc[future["low"] <= lower_bound, "time"]
    else:
        fills = future.loc[future["high"] >= upper_bound, "time"]

    if fills.empty:
        return False, pd.NaT

    return True, fills.iloc[0]


def _confirmed_break_metrics(
    row,
    side,
    level,
    min_break_pct=0.05,
    min_body_ratio=0.5,
    min_close_position=0.6,
):
    """Funcion auxiliar que confirma ruptura estructural con calidad de vela (cierre, cuerpo, posicion y buffer minimo)."""
    if level is None:
        return False, np.nan

    candle_range = max(float(row["high"]) - float(row["low"]), 1e-9)
    candle_body = abs(float(row["close"]) - float(row["open"]))
    body_ratio = candle_body / candle_range

    if side == "bullish":
        break_buffer_pct = ((float(row["close"]) - level) / level) * 100 if level else np.nan
        close_position = (float(row["close"]) - float(row["low"])) / candle_range
        is_confirmed = (
            float(row["close"]) > level
            and float(row["close"]) > float(row["open"])
            and body_ratio >= min_body_ratio
            and close_position >= min_close_position
            and break_buffer_pct >= min_break_pct
        )
    else:
        break_buffer_pct = ((level - float(row["close"])) / level) * 100 if level else np.nan
        close_position = (float(row["high"]) - float(row["close"])) / candle_range
        is_confirmed = (
            float(row["close"]) < level
            and float(row["close"]) < float(row["open"])
            and body_ratio >= min_body_ratio
            and close_position >= min_close_position
            and break_buffer_pct >= min_break_pct
        )

    return is_confirmed, break_buffer_pct


def _latest_swing(levels):
    """Funcion auxiliar que devuelve el swing mas reciente disponible o None si no existe contexto."""
    if not levels:
        return None

    return levels[-1]


def _add_volume_features(df, volume_window=20, min_volume_ratio=1.2):
    """Funcion auxiliar que agrega indicadores de volumen (SMA, ratio, zscore y confirmacion) sobre un DataFrame de velas."""
    result = df.copy()
    rolling_volume = result["volume"].rolling(volume_window)
    result["volume_sma"] = rolling_volume.mean()
    result["volume_std"] = rolling_volume.std(ddof=0).replace(0, np.nan)
    result["volume_ratio"] = result["volume"] / result["volume_sma"]
    result["volume_zscore"] = (
        (result["volume"] - result["volume_sma"]) / result["volume_std"]
    )
    result["volume_confirmed"] = result["volume_ratio"] >= min_volume_ratio
    return result


def build_smc_liquidity_table(
    df,
    swing_window=2,
    major_window=5,
    tolerance_pct=0.001,
):
    """Funcion que construye la tabla de liquidez SMC (major/minor buy/sell), incluyendo barridos y distancia al precio actual."""
    required_columns = {"time", "high", "low", "close"}
    if df.empty or not required_columns.issubset(df.columns):
        return _empty_liquidity_table()

    swings = find_swing_points(df, swing_window=swing_window)
    records = []
    buy_levels = []
    sell_levels = []
    latest_close = float(swings["close"].iloc[-1])

    for row_position, (_, row) in enumerate(swings.iterrows()):
        candidates = []
        if bool(row["swing_high"]):
            candidates.append(("buy", float(row["high"]), buy_levels))
        if bool(row["swing_low"]):
            candidates.append(("sell", float(row["low"]), sell_levels))

        for side, level, known_levels in candidates:
            recent_levels = known_levels[-major_window:] if known_levels else []
            if side == "buy":
                is_major = not recent_levels or level >= max(recent_levels)
            else:
                is_major = not recent_levels or level <= min(recent_levels)

            rank = "major" if is_major else "minor"
            swept, swept_time = _detect_sweep(swings, row_position, side, level)
            touches = _touch_count(recent_levels, level, tolerance_pct)
            distance_pct = ((level - latest_close) / latest_close) * 100

            records.append(
                {
                    "candle_index": row_position,
                    "liquidity_time": row["time"],
                    "liquidity_level": level,
                    "liquidity_side": side,
                    "liquidity_rank": rank,
                    "liquidity_label": f"{rank}_{side}_side_liquidity",
                    "reference_high": float(row["high"]),
                    "reference_low": float(row["low"]),
                    "equal_level_touches": touches,
                    "swept": swept,
                    "swept_time": swept_time,
                    "distance_from_close_pct": distance_pct,
                }
            )

            known_levels.append(level)

    if not records:
        return _empty_liquidity_table()

    liquidity_df = pd.DataFrame.from_records(records, columns=LIQUIDITY_COLUMNS)
    return liquidity_df.sort_values("liquidity_time").reset_index(drop=True)


def build_fvg_table(df, min_gap_pct=0.0):
    """Funcion que detecta Fair Value Gaps alcistas/bajistas, calcula su tamano y marca si el gap fue llenado."""
    required_columns = {"time", "high", "low", "close"}
    if len(df) < 3 or not required_columns.issubset(df.columns):
        return _empty_fvg_table()

    records = []
    latest_close = float(df["close"].iloc[-1])

    for row_position in range(2, len(df)):
        first = df.iloc[row_position - 2]
        third = df.iloc[row_position]

        bullish_gap = float(third["low"]) - float(first["high"])
        if bullish_gap > 0:
            lower_bound = float(first["high"])
            upper_bound = float(third["low"])
            midpoint = (lower_bound + upper_bound) / 2
            gap_pct = (bullish_gap / lower_bound) * 100 if lower_bound else np.nan
            if gap_pct >= min_gap_pct:
                filled, filled_time = _detect_fvg_fill(
                    df,
                    row_position,
                    "bullish",
                    lower_bound,
                    upper_bound,
                )
                records.append(
                    {
                        "candle_index": row_position,
                        "fvg_time": third["time"],
                        "fvg_side": "bullish",
                        "fvg_low": lower_bound,
                        "fvg_high": upper_bound,
                        "fvg_midpoint": midpoint,
                        "gap_size": bullish_gap,
                        "gap_size_pct": gap_pct,
                        "filled": filled,
                        "filled_time": filled_time,
                        "distance_from_close_pct": (
                            (midpoint - latest_close) / latest_close
                        )
                        * 100,
                        "fvg_label": "bullish_fvg",
                    }
                )

        bearish_gap = float(first["low"]) - float(third["high"])
        if bearish_gap > 0:
            lower_bound = float(third["high"])
            upper_bound = float(first["low"])
            midpoint = (lower_bound + upper_bound) / 2
            gap_pct = (bearish_gap / upper_bound) * 100 if upper_bound else np.nan
            if gap_pct >= min_gap_pct:
                filled, filled_time = _detect_fvg_fill(
                    df,
                    row_position,
                    "bearish",
                    lower_bound,
                    upper_bound,
                )
                records.append(
                    {
                        "candle_index": row_position,
                        "fvg_time": third["time"],
                        "fvg_side": "bearish",
                        "fvg_low": lower_bound,
                        "fvg_high": upper_bound,
                        "fvg_midpoint": midpoint,
                        "gap_size": bearish_gap,
                        "gap_size_pct": gap_pct,
                        "filled": filled,
                        "filled_time": filled_time,
                        "distance_from_close_pct": (
                            (midpoint - latest_close) / latest_close
                        )
                        * 100,
                        "fvg_label": "bearish_fvg",
                    }
                )

    if not records:
        return _empty_fvg_table()

    return (
        pd.DataFrame.from_records(records, columns=FVG_COLUMNS)
        .sort_values("fvg_time")
        .reset_index(drop=True)
    )


def build_market_structure_table(
    df,
    swing_window=2,
    min_break_pct=0.05,
    min_body_ratio=0.5,
    min_close_position=0.6,
    volume_window=20,
    min_volume_ratio=1.2,
    require_volume_confirmation=False,
):
    """Funcion que detecta eventos BOS/CHoCH/MSS con niveles protegidos y filtro opcional por confirmacion de volumen."""
    required_columns = {"time", "open", "high", "low", "close", "volume"}
    if df.empty or not required_columns.issubset(df.columns):
        return _empty_market_structure_table()

    swings = find_swing_points(df, swing_window=swing_window)
    swings = _add_volume_features(
        swings,
        volume_window=volume_window,
        min_volume_ratio=min_volume_ratio,
    )
    records = []
    swing_highs = []
    swing_lows = []
    current_trend = None
    protected_high = None
    protected_low = None
    latest_close = float(swings["close"].iloc[-1])

    for row_position, (_, row) in enumerate(swings.iterrows()):
        close_price = float(row["close"])
        volume_confirmed = bool(row["volume_confirmed"]) if not pd.isna(row["volume_confirmed"]) else False
        volume_pass = (not require_volume_confirmation) or volume_confirmed

        latest_high = _latest_swing(swing_highs)
        latest_low = _latest_swing(swing_lows)

        bullish_reference = protected_high if current_trend == "bearish" else latest_high
        bearish_reference = protected_low if current_trend == "bullish" else latest_low

        bullish_break_confirmed, bullish_break_buffer = _confirmed_break_metrics(
            row,
            "bullish",
            None if bullish_reference is None else bullish_reference["level"],
            min_break_pct=min_break_pct,
            min_body_ratio=min_body_ratio,
            min_close_position=min_close_position,
        )
        bearish_break_confirmed, bearish_break_buffer = _confirmed_break_metrics(
            row,
            "bearish",
            None if bearish_reference is None else bearish_reference["level"],
            min_break_pct=min_break_pct,
            min_body_ratio=min_body_ratio,
            min_close_position=min_close_position,
        )

        if current_trend is None:
            if bullish_break_confirmed and bullish_reference is not None and volume_pass:
                bullish_reference["broken"] = True
                current_trend = "bullish"
                protected_low = latest_low
                records.append(
                    {
                        "candle_index": row_position,
                        "event_time": row["time"],
                        "event_price": close_price,
                        "event_side": "bullish",
                        "event_kind": "BOS",
                        "is_mss": False,
                        "broken_level": bullish_reference["level"],
                        "broken_liquidity_time": bullish_reference["time"],
                        "protected_level": (
                            np.nan if protected_low is None else protected_low["level"]
                        ),
                        "protected_level_time": (
                            pd.NaT if protected_low is None else protected_low["time"]
                        ),
                        "break_buffer_pct": bullish_break_buffer,
                        "trend_after": current_trend,
                        "distance_from_close_pct": (
                            (close_price - latest_close) / latest_close
                        )
                        * 100,
                        "structure_label": "bullish_bos",
                    }
                )
            elif bearish_break_confirmed and bearish_reference is not None and volume_pass:
                bearish_reference["broken"] = True
                current_trend = "bearish"
                protected_high = latest_high
                records.append(
                    {
                        "candle_index": row_position,
                        "event_time": row["time"],
                        "event_price": close_price,
                        "event_side": "bearish",
                        "event_kind": "BOS",
                        "is_mss": False,
                        "broken_level": bearish_reference["level"],
                        "broken_liquidity_time": bearish_reference["time"],
                        "protected_level": (
                            np.nan if protected_high is None else protected_high["level"]
                        ),
                        "protected_level_time": (
                            pd.NaT if protected_high is None else protected_high["time"]
                        ),
                        "break_buffer_pct": bearish_break_buffer,
                        "trend_after": current_trend,
                        "distance_from_close_pct": (
                            (close_price - latest_close) / latest_close
                        )
                        * 100,
                        "structure_label": "bearish_bos",
                    }
                )

        elif current_trend == "bullish":
            if bearish_break_confirmed and protected_low is not None and volume_pass:
                protected_low["broken"] = True
                current_trend = "bearish"
                protected_high = latest_high
                records.append(
                    {
                        "candle_index": row_position,
                        "event_time": row["time"],
                        "event_price": close_price,
                        "event_side": "bearish",
                        "event_kind": "CHoCH",
                        "is_mss": True,
                        "broken_level": protected_low["level"],
                        "broken_liquidity_time": protected_low["time"],
                        "protected_level": (
                            np.nan if protected_high is None else protected_high["level"]
                        ),
                        "protected_level_time": (
                            pd.NaT if protected_high is None else protected_high["time"]
                        ),
                        "break_buffer_pct": bearish_break_buffer,
                        "trend_after": current_trend,
                        "distance_from_close_pct": (
                            (close_price - latest_close) / latest_close
                        )
                        * 100,
                        "structure_label": "bearish_choch_mss",
                    }
                )
            elif bullish_break_confirmed and latest_high is not None and volume_pass:
                latest_high["broken"] = True
                if latest_low is not None:
                    protected_low = latest_low
                records.append(
                    {
                        "candle_index": row_position,
                        "event_time": row["time"],
                        "event_price": close_price,
                        "event_side": "bullish",
                        "event_kind": "BOS",
                        "is_mss": False,
                        "broken_level": latest_high["level"],
                        "broken_liquidity_time": latest_high["time"],
                        "protected_level": (
                            np.nan if protected_low is None else protected_low["level"]
                        ),
                        "protected_level_time": (
                            pd.NaT if protected_low is None else protected_low["time"]
                        ),
                        "break_buffer_pct": bullish_break_buffer,
                        "trend_after": current_trend,
                        "distance_from_close_pct": (
                            (close_price - latest_close) / latest_close
                        )
                        * 100,
                        "structure_label": "bullish_bos",
                    }
                )

        else:
            if bullish_break_confirmed and protected_high is not None and volume_pass:
                protected_high["broken"] = True
                current_trend = "bullish"
                protected_low = latest_low
                records.append(
                    {
                        "candle_index": row_position,
                        "event_time": row["time"],
                        "event_price": close_price,
                        "event_side": "bullish",
                        "event_kind": "CHoCH",
                        "is_mss": True,
                        "broken_level": protected_high["level"],
                        "broken_liquidity_time": protected_high["time"],
                        "protected_level": (
                            np.nan if protected_low is None else protected_low["level"]
                        ),
                        "protected_level_time": (
                            pd.NaT if protected_low is None else protected_low["time"]
                        ),
                        "break_buffer_pct": bullish_break_buffer,
                        "trend_after": current_trend,
                        "distance_from_close_pct": (
                            (close_price - latest_close) / latest_close
                        )
                        * 100,
                        "structure_label": "bullish_choch_mss",
                    }
                )
            elif bearish_break_confirmed and latest_low is not None and volume_pass:
                latest_low["broken"] = True
                if latest_high is not None:
                    protected_high = latest_high
                records.append(
                    {
                        "candle_index": row_position,
                        "event_time": row["time"],
                        "event_price": close_price,
                        "event_side": "bearish",
                        "event_kind": "BOS",
                        "is_mss": False,
                        "broken_level": latest_low["level"],
                        "broken_liquidity_time": latest_low["time"],
                        "protected_level": (
                            np.nan if protected_high is None else protected_high["level"]
                        ),
                        "protected_level_time": (
                            pd.NaT if protected_high is None else protected_high["time"]
                        ),
                        "break_buffer_pct": bearish_break_buffer,
                        "trend_after": current_trend,
                        "distance_from_close_pct": (
                            (close_price - latest_close) / latest_close
                        )
                        * 100,
                        "structure_label": "bearish_bos",
                    }
                )

        if bool(row["swing_high"]):
            swing_highs.append(
                {
                    "index": row_position,
                    "time": row["time"],
                    "level": float(row["high"]),
                    "broken": False,
                }
            )

        if bool(row["swing_low"]):
            swing_lows.append(
                {
                    "index": row_position,
                    "time": row["time"],
                    "level": float(row["low"]),
                    "broken": False,
                }
            )

    if not records:
        return _empty_market_structure_table()

    structure_df = pd.DataFrame.from_records(records, columns=MARKET_STRUCTURE_COLUMNS)

    volume_fields = ["volume", "volume_sma", "volume_ratio", "volume_zscore", "volume_confirmed"]
    for field in volume_fields:
        structure_df[field] = structure_df["candle_index"].map(swings[field])

    structure_df["volume_confirmed"] = (
        structure_df["volume_confirmed"].fillna(False).astype(bool)
    )

    return structure_df.sort_values("event_time").reset_index(drop=True)


def build_dealing_range_table(df, swing_window=2, equilibrium_band_pct=5.0):
    """Funcion que calcula el dealing range activo y clasifica el precio actual en premium, discount o equilibrium."""
    required_columns = {"time", "high", "low", "close"}
    if df.empty or not required_columns.issubset(df.columns):
        return _empty_dealing_range_table()

    swings = find_swing_points(df, swing_window=swing_window)
    records = []
    latest_swing_high = None
    latest_swing_low = None

    lower_equilibrium = 50 - (equilibrium_band_pct / 2)
    upper_equilibrium = 50 + (equilibrium_band_pct / 2)

    for row_position, (_, row) in enumerate(swings.iterrows()):
        if bool(row["swing_high"]):
            latest_swing_high = {
                "time": row["time"],
                "level": float(row["high"]),
            }
        if bool(row["swing_low"]):
            latest_swing_low = {
                "time": row["time"],
                "level": float(row["low"]),
            }

        if latest_swing_high is None or latest_swing_low is None:
            continue

        range_high = float(latest_swing_high["level"])
        range_low = float(latest_swing_low["level"])
        if range_high <= range_low:
            continue

        range_size = range_high - range_low
        midpoint = (range_high + range_low) / 2
        current_price = float(row["close"])
        price_position_pct = ((current_price - range_low) / range_size) * 100
        distance_from_eq_pct = ((current_price - midpoint) / midpoint) * 100

        if price_position_pct < lower_equilibrium:
            zone = "discount"
        elif price_position_pct > upper_equilibrium:
            zone = "premium"
        else:
            zone = "equilibrium"

        records.append(
            {
                "candle_index": row_position,
                "range_time": row["time"],
                "range_high": range_high,
                "range_low": range_low,
                "range_midpoint": midpoint,
                "current_price": current_price,
                "price_position_pct": price_position_pct,
                "zone": zone,
                "premium_pct": max(0.0, price_position_pct - 50.0),
                "discount_pct": max(0.0, 50.0 - price_position_pct),
                "range_size": range_size,
                "range_size_pct": (range_size / midpoint) * 100,
                "range_high_time": latest_swing_high["time"],
                "range_low_time": latest_swing_low["time"],
                "distance_from_eq_pct": distance_from_eq_pct,
                "dealing_range_label": f"{zone}_zone",
            }
        )

    if not records:
        return _empty_dealing_range_table()

    return (
        pd.DataFrame.from_records(records, columns=DEALING_RANGE_COLUMNS)
        .sort_values("range_time")
        .reset_index(drop=True)
    )


def _latest_non_empty_row(source_df, time_column):
    """Funcion auxiliar que devuelve la fila mas reciente de una tabla basada en su columna temporal."""
    if source_df.empty:
        return None

    return source_df.sort_values(time_column).iloc[-1]


def _recent_rows_by_candle_index(source_df, latest_candle_index, candle_lookback):
    """Funcion auxiliar que filtra una tabla por una ventana reciente de velas manteniendo el contexto operativo actual."""
    if source_df.empty or candle_lookback is None:
        return source_df.copy()

    min_candle_index = max(0, latest_candle_index - candle_lookback)
    recent_rows = source_df.loc[source_df["candle_index"] >= min_candle_index]
    if recent_rows.empty:
        return source_df.copy()

    return recent_rows.copy()


def _nearest_liquidity_target(liquidity_df, side, current_price):
    """Funcion auxiliar que obtiene el nivel de liquidez no barrido mas cercano por encima o debajo del precio actual."""
    if liquidity_df.empty:
        return None

    unswept_levels = liquidity_df.loc[~liquidity_df["swept"]].copy()
    if unswept_levels.empty:
        unswept_levels = liquidity_df.copy()

    if side == "buy":
        candidates = unswept_levels.loc[unswept_levels["liquidity_level"] > current_price].copy()
        if candidates.empty:
            return None
        candidates["target_distance_pct"] = (
            (candidates["liquidity_level"] - current_price) / current_price
        ) * 100
    else:
        candidates = unswept_levels.loc[unswept_levels["liquidity_level"] < current_price].copy()
        if candidates.empty:
            return None
        candidates["target_distance_pct"] = (
            (current_price - candidates["liquidity_level"]) / current_price
        ) * 100

    return (
        candidates.sort_values(
            by=["target_distance_pct", "liquidity_time"],
            ascending=[True, False],
        )
        .iloc[0]
    )


def _context_bias_from_structure(latest_structure_row, side):
    """Funcion auxiliar que transforma el ultimo evento estructural en un sesgo direccional cuantificable."""
    if latest_structure_row is None:
        return 0.0

    row_side = latest_structure_row["event_side"]
    event_kind = latest_structure_row["event_kind"]
    is_mss = bool(latest_structure_row["is_mss"])
    volume_confirmed = bool(latest_structure_row.get("volume_confirmed", False))

    if row_side == side:
        bias = 1.1 if event_kind == "BOS" else 1.4
        if is_mss:
            bias += 0.3
        if volume_confirmed:
            bias += 0.25
        return bias

    bias = -0.9 if event_kind == "BOS" else -1.2
    if is_mss:
        bias -= 0.2
    if volume_confirmed:
        bias -= 0.15
    return bias


def _context_bias_from_dealing_range(latest_range_row, side):
    """Funcion auxiliar que convierte la zona premium/discount actual en sesgo de probabilidad hacia el siguiente objetivo."""
    if latest_range_row is None:
        return 0.0

    zone = latest_range_row["zone"]
    if zone == "discount":
        return 0.9 if side == "buy" else -0.45
    if zone == "premium":
        return 0.9 if side == "sell" else -0.45
    return 0.15


def _context_bias_from_fvg(fvg_df, current_price, side):
    """Funcion auxiliar que mide soporte direccional usando los FVG recientes y su ubicacion respecto al precio actual."""
    if fvg_df.empty:
        return 0.0

    open_fvgs = fvg_df.loc[~fvg_df["filled"]].copy()
    if open_fvgs.empty:
        open_fvgs = fvg_df.copy()

    if side == "buy":
        relevant = open_fvgs.loc[
            (open_fvgs["fvg_side"] == "bullish") & (open_fvgs["fvg_midpoint"] <= current_price)
        ].copy()
    else:
        relevant = open_fvgs.loc[
            (open_fvgs["fvg_side"] == "bearish") & (open_fvgs["fvg_midpoint"] >= current_price)
        ].copy()

    if relevant.empty:
        return 0.0

    relevant["distance_pct"] = ((relevant["fvg_midpoint"] - current_price).abs() / current_price) * 100
    nearest_fvg = relevant.sort_values("distance_pct", ascending=True).iloc[0]
    distance_bias = float(np.exp(-nearest_fvg["distance_pct"] / 0.8))
    size_bias = min(float(nearest_fvg["gap_size_pct"]), 1.5) / 1.5
    return 0.35 + (0.45 * distance_bias) + (0.25 * size_bias)


def build_immediate_liquidity_probability_table(
    df,
    liquidity_df,
    structure_df,
    fvg_df,
    dealing_range_df,
    candle_lookback=120,
):
    """Funcion que estima la probabilidad relativa de tomar la liquidez inmediata superior o inferior usando confluencias SMC."""
    required_columns = {"close", "high", "low"}
    if df.empty or not required_columns.issubset(df.columns):
        return _empty_liquidity_probability_table()

    latest_candle_index = len(df) - 1
    current_price = float(df["close"].iloc[-1])
    recent_liquidity = _recent_rows_by_candle_index(
        liquidity_df,
        latest_candle_index=latest_candle_index,
        candle_lookback=candle_lookback,
    )
    recent_structure = _recent_rows_by_candle_index(
        structure_df,
        latest_candle_index=latest_candle_index,
        candle_lookback=candle_lookback,
    )
    recent_fvg = _recent_rows_by_candle_index(
        fvg_df,
        latest_candle_index=latest_candle_index,
        candle_lookback=candle_lookback,
    )
    recent_dealing_range = _recent_rows_by_candle_index(
        dealing_range_df,
        latest_candle_index=latest_candle_index,
        candle_lookback=candle_lookback,
    )

    latest_structure_row = _latest_non_empty_row(recent_structure, "event_time")
    latest_range_row = _latest_non_empty_row(recent_dealing_range, "range_time")
    recent_range_pct = (
        ((df["high"] - df["low"]) / df["close"].replace(0, np.nan)).tail(20).mean() * 100
    )
    distance_scale_pct = max(0.35, float(recent_range_pct) * 3 if not pd.isna(recent_range_pct) else 0.35)

    records = []
    for side in ("buy", "sell"):
        target = _nearest_liquidity_target(recent_liquidity, side=side, current_price=current_price)
        if target is None:
            continue

        structure_bias = _context_bias_from_structure(
            latest_structure_row,
            side="bullish" if side == "buy" else "bearish",
        )
        dealing_range_bias = _context_bias_from_dealing_range(
            latest_range_row,
            side=side,
        )
        fvg_bias = _context_bias_from_fvg(
            recent_fvg,
            current_price=current_price,
            side="bullish" if side == "buy" else "bearish",
        )
        distance_score = float(np.exp(-abs(float(target["target_distance_pct"])) / distance_scale_pct))
        rank_bonus = 0.22 if target["liquidity_rank"] == "minor" else 0.15
        touch_bonus = min(float(target["equal_level_touches"]), 4.0) * 0.08
        liquidity_score = distance_score + rank_bonus + touch_bonus
        raw_score = 1.0 + structure_bias + dealing_range_bias + fvg_bias + liquidity_score

        records.append(
            {
                "target_side": side,
                "target_level": float(target["liquidity_level"]),
                "target_rank": target["liquidity_rank"],
                "target_time": target["liquidity_time"],
                "target_distance_pct": float(target["target_distance_pct"]),
                "current_price": current_price,
                "structure_bias": structure_bias,
                "dealing_range_bias": dealing_range_bias,
                "fvg_bias": fvg_bias,
                "distance_score": distance_score,
                "liquidity_score": liquidity_score,
                "raw_score": raw_score,
                "probability_pct": np.nan,
                "target_label": f"next_{side}_liquidity_probability",
            }
        )

    if not records:
        return _empty_liquidity_probability_table()

    probability_df = pd.DataFrame.from_records(records, columns=LIQUIDITY_PROBABILITY_COLUMNS)
    stabilized_scores = np.exp(probability_df["raw_score"] - probability_df["raw_score"].max())
    probability_df["probability_pct"] = (stabilized_scores / stabilized_scores.sum()) * 100
    return probability_df.sort_values("probability_pct", ascending=False).reset_index(drop=True)


def select_recent_liquidity_levels(
    liquidity_df,
    levels_per_group=1,
    prefer_unswept=True,
):
    """Funcion que selecciona niveles de liquidez recientes por lado/rango priorizando niveles no barridos."""
    if liquidity_df.empty:
        return liquidity_df.copy()

    selected_groups = []

    for side in LIQUIDITY_SIDE_ORDER:
        for rank in LIQUIDITY_RANK_ORDER:
            group = liquidity_df.loc[
                (liquidity_df["liquidity_side"] == side)
                & (liquidity_df["liquidity_rank"] == rank)
            ]
            if group.empty:
                continue

            ordered_group = group.sort_values("liquidity_time", ascending=False)
            selected = ordered_group.iloc[0:0]

            if prefer_unswept:
                unswept = ordered_group.loc[~ordered_group["swept"]]
                selected = unswept.head(levels_per_group)

            remaining = levels_per_group - len(selected)
            if remaining > 0:
                fallback = ordered_group.loc[~ordered_group.index.isin(selected.index)]
                selected = pd.concat([selected, fallback.head(remaining)])

            if not selected.empty:
                selected_groups.append(selected)

    if not selected_groups:
        return _empty_liquidity_table()

    return (
        pd.concat(selected_groups)
        .sort_values("liquidity_time", ascending=False)
        .reset_index(drop=True)
    )


def select_recent_fvg_levels(
    fvg_df,
    zones_per_side=2,
    prefer_unfilled=True,
):
    """Funcion que selecciona FVG recientes por lado priorizando zonas no llenadas."""
    if fvg_df.empty:
        return fvg_df.copy()

    selected_groups = []

    for side in FVG_SIDE_ORDER:
        group = fvg_df.loc[fvg_df["fvg_side"] == side]
        if group.empty:
            continue

        ordered_group = group.sort_values("fvg_time", ascending=False)
        selected = ordered_group.iloc[0:0]

        if prefer_unfilled:
            unfilled = ordered_group.loc[~ordered_group["filled"]]
            selected = unfilled.head(zones_per_side)

        remaining = zones_per_side - len(selected)
        if remaining > 0:
            fallback = ordered_group.loc[~ordered_group.index.isin(selected.index)]
            selected = pd.concat([selected, fallback.head(remaining)])

        if not selected.empty:
            selected_groups.append(selected)

    if not selected_groups:
        return _empty_fvg_table()

    return (
        pd.concat(selected_groups)
        .sort_values("fvg_time", ascending=False)
        .reset_index(drop=True)
    )


def select_recent_structure_events(
    structure_df,
    events_per_side=3,
):
    """Funcion que selecciona eventos estructurales recientes por direccion para simplificar visualizacion y lectura."""
    if structure_df.empty:
        return structure_df.copy()

    selected_groups = []

    for side in STRUCTURE_SIDE_ORDER:
        group = structure_df.loc[structure_df["event_side"] == side]
        if group.empty:
            continue

        ordered_group = group.sort_values("event_time", ascending=False)
        selected_groups.append(ordered_group.head(events_per_side))

    if not selected_groups:
        return _empty_market_structure_table()

    return (
        pd.concat(selected_groups)
        .sort_values("event_time", ascending=False)
        .reset_index(drop=True)
    )


def select_latest_dealing_range(dealing_range_df, rows=1):
    """Funcion que retorna el dealing range mas reciente para mostrar el contexto premium/discount vigente."""
    if dealing_range_df.empty:
        return dealing_range_df.copy()

    return (
        dealing_range_df.sort_values("range_time", ascending=False)
        .head(rows)
        .reset_index(drop=True)
    )
