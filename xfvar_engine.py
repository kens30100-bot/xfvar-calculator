from pathlib import Path
from itertools import combinations
import math

import pandas as pd


# ============================================================
# xFVAR 2.1
# Expected Fantasy Value Above Replacement
#
# Roster-slot IL replacement engine architecture
#
# Required data files:
#   league_environment.csv
#   replacement_pools.csv
#
# The engine NEVER invents replacement level or league data.
# ============================================================


MODEL_VERSION = "2.1"

BASE_DIR = Path(__file__).resolve().parent

LEAGUE_FILE = BASE_DIR / "league_environment.csv"
REPLACEMENT_FILE = BASE_DIR / "replacement_pools.csv"


HITTER_CATEGORIES = [
    "R",
    "HR",
    "RBI",
    "SB",
    "AVG",
    "OPS",
]

SP_CATEGORIES = [
    "K",
    "QS",
    "ERA",
    "WHIP",
    "KBB",
]

RP_CATEGORIES = [
    "SV",
    "K",
    "ERA",
    "WHIP",
    "KBB",
]


LOWER_IS_BETTER = {
    "ERA",
    "WHIP",
}


# ============================================================
# GENERAL HELPERS
# ============================================================


def _require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required xFVAR 2.0 data file is missing: "
            f"{path.name}"
        )


def _load_data():
    _require_file(LEAGUE_FILE)
    _require_file(REPLACEMENT_FILE)

    league = pd.read_csv(LEAGUE_FILE)
    replacement = pd.read_csv(REPLACEMENT_FILE)

    return league, replacement


def _require_columns(df, required, file_name):
    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{file_name} is missing required columns: "
            + ", ".join(missing)
        )


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _score_margin(margin):
    """
    H2H category result.

    Win  = 1
    Tie  = .5
    Loss = 0
    """

    if margin > 1e-12:
        return 1.0

    if margin < -1e-12:
        return 0.0

    return 0.5


def _parse_ip(ip):
    """
    Converts standard baseball IP notation into
    true innings.

    Examples:

    53.0 -> 53.0
    79.1 -> 79 + 1/3
    79.2 -> 79 + 2/3

    This is important because 79.2 in baseball
    does NOT mean 79.2 decimal innings.
    """

    if ip is None:
        return 0.0

    text = str(ip).strip()

    if not text:
        return 0.0

    if "." not in text:
        return float(text)

    whole, frac = text.split(".", 1)

    whole = int(whole)

    if frac == "0":
        return float(whole)

    if frac == "1":
        return whole + (1 / 3)

    if frac == "2":
        return whole + (2 / 3)

    # If the source already contains a true decimal,
    # preserve it rather than silently altering it.
    return float(text)


# ============================================================
# ACTIVE SCORING PERIODS
# ============================================================


def _active_weeks(player_stats, league):
    """
    Determines which completed scoring periods belong
    to the player's evaluation window.

    league_environment.csv will contain:

        week
        week_start
        week_end

    A scoring period is eligible if it overlaps the
    evaluation window.

    If an IL stint fully covers a scoring period,
    that period is replacement-filled and therefore
    removed from the player's active xFVAR window.

    Partial-week injury situations can later be flagged
    explicitly in the dataset if necessary.
    """

    start_date = pd.to_datetime(
        player_stats["start_date"]
    )

    end_date = pd.to_datetime(
        player_stats["end_date"]
    )

    weeks = (
        league[
            [
                "week",
                "week_start",
                "week_end",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    weeks["week_start"] = pd.to_datetime(
        weeks["week_start"]
    )

    weeks["week_end"] = pd.to_datetime(
        weeks["week_end"]
    )

    weeks = weeks[
        (weeks["week_end"] >= start_date)
        &
        (weeks["week_start"] <= end_date)
    ].copy()

    il_start = player_stats.get("il_start")
    il_end = player_stats.get("il_end")

    if il_start and il_end:

        il_start = pd.to_datetime(il_start)
        il_end = pd.to_datetime(il_end)

        fully_on_il = (
            (weeks["week_start"] >= il_start)
            &
            (weeks["week_end"] <= il_end)
        )

        weeks = weeks[~fully_on_il]

    return sorted(
        weeks["week"].astype(int).tolist()
    )


def _evaluation_weeks(player_stats, league):
    """
    Full completed scoring periods overlapping the ownership/trade window.

    Unlike _active_weeks(), IL-covered periods remain in this horizon.
    """
    start_date = pd.to_datetime(player_stats["start_date"])
    end_date = pd.to_datetime(player_stats["end_date"])

    weeks = (
        league[
            [
                "week",
                "week_start",
                "week_end",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    weeks["week_start"] = pd.to_datetime(weeks["week_start"])
    weeks["week_end"] = pd.to_datetime(weeks["week_end"])

    weeks = weeks[
        (weeks["week_end"] >= start_date)
        &
        (weeks["week_start"] <= end_date)
    ].copy()

    return sorted(
        weeks["week"].astype(int).tolist()
    )


def _league_time_units(league):
    """
    Length of the frozen replacement-data environment in 7-day units.

    The league environment includes a two-week All-Star scoring period, so
    using calendar days rather than simply counting scoring periods avoids
    treating that 14-day period as if it were only one normal week.
    """
    dates = (
        league[
            [
                "week_start",
                "week_end",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    starts = pd.to_datetime(dates["week_start"])
    ends = pd.to_datetime(dates["week_end"])

    if len(starts) == 0:
        return 0.0

    total_days = (
        ends.max()
        -
        starts.min()
    ).days + 1

    return total_days / 7.0


def _il_time_units(player_stats, league):
    """
    Calendar time spent on IL during the completed evaluation environment,
    expressed in 7-day units.

    Partial scoring periods are included proportionally.
    """
    il_start = player_stats.get("il_start")
    il_end = player_stats.get("il_end")

    if not il_start or not il_end:
        return 0.0

    il_start = pd.to_datetime(il_start)
    il_end = pd.to_datetime(il_end)

    window_start = pd.to_datetime(player_stats["start_date"])
    window_end = pd.to_datetime(player_stats["end_date"])

    league_dates = (
        league[
            [
                "week_start",
                "week_end",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    league_dates["week_start"] = pd.to_datetime(
        league_dates["week_start"]
    )
    league_dates["week_end"] = pd.to_datetime(
        league_dates["week_end"]
    )

    if len(league_dates) == 0:
        return 0.0

    completed_start = max(
        window_start,
        league_dates["week_start"].min(),
    )

    completed_end = min(
        window_end,
        league_dates["week_end"].max(),
    )

    overlap_start = max(
        completed_start,
        il_start,
    )

    overlap_end = min(
        completed_end,
        il_end,
    )

    if overlap_end < overlap_start:
        return 0.0

    missed_days = (
        overlap_end
        -
        overlap_start
    ).days + 1

    return missed_days / 7.0


def _hitter_il_fill_profile(
    pool,
    player_stats,
    league,
):
    """
    Expected replacement-hitter production during the player's IL time.

    Replacement production is estimated from the selected positional pool's
    observed season-to-date workload per roster spot per 7-day unit.

    This fill is added to BOTH:
        actual roster slot = Player while active + Replacement while IL
        baseline slot      = Replacement while active + Replacement while IL

    Therefore IL replacement counting stats cancel in the marginal counting
    categories, but they properly dilute AVG and OPS over the full roster slot.
    """
    missed_units = _il_time_units(
        player_stats,
        league,
    )

    if missed_units <= 0:
        return {
            "PA": 0.0,
            "AB": 0.0,
            "H": 0.0,
            "R": 0.0,
            "HR": 0.0,
            "RBI": 0.0,
            "SB": 0.0,
            "OPS": 0.0,
        }

    env_units = _league_time_units(league)

    if env_units <= 0 or len(pool) == 0:
        raise ValueError(
            "Cannot estimate hitter IL replacement workload."
        )

    denominator = (
        len(pool)
        *
        env_units
    )

    fill = {}

    for category in [
        "PA",
        "AB",
        "H",
        "R",
        "HR",
        "RBI",
        "SB",
    ]:
        fill[category] = (
            pool[category].sum()
            /
            denominator
            *
            missed_units
        )

    total_pa = pool["PA"].sum()

    if total_pa > 0:
        fill["OPS"] = (
            (
                pool["OPS"]
                *
                pool["PA"]
            ).sum()
            /
            total_pa
        )
    else:
        fill["OPS"] = 0.0

    return fill


def _pitcher_il_fill_profile(
    pool,
    player_stats,
    league,
):
    """
    Expected replacement-pitcher raw production during the player's IL time.

    Uses the selected positional pool's observed season-to-date production per
    roster spot per 7-day unit.

    The fill is shared by actual and baseline roster-slot lines. Counting-stat
    value therefore still comes only from the player's active above-replacement
    contribution, while ERA/WHIP/KBB correctly include replacement innings
    accumulated during the IL absence.
    """
    missed_units = _il_time_units(
        player_stats,
        league,
    )

    if missed_units <= 0:
        return {
            "IP": 0.0,
            "H_ALLOWED": 0.0,
            "ER": 0.0,
            "BB": 0.0,
            "K": 0.0,
            "SV": 0.0,
            "QS": 0.0,
        }

    env_units = _league_time_units(league)

    if env_units <= 0 or len(pool) == 0:
        raise ValueError(
            "Cannot estimate pitcher IL replacement workload."
        )

    work = pool.copy()

    work["IP_TRUE"] = work["IP"].apply(
        _parse_ip
    )

    denominator = (
        len(work)
        *
        env_units
    )

    fill = {
        "IP": (
            work["IP_TRUE"].sum()
            /
            denominator
            *
            missed_units
        ),
        "H_ALLOWED": (
            work["H_ALLOWED"].sum()
            /
            denominator
            *
            missed_units
        ),
        "ER": (
            work["ER"].sum()
            /
            denominator
            *
            missed_units
        ),
        "BB": (
            work["BB"].sum()
            /
            denominator
            *
            missed_units
        ),
        "K": (
            work["K"].sum()
            /
            denominator
            *
            missed_units
        ),
        "SV": (
            work["SV"].sum()
            /
            denominator
            *
            missed_units
        ),
        "QS": (
            work["QS"].sum()
            /
            denominator
            *
            missed_units
        ),
    }

    return fill


def _combine_hitter_slot(
    player_stats,
    active_replacement,
    il_fill,
):
    """
    Build full roster-slot hitter lines.

    actual_slot:
        Player's real active production + replacement production while IL

    baseline_slot:
        Active-window replacement equivalent + the same IL replacement fill
    """
    player_pa = _safe_float(
        player_stats.get("PA")
    )

    player_ab = _safe_float(
        player_stats.get("AB")
    )

    player_h = _safe_float(
        player_stats.get("H")
    )

    player_ops = _safe_float(
        player_stats.get("OPS")
    )

    fill_pa = il_fill["PA"]
    fill_ab = il_fill["AB"]
    fill_h = il_fill["H"]
    fill_ops = il_fill["OPS"]

    actual_pa = player_pa + fill_pa
    actual_ab = player_ab + fill_ab
    actual_h = player_h + fill_h

    if actual_pa > 0:
        actual_ops = (
            player_ops
            *
            player_pa
            +
            fill_ops
            *
            fill_pa
        ) / actual_pa
    else:
        actual_ops = 0.0

    baseline_pa = (
        active_replacement["PA"]
        +
        fill_pa
    )

    baseline_ab = (
        active_replacement["AB"]
        +
        fill_ab
    )

    baseline_h = (
        active_replacement["H"]
        +
        fill_h
    )

    if baseline_pa > 0:
        baseline_ops = (
            active_replacement["OPS"]
            *
            active_replacement["PA"]
            +
            fill_ops
            *
            fill_pa
        ) / baseline_pa
    else:
        baseline_ops = 0.0

    actual_stats = dict(player_stats)

    actual_stats["PA"] = actual_pa
    actual_stats["AB"] = actual_ab
    actual_stats["H"] = actual_h
    actual_stats["OPS"] = actual_ops

    baseline_profile = dict(
        active_replacement
    )

    baseline_profile["PA"] = baseline_pa
    baseline_profile["AB"] = baseline_ab
    baseline_profile["H"] = baseline_h
    baseline_profile["AVG"] = (
        baseline_h / baseline_ab
        if baseline_ab > 0
        else 0.0
    )
    baseline_profile["OPS"] = baseline_ops

    return actual_stats, baseline_profile


def _combine_pitcher_slot(
    player_stats,
    active_replacement,
    il_fill,
):
    """
    Build full roster-slot pitcher raw lines for ERA/WHIP/KBB.
    """
    player_raw = _pitcher_raw_stats(
        player_stats
    )

    actual_raw = {}

    baseline_raw = {}

    for key in [
        "IP",
        "H_ALLOWED",
        "ER",
        "BB",
        "K",
    ]:
        actual_raw[key] = (
            player_raw[key]
            +
            il_fill[key]
        )

        baseline_raw[key] = (
            active_replacement[key]
            +
            il_fill[key]
        )

    actual_stats = dict(
        player_stats
    )

    actual_stats["IP"] = actual_raw["IP"]
    actual_stats["H"] = actual_raw["H_ALLOWED"]
    actual_stats["ER"] = actual_raw["ER"]
    actual_stats["BB"] = actual_raw["BB"]
    actual_stats["K"] = actual_raw["K"]

    baseline_profile = dict(
        active_replacement
    )

    baseline_profile["IP"] = baseline_raw["IP"]
    baseline_profile["H_ALLOWED"] = (
        baseline_raw["H_ALLOWED"]
    )
    baseline_profile["ER"] = baseline_raw["ER"]
    baseline_profile["BB"] = baseline_raw["BB"]
    baseline_profile["K"] = baseline_raw["K"]

    return actual_stats, baseline_profile


# ============================================================
# REPLACEMENT POOLS
# ============================================================


def _get_replacement_rows(
    replacement,
    position,
):
    """
    replacement_pools.csv contains the frozen players
    selected for each xFVAR replacement band.

    We do NOT dynamically substitute random free agents.

    Required flags:

        eligible = 1
        selected = 1

    Example:

        position = 1B
        top 8 active FA 1B selected for xFVAR 2.0
    """

    pool = replacement[
        (
            replacement["position"]
            .astype(str)
            .str.upper()
            ==
            position.upper()
        )
        &
        (
            replacement["eligible"]
            .astype(int)
            ==
            1
        )
        &
        (
            replacement["selected"]
            .astype(int)
            ==
            1
        )
    ].copy()

    required_counts = {
        "C": 5,
        "1B": 8,
        "2B": 8,
        "3B": 8,
        "SS": 8,
        "OF": 12,
        "UTIL": 12,
        "SP": 12,
        "RP": 8,
    }

    target = required_counts[position]

    # Frozen rule:
    # If fewer than 12 legitimate SPs exist,
    # do not pad the pool with relievers.
    if position == "SP":

        if len(pool) == 0:
            raise ValueError(
                "No legitimate active starter-role "
                "replacement pitchers are stored."
            )

    else:

        if len(pool) != target:
            raise ValueError(
                f"{position} replacement pool should contain "
                f"{target} selected players, but contains "
                f"{len(pool)}."
            )

    return pool


# ============================================================
# COMPOSITE REPLACEMENT PROFILES
# ============================================================


def _hitter_replacement_profile(
    pool,
    player_stats,
):
    """
    Creates one composite replacement hitter.

    Counting stats scale to equivalent PA.

    AVG uses H/AB.

    OPS uses PA-weighted OPS.
    """

    required = [
        "PA",
        "AB",
        "H",
        "R",
        "HR",
        "RBI",
        "SB",
        "OPS",
    ]

    _require_columns(
        pool,
        required,
        "replacement_pools.csv",
    )

    total_pa = pool["PA"].sum()
    total_ab = pool["AB"].sum()

    if total_pa <= 0 or total_ab <= 0:
        raise ValueError(
            "Invalid hitter replacement workload."
        )

    player_pa = _safe_float(
        player_stats.get("PA")
    )

    player_ab = _safe_float(
        player_stats.get("AB")
    )

    profile = {}

    # Counting rates per PA
    for category in [
        "R",
        "HR",
        "RBI",
        "SB",
    ]:

        rate = (
            pool[category].sum()
            /
            total_pa
        )

        profile[category] = (
            rate * player_pa
        )

    # AVG workload
    replacement_avg = (
        pool["H"].sum()
        /
        total_ab
    )

    profile["AB"] = player_ab

    profile["H"] = (
        replacement_avg
        *
        player_ab
    )

    profile["AVG"] = replacement_avg

    # OPS is PA-weighted
    replacement_ops = (
        (
            pool["OPS"]
            *
            pool["PA"]
        ).sum()
        /
        total_pa
    )

    profile["PA"] = player_pa
    profile["OPS"] = replacement_ops

    return profile


def _pitcher_replacement_profile(
    pool,
    player_stats,
    role,
):
    """
    SP replacement:
        scale to equivalent GS

    RP replacement:
        scale to equivalent appearances

    Ratio workload is allowed to differ from the
    evaluated player's IP because replacement pitchers
    may throw a different amount per opportunity.
    """

    required = [
        "G",
        "GS",
        "IP",
        "H_ALLOWED",
        "ER",
        "BB",
        "K",
        "SV",
        "QS",
    ]

    _require_columns(
        pool,
        required,
        "replacement_pools.csv",
    )

    work = pool.copy()

    work["IP_TRUE"] = work["IP"].apply(
        _parse_ip
    )

    if role == "SP":

        denominator = work["GS"].sum()

        player_opportunities = _safe_float(
            player_stats.get("GS")
        )

        if denominator <= 0:
            raise ValueError(
                "SP replacement pool contains "
                "no starting opportunities."
            )

    else:

        denominator = work["G"].sum()

        player_opportunities = _safe_float(
            player_stats.get("G")
        )

        if denominator <= 0:
            raise ValueError(
                "RP replacement pool contains "
                "no relief appearances."
            )

    scale = (
        player_opportunities
        /
        denominator
    )

    profile = {
        "IP": (
            work["IP_TRUE"].sum()
            *
            scale
        ),
        "H_ALLOWED": (
            work["H_ALLOWED"].sum()
            *
            scale
        ),
        "ER": (
            work["ER"].sum()
            *
            scale
        ),
        "BB": (
            work["BB"].sum()
            *
            scale
        ),
        "K": (
            work["K"].sum()
            *
            scale
        ),
        "SV": (
            work["SV"].sum()
            *
            scale
        ),
        "QS": (
            work["QS"].sum()
            *
            scale
        ),
    }

    ip = profile["IP"]

    if ip > 0:

        profile["ERA"] = (
            9
            *
            profile["ER"]
            /
            ip
        )

        profile["WHIP"] = (
            (
                profile["H_ALLOWED"]
                +
                profile["BB"]
            )
            /
            ip
        )

    else:

        profile["ERA"] = 0
        profile["WHIP"] = 0

    if profile["BB"] > 0:

        profile["KBB"] = (
            profile["K"]
            /
            profile["BB"]
        )

    else:

        profile["KBB"] = math.inf

    return profile


# ============================================================
# HISTORICAL PAIR CONSTRUCTION
# ============================================================


def _weekly_rows(
    league,
    active_weeks,
):
    return league[
        league["week"]
        .astype(int)
        .isin(active_weeks)
    ].copy()


def _week_pairs(df):
    """
    Generates the 28 unique team-vs-team pairs
    from an eight-team scoring period.

    Each pair is evaluated from BOTH perspectives,
    then averaged.

    This preserves the intended:

        C(8,2) = 28 comparisons per week

    without giving arbitrary value to alphabetical
    team ordering.
    """

    records = list(
        df.to_dict("records")
    )

    return combinations(records, 2)


# ============================================================
# COUNTING CATEGORY ENGINE
# ============================================================


def _counting_category_value(
    league,
    active_weeks,
    category,
    total_delta,
):
    """
    Converts above-replacement counting production
    into expected H2H category wins.

    Player production above replacement is distributed
    across active scoring periods.

    The historical league environment then determines
    how often that amount changes a category result.
    """

    W = len(active_weeks)

    if W == 0:
        return 0.0

    weekly_delta = (
        total_delta
        /
        W
    )

    # xFVAR 2.0 calibration environment is always the full frozen
    # Weeks 1-19 league environment. The player's active_weeks controls
    # production/opportunity volume (W), not which historical matchup
    # margins define category leverage.
    env = league.copy()

    pair_effects = []

    for week in sorted(league["week"].astype(int).unique()):

        week_df = env[
            env["week"].astype(int)
            ==
            int(week)
        ]

        for a, b in _week_pairs(week_df):

            a_value = _safe_float(
                a[category]
            )

            b_value = _safe_float(
                b[category]
            )

            margin_ab = (
                a_value - b_value
            )

            margin_ba = (
                b_value - a_value
            )

            effect_a = (
                _score_margin(
                    margin_ab
                    +
                    weekly_delta
                )
                -
                _score_margin(
                    margin_ab
                )
            )

            effect_b = (
                _score_margin(
                    margin_ba
                    +
                    weekly_delta
                )
                -
                _score_margin(
                    margin_ba
                )
            )

            pair_effects.append(
                (
                    effect_a
                    +
                    effect_b
                )
                /
                2
            )

    if not pair_effects:
        return 0.0

    expected_per_week = (
        sum(pair_effects)
        /
        len(pair_effects)
    )

    return (
        W
        *
        expected_per_week
    )


# ============================================================
# AVG ENGINE
# ============================================================


def _avg_value(
    league,
    active_weeks,
    player_stats,
    replacement_profile,
):
    W = len(active_weeks)

    if W == 0:
        return 0.0

    player_h_week = (
        _safe_float(
            player_stats["H"]
        )
        /
        W
    )

    player_ab_week = (
        _safe_float(
            player_stats["AB"]
        )
        /
        W
    )

    repl_h_week = (
        replacement_profile["H"]
        /
        W
    )

    repl_ab_week = (
        replacement_profile["AB"]
        /
        W
    )

    # xFVAR 2.0 calibration environment is always the full frozen
    # Weeks 1-19 league environment. The player's active_weeks controls
    # production/opportunity volume (W), not which historical matchup
    # margins define category leverage.
    env = league.copy()

    effects = []

    for week in sorted(league["week"].astype(int).unique()):

        week_df = env[
            env["week"].astype(int)
            ==
            int(week)
        ]

        for a, b in _week_pairs(week_df):

            def adjusted_avg(row):

                team_h = _safe_float(
                    row["H"]
                )

                team_ab = _safe_float(
                    row["AB"]
                )

                new_h = (
                    team_h
                    -
                    repl_h_week
                    +
                    player_h_week
                )

                new_ab = (
                    team_ab
                    -
                    repl_ab_week
                    +
                    player_ab_week
                )

                if new_ab <= 0:
                    return 0.0

                return (
                    new_h
                    /
                    new_ab
                )

            a_old = (
                _safe_float(a["H"])
                /
                _safe_float(a["AB"])
            )

            b_old = (
                _safe_float(b["H"])
                /
                _safe_float(b["AB"])
            )

            a_new = adjusted_avg(a)
            b_new = adjusted_avg(b)

            old_ab = (
                a_old - b_old
            )

            old_ba = (
                b_old - a_old
            )

            new_ab = (
                a_new - b_old
            )

            new_ba = (
                b_new - a_old
            )

            effect_a = (
                _score_margin(new_ab)
                -
                _score_margin(old_ab)
            )

            effect_b = (
                _score_margin(new_ba)
                -
                _score_margin(old_ba)
            )

            effects.append(
                (
                    effect_a
                    +
                    effect_b
                )
                /
                2
            )

    return (
        W
        *
        sum(effects)
        /
        len(effects)
        if effects
        else 0.0
    )


# ============================================================
# OPS ENGINE
# ============================================================


def _ops_value(
    league,
    active_weeks,
    player_stats,
    replacement_profile,
):
    """
    xFVAR 2.0 uses a documented workload approximation
    for team OPS because Yahoo matchup screenshots do
    not expose team OBP and SLG components.

    league_environment.csv therefore contains a frozen
    OPS_WEIGHT field.

    We do NOT silently substitute a new approximation
    if OPS_WEIGHT is missing.
    """

    if "OPS_WEIGHT" not in league.columns:

        raise ValueError(
            "league_environment.csv requires OPS_WEIGHT "
            "for the frozen xFVAR 2.0 OPS approximation."
        )

    W = len(active_weeks)

    if W == 0:
        return 0.0

    player_pa_week = (
        _safe_float(
            player_stats["PA"]
        )
        /
        W
    )

    player_ops = _safe_float(
        player_stats["OPS"]
    )

    repl_pa_week = (
        replacement_profile["PA"]
        /
        W
    )

    repl_ops = (
        replacement_profile["OPS"]
    )

    # xFVAR 2.0 calibration environment is always the full frozen
    # Weeks 1-19 league environment. The player's active_weeks controls
    # production/opportunity volume (W), not which historical matchup
    # margins define category leverage.
    env = league.copy()

    effects = []

    for week in sorted(league["week"].astype(int).unique()):

        week_df = env[
            env["week"].astype(int)
            ==
            int(week)
        ]

        for a, b in _week_pairs(week_df):

            def adjusted_ops(row):

                baseline_ops = _safe_float(
                    row["OPS"]
                )

                weight = _safe_float(
                    row["OPS_WEIGHT"]
                )

                if weight <= 0:
                    raise ValueError(
                        "OPS_WEIGHT must be positive."
                    )

                numerator = (
                    baseline_ops
                    *
                    weight
                    -
                    repl_ops
                    *
                    repl_pa_week
                    +
                    player_ops
                    *
                    player_pa_week
                )

                denominator = (
                    weight
                    -
                    repl_pa_week
                    +
                    player_pa_week
                )

                if denominator <= 0:
                    return baseline_ops

                return (
                    numerator
                    /
                    denominator
                )

            a_old = _safe_float(
                a["OPS"]
            )

            b_old = _safe_float(
                b["OPS"]
            )

            a_new = adjusted_ops(a)
            b_new = adjusted_ops(b)

            effect_a = (
                _score_margin(
                    a_new
                    -
                    b_old
                )
                -
                _score_margin(
                    a_old
                    -
                    b_old
                )
            )

            effect_b = (
                _score_margin(
                    b_new
                    -
                    a_old
                )
                -
                _score_margin(
                    b_old
                    -
                    a_old
                )
            )

            effects.append(
                (
                    effect_a
                    +
                    effect_b
                )
                /
                2
            )

    return (
        W
        *
        sum(effects)
        /
        len(effects)
        if effects
        else 0.0
    )


# ============================================================
# PITCHING RATIO ENGINE
# ============================================================


def _pitcher_raw_stats(
    player_stats,
):
    ip = _parse_ip(
        player_stats.get("IP", 0)
    )

    return {
        "IP": ip,
        "H_ALLOWED": _safe_float(
            player_stats.get("H", 0)
        ),
        "ER": _safe_float(
            player_stats.get("ER", 0)
        ),
        "BB": _safe_float(
            player_stats.get("BB", 0)
        ),
        "K": _safe_float(
            player_stats.get("K", 0)
        ),
    }


def _team_pitching_raw(row):
    """
    Reconstructs approximate weekly pitching raw totals
    from Yahoo team-category values.

    ER:
        ERA * IP / 9

    BB:
        K / KBB

    Baserunners:
        WHIP * IP

    Hits allowed:
        baserunners - BB
    """

    ip = _parse_ip(
        row["IP"]
    )

    era = _safe_float(
        row["ERA"]
    )

    whip = _safe_float(
        row["WHIP"]
    )

    k = _safe_float(
        row["K"]
    )

    kbb = _safe_float(
        row["KBB"]
    )

    er = (
        era
        *
        ip
        /
        9
    )

    if kbb > 0:

        bb = (
            k
            /
            kbb
        )

    else:

        bb = 0.0

    baserunners = (
        whip
        *
        ip
    )

    h_allowed = max(
        baserunners
        -
        bb,
        0.0,
    )

    return {
        "IP": ip,
        "ER": er,
        "BB": bb,
        "K": k,
        "H_ALLOWED": h_allowed,
    }


def _pitching_ratio(
    raw,
    category,
):
    ip = raw["IP"]

    if ip <= 0:
        return 0.0

    if category == "ERA":

        return (
            9
            *
            raw["ER"]
            /
            ip
        )

    if category == "WHIP":

        return (
            (
                raw["H_ALLOWED"]
                +
                raw["BB"]
            )
            /
            ip
        )

    if category == "KBB":

        if raw["BB"] <= 0:

            if raw["K"] > 0:
                return math.inf

            return 0.0

        return (
            raw["K"]
            /
            raw["BB"]
        )

    raise ValueError(
        f"Unknown pitching ratio: {category}"
    )


def _pitching_ratio_value(
    league,
    active_weeks,
    player_stats,
    replacement_profile,
    category,
):
    W = len(active_weeks)

    if W == 0:
        return 0.0

    player = _pitcher_raw_stats(
        player_stats
    )

    repl = {
        "IP": replacement_profile["IP"],
        "H_ALLOWED": replacement_profile[
            "H_ALLOWED"
        ],
        "ER": replacement_profile["ER"],
        "BB": replacement_profile["BB"],
        "K": replacement_profile["K"],
    }

    player_week = {
        key: value / W
        for key, value
        in player.items()
    }

    repl_week = {
        key: value / W
        for key, value
        in repl.items()
    }

    # xFVAR 2.0 calibration environment is always the full frozen
    # Weeks 1-19 league environment. The player's active_weeks controls
    # production/opportunity volume (W), not which historical matchup
    # margins define category leverage.
    env = league.copy()

    effects = []

    for week in sorted(league["week"].astype(int).unique()):

        week_df = env[
            env["week"].astype(int)
            ==
            int(week)
        ]

        for a, b in _week_pairs(week_df):

            a_raw = _team_pitching_raw(a)
            b_raw = _team_pitching_raw(b)

            def adjusted(raw):

                result = {}

                for key in [
                    "IP",
                    "H_ALLOWED",
                    "ER",
                    "BB",
                    "K",
                ]:

                    result[key] = (
                        raw[key]
                        -
                        repl_week[key]
                        +
                        player_week[key]
                    )

                result["IP"] = max(
                    result["IP"],
                    0.000001,
                )

                result["H_ALLOWED"] = max(
                    result["H_ALLOWED"],
                    0.0,
                )

                result["ER"] = max(
                    result["ER"],
                    0.0,
                )

                result["BB"] = max(
                    result["BB"],
                    0.0,
                )

                result["K"] = max(
                    result["K"],
                    0.0,
                )

                return result

            a_old = _pitching_ratio(
                a_raw,
                category,
            )

            b_old = _pitching_ratio(
                b_raw,
                category,
            )

            a_new = _pitching_ratio(
                adjusted(a_raw),
                category,
            )

            b_new = _pitching_ratio(
                adjusted(b_raw),
                category,
            )

            if category in LOWER_IS_BETTER:

                old_ab = (
                    b_old - a_old
                )

                old_ba = (
                    a_old - b_old
                )

                new_ab = (
                    b_old - a_new
                )

                new_ba = (
                    a_old - b_new
                )

            else:

                old_ab = (
                    a_old - b_old
                )

                old_ba = (
                    b_old - a_old
                )

                new_ab = (
                    a_new - b_old
                )

                new_ba = (
                    b_new - a_old
                )

            effect_a = (
                _score_margin(new_ab)
                -
                _score_margin(old_ab)
            )

            effect_b = (
                _score_margin(new_ba)
                -
                _score_margin(old_ba)
            )

            effects.append(
                (
                    effect_a
                    +
                    effect_b
                )
                /
                2
            )

    return (
        W
        *
        sum(effects)
        /
        len(effects)
        if effects
        else 0.0
    )


# ============================================================
# HIT FOR HITTER
# ============================================================


def _calculate_hitter(
    player_stats,
    league,
    replacement,
):
    position = (
        player_stats["position"]
        .upper()
    )

    pool = _get_replacement_rows(
        replacement,
        position,
    )

    # Replacement equivalent to the player's ACTUAL active workload.
    active_repl = _hitter_replacement_profile(
        pool,
        player_stats,
    )

    active_weeks = _active_weeks(
        player_stats,
        league,
    )

    evaluation_weeks = _evaluation_weeks(
        player_stats,
        league,
    )

    il_fill = _hitter_il_fill_profile(
        pool,
        player_stats,
        league,
    )

    slot_player_stats, slot_repl = (
        _combine_hitter_slot(
            player_stats,
            active_repl,
            il_fill,
        )
    )

    category_values = {}

    # Counting stats:
    # IL replacement production is on both sides, so it cancels.
    # Value comes from the player's actual active production above
    # replacement during his active workload.
    for category in [
        "R",
        "HR",
        "RBI",
        "SB",
    ]:

        delta = (
            _safe_float(
                player_stats[category]
            )
            -
            _safe_float(
                active_repl[category]
            )
        )

        category_values[category] = (
            _counting_category_value(
                league,
                evaluation_weeks,
                category,
                delta,
            )
        )

    # Rate stats:
    # Player while active + replacement while IL
    # versus replacement across both portions.
    category_values["AVG"] = (
        _avg_value(
            league,
            evaluation_weeks,
            slot_player_stats,
            slot_repl,
        )
    )

    category_values["OPS"] = (
        _ops_value(
            league,
            evaluation_weeks,
            slot_player_stats,
            slot_repl,
        )
    )

    xfvar = sum(
        category_values.values()
    )

    return {
        "xfvar": xfvar,
        "category_values": category_values,
        "replacement_position": position,
        "active_weeks": active_weeks,
        "evaluation_weeks": evaluation_weeks,
        "il_time_units": _il_time_units(
            player_stats,
            league,
        ),
        "il_replacement_profile": il_fill,
        "replacement_profile": [
            {
                "Metric": key,
                "Replacement": value,
            }
            for key, value
            in active_repl.items()
        ],
    }


# ============================================================
# STARTING PITCHER
# ============================================================


def _calculate_sp(
    player_stats,
    league,
    replacement,
):
    pool = _get_replacement_rows(
        replacement,
        "SP",
    )

    # Replacement equivalent to the player's ACTUAL starts.
    active_repl = _pitcher_replacement_profile(
        pool,
        player_stats,
        "SP",
    )

    active_weeks = _active_weeks(
        player_stats,
        league,
    )

    evaluation_weeks = _evaluation_weeks(
        player_stats,
        league,
    )

    il_fill = _pitcher_il_fill_profile(
        pool,
        player_stats,
        league,
    )

    slot_player_stats, slot_repl = (
        _combine_pitcher_slot(
            player_stats,
            active_repl,
            il_fill,
        )
    )

    category_values = {}

    # SP counting value comes only from the player's actual starts
    # above replacement. IL replacement K/QS cancel against baseline.
    category_values["K"] = (
        _counting_category_value(
            league,
            evaluation_weeks,
            "K",
            (
                _safe_float(
                    player_stats["K"]
                )
                -
                active_repl["K"]
            ),
        )
    )

    category_values["QS"] = (
        _counting_category_value(
            league,
            evaluation_weeks,
            "QS",
            (
                _safe_float(
                    player_stats["QS"]
                )
                -
                active_repl["QS"]
            ),
        )
    )

    # Ratios are based on the full roster-slot line:
    # Player while active + replacement innings while IL.
    for category in [
        "ERA",
        "WHIP",
        "KBB",
    ]:

        category_values[category] = (
            _pitching_ratio_value(
                league,
                evaluation_weeks,
                slot_player_stats,
                slot_repl,
                category,
            )
        )

    xfvar = sum(
        category_values.values()
    )

    return {
        "xfvar": xfvar,
        "category_values": category_values,
        "replacement_position": "SP",
        "active_weeks": active_weeks,
        "evaluation_weeks": evaluation_weeks,
        "il_time_units": _il_time_units(
            player_stats,
            league,
        ),
        "il_replacement_profile": il_fill,
        "replacement_profile": [
            {
                "Metric": key,
                "Replacement": value,
            }
            for key, value
            in active_repl.items()
        ],
    }


# ============================================================
# RELIEF PITCHER
# ============================================================


def _calculate_rp(
    player_stats,
    league,
    replacement,
):
    pool = _get_replacement_rows(
        replacement,
        "RP",
    )

    # Replacement equivalent to the player's ACTUAL appearances.
    active_repl = _pitcher_replacement_profile(
        pool,
        player_stats,
        "RP",
    )

    active_weeks = _active_weeks(
        player_stats,
        league,
    )

    evaluation_weeks = _evaluation_weeks(
        player_stats,
        league,
    )

    il_fill = _pitcher_il_fill_profile(
        pool,
        player_stats,
        league,
    )

    slot_player_stats, slot_repl = (
        _combine_pitcher_slot(
            player_stats,
            active_repl,
            il_fill,
        )
    )

    category_values = {}

    # RP counting value comes only from actual active appearances
    # above replacement. IL replacement stats cancel.
    category_values["SV"] = (
        _counting_category_value(
            league,
            evaluation_weeks,
            "SV",
            (
                _safe_float(
                    player_stats["SV"]
                )
                -
                active_repl["SV"]
            ),
        )
    )

    category_values["K"] = (
        _counting_category_value(
            league,
            evaluation_weeks,
            "K",
            (
                _safe_float(
                    player_stats["K"]
                )
                -
                active_repl["K"]
            ),
        )
    )

    for category in [
        "ERA",
        "WHIP",
        "KBB",
    ]:

        category_values[category] = (
            _pitching_ratio_value(
                league,
                evaluation_weeks,
                slot_player_stats,
                slot_repl,
                category,
            )
        )

    xfvar = sum(
        category_values.values()
    )

    return {
        "xfvar": xfvar,
        "category_values": category_values,
        "replacement_position": "RP",
        "active_weeks": active_weeks,
        "evaluation_weeks": evaluation_weeks,
        "il_time_units": _il_time_units(
            player_stats,
            league,
        ),
        "il_replacement_profile": il_fill,
        "replacement_profile": [
            {
                "Metric": key,
                "Replacement": value,
            }
            for key, value
            in active_repl.items()
        ],
    }


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================


def calculate_xfvar(player_stats):
    """
    Main function called by app.py.

    Same input + same frozen data =
    same xFVAR every time.
    """

    league, replacement = _load_data()

    league_required = [
        "week",
        "week_start",
        "week_end",
        "team",
        "H",
        "AB",
        "R",
        "HR",
        "RBI",
        "SB",
        "AVG",
        "OPS",
        "OPS_WEIGHT",
        "IP",
        "SV",
        "K",
        "ERA",
        "WHIP",
        "KBB",
        "QS",
    ]

    replacement_required = [
        "position",
        "player",
        "eligible",
        "selected",
    ]

    _require_columns(
        league,
        league_required,
        "league_environment.csv",
    )

    _require_columns(
        replacement,
        replacement_required,
        "replacement_pools.csv",
    )

    player_type = (
        str(
            player_stats.get(
                "player_type",
                ""
            )
        )
        .upper()
    )

    if player_type == "HITTER":

        return _calculate_hitter(
            player_stats,
            league,
            replacement,
        )

    if player_type == "SP":

        return _calculate_sp(
            player_stats,
            league,
            replacement,
        )

    if player_type == "RP":

        return _calculate_rp(
            player_stats,
            league,
            replacement,
        )

    raise ValueError(
        "player_type must be Hitter, SP, or RP."
    )
