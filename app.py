import streamlit as st
import pandas as pd

# ---------------------------------------------------------
# xFVAR 2.0 — Streamlit Front End
# ---------------------------------------------------------

st.set_page_config(
    page_title="xFVAR 2.0 Calculator",
    page_icon="⚾",
    layout="wide",
)

st.title("⚾ xFVAR 2.0 Calculator")
st.caption("Expected Fantasy Value Above Replacement")

st.info(
    "xFVAR estimates how many additional expected H2H category wins "
    "a player provides compared with a realistically available "
    "replacement player at the same usable roster position."
)

# ---------------------------------------------------------
# Attempt to load the calculation engine
# ---------------------------------------------------------

try:
    from xfvar_engine import calculate_xfvar
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("xFVAR 2.0")

mode = st.sidebar.radio(
    "Calculator Mode",
    [
        "Single Player",
        "Trade Calculator",
        "Model Information",
    ],
)

st.sidebar.markdown("---")

st.sidebar.write("**Frozen model version:** 2.0")
st.sidebar.write("**League environment:** Weeks 1–19")
st.sidebar.write("**League size:** 8 teams")
st.sidebar.write("**Scoring:** 6×6 H2H categories")


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def number_input(label, value=0.0, step=1.0, fmt=None):
    kwargs = {
        "label": label,
        "value": value,
        "step": step,
    }

    if fmt:
        kwargs["format"] = fmt

    return st.number_input(**kwargs)


# ---------------------------------------------------------
# SINGLE PLAYER CALCULATOR
# ---------------------------------------------------------

if mode == "Single Player":

    st.header("Single Player Calculator")

    col1, col2, col3 = st.columns(3)

    with col1:
        player_name = st.text_input(
            "Player Name",
            placeholder="Shohei Ohtani",
        )

    with col2:
        player_type = st.selectbox(
            "Player Type",
            ["Hitter", "Starting Pitcher", "Relief Pitcher"],
        )

    with col3:

        if player_type == "Hitter":
            position = st.selectbox(
                "Position",
                ["C", "1B", "2B", "3B", "SS", "OF", "UTIL"],
            )

        elif player_type == "Starting Pitcher":
            position = "SP"
            st.text_input("Position", value="SP", disabled=True)

        else:
            position = "RP"
            st.text_input("Position", value="RP", disabled=True)


    st.subheader("Evaluation Window")

    date_col1, date_col2 = st.columns(2)

    with date_col1:
        start_date = st.date_input("Start Date")

    with date_col2:
        end_date = st.date_input("End Date")


    # -----------------------------------------------------
    # IL periods
    # -----------------------------------------------------

    st.subheader("IL Information")

    had_il = st.checkbox(
        "Player had an IL stint during this evaluation period"
    )

    il_start = None
    il_end = None

    if had_il:
        il_col1, il_col2 = st.columns(2)

        with il_col1:
            il_start = st.date_input(
                "IL Start Date",
                key="il_start",
            )

        with il_col2:
            il_end = st.date_input(
                "IL End Date",
                key="il_end",
            )


    # -----------------------------------------------------
    # HITTER INPUTS
    # -----------------------------------------------------

    if player_type == "Hitter":

        st.subheader("Hitting Statistics")

        row1 = st.columns(4)

        with row1[0]:
            PA = number_input("PA", 0, 1)

        with row1[1]:
            AB = number_input("AB", 0, 1)

        with row1[2]:
            H = number_input("H", 0, 1)

        with row1[3]:
            R = number_input("R", 0, 1)


        row2 = st.columns(4)

        with row2[0]:
            HR = number_input("HR", 0, 1)

        with row2[1]:
            RBI = number_input("RBI", 0, 1)

        with row2[2]:
            SB = number_input("SB", 0, 1)

        with row2[3]:
            BB = number_input("BB", 0, 1)


        row3 = st.columns(4)

        with row3[0]:
            AVG = number_input(
                "AVG",
                0.000,
                0.001,
                "%.3f",
            )

        with row3[1]:
            OBP = number_input(
                "OBP",
                0.000,
                0.001,
                "%.3f",
            )

        with row3[2]:
            SLG = number_input(
                "SLG",
                0.000,
                0.001,
                "%.3f",
            )

        with row3[3]:
            OPS = number_input(
                "OPS",
                0.000,
                0.001,
                "%.3f",
            )


        player_stats = {
            "player_name": player_name,
            "player_type": "Hitter",
            "position": position,
            "PA": PA,
            "AB": AB,
            "H": H,
            "R": R,
            "HR": HR,
            "RBI": RBI,
            "SB": SB,
            "BB": BB,
            "AVG": AVG,
            "OBP": OBP,
            "SLG": SLG,
            "OPS": OPS,
        }


    # -----------------------------------------------------
    # STARTING PITCHER INPUTS
    # -----------------------------------------------------

    elif player_type == "Starting Pitcher":

        st.subheader("Starting Pitcher Statistics")

        row1 = st.columns(4)

        with row1[0]:
            GS = number_input("GS", 0, 1)

        with row1[1]:
            IP = number_input(
                "IP",
                0.0,
                0.1,
                "%.1f",
            )

        with row1[2]:
            H_allowed = number_input(
                "Hits Allowed",
                0,
                1,
            )

        with row1[3]:
            ER = number_input(
                "Earned Runs",
                0,
                1,
            )


        row2 = st.columns(4)

        with row2[0]:
            BB = number_input(
                "Walks",
                0,
                1,
            )

        with row2[1]:
            K = number_input(
                "Strikeouts",
                0,
                1,
            )

        with row2[2]:
            QS = number_input(
                "Quality Starts",
                0,
                1,
            )

        with row2[3]:
            HBP = number_input(
                "HBP",
                0,
                1,
            )


        calculated_era = (
            9 * ER / IP
            if IP > 0
            else 0
        )

        calculated_whip = (
            (H_allowed + BB) / IP
            if IP > 0
            else 0
        )

        calculated_kbb = (
            K / BB
            if BB > 0
            else 0
        )


        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Calculated ERA",
            f"{calculated_era:.2f}",
        )

        metric2.metric(
            "Calculated WHIP",
            f"{calculated_whip:.2f}",
        )

        metric3.metric(
            "Calculated K/BB",
            f"{calculated_kbb:.2f}",
        )


        player_stats = {
            "player_name": player_name,
            "player_type": "SP",
            "position": "SP",
            "GS": GS,
            "IP": IP,
            "H": H_allowed,
            "ER": ER,
            "BB": BB,
            "K": K,
            "QS": QS,
            "HBP": HBP,
            "ERA": calculated_era,
            "WHIP": calculated_whip,
            "KBB": calculated_kbb,
        }


    # -----------------------------------------------------
    # RELIEF PITCHER INPUTS
    # -----------------------------------------------------

    else:

        st.subheader("Relief Pitcher Statistics")

        row1 = st.columns(4)

        with row1[0]:
            G = number_input(
                "Games",
                0,
                1,
            )

        with row1[1]:
            IP = number_input(
                "IP",
                0.0,
                0.1,
                "%.1f",
            )

        with row1[2]:
            H_allowed = number_input(
                "Hits Allowed",
                0,
                1,
            )

        with row1[3]:
            ER = number_input(
                "Earned Runs",
                0,
                1,
            )


        row2 = st.columns(4)

        with row2[0]:
            BB = number_input(
                "Walks",
                0,
                1,
            )

        with row2[1]:
            K = number_input(
                "Strikeouts",
                0,
                1,
            )

        with row2[2]:
            SV = number_input(
                "Saves",
                0,
                1,
            )

        with row2[3]:
            HBP = number_input(
                "HBP",
                0,
                1,
            )


        calculated_era = (
            9 * ER / IP
            if IP > 0
            else 0
        )

        calculated_whip = (
            (H_allowed + BB) / IP
            if IP > 0
            else 0
        )

        calculated_kbb = (
            K / BB
            if BB > 0
            else 0
        )


        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Calculated ERA",
            f"{calculated_era:.2f}",
        )

        metric2.metric(
            "Calculated WHIP",
            f"{calculated_whip:.2f}",
        )

        metric3.metric(
            "Calculated K/BB",
            f"{calculated_kbb:.2f}",
        )


        player_stats = {
            "player_name": player_name,
            "player_type": "RP",
            "position": "RP",
            "G": G,
            "IP": IP,
            "H": H_allowed,
            "ER": ER,
            "BB": BB,
            "K": K,
            "SV": SV,
            "HBP": HBP,
            "ERA": calculated_era,
            "WHIP": calculated_whip,
            "KBB": calculated_kbb,
        }


    player_stats["start_date"] = str(start_date)
    player_stats["end_date"] = str(end_date)

    player_stats["il_start"] = (
        str(il_start)
        if il_start
        else None
    )

    player_stats["il_end"] = (
        str(il_end)
        if il_end
        else None
    )


    st.markdown("---")


    # -----------------------------------------------------
    # CALCULATE BUTTON
    # -----------------------------------------------------

    if st.button(
        "Calculate xFVAR 2.0",
        type="primary",
        use_container_width=True,
    ):

        if not ENGINE_AVAILABLE:

            st.error(
                "The xFVAR calculation engine has not been "
                "installed yet. Add xfvar_engine.py to the "
                "repository before calculations can run."
            )

        elif not player_name:

            st.warning(
                "Please enter a player name."
            )

        else:

            try:

                result = calculate_xfvar(
                    player_stats
                )

                st.success(
                    f"{player_name}: "
                    f"{result['xfvar']:.2f} xFVAR"
                )

                st.metric(
                    "xFVAR 2.0",
                    f"{result['xfvar']:.2f}",
                )


                if "replacement_position" in result:

                    st.write(
                        "**Replacement position:**",
                        result[
                            "replacement_position"
                        ],
                    )


                if "category_values" in result:

                    st.subheader(
                        "Category Breakdown"
                    )

                    breakdown = pd.DataFrame(
                        [
                            {
                                "Category": category,
                                "xFVAR": value,
                            }
                            for category, value
                            in result[
                                "category_values"
                            ].items()
                        ]
                    )

                    st.dataframe(
                        breakdown,
                        use_container_width=True,
                        hide_index=True,
                    )


                if "replacement_profile" in result:

                    st.subheader(
                        "Replacement-Level Comparison"
                    )

                    st.dataframe(
                        pd.DataFrame(
                            result[
                                "replacement_profile"
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )


            except Exception as e:

                st.error(
                    "Calculation error:"
                )

                st.exception(e)


# ---------------------------------------------------------
# TRADE CALCULATOR
# ---------------------------------------------------------

elif mode == "Trade Calculator":

    st.header("Trade Calculator")

    st.write(
        "This section will compare the combined xFVAR "
        "received by each side of a trade."
    )

    st.info(
        "The single-player xFVAR engine will be completed "
        "first. Trade calculations will then call the exact "
        "same frozen xFVAR 2.0 engine for every player."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Team A Receives")

        team_a = st.text_area(
            "Players",
            placeholder=(
                "Shohei Ohtani — UTIL\n"
                "Player 2 — SP"
            ),
            key="team_a",
        )

    with col2:

        st.subheader("Team B Receives")

        team_b = st.text_area(
            "Players",
            placeholder=(
                "Rafael Devers — 1B\n"
                "Hunter Brown — SP"
            ),
            key="team_b",
        )


# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------

else:

    st.header("xFVAR 2.0 Methodology")

    st.markdown(
        """
### What xFVAR measures

**xFVAR — Expected Fantasy Value Above Replacement**

estimates the number of additional expected H2H category
wins a fantasy player provides compared with a realistically
available replacement player at the same usable roster position.

**1.00 xFVAR ≈ one expected category win above replacement.**

---

### League

**8 teams**

Hitting roster:

- 1 C
- 1 1B
- 1 2B
- 1 3B
- 1 SS
- 1 IF
- 3 OF
- 2 UTIL

Pitching roster:

- 5 SP
- 3 RP
- 2 P
- 4 bench spots

---

### Hitting Categories

R · HR · RBI · SB · AVG · OPS

### Pitching Categories

SV · K · ERA · WHIP · K/BB · QS

---

### Replacement Bands

| Position | Replacement Pool |
|---|---|
| C | Top 5 active FA catchers |
| 1B | Top 8 active FA 1B |
| 2B | Top 8 active FA 2B |
| 3B | Top 8 active FA 3B |
| SS | Top 8 active FA SS |
| OF | Top 12 active FA OF |
| UTIL | Top 12 active FA hitters |
| SP | Top 12 active FA starter-role pitchers |
| RP | Top 8 active FA reliever-role pitchers |

Players on IL, NA, season-ending injury status, or otherwise
unavailable are excluded from replacement pools.

---

### League Environment

xFVAR 2.0 uses the league's actual completed scoring periods.

Current frozen calibration:

**Weeks 1–19**

**152 team-period observations**

For every category and week, all possible team-vs-team margins
are used to estimate how additional production changes expected
category outcomes.

---

### Important Rules

- No star bonus
- No catcher bonus
- No closer bonus
- No manual saves multiplier
- No name-value adjustment
- No trade-value adjustment
- Negative xFVAR is allowed
- IL time is filled by replacement production
- SP and RP use separate replacement ecosystems
- SP counting value is based on starting opportunities
- RP value is based on realistic relief opportunities
- AVG is workload-weighted using H and AB
- Pitching ratios are workload-weighted using IP
- xFVAR 2.0 is frozen unless an objective mathematical or
  implementation flaw is discovered
"""
    )
