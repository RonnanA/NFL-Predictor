import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from sklearn.pipeline import Pipeline

from features import (
    data_cleanup,
    build_team_snapshots,
    build_matchup_frame,
    _extract_team_view,
    _add_rest_days,
    _add_rolling_avgs,
    _add_season_to_date_avgs,
    _add_streak,
    STAT_COLS,
    ROLLING_WINDOW,
)
from model import FEATURE_COLS
from predict import make_prediction, get_latest_snapshot, _build_matchup_row


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def make_raw_game(
    week=1, season=2024,
    home_team="KC", away_team="BUF",
    home_score=27, away_score=20,
    date="2024-09-08",
):
    """Minimal raw game row matching the processed CSV schema."""
    return {
        "week": week, "week_day": "Sun", "event_date": date,
        "game_time": "1:00PM", "home_team": home_team, "away_team": away_team,
        "home_score": home_score, "away_score": away_score,
        "home_first_downs": 20, "away_first_downs": 18,
        "home_rush_att": 25, "away_rush_att": 22,
        "home_rush_yds": 120, "away_rush_yds": 90,
        "home_rush_tds": 1, "away_rush_tds": 1,
        "home_pass_cmp": 20, "away_pass_cmp": 18,
        "home_pass_att": 30, "away_pass_att": 28,
        "home_pass_yds": 250, "away_pass_yds": 220,
        "home_pass_tds": 2, "away_pass_tds": 1,
        "home_pass_int": 0, "away_pass_int": 1,
        "home_sacks": 2, "away_sacks": 1,
        "home_sack_yds": 12, "away_sack_yds": 6,
        "home_net_pass_yards": 238, "away_net_pass_yards": 214,
        "home_total_yards": 358, "away_total_yards": 304,
        "home_fmbl": 0, "away_fmbl": 1,
        "home_fmbl_yds": 0, "away_fmbl_yds": 5,
        "home_turnovers": 0, "away_turnovers": 2,
        "home_pen": 5, "away_pen": 6,
        "home_pen_yds": 40, "away_pen_yds": 50,
        "home_third_d": 8, "away_third_d": 7,
        "home_thrid_d_conv": 5, "away_thrid_d_conv": 3,
        "home_fourth_d": 1, "away_fourth_d": 0,
        "home_fourth_d_conv": 1, "away_fourth_d_conv": 0,
        "home_time_of_possession": "32:00", "away_time_of_possession": "28:00",
        "url": "http://example.com",
    }


@pytest.fixture
def single_game_df():
    return pd.DataFrame([make_raw_game()])


@pytest.fixture
def multi_game_df():
    """
    6 games across 2 seasons for KC and BUF.
    Enough to test rolling windows, streaks, and season boundaries.
    """
    rows = [
        # 2023 season - KC wins all 3
        make_raw_game(week=1, season=2023, home_team="KC", away_team="BUF", home_score=30, away_score=20, date="2023-09-10"),
        make_raw_game(week=2, season=2023, home_team="BUF", away_team="KC", home_score=14, away_score=27, date="2023-09-17"),
        make_raw_game(week=3, season=2023, home_team="KC", away_team="BUF", home_score=24, away_score=17, date="2023-09-24"),
        # 2024 season - BUF wins first, KC wins next two
        make_raw_game(week=1, season=2024, home_team="BUF", away_team="KC", home_score=28, away_score=21, date="2024-09-08"),
        make_raw_game(week=2, season=2024, home_team="KC", away_team="BUF", home_score=27, away_score=20, date="2024-09-15"),
        make_raw_game(week=3, season=2024, home_team="BUF", away_team="KC", home_score=17, away_score=24, date="2024-09-22"),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def cleaned_multi_df(multi_game_df):
    dfs = []
    for season in [2023, 2024]:
        raw = multi_game_df[multi_game_df["week_day"].notna()].copy()
        raw = raw[multi_game_df["week"].isin([1, 2, 3])]
        cleaned = data_cleanup(raw[raw["event_date"].str.startswith(str(season))], season)
        dfs.append(cleaned)
    return pd.concat(dfs).reset_index(drop=True)


@pytest.fixture
def snapshots_df(cleaned_multi_df):
    return build_team_snapshots(cleaned_multi_df)


# ─────────────────────────────────────────────
# data_cleanup
# ─────────────────────────────────────────────

class TestDataCleanup:

    def test_adds_season_column(self, single_game_df):
        result = data_cleanup(single_game_df, 2024)
        assert "season" in result.columns
        assert (result["season"] == 2024).all()

    def test_drops_postseason_rows(self):
        rows = [make_raw_game(week=1)]
        for label in ["WildCard", "Divisional", "ConfChamp", "SuperBowl"]:
            rows.append({**make_raw_game(), "week": label})
        df = pd.DataFrame(rows)
        result = data_cleanup(df, 2024)
        assert len(result) == 1
        assert result["week"].iloc[0] == 1

    def test_drops_metadata_columns(self, single_game_df):
        result = data_cleanup(single_game_df, 2024)
        for col in ["url", "week_day", "game_time",
                    "home_time_of_possession", "away_time_of_possession"]:
            assert col not in result.columns

    def test_keeps_event_date(self, single_game_df):
        result = data_cleanup(single_game_df, 2024)
        assert "event_date" in result.columns

    def test_numeric_coercion(self):
        row = make_raw_game()
        row["home_score"] = "27"
        row["away_score"] = "20"
        df = pd.DataFrame([row])
        result = data_cleanup(df, 2024)
        assert result["home_score"].dtype in [np.float64, np.int64]

    def test_drops_rows_missing_scores(self):
        rows = [make_raw_game(), {**make_raw_game(), "home_score": None}]
        df = pd.DataFrame(rows)
        result = data_cleanup(df, 2024)
        assert len(result) == 1

    def test_does_not_mutate_input(self, single_game_df):
        original_cols = list(single_game_df.columns)
        data_cleanup(single_game_df, 2024)
        assert list(single_game_df.columns) == original_cols


# ─────────────────────────────────────────────
# _extract_team_view
# ─────────────────────────────────────────────

class TestExtractTeamView:

    def test_home_perspective_columns(self, single_game_df):
        cleaned = data_cleanup(single_game_df, 2024)
        view = _extract_team_view(cleaned, "home")
        assert "team" in view.columns
        assert "score" in view.columns
        assert "opp_score" in view.columns
        assert "won" in view.columns

    def test_home_team_assigned_correctly(self, single_game_df):
        cleaned = data_cleanup(single_game_df, 2024)
        view = _extract_team_view(cleaned, "home")
        assert view["team"].iloc[0] == "KC"

    def test_away_team_assigned_correctly(self, single_game_df):
        cleaned = data_cleanup(single_game_df, 2024)
        view = _extract_team_view(cleaned, "away")
        assert view["team"].iloc[0] == "BUF"

    def test_won_flag_home_winner(self, single_game_df):
        cleaned = data_cleanup(single_game_df, 2024)
        home_view = _extract_team_view(cleaned, "home")
        away_view = _extract_team_view(cleaned, "away")
        # KC 27 > BUF 20 → home won, away lost
        assert home_view["won"].iloc[0] == 1
        assert away_view["won"].iloc[0] == 0

    def test_won_flag_away_winner(self):
        row = make_raw_game(home_score=14, away_score=28)
        cleaned = data_cleanup(pd.DataFrame([row]), 2024)
        home_view = _extract_team_view(cleaned, "home")
        away_view = _extract_team_view(cleaned, "away")
        assert home_view["won"].iloc[0] == 0
        assert away_view["won"].iloc[0] == 1


# ─────────────────────────────────────────────
# _add_rest_days
# ─────────────────────────────────────────────

class TestAddRestDays:

    def test_first_game_gets_default(self, cleaned_multi_df):
        long = pd.concat([
            _extract_team_view(cleaned_multi_df, "home"),
            _extract_team_view(cleaned_multi_df, "away"),
        ]).sort_values(["team", "season", "week"]).reset_index(drop=True)

        result = _add_rest_days(long)
        kc_week1 = result[(result["team"] == "KC") & (result["season"] == 2023) & (result["week"] == 1)]
        assert kc_week1["rest_days"].iloc[0] == 7  # REST_DAYS_DEFAULT

    def test_rest_days_calculated_correctly(self, cleaned_multi_df):
        long = pd.concat([
            _extract_team_view(cleaned_multi_df, "home"),
            _extract_team_view(cleaned_multi_df, "away"),
        ]).sort_values(["team", "season", "week"]).reset_index(drop=True)

        result = _add_rest_days(long)
        kc_2023 = result[(result["team"] == "KC") & (result["season"] == 2023)].sort_values("week")
        # week 1: 2023-09-10, week 2 (away): 2023-09-17 → 7 days
        assert kc_2023.iloc[1]["rest_days"] == 7

    def test_no_negative_rest_days(self, cleaned_multi_df):
        long = pd.concat([
            _extract_team_view(cleaned_multi_df, "home"),
            _extract_team_view(cleaned_multi_df, "away"),
        ]).sort_values(["team", "season", "week"]).reset_index(drop=True)
        result = _add_rest_days(long)
        assert (result["rest_days"] >= 0).all()


# ─────────────────────────────────────────────
# _add_rolling_avgs
# ─────────────────────────────────────────────

class TestAddRollingAvgs:

    @pytest.fixture
    def long_df(self, cleaned_multi_df):
        long = pd.concat([
            _extract_team_view(cleaned_multi_df, "home"),
            _extract_team_view(cleaned_multi_df, "away"),
        ]).sort_values(["team", "season", "week"]).reset_index(drop=True)
        return _add_rest_days(long)

    def test_rolling_cols_created(self, long_df):
        result = _add_rolling_avgs(long_df)
        for stat in STAT_COLS:
            assert f"roll_{stat}" in result.columns

    def test_first_game_rolling_is_nan(self, long_df):
        """First game has no prior games so rolling avg should be NaN."""
        result = _add_rolling_avgs(long_df)
        kc_first = result[(result["team"] == "KC")].sort_values(["season", "week"]).iloc[0]
        assert pd.isna(kc_first["roll_score"])

    def test_rolling_excludes_current_game(self, long_df):
        """
        The rolling avg at game N should only use games 1..N-1.
        For KC: game 1 score=30. Game 2 roll_score should equal 30, not
        include game 2's score.
        """
        result = _add_rolling_avgs(long_df)
        kc = result[(result["team"] == "KC")].sort_values(["season", "week"])
        # second KC game rolling avg should be the first game's score only
        assert kc.iloc[1]["roll_score"] == pytest.approx(kc.iloc[0]["score"])

    def test_rolling_window_respects_size(self, long_df):
        """After enough games, rolling avg should only use last ROLLING_WINDOW games."""
        result = _add_rolling_avgs(long_df)
        kc = result[(result["team"] == "KC")].sort_values(["season", "week"])
        if len(kc) > ROLLING_WINDOW:
            # manual check: avg of prior ROLLING_WINDOW scores
            idx = ROLLING_WINDOW
            expected = kc.iloc[:idx]["score"].mean()
            assert kc.iloc[idx]["roll_score"] == pytest.approx(expected, rel=1e-3)


# ─────────────────────────────────────────────
# _add_streak
# ─────────────────────────────────────────────

class TestAddStreak:

    @pytest.fixture
    def long_df(self, cleaned_multi_df):
        long = pd.concat([
            _extract_team_view(cleaned_multi_df, "home"),
            _extract_team_view(cleaned_multi_df, "away"),
        ]).sort_values(["team", "season", "week"]).reset_index(drop=True)
        return long

    def test_streak_col_created(self, long_df):
        result = _add_streak(long_df)
        assert "streak" in result.columns

    def test_first_game_streak_is_zero(self, long_df):
        result = _add_streak(long_df)
        kc_first = result[(result["team"] == "KC")].sort_values(["season", "week"]).iloc[0]
        assert kc_first["streak"] == 0

    def test_win_streak_increments(self, long_df):
        """KC wins games 1 and 2 in 2023, so streak before game 3 should be +2."""
        result = _add_streak(long_df)
        kc_2023 = result[(result["team"] == "KC") & (result["season"] == 2023)].sort_values("week")
        assert kc_2023.iloc[2]["streak"] == 2

    def test_loss_streak_decrements(self, long_df):
        """BUF loses games 1 and 2 in 2023, so streak before game 3 should be -2."""
        result = _add_streak(long_df)
        buf_2023 = result[(result["team"] == "BUF") & (result["season"] == 2023)].sort_values("week")
        assert buf_2023.iloc[2]["streak"] == -2

    def test_streak_resets_each_season(self, long_df):
        """Streak at week 1 of a new season should be 0 regardless of prior season."""
        result = _add_streak(long_df)
        kc_2024_w1 = result[
            (result["team"] == "KC") &
            (result["season"] == 2024) &
            (result["week"] == 1)
        ]
        assert kc_2024_w1["streak"].iloc[0] == 0

    def test_streak_switches_sign_on_result_change(self):
        """W, W, L should give streak of -1 before the 4th game."""
        won = pd.Series([1, 1, 0, 1])
        long = pd.DataFrame({
            "team": ["KC"] * 4,
            "season": [2024] * 4,
            "week": [1, 2, 3, 4],
            "won": won,
        })
        result = _add_streak(long)
        assert result.iloc[3]["streak"] == -1


# ─────────────────────────────────────────────
# build_team_snapshots
# ─────────────────────────────────────────────

class TestBuildTeamSnapshots:

    def test_output_has_one_row_per_team_per_game(self, cleaned_multi_df, snapshots_df):
        # 6 games × 2 teams per game = 12 team-game rows
        assert len(snapshots_df) == 12

    def test_required_columns_present(self, snapshots_df):
        expected = (
            ["team", "season", "week", "streak", "rest_days"] +
            [f"roll_{s}" for s in STAT_COLS] +
            [f"std_{s}"  for s in STAT_COLS]
        )
        for col in expected:
            assert col in snapshots_df.columns, f"missing column: {col}"

    def test_event_date_dropped(self, snapshots_df):
        assert "event_date" not in snapshots_df.columns

    def test_raw_stat_cols_dropped(self, snapshots_df):
        for stat in STAT_COLS:
            assert stat not in snapshots_df.columns

    def test_no_duplicate_team_week_season(self, snapshots_df):
        dupes = snapshots_df.duplicated(subset=["team", "season", "week"])
        assert not dupes.any()


# ─────────────────────────────────────────────
# build_matchup_frame
# ─────────────────────────────────────────────

class TestBuildMatchupFrame:

    @pytest.fixture
    def matchup_df(self, cleaned_multi_df, snapshots_df):
        return build_matchup_frame(cleaned_multi_df, snapshots_df)

    def test_has_label_column(self, matchup_df):
        assert "home_team_wins" in matchup_df.columns

    def test_label_is_binary(self, matchup_df):
        assert set(matchup_df["home_team_wins"].unique()).issubset({0, 1})

    def test_diff_columns_present(self, matchup_df):
        for stat in STAT_COLS:
            assert f"diff_roll_{stat}" in matchup_df.columns
            assert f"diff_std_{stat}"  in matchup_df.columns

    def test_rest_day_columns_not_diffed(self, matchup_df):
        assert "home_rest_days" in matchup_df.columns
        assert "away_rest_days" in matchup_df.columns
        assert "diff_rest_days" not in matchup_df.columns

    def test_no_lookahead(self, cleaned_multi_df, snapshots_df):
        """
        Features for a game in week N should only use snapshots
        from weeks strictly less than N in the same season.
        """
        matchup_df = build_matchup_frame(cleaned_multi_df, snapshots_df)
        # week 1 rows should either be NaN (dropped) or use prior season data only
        # since they're dropna'd, week 1 rows from 2023 should not appear
        # (no prior season data available for 2023)
        w1_2023 = matchup_df[(matchup_df["season"] == 2023) & (matchup_df["week"] == 1)]
        assert len(w1_2023) == 0

    def test_feature_cols_match_model(self, matchup_df):
        """All FEATURE_COLS from model.py must be present in the matchup frame."""
        for col in FEATURE_COLS:
            assert col in matchup_df.columns, f"missing model feature: {col}"


# ─────────────────────────────────────────────
# predict
# ─────────────────────────────────────────────

class TestGetLatestSnapshot:

    def test_returns_most_recent_row(self, snapshots_df):
        snap = get_latest_snapshot(snapshots_df, "KC")
        assert snap is not None
        kc_rows = snapshots_df[snapshots_df["team"] == "KC"].sort_values(["season", "week"])
        assert snap["week"] == kc_rows.iloc[-1]["week"]
        assert snap["season"] == kc_rows.iloc[-1]["season"]

    def test_returns_none_for_unknown_team(self, snapshots_df):
        result = get_latest_snapshot(snapshots_df, "INVALID")
        assert result is None


class TestBuildMatchupRow:

    @pytest.fixture
    def snap(self, snapshots_df):
        return get_latest_snapshot(snapshots_df, "KC")

    def test_returns_dataframe(self, snap):
        row = _build_matchup_row(snap, snap)
        assert isinstance(row, pd.DataFrame)

    def test_columns_match_feature_cols(self, snap):
        row = _build_matchup_row(snap, snap)
        assert list(row.columns) == FEATURE_COLS

    def test_same_team_diffs_are_zero(self, snap):
        """Diffing a team against itself should give 0 for all diff columns."""
        row = _build_matchup_row(snap, snap)
        diff_cols = [c for c in FEATURE_COLS if c.startswith("diff_")]
        for col in diff_cols:
            assert row[col].iloc[0] == pytest.approx(0.0), f"{col} should be 0"


class TestMakePrediction:

    @pytest.fixture
    def mock_pipe(self):
        pipe = MagicMock(spec=Pipeline)
        pipe.predict_proba.return_value = np.array([[0.35, 0.65]])
        return pipe

    def test_returns_dict(self, mock_pipe, snapshots_df):
        result = make_prediction("KC", "BUF", mock_pipe, snapshots_df)
        assert isinstance(result, dict)

    def test_result_keys(self, mock_pipe, snapshots_df):
        result = make_prediction("KC", "BUF", mock_pipe, snapshots_df)
        assert "home_team" in result
        assert "away_team" in result
        assert "home_win_prob" in result
        assert "predicted_winner" in result
        assert "confidence" in result

    def test_home_win_prob_matches_mock(self, mock_pipe, snapshots_df):
        result = make_prediction("KC", "BUF", mock_pipe, snapshots_df)
        assert result["home_win_prob"] == pytest.approx(0.65, rel=1e-3)

    def test_predicted_winner_home(self, mock_pipe, snapshots_df):
        # prob=0.65 → home team wins
        result = make_prediction("KC", "BUF", mock_pipe, snapshots_df)
        assert result["predicted_winner"] == "KC"

    def test_predicted_winner_away(self, snapshots_df):
        pipe = MagicMock(spec=Pipeline)
        pipe.predict_proba.return_value = np.array([[0.70, 0.30]])
        result = make_prediction("KC", "BUF", pipe, snapshots_df)
        assert result["predicted_winner"] == "BUF"

    def test_confidence_is_always_above_50(self, mock_pipe, snapshots_df):
        result = make_prediction("KC", "BUF", mock_pipe, snapshots_df)
        assert result["confidence"] >= 0.5

    def test_unknown_team_returns_error(self, mock_pipe, snapshots_df):
        result = make_prediction("KC", "INVALID", mock_pipe, snapshots_df)
        assert "error" in result

    def test_prob_is_between_0_and_1(self, mock_pipe, snapshots_df):
        result = make_prediction("KC", "BUF", mock_pipe, snapshots_df)
        assert 0.0 <= result["home_win_prob"] <= 1.0