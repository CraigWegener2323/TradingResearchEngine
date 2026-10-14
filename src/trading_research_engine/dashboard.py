"""Streamlit viewer for isolated hypothetical forward-test sessions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    import plotly.express as px
    import streamlit as st

    st.set_page_config(page_title="Forward Test Monitor", layout="wide")
    st.title("Hypothetical Forward-Test Monitor")
    st.caption(
        "Simulation only. This dashboard has no broker connection and cannot "
        "submit live orders."
    )
    root = Path(st.sidebar.text_input("Forward session folder", r"runs\forward_tests"))
    if st.sidebar.button("Refresh"):
        st.rerun()

    if not root.is_dir():
        st.info(f"No forward-test folder found at {root}.")
        return
    state_paths = sorted(
        root.glob("*/forward_state.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not state_paths:
        st.info("No persisted forward-test sessions were found.")
        return

    sessions: list[tuple[Path, dict[str, object]]] = []
    for path in state_paths:
        with path.open("r", encoding="utf-8") as file:
            sessions.append((path.parent, json.load(file)))

    active_rows = []
    for directory, state in sessions:
        active_rows.append(
            {
                "session": directory.name,
                "strategy": state["active_strategy"]["name"],
                "version": state["active_strategy"]["version"],
                "status": "finalized" if state["finalized"] else "active",
                "last event": state["last_event_timestamp"],
                "equity": state["metrics"].get("ending_equity"),
                "max drawdown": state["metrics"].get("max_drawdown_dollars"),
            }
        )
    st.subheader("Active strategies")
    st.dataframe(pd.DataFrame(active_rows), use_container_width=True, hide_index=True)

    selected_name = st.selectbox(
        "Session details",
        [directory.name for directory, _ in sessions],
    )
    directory, state = next(
        (directory, state)
        for directory, state in sessions
        if directory.name == selected_name
    )
    metrics = state["metrics"]
    st.subheader(f"{state['active_strategy']['name']} — {selected_name}")
    st.caption(
        f"Version {state['active_strategy']['version']} | "
        f"{'Finalized' if state['finalized'] else 'Running'} | "
        f"Last event: {state['last_event_timestamp'] or 'None'}"
    )
    metric_columns = st.columns(6)
    metric_columns[0].metric("Hypothetical equity", _format_currency(metrics.get("ending_equity")))
    metric_columns[1].metric("Realized P&L", _format_currency(state["realized_pnl"]))
    metric_columns[2].metric("Unrealized P&L", _format_currency(state["unrealized_pnl"]))
    metric_columns[3].metric("Expectancy / trade", _format_currency(metrics.get("expectancy_per_trade")))
    metric_columns[4].metric("Max drawdown", _format_currency(metrics.get("max_drawdown_dollars")))
    metric_columns[5].metric("Completed trades", str(metrics.get("trade_count", 0)))

    position = state.get("active_position")
    st.subheader("Simulated position")
    if position is None:
        st.write("Flat")
    else:
        st.json(position)

    equity_path = directory / "forward_equity.csv"
    if equity_path.is_file() and equity_path.stat().st_size:
        equity = pd.read_csv(equity_path)
        if not equity.empty:
            equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True)
            st.subheader("Hypothetical equity and drawdown")
            figure = px.line(
                equity,
                x="timestamp",
                y="equity",
                title="Equity (realized plus marked-to-market open position)",
            )
            st.plotly_chart(figure, use_container_width=True)
            drawdown_figure = px.area(
                equity,
                x="timestamp",
                y="drawdown_dollars",
                title="Drawdown in dollars",
            )
            st.plotly_chart(drawdown_figure, use_container_width=True)

    events_path = directory / "forward_events.jsonl"
    if events_path.is_file() and events_path.stat().st_size:
        events = pd.read_json(events_path, lines=True)
        st.subheader("Signals and execution log")
        filters = ["all"] + sorted(events["event_type"].dropna().unique().tolist())
        selected_type = st.selectbox("Event type", filters)
        if selected_type != "all":
            events = events.loc[events["event_type"] == selected_type]
        st.dataframe(events.iloc[::-1], use_container_width=True, hide_index=True)

    for filename, title in (
        ("forward_orders.csv", "Simulated orders"),
        ("forward_trades.csv", "Closed hypothetical trades"),
    ):
        table_path = directory / filename
        if table_path.is_file() and table_path.stat().st_size:
            table = pd.read_csv(table_path)
            with st.expander(title):
                st.dataframe(table, use_container_width=True, hide_index=True)


def _format_currency(value: object) -> str:
    return "—" if value is None or pd.isna(value) else f"${float(value):,.2f}"


if __name__ == "__main__":
    main()
