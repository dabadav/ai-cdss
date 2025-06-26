# Import your CDSS code (adjust import as needed)
import sys

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append("../../src")  # Adjust if needed
from ai_cdss.loaders import DataLoader
from rgs_interface.data.interface import DatabaseInterface

st.set_page_config(layout="wide")
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# --- Sidebar: Study & Patient Selection ---
def sidebar_selection(loader):
    st.sidebar.header("Study & Patient Selection")
    study_id = st.sidebar.selectbox("Select Study", [2])  # You can make this dynamic
    selected_patient = None
    patient_list = []
    if study_id:
        try:
            patient_list = loader.fetch_and_validate_patients(study_ids=[study_id])
            if patient_list:
                selected_patient = st.sidebar.selectbox("Select Patient", patient_list)
            else:
                st.sidebar.warning(f"No patients found for study {study_id}")
        except Exception as e:
            st.sidebar.error(f"Error fetching patients for study {study_id}: {e}")
    else:
        st.sidebar.info("Select a study to view patients and their prescriptions.")
    return study_id, selected_patient, patient_list


# --- Data Fetching ---
def fetch_data(db, selected_patient):
    prescriptions = db.fetch_prescription_staging(patient_id=selected_patient)
    metrics = db.fetch_recsys_metrics(patient_id=selected_patient)
    return prescriptions, metrics


# --- Week Selection ---
def week_selector():
    st.write("Select Week")
    cols = st.columns(12)
    selected_week = None
    for i, col in enumerate(cols):  # 0 to 11
        with col:
            if st.button(f"{i}"):
                selected_week = i
    return selected_week


# --- Calendar View ---
def show_calendar(week_prescriptions):
    if (
        "PROTOCOL_ID" in week_prescriptions.columns
        and "WEEKDAY" in week_prescriptions.columns
    ):
        calendar = week_prescriptions.copy()
        calendar["PROTOCOL_ID"] = calendar["PROTOCOL_ID"].astype(str)
        # Map full weekday names to two-letter abbreviations
        weekday_map = {
            "MONDAY": "MON",
            "TUESDAY": "TUE",
            "WEDNESDAY": "WED",
            "THURSDAY": "THU",
            "FRIDAY": "FRI",
            "SATURDAY": "SAT",
            "SUNDAY": "SUN",
        }
        calendar["WEEKDAY"] = calendar["WEEKDAY"].map(weekday_map)
        weekday_order = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
        protocol_ids = calendar["PROTOCOL_ID"].unique().tolist()

        # Build a set of (protocol, weekday) pairs that are scheduled
        scheduled = set(zip(calendar["PROTOCOL_ID"], calendar["WEEKDAY"]))

        # Build HTML table with dark theme and strong highlight
        html = """
        <style>
        table.calendar-table {
            border-collapse: collapse;
            width: 90%;
            font-size: 0.9em;
            background-color: #23272f;
            color: #f1f1f1;
        }
        table.calendar-table th, table.calendar-table td {
            border: 1px solid #444;
            padding: 8px;
            text-align: center;
            width: 50px; /* Set all columns to the same width */
            min-width: 50px;
            max-width: 50px;
        }
        table.calendar-table th {
            background-color: #14532d;
            color: #fff;
        }
        table.calendar-table tr {
            background-color: #23272f;
        }
        table.calendar-table tr:nth-child(even) {
            background-color: #2c313a;
        }
        table.calendar-table td.scheduled {
            background-color: #00ff99 !important;
            color: #23272f !important;
            font-weight: bold;
            font-size: 0.9em;
            box-shadow: 0 0 8px #00ff99;
            border: 2px solid #fff;
        }
        </style>
        <table class="calendar-table">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>MON</th>
                    <th>TUE</th>
                    <th>WED</th>
                    <th>THU</th>
                    <th>FRI</th>
                    <th>SAT</th>
                    <th>SUN</th>
                </tr>
            </thead>
            <tbody>
        """
        for pid in protocol_ids:
            html += "<tr>"
            html += f"<td>{pid}</td>"
            for day in weekday_order:
                if (pid, day) in scheduled:
                    html += '<td class="scheduled"></td>'
                else:
                    html += "<td></td>"
            html += "</tr>"
        html += """
            </tbody>
        </table>
        """
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.dataframe(week_prescriptions)
        st.info(
            "Prescription data missing PROTOCOL_ID or WEEKDAY columns for calendar view."
        )


# --- Metrics Facet Barplot ---
def plot_metrics_facet_barplot(week_metrics, prescriptions):
    if "METRIC_KEY" in week_metrics.columns:
        plot_data = week_metrics.copy()
        plot_data = plot_data[
            plot_data["METRIC_KEY"].isin(["delta_dm", "adherence", "ppf", "score"])
        ]
        plot_data["PROTOCOL_ID"] = plot_data["PROTOCOL_ID"].astype(str)
        plot_data["METRIC_KEY"] = plot_data["METRIC_KEY"].map(
            {
                "delta_dm": "Delta DM",
                "adherence": "Adherence",
                "ppf": "PPF",
                "score": "SCORE",
            }
        )

        # Sort by score in ascending order
        # First, get the score values for each protocol
        score_data = plot_data[plot_data["METRIC_KEY"] == "SCORE"].copy()
        score_data["METRIC_VALUE"] = pd.to_numeric(
            score_data["METRIC_VALUE"], errors="coerce"
        )
        score_data = score_data.sort_values("METRIC_VALUE", ascending=True)
        protocol_order = score_data["PROTOCOL_ID"].tolist()

        # Create a categorical type with the sorted order
        plot_data["PROTOCOL_ID"] = pd.Categorical(
            plot_data["PROTOCOL_ID"], categories=protocol_order, ordered=True
        )

        # Calculate mean for SCORE
        score_data = plot_data[plot_data["METRIC_KEY"] == "SCORE"].copy()
        score_data["METRIC_VALUE"] = pd.to_numeric(
            score_data["METRIC_VALUE"], errors="coerce"
        )
        score_mean = score_data["METRIC_VALUE"].mean()
        closest_protocol = score_data.iloc[
            (score_data["METRIC_VALUE"] - score_mean).abs().argmin()
        ]["PROTOCOL_ID"]

        # Add mean and closest_protocol columns to plot_data
        plot_data["mean_score"] = None
        plot_data.loc[plot_data["METRIC_KEY"] == "SCORE", "mean_score"] = score_mean
        plot_data["is_closest"] = False
        plot_data.loc[
            (plot_data["METRIC_KEY"] == "SCORE")
            & (plot_data["PROTOCOL_ID"] == closest_protocol),
            "is_closest",
        ] = True

        # 1. Get prescribed protocol IDs as strings
        prescribed_protocols = set(prescriptions["PROTOCOL_ID"].astype(str).unique())

        # 2. Add a boolean column to plot_data
        plot_data["is_prescribed"] = (
            plot_data["PROTOCOL_ID"].astype(str).isin(prescribed_protocols)
        )

        # 3. Update the bar chart color encoding
        bars = (
            alt.Chart(plot_data)
            .mark_bar()
            .encode(
                x=alt.X("PROTOCOL_ID:N", title="Protocol ID", sort=protocol_order),
                y=alt.Y(
                    "METRIC_VALUE:Q",
                    title=None,
                    scale=alt.Scale(domain=[0, 1], zero=True),
                ),
                color=alt.condition(
                    "datum.is_prescribed", alt.value("green"), alt.value("steelblue")
                ),
                tooltip=["PROTOCOL_ID", "METRIC_KEY", "METRIC_VALUE", "is_prescribed"],
            )
        )

        # Horizontal mean line (only for SCORE)
        mean_hline = (
            alt.Chart(plot_data)
            .mark_rule(color="red", strokeDash=[5, 5], strokeWidth=2)
            .encode(y="mean_score:Q")
            .transform_filter(alt.datum.METRIC_KEY == "SCORE")
        )

        # Vertical line at protocol closest to mean (only for SCORE)
        # mean_vline = (
        #     alt.Chart(plot_data)
        #     .mark_rule(color="orange", strokeDash=[2, 2], strokeWidth=2)
        #     .encode(x=alt.X("PROTOCOL_ID:N", sort=protocol_order))
        #     .transform_filter(
        #         (alt.datum.METRIC_KEY == "SCORE") & (alt.datum.is_closest == True)
        #     )
        # )

        # Layer and facet
        chart = (
            alt.layer(bars, mean_hline)
            # alt.layer(bars, mean_hline, mean_vline)
            .properties(width=500, height=100)
            .facet(
                row=alt.Row(
                    "METRIC_KEY:N",
                    title=None,
                    sort=["SCORE", "PPF", "Delta DM", "Adherence"],
                )
            )
            .properties(title="Protocol Metrics (Red dash = mean)")
        )

        st.altair_chart(chart, use_container_width=True)
    else:
        st.error(
            "No 'METRIC_KEY' column found in metrics data. Available columns: "
            + str(list(week_metrics.columns))
        )


# --- Main Dashboard Logic ---
def main():
    db = DatabaseInterface()
    loader = DataLoader(rgs_mode="plus")
    study_id, selected_patient, patient_list = sidebar_selection(loader)

    if selected_patient:
        prescriptions, metrics = fetch_data(db, selected_patient)
        if prescriptions is not None and not prescriptions.empty:
            selected_week = week_selector()
            if selected_week is not None:
                if "WEEKS_SINCE_START" in prescriptions.columns:
                    week_prescriptions = prescriptions[
                        prescriptions["WEEKS_SINCE_START"] == selected_week
                    ]
                else:
                    week_prescriptions = prescriptions
                    st.info(
                        "No WEEKS_SINCE_START information in prescriptions data. Showing all prescriptions."
                    )
                # --- Two-column layout ---
                col1, col2 = st.columns([1, 1.5])
                with col1:
                    if not week_prescriptions.empty:
                        show_calendar(week_prescriptions)
                        st.write(
                            f"Found {len(week_prescriptions)} prescriptions for Week {selected_week}"
                        )
                    else:
                        st.info(f"No prescriptions found for Week {selected_week}")
                with col2:
                    # Metrics facet barplot
                    if metrics is not None and not metrics.empty:
                        # Join metrics with prescriptions for the selected week
                        if not week_prescriptions.empty:
                            week_metrics = metrics.merge(
                                week_prescriptions[
                                    ["RECOMMENDATION_ID", "PATIENT_ID", "PROTOCOL_ID"]
                                ].drop_duplicates(),
                                on=["RECOMMENDATION_ID", "PATIENT_ID", "PROTOCOL_ID"],
                                how="left",
                            )
                        else:
                            week_metrics = metrics[
                                metrics["PATIENT_ID"] == selected_patient
                            ]
                        if not week_metrics.empty:
                            plot_metrics_facet_barplot(week_metrics, prescriptions)
                        else:
                            st.info("No metrics data found for the selected week.")
                    else:
                        st.info(f"No metrics found for patient {selected_patient}")
        else:
            st.info(f"No prescriptions found for patient {selected_patient}")
    else:
        st.info(
            "Please select a study and patient from the sidebar to view prescriptions."
        )


if __name__ == "__main__":
    main()
