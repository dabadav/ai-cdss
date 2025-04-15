import pandas as pd
import plotly.subplots as sp
import plotly.graph_objs as go
import math

scores_df = pd.read_csv('nest_scores.csv', index_col=0)

# Assuming scores_df is your DataFrame
metrics = ['DM_VALUE', 'ADHERENCE', 'PPF']
relative_weeks = sorted(scores_df['RELATIVE_WEEK'].unique())
weeks_per_page = 10
total_pages = math.ceil(len(relative_weeks) / weeks_per_page)

for page in range(total_pages):
    # Get weeks for this page
    start = page * weeks_per_page
    end = start + weeks_per_page
    weeks_chunk = relative_weeks[start:end]

    # Create subplots
    fig = sp.make_subplots(
        rows=len(metrics),
        cols=len(weeks_chunk),
        subplot_titles=[f'Week {week}' for week in weeks_chunk],
        shared_yaxes=True,
        horizontal_spacing=0.02,
        vertical_spacing=0.1
    )

    # Add histograms
    for row, metric in enumerate(metrics, start=1):
        for col, week in enumerate(weeks_chunk, start=1):
            data = scores_df[scores_df['RELATIVE_WEEK'] == week][metric]
            fig.add_trace(
                go.Histogram(
                    x=data,
                    nbinsx=15,
                    marker=dict(line=dict(width=1, color='black')),
                    opacity=0.75,
                    showlegend=False
                ),
                row=row,
                col=col
            )

    # Update layout
    fig.update_layout(
        height=400 * len(metrics),
        width=300 * len(weeks_chunk),
        title_text=f'Distributions of DM, Adherence, and PPF per Relative Week (Weeks {weeks_chunk[0]}–{weeks_chunk[-1]})',
        barmode='overlay'
    )

    # Export to HTML
    fig.write_html(f"histograms_per_week_page_{page + 1}.html")

    # Optionally display in notebook
    # fig.show()
