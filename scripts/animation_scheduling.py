import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import math
# Adjusting figure size and layout to ensure the legend is not cut off
# Adding the number of protocols (n) to the parameter display

# Parameters
days = 7
protocols_per_day = 5
protocols = [101, 102, 103, 104, 105]
n = len(protocols)
total_slots = days * protocols_per_day

# Repeat protocols to fill the schedule
repeated_protocols = (protocols * math.ceil(total_slots / len(protocols)))[:total_slots]

# Prepare schedule and animation frames
schedule = {day: [] for day in range(days)}
frames = []

# Simulate scheduling and capture frames
for i, protocol in enumerate(repeated_protocols):
    day = i % days
    if protocol not in schedule[day] and len(schedule[day]) < protocols_per_day:
        schedule[day].append(protocol)
    frames.append({d: list(schedule[d]) for d in range(days)})

# Assign unique colors to each protocol
protocol_colors = {
    101: '#1f77b4',
    102: '#ff7f0e',
    103: '#2ca02c',
    104: '#d62728',
    105: '#9467bd',
}

# Initialize plot
fig, ax = plt.subplots(figsize=(14, 7))
fig.subplots_adjust(right=0.75)  # Leave space for legend and text

def update(frame_index):
    ax.clear()
    ax.set_xlim(-0.5, days - 0.5)
    ax.set_ylim(0, protocols_per_day + 1)
    ax.set_xticks(range(days))
    ax.set_xticklabels([f"Day {i+1}" for i in range(days)])
    ax.set_ylabel("Scheduled Protocols")
    ax.set_title("Protocol Scheduling Animation")

    # Draw bars
    current_schedule = frames[frame_index]
    for day, protocols_today in current_schedule.items():
        for level, protocol in enumerate(protocols_today):
            ax.bar(day, 1, bottom=level, color=protocol_colors[protocol], edgecolor='black')

    # Add legend and parameter box
    legend_patches = [mpatches.Patch(color=color, label=f'Protocol {pid}') for pid, color in protocol_colors.items()]
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.15, 1))
    ax.text(1.15, 0.7,
            f"Parameters:\nDays: {days}\nProtocols/Day: {protocols_per_day}\nTotal Protocols: {n}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8))

    return []

# Create and save animation
ani = animation.FuncAnimation(fig, update, frames=len(frames), repeat=False, interval=500)
ani.save("/mnt/data/protocol_scheduling_full.gif", writer="pillow")

"/mnt/data/protocol_scheduling_full.gif"
