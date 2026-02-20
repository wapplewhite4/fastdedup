#!/usr/bin/env python3
"""Create benchmark visualization"""
import json
import matplotlib.pyplot as plt

with open('benchmarks/results.json', 'r') as f:
    data = json.load(f)

tools = []
times = []

for result in data['results']:
    command = result['command']
    if 'pandas' in command:
        tools.append('Pandas')
    elif 'polars' in command:
        tools.append('Polars')
    elif 'streaming' in command:
        tools.append('Streaming\nPython')
    else:
        tools.append('Rust\nfastdedup')

    times.append(result['mean'])

# Create bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(tools, times, color=['#e74c3c', '#3498db', '#95a5a6', '#2ecc71'])

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}s',
            ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('Time (seconds)', fontsize=12)
ax.set_title('fastdedup Performance Comparison', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(times) * 1.15)

# Add speedup annotation
rust_time = times[-1]
pandas_time = times[0]
speedup = pandas_time / rust_time

ax.text(0.5, 0.95, f'{speedup:.1f}x faster than Pandas',
        transform=ax.transAxes,
        ha='center', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('benchmarks/comparison_chart.png', dpi=300, bbox_inches='tight')
print("✓ Saved chart to benchmarks/comparison_chart.png")
