import numpy as np
import matplotlib.pyplot as plt

# 构造数据
labels = ['L1', 'L2', 'L3', 'L4', 'L5']
data_a = [20, 34, 30, 35, 27]
data_b = [25, 32, 34, 20, 25]
data_c = [12, 20, 24, 17, 16]

x = np.arange(len(labels))
width = .25

plt.rcParams['font.family'] = "Times New Roman"
# plots

fig, ax = plt.subplots(figsize=(5, 3), dpi=200)
# plt.grid(axis="y",ls='--',alpha=0.5)
bar_a = ax.bar(x - width / 2, data_a, width, label='category_A', color='#130074', ec='black', lw=.5)
bar_b = ax.bar(x + width / 2, data_b, width, label='category_B', color='#CB181B', ec='black', lw=.5)
bar_c = ax.bar(x+width*3/2, data_c,width,label='category_C',color='#008B45',ec='black',lw=.5)

# 定制化设计
ax.tick_params(axis='x', direction='in', bottom=False)
ax.tick_params(axis='y', direction='out', labelsize=8, length=3)
ax.set_xticks(x + 1)
ax.set_xticklabels(labels, size=10)
ax.set_ylim(bottom=0, top=40)
ax.set_yticks(np.arange(0, 50, step=5))


for spine in ['top', 'right']:
        ax.spines[spine].set_color('none')

ax.legend(fontsize=7, frameon=False)

text_font = {'size': '14', 'weight': 'bold', 'color': 'black'}
ax.text(.03, .93, "(a)", transform=ax.transAxes, fontdict=text_font, zorder=4)
ax.text(.87, -.08, '\nVisualization by DataCharm', transform=ax.transAxes,
        ha='center', va='center', fontsize=5, color='black', fontweight='bold', family='Roboto Mono')
# plt.savefig(r'E:\Data_resourses\DataCharm 公众号\Python\学术图表绘制\bar_class.png', width=5, height=3,
#             dpi=900, bbox_inches='tight')
plt.show()