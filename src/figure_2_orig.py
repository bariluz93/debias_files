import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font="Verdana")
sns.set_style("white")

df = pd.DataFrame([[-0.01, 1.5, 'de-encoder-hard'],
                   [-0.02, 2.8, 'he-encoder-hard'],
                   [-0.02, 0.2, 'ru-encoder-hard'],

                   [0, 0, ''],

                   [0, -0.1, 'de-decoder-in-hard'],
                   [-0.04, -0.1, 'he-decoder-in-hard'],

                   [0, 0, ' '],

                   [0, -0.1, 'de-decoder-out-hard'],
                   [-0.04, -0.1, 'he-decoder-out-hard'],

                   [0, 0, '  '],
                   [0, 0, '   '],

                   [-0.05, -2.8, 'de-encoder-inlp'],
                   [-0.1, -3.4, 'he-encoder-inlp'],
                   [-0.02, -8.6, 'ru-encoder-inlp'],

                   [0, 0, '    '],

                   [-2.95, 2.7, 'de-decoder-in-inlp'],
                   [-0.62, 1, 'he-decoder-in-inlp'],

                   [0, 0, '     '],

                   [-2.82, 3.7, 'de-decoder-out-inlp'],
                   [-0.62, 1, 'he-decoder-out-inlp']],

                  columns=['delta BLEU', 'delta acc', 'lang'])

ax = sns.barplot(x=df["lang"], y=df["delta acc"], color=sns.color_palette()[1])
ax = sns.barplot(x=df["lang"], y=df["delta BLEU"], color=sns.color_palette()[0])

# plt.plot([0, 0], [0, 5], linewidth=2)
ax.axhline(0, color='black')
ax.set(xlabel='', ylabel='relative change')

# plt.xticks(rotation=90)

labels = ['$\Delta$ acc', '$\Delta$ BLEU']
# plt.legend(loc='upper left', labels=['$\Delta$ acc', '$\Delta$ BLEU'], facecolor='w')
handles = []
for color, label in zip([sns.color_palette()[1], sns.color_palette()[0]], labels):
    handles.append(plt.scatter([], [], c=color, label=label))
plt.legend(handles=handles)

for ind, bars in enumerate(ax.patches):
    if ind < 20:
        c = color = sns.color_palette()[1]
        ext = 0.08
    else:
        c = color = sns.color_palette()[0]
        ext = -0.4

    skip = [3, 6, 9, 10, 14, 17, 23, 26, 29, 30, 34, 37]
    if ind not in skip:
        if ind in [4, 5, 7, 8]:
            ax.text(bars.get_x(), bars.get_height() + 0.25, bars.get_height(), fontsize=7, color=c)
        elif ind in [11, 12, 13]:
            ax.text(bars.get_x(), bars.get_height() - 0.5, bars.get_height(), fontsize=7, color=c)
        elif ind in [31, 32, 33]:
            ax.text(bars.get_x(), bars.get_height() + 0.3, bars.get_height(), fontsize=7, color=c)
        else:
            ax.text(bars.get_x() + 0.15, bars.get_height() + ext, bars.get_height(), fontsize=7, color=c)

plt.xticks([0, 1, 2, 4, 5, 7, 8, 11, 12, 13, 15, 16, 18, 19])

ax.set_xticklabels(['de', 'he', 'ru', 'de', 'he', 'de', 'he', 'de', 'he', 'ru', 'de', 'he', 'de', 'he'])

ax.text(0.5, -11, 'Enc', fontsize=9)
ax.text(3.5, -11, 'Dec_in', fontsize=9)
ax.text(6.5, -11, 'Dec_out', fontsize=9)

ax.text(11.5, -11, 'Enc', fontsize=9)
ax.text(14.5, -11, 'Dec_in', fontsize=9)
ax.text(17.5, -11, 'Dec_out', fontsize=9)

ax.text(1.7, 4.5, 'Hard Debiasing', fontsize=12)
ax.text(14.5, 4.5, 'INLP', fontsize=12)

ax.axvline(9.5, ls='--', c='black')

plt.tight_layout()

plt.savefig("acc-bleu.pdf")

plt.show()

bars.get_x()

bars.get_height()
bars.get_width()

for bars in ax.patches:
    print(bars.get_x(),bars.get_height(),bars.get_height())

len(ax.patches)