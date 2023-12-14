import matplotlib.pyplot as plt
import pandas as pd

# Load CSV files
df1 = pd.read_csv('average_star_all.csv')
df2 = pd.read_csv('average_star_mw.csv')
df3 = pd.read_csv('average_with_block.csv')
df4 = pd.read_csv('average_without_block.csv')

# Plot settings for larger and bold text
plt.rcParams.update({'font.size': 30, 'font.weight': 'bold'})

# First plot
plt.figure()
plt.plot(df1['avg_fractions'], df1['avg_f1_star'], label='F1* mw, brand, potential modelid', linestyle='-', marker='o', linewidth=3.5)
plt.plot(df2['avg_fractions'], df2['avg_f1_star'], label='F1* only mw', linestyle='-', marker='x', linewidth=3.5)

plt.xlabel('Average Fraction of Comparisons', fontweight='bold', fontsize=30)
plt.ylabel('Average F1 Score', fontweight='bold', fontsize=30)
plt.legend()
plt.title('Comparison of F1 star Scores different binary matrices')

# second plot
plt.figure()
plt.plot(df3['avg_fractions'], df3['avg_f1'], label='F1 with blocking', linestyle='-', marker='o', linewidth=3.5)
plt.plot(df4['avg_fractions'], df4['avg_f1'], label='F1 without blocking', linestyle='-', marker='x', linewidth=3.5)

plt.xlabel('Average Fraction of Comparisons', fontweight='bold', fontsize=30)
plt.ylabel('Average F1 Score', fontweight='bold', fontsize=30)
plt.legend()
plt.title('Comparison of F1 Scores with and without blocking')

plt.show()