import matplotlib.pyplot as plt
import seaborn as sns

def line_plot(data, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    ax = data.plot(kind='line', color='blue', marker='o')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    