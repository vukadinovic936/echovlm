import matplotlib.pyplot as plt
import os

def save_plot(steps, values, ylabel, filename, run_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, '-*')
    plt.xlabel('Step')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} vs Step')
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, filename), dpi=150)
    plt.close()
