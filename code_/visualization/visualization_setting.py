import matplotlib.pyplot as plt
import seaborn as sns
import os


def set_plot_style(
    font_family="sans-serif",
    font_sans_serif="Arial",
    title_size=20,
    label_size=16,
    tick_size=14,
    legend_size=14,
    seaborn_style=None,
    seaborn_palette=None
):
    """Configure Matplotlib and Seaborn global font and style settings with optional customization."""
    
    sns.set_style(seaborn_style)  # Options: "white", "darkgrid", "whitegrid", "ticks", "dark"
    sns.set_palette(seaborn_palette)  # Options: "deep", "muted", "bright", "colorblind", etc.

    # Set Matplotlib font configurations
    plt.rc("font", family=font_family)  # Correct way to set the font family
    plt.rc("font", **{"sans-serif": [font_sans_serif]})  # Set sans-serif font

    plt.rc("axes", titlesize=title_size)  # Title font size
    plt.rc("axes", labelsize=label_size)  # X and Y label font size
    plt.rc("xtick", labelsize=tick_size)  # X-tick font size
    plt.rc("ytick", labelsize=tick_size)  # Y-tick font size
    plt.rc("legend", fontsize=legend_size)  # Legend font size

    # Apply Seaborn context settings for consistent font sizes
    sns.set_context("notebook", rc={
        "axes.titlesize": title_size,
        "axes.labelsize": label_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": legend_size,
    })




def save_img_path(folder_path:str, file_name:str)->None:
    visualization_folder_path =  folder_path
    os.makedirs(visualization_folder_path, exist_ok=True)    
    fname = file_name
    plt.savefig(visualization_folder_path / fname, dpi=600)