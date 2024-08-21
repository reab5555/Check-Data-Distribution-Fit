import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest, norm, expon, lognorm, chi2, beta, gamma, uniform, pareto, cauchy, t, \
    weibull_min, laplace, logistic, burr, invgamma, invgauss, gompertz, triang, loglaplace, levy, gumbel_r, gumbel_l, \
    rayleigh, powerlaw
import warnings
import tempfile

# Suppress specific runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Function to check distribution type
def check_distribution(target_column):
    data = target_column.dropna()

    # Distribution dictionaries
    distributions = {
        'Normal': norm,
        'Exponential': expon,
        'Lognormal': lognorm,
        'Chi-square': chi2,
        'Beta': beta,
        'Gamma': gamma,
        'Uniform': uniform,
        'Pareto': pareto,
        'Cauchy': cauchy,
        'Student\'s t': t,
        'Weibull': weibull_min,
        'Laplace': laplace,
        'Logistic': logistic,
        'Burr': burr,
        'Inverse Gamma': invgamma,
        'Inverse Gaussian': invgauss,
        'Gompertz': gompertz,
        'Triangular': triang,
        'Log-Laplace': loglaplace,
        'Levy': levy,
        'Gumbel Right': gumbel_r,
        'Gumbel Left': gumbel_l,
        'Rayleigh': rayleigh,
        'Powerlaw': powerlaw
    }

    # Colors for top 3 distributions and the normal distribution
    distribution_colors = ['gold', 'silver', 'brown']
    normal_color = 'red'
    actual_data_color = 'black'

    # List to store results
    results = []

    # Test each distribution with error handling
    for name, distribution in distributions.items():
        try:
            params = distribution.fit(data)
            D, p_value = kstest(data, distribution.cdf, args=params)
            results.append((name, p_value, params))
        except Exception as e:
            print(f"Skipping {name} distribution due to an error: {e}")

    # Sort by p-value and get top 3
    results.sort(key=lambda x: x[1], reverse=True)
    top_3_results = results[:3]

    # Create a single plot for all distributions
    plt.figure(figsize=(12, 8), dpi=400)

    # Plot the original data distribution as a histogram
    sns.histplot(data, kde=False, stat="density", bins=50, color=actual_data_color, label='Actual Data Distribution')

    # Overlay the actual data KDE line
    sns.kdeplot(data, color=actual_data_color, lw=2, label='Actual Data Distribution Line')

    # Overlay the top 3 best fit distributions
    for i, (name, p_value, params) in enumerate(top_3_results):
        best_fit_data = np.linspace(min(data), max(data), 1000)
        pdf = distributions[name].pdf(best_fit_data, *params)
        p_value_text = "<0.001" if p_value < 0.001 else f"{p_value:.5f}"
        plt.plot(best_fit_data, pdf, color=distribution_colors[i], lw=2, label=f'{name} Fit (p-value={p_value_text})')

    plt.title("Top 3 Best Fit Distributions Overlaid")
    plt.legend()

    # Save plot to temporary file
    temp_file_fit = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file_fit.name)
    plt.close()

    best_fit_plot = temp_file_fit.name

    # Prepare result text with top 3 distributions and p-values
    result_text = "Top 3 best matched distributions:\n"
    for i, (name, p_value, _) in enumerate(top_3_results):
        p_value_text = "<0.001" if p_value < 0.001 else f"{p_value:.5f}"
        result_text += f"Top {i + 1}: {name} with a p-value of {p_value_text}\n"

    # Add disclaimer about p-value significance
    result_text += "\nDisclaimer: A significant p-value (below 0.05) indicates that the distribution does not conform well to the actual data distribution."

    # Normal distribution comparison
    mean, std = norm.fit(data)
    normal_best_fit_data = np.linspace(min(data), max(data), 1000)
    normal_pdf = norm.pdf(normal_best_fit_data, mean, std)
    ks_stat, normal_p_value = kstest(data, 'norm', args=(mean, std))

    p_value_text = "<0.001" if normal_p_value < 0.001 else f"{normal_p_value:.5f}"

    plt.figure(figsize=(12, 8), dpi=400)
    sns.histplot(data, kde=False, stat="density", bins=50, color=actual_data_color, label='Actual Data Distribution')
    sns.kdeplot(data, color=actual_data_color, lw=2, label='Actual Data Distribution Line')
    plt.plot(normal_best_fit_data, normal_pdf, color=normal_color, lw=2, label=f'Normal Fit (p-value={p_value_text})')
    plt.title("Comparison with Normal Distribution")
    plt.legend()

    # Save plot to temporary file
    temp_file_normal = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file_normal.name)
    plt.close()

    normal_comparison_plot = temp_file_normal.name

    return result_text, best_fit_plot, normal_comparison_plot


# Function to load the CSV file and extract numeric column names
def load_file(file):
    df = pd.read_csv(file.name)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return gr.update(choices=numeric_columns), df


# Function to analyze the selected column
def analyze_column(selected_column, df):
    result_text, best_fit_plot, normal_comparison_plot = check_distribution(df[selected_column])
    return result_text, best_fit_plot, normal_comparison_plot


# Define the Gradio app layout
with gr.Blocks() as demo:
    gr.Markdown("# Data Distribution Fit\n")

    file_input = gr.File(label="Upload CSV File")
    column_selector = gr.Dropdown(label="Select Target Column", choices=[])
    analyze_button = gr.Button("Fit")
    output_text = gr.Textbox(label="Results")
    best_fit_plot_output = gr.Image(label="Best Fit Distributions")
    normal_comparison_output = gr.Image(label="Comparison with Normal Distribution")

    # State management
    df_state = gr.State(None)

    # Load the file and populate the dropdown
    file_input.upload(load_file, inputs=file_input, outputs=[column_selector, df_state])

    # Perform analysis on the selected column
    analyze_button.click(analyze_column, inputs=[column_selector, df_state],
                         outputs=[output_text, best_fit_plot_output, normal_comparison_output])

demo.launch()
