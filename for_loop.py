from main import *
import numpy as np
import matplotlib.pyplot as plt

def loop_bands(products, number_hashes, true_pairs, tot_comparisons, shingle_size, threshold, alpha, beta, gamma, mu):
    bands_list = [b for b in range(1, (number_hashes + 1)) if number_hashes % b == 0]
    binary_matrix = get_binary_matrix(products, shingle_size)
    signature_matrix = get_signature_matrix(binary_matrix, number_hashes)

    pcs = []
    f1_scores = []
    pqs = []
    fractions = []
    f1_star_scores = []

    for bands in bands_list:
        rows = number_hashes // bands
        t_score = (1 / bands) ** (1 / rows)
        print(f"The t score = {t_score}")

        candidate_pairs = perform_LSH(products, signature_matrix, bands, rows)
        comparisons_made = len(candidate_pairs)
        fractions.append(comparisons_made / tot_comparisons)
        print(f"The amount of comparisons made = {comparisons_made}")
        PQ, PC, F1_star = get_performance_LSH(true_pairs, candidate_pairs)
        pre_dissimilarity_matrix = pre_dis_mat(products, candidate_pairs)
        PQ_predismat, PC_predismat, F1_star_predismat = get_performance_predismat(products, pre_dissimilarity_matrix,
                                                                                  true_pairs)
        print_before_clustering(PQ, PC, F1_star, PQ_predismat, PC_predismat, F1_star_predismat)
        predicted_pairs = get_predicted_pairs(products, pre_dissimilarity_matrix, threshold, shingle_size, alpha, beta,
                                              gamma, mu)
        TN, TP, FN, FP, F1, precision, recall = get_final_performance(products, predicted_pairs, true_pairs)

        pcs.append(recall)
        pqs.append(precision)
        f1_scores.append(F1)
        f1_star_scores.append(F1_star)

    plt.figure()
    plt.plot(fractions, pcs, label='Pair Completeness')
    plt.plot(fractions, f1_scores, label='F1 Score')
    plt.plot(fractions, pqs, label='Pair Quality')
    plt.plot(fractions, f1_star_scores, label='F1 Star')
    plt.xlabel('Fraction of comparisons')
    plt.ylabel('Metric value')
    plt.legend()
    plt.show()

def loop_parameters_plot(products, true_pairs, signature_matrix, bands, rows, t_score, tot_comparisons, shingle_size):
    print(f"The t score = {t_score}")
    candidate_pairs = perform_LSH(products, signature_matrix, bands, rows)
    comparisons_made = len(candidate_pairs)
    print(f"The fraction of comparisons made = {comparisons_made / tot_comparisons}")
    print(f"The amount of comparisons made = {comparisons_made}")
    PQ, PC, F1_star = get_performance_LSH(true_pairs, candidate_pairs)
    pre_dissimilarity_matrix = pre_dis_mat(products, candidate_pairs)
    PQ_predismat, PC_predismat, F1_star_predismat = get_performance_predismat(products, pre_dissimilarity_matrix,
                                                                              true_pairs)
    print_before_clustering(PQ, PC, F1_star, PQ_predismat, PC_predismat, F1_star_predismat)

    alpha_values = [0.05, 0.1, 0.15]
    beta_values = [0.4, 0.5, 0.6]
    gamma_values = [0.1, 0.2, 0.3]
    mu_values = [0.5, 0.6, 0.7]
    threshold_values = [0.1, 0.2, 0.3, 0.9]

    def perform_calculations(param1, param2, main_param_index, main_param_values):
        results = []
        for main_param in main_param_values:
            if main_param_index == 0:
                predicted_pairs = get_predicted_pairs(products, pre_dissimilarity_matrix, param2, shingle_size,
                                                      main_param, param1, gamma, mu)
                TN, TP, FN, FP, F1, precision, recall = get_final_performance(products, predicted_pairs, true_pairs)
                results.append((recall, precision, F1))
            elif main_param_index == 1:
                predicted_pairs = get_predicted_pairs(products, pre_dissimilarity_matrix, param2, shingle_size, param1,
                                                      main_param, gamma, mu)
                TN, TP, FN, FP, F1, precision, recall = get_final_performance(products, predicted_pairs, true_pairs)
                results.append((recall, precision, F1))
            else:
                predicted_pairs = get_predicted_pairs(products, pre_dissimilarity_matrix, main_param, shingle_size,
                                                      param1, param2, gamma, mu)
                TN, TP, FN, FP, F1, precision, recall = get_final_performance(products, predicted_pairs, true_pairs)
                results.append((recall, precision, F1))
        return zip(*results)

    def plot_subplots(fig, axes, main_param_values, secondary_params, title, main_param_index, num_rows, num_cols):
        for idx, (param1, param2) in enumerate(secondary_params):
            i, j = divmod(idx, num_cols)
            pcs, pqs, f1_scores = perform_calculations(param1, param2, main_param_index, main_param_values)

            ax = axes[i][j]  # Adjust indexing for subplots
            ax.plot(main_param_values, pcs, label='Pair Completeness')
            ax.plot(main_param_values, pqs, label='Pair Quality')
            ax.plot(main_param_values, f1_scores, label='F1 Score')
            ax.set_title(f'Secondary Params: {param1}, {param2}')
            ax.legend()

        fig.suptitle(title)
        plt.tight_layout()

    beta_threshold_product = list(itertools.product(beta_values, threshold_values))
    alpha_threshold_product = list(itertools.product(alpha_values, threshold_values))
    alpha_beta_product = list(itertools.product(alpha_values, beta_values))

    fig_alpha, axes_alpha = plt.subplots(len(beta_values), len(threshold_values), figsize=(15, 15))
    plot_subplots(fig_alpha, axes_alpha, alpha_values, beta_threshold_product, 'Alpha Plots', 0, len(beta_values),
                  len(threshold_values))

    fig_beta, axes_beta = plt.subplots(len(alpha_values), len(threshold_values), figsize=(15, 15))
    plot_subplots(fig_beta, axes_beta, beta_values, alpha_threshold_product, 'Beta Plots', 1, len(alpha_values),
                  len(threshold_values))

    fig_threshold, axes_threshold = plt.subplots(len(alpha_values), len(beta_values), figsize=(15, 15))
    plot_subplots(fig_threshold, axes_threshold, threshold_values, alpha_beta_product, 'Threshold Plots', 2,
                  len(alpha_values), len(beta_values))

    plt.show()


#-----------------------------------------------------------------------------------------------------------------------
# Set the seed for reproducibility
random_seed = 123
random.seed(random_seed)  # Seed for Python's built-in random module
np.random.seed(random_seed)  # Seed for NumPy's random module

number_hashes = 840
bands = 60
rows = number_hashes // bands
t_score = (1 / bands) ** (1 / rows)

shingle_size = 3
threshold = 0.1

q = 3
alpha = 0.1
beta = 0.5
gamma = 0.2
mu = 0.65

products = load_data()
tot_comparisons = math.comb(len(products), 2)
true_pairs = get_true_pairs(products)

loop_bands(products, number_hashes, true_pairs, tot_comparisons, shingle_size, threshold, alpha, beta, gamma, mu)
loop_parameters_plot(products, true_pairs, signature_matrix, bands, rows, t_score, tot_comparisons, shingle_size)



print("bug_analyse")

