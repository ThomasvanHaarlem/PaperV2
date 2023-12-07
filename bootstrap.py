from main import *
import numpy as np

def bootstrap_sample(data):
    """ Generate a bootstrap sample from the dataset """
    n = len(data)
    return [random.choice(data) for _ in range(n)]

random_seed = 123
random.seed(random_seed)  # Seed for Python's built-in random module
np.random.seed(random_seed)  # Seed for NumPy's random module

shingle_size = 3
number_hashes = 600
bands = 60
threshold = 0.6
rows = number_hashes // bands
t_score = (1 / bands) ** (1 / rows)
print(f"The t score = {t_score}")

products = load_data()


n_bootstrap_samples = 5

for i in range(n_bootstrap_samples):
    boot_products = bootstrap_sample(products)
    true_pairs = get_true_pairs(boot_products)
    binary_matrix = get_binary_matrix(boot_products, shingle_size)
    signature_matrix = get_signature_matrix(binary_matrix, number_hashes)
    candidate_pairs = perform_LSH(boot_products, signature_matrix, bands, rows)
    comparisons_made = len(candidate_pairs)
    print(f"The amount of comparisons made = {comparisons_made}")
    PQ, PC, F1_star = get_performance_LSH(true_pairs, candidate_pairs)
    pre_dissimilarity_matrix = pre_dis_mat(boot_products, candidate_pairs)
    PQ_predismat, PC_predismat, F1_star_predismat = get_performance_predismat(boot_products, pre_dissimilarity_matrix,
                                                                              true_pairs)
    print_before_clustering(PQ, PC, F1_star, PQ_predismat, PC_predismat, F1_star_predismat)
    predicted_pairs = get_predicted_pairs(boot_products, pre_dissimilarity_matrix, threshold, shingle_size)
    TN, TP, FN, FP, F1, precision, recall = get_final_performance(boot_products, predicted_pairs, true_pairs)

print("END_BUG")