from main import *
import numpy as np
import matplotlib.pyplot as plt

# Set the seed for reproducibility
random_seed = 123
random.seed(random_seed)  # Seed for Python's built-in random module
np.random.seed(random_seed)  # Seed for NumPy's random module

number_hashes = 840
bands_list = [b for b in range(1,(number_hashes + 1)) if number_hashes % b==0]
shingle_size = 3
threshold = 0.7

q = 3
alpha = 0.6
beta = 0.01
gamma = 0.2
mu = 0.65

products = load_data()
true_pairs = get_true_pairs(products)
binary_matrix = get_binary_matrix(products, shingle_size)
signature_matrix = get_signature_matrix(binary_matrix, number_hashes)

pcs = []
f1_scores = []
pqs = []
fractions = []
tot_comparisons = math.comb(len(products), 2)

for bands in bands_list:
    rows = number_hashes // bands
    t_score = (1 / bands) ** (1 / rows)
    print(f"The t score = {t_score}")

    candidate_pairs = perform_LSH(products, signature_matrix, bands, rows)
    comparisons_made = len(candidate_pairs)
    fractions.append(comparisons_made/tot_comparisons)
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

plt.figure()
plt.plot(fractions, pcs, label='Pair Completeness')
plt.plot(fractions, f1_scores, label='F1 Score')
plt.plot(fractions, pqs, label='PQ')
plt.xlabel('Fraction of comparisons')
plt.ylabel('Metric value')
plt.legend()
plt.show()

print("bug_analyse")

