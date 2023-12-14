from main import *
import numpy as np
import pandas as pd

def bootstrap_sample(data):
    """ Generate a bootstrap sample and separate the data into a training and a test set. """
    n = len(data)
    # Create a bootstrap sample with replacement
    bootstrap_sample = [random.choice(data) for _ in range(n)]

    # Convert to set for efficient lookup
    bootstrap_set = set(bootstrap_sample)

    # Create training set (unique observations in the bootstrap sample)
    train_set = list(bootstrap_set)

    # Create test set (observations not included in the bootstrap sample)
    test_set = [item for item in data if item not in bootstrap_set]

    return train_set, test_set

def bootstrap_all(products, n_bootstrap_samples, shingle_size, number_hashes, bands, rows, threshold, alpha, beta, gamma, mu):
    for i in range(n_bootstrap_samples):
        products_train, products_test = bootstrap_sample(products)
        true_pairs = get_true_pairs(products_train)
        tot_comparisons = math.comb(len(products_train), 2)
        binary_matrix = get_binary_matrix(products_train, shingle_size)
        signature_matrix = get_signature_matrix(binary_matrix, number_hashes)
        candidate_pairs = perform_LSH(products_train, signature_matrix, bands, rows)
        comparisons_made = len(candidate_pairs)
        print(f"The amount of comparisons made = {comparisons_made}")
        print(f"The fraction of comparisons made = {comparisons_made / tot_comparisons}")
        PQ_star, PC_star, F1_star = get_performance_LSH(true_pairs, candidate_pairs)
        pre_dissimilarity_matrix = pre_dis_mat(products_train, candidate_pairs)
        dismat_pairs, PQ_predismat, PC_predismat, F1_star_predismat = get_performance_predismat(products_train,
                                                                                                pre_dissimilarity_matrix,
                                                                                                true_pairs)
        predicted_pairs = get_predicted_pairs(products_train, pre_dissimilarity_matrix, threshold, shingle_size, alpha,
                                              beta,
                                              gamma, mu)
        TN, TP, FN, FP, F1_final, PQ_final, PC_final = get_final_performance(products_train, predicted_pairs,
                                                                             true_pairs)
        print_performance(candidate_pairs, dismat_pairs, predicted_pairs, true_pairs, PQ_star, PC_star, F1_star,
                          PQ_predismat, PC_predismat, F1_star_predismat, F1_final, TN, TP, FN, FP, PQ_final, PC_final)


def bootstrap_and_plot(products, number_hashes, n_bootstrap_samples, shingle_size, threshold, alpha, beta, gamma, mu):
    #bands_list = [b for b in range(1, (number_hashes + 1)) if number_hashes % b == 0]
    #bands_list = [1, 20, 30, 60, 70, 84, 90, 126, 140, 180, 210, 252, 315, 420, 630, 1260]
    #bands_list = [1, 30, 70, 90, 126, 140, 180, 210, 252, 315, 420, 630]
    bands_list = [1, 30, 70, 90, 126, 140, 180, 210, 252, 315]

    # Initialize lists to store average metrics for each fraction of comparisons
    avg_pcs = []
    avg_pqs = []
    avg_f1_scores = []
    avg_pcs_star = []
    avg_pqs_star = []
    avg_f1_star_scores = []
    avg_fractions = []
    avg_reduction = []

    for bands in bands_list:
        rows = number_hashes // bands
        t_score = (1 / bands) ** (1 / rows)
        print("------------------------------------")
        print(f"The number of bands = {bands}")

        # Temporary lists to hold metrics for each bootstrap sample
        pcs_temp = []
        pqs_temp = []
        f1_scores_temp = []
        pcs_star_temp = []
        pqs_star_temp = []
        f1_star_scores_temp = []
        fractions_temp = []
        reduction_blocking_temp = []

        for _ in range(n_bootstrap_samples):
            print("-----")
            print(f"Bootstrap sample {_ + 1}")
            products_train, products_test = bootstrap_sample(products)
            true_pairs = get_true_pairs(products_test)
            tot_comparisons = math.comb(len(products_test), 2)
            binary_matrix = get_binary_matrix(products_test, shingle_size)
            signature_matrix = get_signature_matrix(binary_matrix, number_hashes)

            candidate_pairs = perform_LSH(products_test, signature_matrix, bands, rows)
            comparisons_made = len(candidate_pairs)
            fraction = comparisons_made / tot_comparisons
            print(f"The fraction of comparisons made = {fraction}")

            PQ_star, PC_star, F1_star = get_performance_LSH(true_pairs, candidate_pairs)
            pre_dissimilarity_matrix = pre_dis_mat(products_test, candidate_pairs)
            dismat_pairs, PQ_predismat, PC_predismat, F1_star_predismat = get_performance_predismat(products_test,
                                                                                                    pre_dissimilarity_matrix,
                                                                                                    true_pairs)
            predicted_pairs = get_predicted_pairs(products_test, pre_dissimilarity_matrix, threshold, shingle_size,
                                                  alpha, beta, gamma, mu)
            TN, TP, FN, FP, F1_final, PQ_final, PC_final = get_final_performance(products_test, predicted_pairs,
                                                                                 true_pairs)
            print_performance(candidate_pairs, dismat_pairs, predicted_pairs, true_pairs, PQ_star, PC_star, F1_star,
                              PQ_predismat, PC_predismat, F1_star_predismat, F1_final, TN, TP, FN, FP, PQ_final,
                              PC_final)
            reduction_blocking = 1 - len(dismat_pairs) / len(candidate_pairs)
            print(f"The reduction in comparisons made by blocking = {reduction_blocking}")

            pcs_temp.append(PC_final)
            pqs_temp.append(PQ_final)
            f1_scores_temp.append(F1_final)
            pcs_star_temp.append(PC_star)
            pqs_star_temp.append(PQ_star)
            f1_star_scores_temp.append(F1_star)
            fractions_temp.append(fraction)
            reduction_blocking_temp.append(reduction_blocking)

        # Calculate the average metrics for the current fraction
        avg_pcs.append(np.mean(pcs_temp))
        avg_pqs.append(np.mean(pqs_temp))
        avg_f1_scores.append(np.mean(f1_scores_temp))
        avg_pcs_star.append(np.mean(pcs_star_temp))
        avg_pqs_star.append(np.mean(pqs_star_temp))
        avg_f1_star_scores.append(np.mean(f1_star_scores_temp))
        avg_fractions.append(np.mean(fractions_temp))
        avg_reduction.append(np.mean(reduction_blocking_temp))

    data = {
        'avg_f1': avg_f1_scores,
        #'avg_f1_star': avg_f1_star_scores,
        'avg_fractions': avg_fractions,
        # Add any other averages you want to save
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    #df.to_csv('average_with_block.csv', index=False)

    print(f"Average reduction by blocking = {np.mean(avg_reduction)}")
    # Plotting
    # Plot settings for larger and bold text
    plt.rcParams.update({'font.size': 30, 'font.weight': 'bold'})

    # First plot
    plt.figure()
    plt.plot(avg_fractions, avg_pcs, label='Pair Completeness', linestyle='-', marker='o', linewidth=3.5)
    plt.plot(avg_fractions, avg_pqs, label='Pair Quality', linestyle='-', marker='x', linewidth=3.5)
    plt.plot(avg_fractions, avg_f1_scores, label='F1 Score', linestyle='-', marker='+', linewidth=3.5)
    plt.xlabel('Average Fraction of Comparisons', fontweight='bold', fontsize=30)
    plt.ylabel('Average Metric Value', fontweight='bold', fontsize=30)
    plt.legend()
    plt.title('Average Metrics for PQ, PC, F1')

    # Second plot
    plt.figure()
    plt.plot(avg_fractions, avg_pcs_star, label='Pair Completeness *', linestyle='-', marker='o', linewidth=3.5)
    plt.plot(avg_fractions, avg_pqs_star, label='Pair Quality *', linestyle='-', marker='x', linewidth=3.5)
    plt.plot(avg_fractions, avg_f1_star_scores, label='F1*', linestyle='-', marker='+', linewidth=3.5)
    plt.xlabel('Average Fraction of Comparisons', fontweight='bold', fontsize=30)
    plt.ylabel('Average Metric Value', fontweight='bold', fontsize=30)
    plt.legend()
    plt.title('Average Metrics for PQ Star, PC Star, F1 Star')

    plt.show()


########## MAIN ##########
random_seed = 123
random.seed(random_seed)  # Seed for Python's built-in random module
np.random.seed(random_seed)  # Seed for NumPy's random module


number_hashes = 1260
# bands = 100
# threshold = 0.6
# rows = number_hashes // bands
# t_score = (1 / bands) ** (1 / rows)
# print(f"The t score = {t_score}")

shingle_size = 3
threshold = 0.5
alpha = 0.2
beta = 0.2
gamma = 0.05
mu = 0.4

products = load_data()
n_bootstrap_samples = 5

bootstrap_and_plot(products, number_hashes, n_bootstrap_samples, shingle_size, threshold, alpha, beta, gamma, mu)

print("END_BUG")