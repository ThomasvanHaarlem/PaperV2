import json
import math
import random
import sympy
import numpy as np
import itertools
from product import Product
from tabulate import tabulate
from MSM import msm

from collections import Counter
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.cluster import AgglomerativeClustering
from Levenshtein import distance
from itertools import product as cartesian_product
import re


def get_true_pairs(products):
    true_pairs = set()
    for i, product1 in enumerate(products):
        for j, product2 in enumerate(products[i + 1:], start=i + 1):
            if product1.model_id == product2.model_id:
                true_pairs.add((i, j))

    return true_pairs


def get_unique_words(products):
    unique_model_words = set()
    unique_brands = set()
    unique_sizes = set()
    unique_potential_model_ids = set()

    update_brands(products)

    for tv in products:
        unique_model_words.update(tv.model_words_title)  # Use update to add elements of a set
        unique_brands.add(tv.brand)
        unique_sizes.add(tv.size_class)
        unique_potential_model_ids.add(tv.potential_model_id)

    unique_words = list(unique_model_words.union(unique_brands, unique_potential_model_ids))

    unique_words.remove("Brand not found") if "Brand not found" in unique_words else None
    unique_words.remove("size class not found") if "size class not found" in unique_words else None
    unique_words.remove("Not found") if "Not found" in unique_words else None

    return unique_words


def get_binary_matrix(products, shingle_size):
    unique_words = get_unique_words(products)
    binary_matrix = np.zeros((len(unique_words), len(products)))

    for i, word in enumerate(unique_words):
        for j, tv in enumerate(products):
            if (word in tv.model_words_title or
                    word in tv.brand or word in tv.size_class or word in tv.potential_model_id):
                binary_matrix[i][j] = 1

    return binary_matrix


def hash_function_generator(a, b, prime):
    return lambda x: (a + b * x) % prime


def get_signature_matrix(binary_matrix, number_hashes):
    n, p = binary_matrix.shape
    signature_matrix = np.full((number_hashes, p), np.inf)
    prime = sympy.nextprime(n * 2)

    # Generate the list of hash functions
    hash_functions = [hash_function_generator(random.randrange(1, prime),
                                              random.randrange(1, prime),
                                              prime) for _ in range(number_hashes)]

    hash_values_dict = {}

    for row_idx, row in enumerate(binary_matrix):
        for h_func in hash_functions:
            if row_idx not in hash_values_dict:
                hash_values_dict[row_idx] = []

            hash_values_dict[row_idx].append(h_func(row_idx))

        for col_idx, element in enumerate(row):
            if element == 1:
                # Update the signature matrix if the hash value is smaller
                for h_idx, hash_value in enumerate(hash_values_dict[row_idx]):
                    if hash_value < signature_matrix[h_idx, col_idx]:
                        signature_matrix[h_idx, col_idx] = hash_value

    print("bug_sigmat")
    return signature_matrix


def hash_function_band(product_vector):
    # Concatenate values within the band to form the bucket number
    product_vector = [int(value) if isinstance(value, np.float64) else value for value in product_vector]
    bucket_number = int(''.join(map(str, product_vector)))
    return bucket_number


def perform_LSH(products, signature_matrix, bands, rows):
    candidate_groups = []
    for i in range(bands):
        index_dict = {}
        bucket_numbers = []
        start_row = i * rows
        band = signature_matrix[start_row: start_row + rows]

        for i_product in range(len(products)):
            product_vector = band[:, i_product]
            bucket_number = hash_function_band(product_vector)
            bucket_numbers.append(bucket_number)

        for j, num in enumerate(bucket_numbers):
            if num in index_dict:
                index_dict[num].append(j)
            else:
                index_dict[num] = [j]

        for indices in index_dict.values():
            if (len(indices)) > 1:
                candidate_groups.append(indices)

    candidate_pairs = set()
    for group in candidate_groups:
        # Generate all pairs within this group and add them to the set
        for pair in itertools.combinations(group, 2):
            candidate_pairs.add(tuple(sorted(pair)))  # Sorting ensures (1, 2) and (2, 1) are treated as the same pair

    print("bug_LSH")
    return candidate_pairs


def get_performance_LSH(true_pairs, candidate_pairs):
    duplicates_found = 0
    for pair in candidate_pairs:
        if pair in true_pairs:
            duplicates_found += 1

    PQ = duplicates_found / len(candidate_pairs)
    PC = duplicates_found / len(true_pairs)

    F1_star = (2 * PQ * PC) / (PQ + PC)

    return PQ, PC, F1_star


def update_brands(products):
    # Create set of brands mentioned
    brands = set()
    for product in products:
        if product.brand is not None:
            brands.add(product.brand)

    for tv in products:
        if tv.brand is None:
            for brand in brands:
                if brand in tv.title:
                    tv.brand = brand
                    break


def pre_dis_mat(products, candidate_pairs):
    dis_matrix = np.full((len(products), len(products)), 1.0)
    update_brands(products)

    for i, product1 in enumerate(products):
        for j, product2 in enumerate(products[i:], start=i):

            # Check if both brands are known and different
            if (product1.brand != "Brand not found" and product2.brand != "Brand not found"
                    and product1.brand != product2.brand):
                dis_matrix[i][j] = dis_matrix[j][i] = np.inf

            # Check if products are from the same shop
            elif product1.shop == product2.shop:
                dis_matrix[i][j] = dis_matrix[j][i] = np.inf

            # Check if they are never mentioned as candidate
            elif (i, j) not in candidate_pairs and (j, i) not in candidate_pairs:
                dis_matrix[i][j] = dis_matrix[j][i] = np.inf

    print("bug_pre")
    return dis_matrix


def get_performance_predismat(products, pre_dissimilarity_matrix, true_pairs):
    duplicates_found = 0
    dismat_pairs = set()

    for i in range(len(products)):
        for j in range(i, len(products)):
            if np.isfinite(pre_dissimilarity_matrix[i][j]):
                dismat_pairs.add(tuple(sorted((i, j))))

    for pair in dismat_pairs:
        if pair in true_pairs:
            duplicates_found += 1

    PQ_predismat = duplicates_found / len(dismat_pairs)
    PC_predismat = duplicates_found / len(true_pairs)

    F1_star_predismat = (2 * PQ_predismat * PC_predismat) / (PQ_predismat + PC_predismat)

    return PQ_predismat, PC_predismat, F1_star_predismat

def get_predicted_pairs(products, dis_mat, threshold, shingle_size, alpha, beta, gamma, mu):
    for i, product1 in enumerate(products):
        for j, product2 in enumerate(products[i:], start=i):
            if not np.isinf(dis_mat[i][j]):
                ## msm does not yield better f1 than just singles
                # dis_mat[i][j] = 1 - msm(product1, product2, shingle_size, alpha, beta, gamma, mu)

                shingles1 = product1.get_shingles_title(shingle_size)
                shingles2 = product2.get_shingles_title(shingle_size)

                count = sum(1 for shingle in shingles1 if shingle in shingles2)
                similarity = count / min(len(shingles1), len(shingles2)) if shingles1 or shingles2 else 0
                if product1.size_class == product2.size_class:
                    similarity += alpha

                if (product1.potential_model_id == product2.potential_model_id and
                        product1.potential_model_id != "Not found" and product2.potential_model_id != "Not found"):
                    similarity += beta

                if similarity > 1:
                    similarity = 1

                dis_mat[i][j] = dis_mat[j][i] = 1 - similarity
    predicted_pairs = set()

    # Does not yield beter f1 than just finding the pairs
    ### Clusters ###
    # # Replace np.inf values with a very large number
    # dis_mat = np.where(dis_mat == np.inf, 1e8, dis_mat)
    #
    # # Create the clustering model
    # model = AgglomerativeClustering(metric='precomputed', linkage='single',
    #                                 distance_threshold=threshold, n_clusters=None)
    # model.fit_predict(dis_mat)
    #
    # # Mapping cluster labels to original product indices
    # clusters = {}
    # for index, label in enumerate(model.labels_):
    #     clusters.setdefault(label, []).append(index)
    #
    # for key in clusters:
    #     if len(clusters[key]) > 1:
    #         for i, product_index in enumerate(clusters[key]):
    #             for j in range(i+1, len(clusters[key])):
    #                 predicted_pairs.add((product_index, clusters[key][j]))
    ### Clusters ###


    for i in range(len(products)):
        for j in range(i, len(products)):
            if dis_mat[i][j] < threshold:
                predicted_pairs.add(tuple(sorted((i, j))))

    return predicted_pairs


def get_final_performance(products, predicted_pairs, true_pairs):
    TP = 0
    FN = 0
    for pair in predicted_pairs:
        if pair in true_pairs:
            TP += 1

    for pair in true_pairs:
        if pair not in predicted_pairs:
            FN += 1

    FP = len(predicted_pairs) - TP
    TN = math.comb(len(products), 2) - TP - FN - FP

    F1 = (2 * TP) / (2 * TP + FP + FN)

    precision = TP / (TP + FP)  # quality
    recall = TP / (TP + FN)     # completeness

    PQ = TP / len(predicted_pairs)
    PC = TP / len(true_pairs)

    # Print things:
    final_performance = [
        ["F1", F1],
        ["precision", precision],
        ["recall", recall],
    ]
    table_final = tabulate(final_performance, headers=["", "score"], tablefmt="grid")

    print("final scores")
    print(table_final)

    print(f"PQ = {PQ}, PC = {PC}")

    confusion_matrix = [
        ["TN:", TN, "FP:", FP],
        ["FN:", FN, "TP:", TP], ]

    # Print the confusion matrix
    for row in confusion_matrix:
        print("\t".join(map(str, row)))

    return TN, TP, FN, FP, F1, precision, recall


def print_before_clustering(PQ, PC, F1_star, PQ_predismat, PC_predismat, F1_star_predismat):
    performance_LSH = [
        ["PQ", PQ, PQ_predismat],
        ["PC", PC, PC_predismat],
        ["F1 star", F1_star, F1_star_predismat], ]
    table_LSH = tabulate(performance_LSH, headers=["", "before", "after"], tablefmt="grid")

    print("scores before clustering (before and after checking for brand and shop)")
    print(table_LSH)


def load_data():
    with open("TVs-all-merged.json") as json_file:
        tv_data = json.load(json_file)

    products = []

    for model_id, product_list in tv_data.items():
        for product in product_list:
            # Extracting relevant information
            product_instance = Product(
                model_id=model_id,
                shop=product.get('shop'),
                title=product.get('title'),
                url=product.get('url'),
                features=product.get('featuresMap', {})
            )
            products.append(product_instance)
    return products


########################################################################################################################
if __name__ == "__main__":
    # Set the seed for reproducibility
    random_seed = 123
    random.seed(random_seed)  # Seed for Python's built-in random module
    np.random.seed(random_seed)  # Seed for NumPy's random module

    shingle_size = 3
    number_hashes = 600
    bands = 120
    threshold = 0.1
    rows = number_hashes // bands
    t_score = (1 / bands) ** (1 / rows)
    print(f"The t score = {t_score}")

    q = 3
    alpha = 0.05
    beta = 0.5
    gamma = 0.2
    mu = 0.65

    products = load_data()
    for product in products:
        product.get_potential_model_id()
    true_pairs = get_true_pairs(products)
    binary_matrix = get_binary_matrix(products, shingle_size)
    signature_matrix = get_signature_matrix(binary_matrix, number_hashes)
    candidate_pairs = perform_LSH(products, signature_matrix, bands, rows)
    comparisons_made = len(candidate_pairs)
    print(f"The amount of comparisons made = {comparisons_made}")
    PQ, PC, F1_star = get_performance_LSH(true_pairs, candidate_pairs)
    pre_dissimilarity_matrix = pre_dis_mat(products, candidate_pairs)
    PQ_predismat, PC_predismat, F1_star_predismat = get_performance_predismat(products, pre_dissimilarity_matrix,
                                                                              true_pairs)
    print_before_clustering(PQ, PC, F1_star, PQ_predismat, PC_predismat, F1_star_predismat)
    predicted_pairs = get_predicted_pairs(products, pre_dissimilarity_matrix, threshold, shingle_size, alpha, beta,
                                          gamma, mu)
    TN, TP, FN, FP, F1, precision, recall = get_final_performance(products, predicted_pairs, true_pairs)
    print("END_BUG")
