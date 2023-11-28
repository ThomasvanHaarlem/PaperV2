import json
import random
import sympy
import numpy as np
from product import Product
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.cluster import AgglomerativeClustering
from Levenshtein import distance
from itertools import product as cartesian_product
import re


def get_unique_model_words(products):
    unique_model_words = set()

    for tv in products:
        unique_model_words.update(tv.model_words_title)  # Use update to add elements of a set

    return unique_model_words


def get_binary_matrix(products):
    unique_model_words = list(get_unique_model_words(products))
    binary_matrix = np.zeros((len(unique_model_words), len(products)))

    for i, word in enumerate(unique_model_words):
        for j, tv in enumerate(products):
            if word in tv.model_words_title:
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
    candidate_groups = set()
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
                candidate_groups.add(indices)

    print("bug_LSH")
    return candidate_groups


########################################################################################################################
# Set the seed for reproducibility
random_seed = 123
random.seed(random_seed)  # Seed for Python's built-in random module
np.random.seed(random_seed)  # Seed for NumPy's random module

with open("TVs-all-merged.json") as json_file:
    tv_data = json.load(json_file)

products = []
duplicates_dict = {}

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

        if model_id not in duplicates_dict:
            duplicates_dict[model_id] = [product_instance]
        else:
            duplicates_dict[model_id].append(product_instance)

shingle_size = 3
number_hashes = 600
bands = 30
threshold = 0.9
rows = number_hashes // bands
t_score = (1 / bands) ** (1 / rows)
print(f"The t score = {t_score}")

binary_matrix = get_binary_matrix(products)
signature_matrix = get_signature_matrix(binary_matrix, number_hashes)
candidate_pairs = perform_LSH(products, signature_matrix, bands, rows)

print("END_BUG")
