from product import Product
import numpy as np
from itertools import product as cartesian_product
from Levenshtein import distance as lev
import re

def key_similarity(product1, product2, q):
    shingles_keys1 = product1.get_shingles_key(q)
    shingles_keys2 = product2.get_shingles_key(q)

    keys_sim_dict = {}

    for key1, shingles1 in shingles_keys1.items():
        if key1 not in keys_sim_dict:
            keys_sim_dict[key1] = {}

        for key2, shingles2 in shingles_keys2.items():
            count = sum(1 for shingle in shingles1 if not shingle in shingles2)
            count += sum(1 for shingle in shingles2 if not shingle in shingles1)
            n1 = len(shingles1)
            n2 = len(shingles2)

            # Check to prevent division by zero
            if (n1 + n2) > 0:
                keys_sim_dict[key1][key2] = (n1 + n2 - count) / (n1 + n2)
            else:
                keys_sim_dict[key1][key2] = 0

    return keys_sim_dict


def value_similarity(product1, product2, key1, key2, q):
    shingles_values1 = product1.get_shingles_values(key1, q)
    shingles_values2 = product2.get_shingles_values(key2, q)

    shingles1 = shingles_values1[key1]
    shingles2 = shingles_values2[key2]

    count = sum(1 for shingle in shingles1 if not shingle in shingles2)
    count += sum(1 for shingle in shingles2 if not shingle in shingles1)
    n1 = len(shingles1)
    n2 = len(shingles2)

    # Check to prevent division by zero
    if (n1 + n2) > 0:
        value_sim = (n1 + n2 - count) / (n1 + n2)
    else:
        value_sim = 0

    return value_sim


def model_word_perc_features(product1, product2):
    all_keys1 = list(product1.features.keys())
    all_keys2 = list(product2.features.keys())

    dict1 = product1.MW_features(all_keys1)
    dict2 = product2.MW_features(all_keys2)

    mw1 = set()
    mw2 = set()

    for key1 in all_keys1:
        if dict1[key1] is not None:
            mw1.update(dict1[key1])

    for key2 in all_keys2:
        if dict2[key2] is not None:
            mw2.update(dict2[key2])

    count = sum(1 for mw in mw1 if mw in mw2)
    similarity = count / max(len(mw1), len(mw2)) if mw1 or mw2 else 0

    return similarity


def cos_sim(product1, product2):
    title1 = product1.title
    title2 = product2.title

    words1 = set(title1.split())
    words2 = set(title2.split())

    intersection = words1.intersection(words2)

    size1 = len(words1)
    size2 = len(words2)
    size_intersection = len(intersection)

    if size1 == 0 or size2 == 0:
        return 0  # To handle cases where one or both sets are empty
    cosine_similarity = size_intersection / (np.sqrt(size1) * np.sqrt(size2))

    return cosine_similarity


def non_numeric_part(word):
    # Extract non-numeric part of the word
    return re.sub(r'\d', '', word)


def numeric_part(word):
    # Extract numeric part of the word
    return re.sub(r'[^\d]', '', word)


def title_mw_sim(product1, product2, alpha, beta):
    cosine_similarity = cos_sim(product1, product2)

    if cosine_similarity > alpha:
        return 1

    model_words1 = set(product1.model_words_title)
    model_words2 = set(product2.model_words_title)

    pairs = list(cartesian_product(model_words1, model_words2))
    lv_distance_tot = 0

    for pair in pairs:
        mw1, mw2 = pair
        non_numeric1 = non_numeric_part(mw1)
        non_numeric2 = non_numeric_part(mw2)
        lv_distance = lev(non_numeric1, non_numeric2)
        lv_distance_tot += lv_distance

        if lv_distance < beta and numeric_part(mw1) != numeric_part(mw2):
            return -1

    avg_lv_sim = lv_distance_tot / (len(pairs))
    final_sim = beta * cosine_similarity + (1 - beta) * avg_lv_sim

    return final_sim


def min_features(product1, product2):
    count1 = len(product1.features)
    count2 = len(product2.features)

    return min(count1, count2)


def msm(product1, product2, q, alpha, beta, gamma, mu):
    sim = 0
    avg_sim = 0
    m = 0
    w = 0
    nmk1 = list(product1.features.keys())
    nmk2 = list(product2.features.keys())
    keys_sim_dict = key_similarity(product1, product2, q)
    for key1 in product1.features.keys():
        for key2 in product2.features.keys():
            if keys_sim_dict[key1][key2] > gamma:
                value_sim = value_similarity(product1, product2, key1, key2, q)
                weight = keys_sim_dict[key1][key2]
                sim += weight * value_sim
                m += 1
                w += weight

                if key1 in nmk1:
                    nmk1.remove(key1)
                if key2 in nmk2:
                    nmk2.remove(key2)
    if w > 0:
        avg_sim = sim / w

    mw_percentage = model_word_perc_features(product1, product2)
    title_sim = title_mw_sim(product1, product2, alpha, beta)

    if title_sim == -1:
        theta1 = m / min_features(product1, product2)
        theta2 = 1 - theta1
        h_sim = theta1 * avg_sim + theta2 * mw_percentage
    else:
        theta1 = (1 - mu) * m / min_features(product1, product2)
        theta2 = 1 - mu - theta1
        h_sim = theta1 * avg_sim + theta2 * mw_percentage + mu * title_sim


    return h_sim
