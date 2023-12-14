from main import *

products = load_data()

def features_count(products):
    feature_counter = Counter()
    for product in products:
        feature_counter.update(product.features.keys())

    return feature_counter

def find_size(products):
    for tv in products:
        for key, item in tv.features.items():
            if "size" in key:
                print(key, item)

def size_count(products):
    size_counter = {}
    for i, product in enumerate(products):
        size = product.size_class
        if size not in size_counter:
            size_counter[size] = [1, i]
        else:
            size_counter[size][0] += 1

    return size_counter

def brand_counter(products):
    brand_counter = {}
    for i, product in enumerate(products):
        brand = product.brand
        if brand not in brand_counter:
            brand_counter[brand] = [1, [i]]
        else:
            brand_counter[brand][0] += 1
            brand_counter[brand][1].append(i)

    return brand_counter

def analysis_model_ids(products):
    min_length = 20
    max_length = 0
    for i, product in enumerate(products):
        model_id = product.model_id

        if len(model_id) < min_length:
            min_length = len(model_id)
        if len(model_id) > max_length:
            max_length = len(model_id)

    return min_length, max_length

# f_counts = features_count(products)
# w_counts = countFeatureKeysWebsite(products)
# find_size(products)
# s_counts = size_count(products)
# b_counts = brand_counter(products)
min_length, max_length = analysis_model_ids(products)


print("analyse")