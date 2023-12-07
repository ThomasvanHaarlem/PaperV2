from main import *

products = load_data()

def features_count(products):
    feature_counter = Counter()
    for product in products:
        feature_counter.update(product.features.keys())

    return feature_counter


def countFeatureKeysWebsite(products):  # Van RENZO!
    website_key_counts = {}
    for tv in products:
        shop = tv.shop
        features = tv.features

        if shop not in website_key_counts:
            website_key_counts[shop] = {}

        for key in features:
            website_key_counts[shop].setdefault(key, 0)
            website_key_counts[shop][key] += 1
    return website_key_counts

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


# f_counts = features_count(products)
# w_counts = countFeatureKeysWebsite(products)
# find_size(products)
# s_counts = size_count(products)
b_counts = brand_counter(products)


print("analyse")