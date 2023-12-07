import re


class Product:
    """
    Class for each tv
    """

    def __init__(self, model_id, shop, title, url, features):
        self.model_id = model_id
        self.shop = shop
        self.title = self.clean_title(title)
        self.url = url
        self.features = self.clean_features(features)
        self.brand = self.get_brand()
        self.model_words_title = self.get_model_words_title()
        self.size_class = self.get_size()

    def __repr__(self):
        return f"Product(model_id={self.model_id}, shop={self.shop}, title={self.title})"

    def clean_title(self, title):
        title = title.lower()
        title = re.sub(r'\s*-\s*hz|\s*hertz|\s+hz', 'hz', title)
        title = re.sub(r'\s*-?\s*inches?|\s*-\s*inch|\s*inch|\s+inch|\s*"', 'inch', title)
        return title

    def clean_features(self, features):
        cleaned_features = {}
        for key, value in features.items():
            cleaned_key = key.lower()
            if isinstance(value, str):
                cleaned_value = value.lower()
                cleaned_value = re.sub(r'\s*-\s*hz|\s*hertz|\s+hz', 'hz', cleaned_value)
                cleaned_value = re.sub(r'\s*-?\s*inches?|\s*-\s*inch|\s*inch|\s+inch|\s*"', 'inch', cleaned_value)
            else:
                cleaned_value = value
            cleaned_features[cleaned_key] = cleaned_value
        return cleaned_features

    def get_brand(self):
        # Check for 'Brand' key in features and return in lowercase
        if 'brand' in self.features:
            return self.features['brand'].split()[0].lower()
        # If 'Brand' is not available, check for 'Brand Name'
        elif 'brand name' in self.features:
            return self.features['brand name'].split()[0].lower()
        elif 'brand name:' in self.features:
            return self.features['brand name:'].split()[0].lower()
        else:
            return "Brand not found"

    def get_model_words_title(self):
        """
        Find all model words in a title. A model word contains at least two of the following types:
        alphanumerical, numerical, and special characters.
        does not yet include "
        """
        regex = r'([a-zA-Z0-9]*(?:(?:[0-9]+(?:\.[0-9]+)?[^0-9, ()]+)|(?:[^0-9, ()]+[0-9]+(?:\.[0-9]+)?))[a-zA-Z0-9]*)'

        model_words_title = re.findall(regex, self.title)
        # Remove items that are just numbers
        model_words_title = [word for word in model_words_title if not word.isdigit()]

        return set(model_words_title)

    def get_shingles_title(self, shingle_size):
        shingle_title = self.title.replace(" ", "")
        self.shingles = set()
        for i in range(len(shingle_title) - shingle_size):
            self.shingles.add(shingle_title[i:i+shingle_size])

        return self.shingles

    def get_size(self):
        self.size_class = None
        for key, item in self.features.items():
            if "size" in key:
                match = re.search(r'\d+(\.\d+)?', item)
                if match:
                    size_value = float(match.group())
                    self.size_class = self.determine_size_class(size_value)
                    break

        return "size class: " + str(self.size_class) if self.size_class is not None else "size class not found"
    @staticmethod
    def determine_size_class(size_value):
        size_ranges = [
            (0, 10.5), (10.5, 20.5), (21, 30.5), (30.5, 35.5),
            (35.5, 40.5), (40.5, 45.5), (45.5, 50.5), (50.5, 55.5),
            (55.5, 60.5), (60.5, 65.5), (65.5, 70.5), (70.5, 80.5),
            (80.5, 90.5), (90.5, float('inf'))
        ]
        for i, (start, end) in enumerate(size_ranges):
            if start <= size_value < end:
                return i
        return len(size_ranges) - 1  # For sizes >= 90.5


