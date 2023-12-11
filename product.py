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
        self.potential_model_id = self.get_potential_model_id()
        self.model_words_features = None
        self.shingles_keys = None
        self.shingles_values = None

    def __repr__(self):
        return f"Product(model_id={self.model_id}, shop={self.shop}, title={self.title})"

    def clean_title(self, title):
        title = title.lower()
        title = re.sub(r'\s*-\s*hz|\s*hertz|\s+hz', 'hz', title)
        title = re.sub(r'\s*-?\s*inches?|\s*-\s*inch|\s*inch|\s+inch|\s*"|\d+\s*in\b|\s*\'|\s*”', 'inch', title)
        title = title.replace("inchinch", "inch")
        title = title.replace("-", "")
        title = title.replace("/", "")
        return title

    def clean_features(self, features):
        cleaned_features = {}
        for key, value in features.items():
            cleaned_key = key.lower()
            if isinstance(value, str):
                cleaned_value = value.lower()
                cleaned_value = re.sub(r'\s*-\s*hz|\s*hertz|\s+hz', 'hz', cleaned_value)
                cleaned_value = re.sub(r'\s*-?\s*inches?|\s*-\s*inch|\s*inch|\s+inch|\s*"|\d+\s*in\b|\s*\'|\s*”', 'inch', cleaned_value)
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
        """
        # regex to find all words
        regex = r'([a-zA-Z0-9]*(?:(?:[0-9]+(?:\.[0-9]+)?[^0-9, ()]+)|(?:[^0-9, ()]+[0-9]+(?:\.[0-9]+)?))[a-zA-Z0-9]*)'

        model_words_title = re.findall(regex, self.title)
        # Remove items that are just numbers
        model_words_title = [word for word in model_words_title if not word.isdigit()]

        return set(model_words_title)

    def get_shingles_title(self, shingle_size):
        shingle_title = self.title.replace(" ", "")
        # Remove shop from title
        shingle_title = shingle_title.replace(self.shop, "")
        shingle_title = shingle_title.replace("-bestbuy", "")

        self.shingles = set()
        for i in range(len(shingle_title) - shingle_size):
            self.shingles.add(shingle_title[i:i+shingle_size])

        return self.shingles

    def get_size(self):
        self.size_class = None
        size_found = False

        for key, item in self.features.items():
            if "size" in key:
                match = re.search(r'\d+(\.\d+)?', item)
                if match:
                    size_value = float(match.group())
                    self.size_class = self.determine_size_class(size_value)
                    size_found = True
                    break

        if not size_found:
            matches_title = re.findall(r'\d+(?:\.\d+)?(?=inch)', self.title)
            if matches_title:
                size_value = float(matches_title[0])
                self.size_class = self.determine_size_class(size_value)

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

    def MW_features(self, important_features):
        if self.model_words_features is None:
            self.model_words_features = {}
        # Just keep numbers
        regex = r'\b\d+(?:[.,]\d+)?[a-zA-Z]*\b'

        for key in important_features:
            if key not in self.model_words_features.keys():
                value = self.features[key]
                matches = re.findall(regex, value)

                processed_matches = []

                # Remove text part from each match and add to processed_matches
                for match in matches:
                    match = re.sub(r'[a-zA-Z]+$', '', match)
                    processed_matches.append(match)

                # Assign processed matches to the key, or a default value if no matches
                self.model_words_features[key] = processed_matches if processed_matches else None

        return self.model_words_features

    def get_shingles_key(self, q):
        if self.shingles_keys is None:
            self.shingles_keys = {}
            for key in self.features.keys():
                shingles = [key[i:i + q] for i in range(0, len(key) - q + 1)]
                self.shingles_keys[key] = shingles

        return self.shingles_keys

    def get_shingles_values(self, key, q):
        if self.shingles_values is None:
            self.shingles_values = {}

        if key not in self.shingles_values:
            value = self.features[key]
            shingles = [value[i:i + q] for i in range(0, len(value) - q + 1)]
            self.shingles_values[key] = shingles
            return self.shingles_values
        else:
            return self.shingles_values

    def get_potential_model_id(self):
        """
        Get a potential model id from the title
        by finding all words that contain at least one letter and one number
        checking whether the word is at least 4 characters long
        removing all words that contain inch or hz or 1080p or 720p or 2160p or 3dready
        removing all words that are just letters or just numbers
        removing all words that contain series
        """
        regex = r'(?=.*[a-zA-Z])(?=.*[0-9])\w+'
        potential_model_id = re.findall(regex, self.title)
        potential_model_id = [word for word in potential_model_id if len(word) >= 4]
        potential_model_id = [word for word in potential_model_id if
                              'inch' not in word and 'hz' not in word and '1080p' not in word and '720p' not in word
                              and '2160p' not in word and '3dready' not in word]
        potential_model_id = [word for word in potential_model_id if not word.isdigit() and not word.isalpha()]
        potential_model_id = [word for word in potential_model_id if 'series' not in word]
        if len(potential_model_id) > 1:
            potential_model_id = max(potential_model_id, key=len)
        else:
            potential_model_id = potential_model_id[0] if len(potential_model_id) > 0 else "Not found"

        return potential_model_id
