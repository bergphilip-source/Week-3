import math
import re
from email import policy
from email.parser import BytesParser
from pathlib import Path


MONEY_WORDS = {
    "cash",
    "credit",
    "earn",
    "free",
    "loan",
    "money",
    "payment",
    "prize",
}
COMMERCIAL_AD_TERMS = {
    "buy",
    "coupon",
    "deal",
    "discount",
    "offer",
    "promotion",
    "promo",
    "sale",
    "save",
    "shop",
}
BULK_SENDING_PATTERN = re.compile(
    r"\b(?:unsubscribe|mailing\s+list|newsletter|bulk\s+email|"
    r"email\s+preferences|opt[\s-]+out)\b",
    re.IGNORECASE,
)
FEATURE_NAMES = (
    "word_count",
    "exclamations",
    "money_words",
    "all_caps",
    "commercial_ads",
    "bulk_sending",
)
CLASSES = ("Spam", "Non-Spam")
MIN_VARIANCE = 0.25


def extract_features(text):
    words = text.split()
    word_count = len(words)
    exclamations = text.count("!")
    money_words = sum(
        word.strip(".,!?;:()[]{}\"'").lower() in MONEY_WORDS
        for word in words
    )
    all_caps = sum(
        word.isalpha() and len(word) > 1 and word.isupper()
        for word in words
    )
    commercial_ads = sum(
        word.strip(".,!?;:()[]{}\"'").lower() in COMMERCIAL_AD_TERMS
        for word in words
    )
    bulk_sending = len(BULK_SENDING_PATTERN.findall(text))

    return {
        "word_count": word_count,
        "exclamations": exclamations,
        "money_words": money_words,
        "all_caps": all_caps,
        "commercial_ads": commercial_ads,
        "bulk_sending": bulk_sending,
    }


def read_email(path):
    message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    body = message.get_body(preferencelist=("plain", "html"))
    body_text = body.get_content() if body else ""
    return f"{message.get('subject', '')} {body_text}"


def train_model(dataset_folder):
    examples = {label: [] for label in CLASSES}

    for label in CLASSES:
        email_files = sorted((dataset_folder / label).glob("*.eml"))
        if not email_files:
            raise ValueError(
                f"No .eml training emails found in "
                f"{dataset_folder / label}"
            )
        examples[label] = [
            extract_features(read_email(path)) for path in email_files
        ]

    model = {}
    total_examples = sum(len(class_examples) for class_examples in examples.values())
    for label, class_examples in examples.items():
        means = {}
        variances = {}
        for feature in FEATURE_NAMES:
            values = [math.log1p(row[feature]) for row in class_examples]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            means[feature] = mean
            variances[feature] = max(variance, MIN_VARIANCE)

        model[label] = {
            "prior": len(class_examples) / total_examples,
            "means": means,
            "variances": variances,
            "count": len(class_examples),
        }

    return model


def classify(features, model):
    scores = {}
    for label, class_model in model.items():
        score = math.log(class_model["prior"])
        for feature in FEATURE_NAMES:
            value = math.log1p(features[feature])
            mean = class_model["means"][feature]
            variance = class_model["variances"][feature]
            score += -0.5 * math.log(2 * math.pi * variance)
            score -= (value - mean) ** 2 / (2 * variance)
        scores[label] = score

    return max(scores, key=scores.get)


def main():
    dataset_folder = Path(__file__).resolve().parent
    model = train_model(dataset_folder)

    print(
        "Trained on "
        f"{model['Spam']['count']} spam and "
        f"{model['Non-Spam']['count']} non-spam emails."
    )
    email = input("Paste the email text: ")
    features = extract_features(email)

    print(f"Word count: {features['word_count']}")
    print(f"Exclamations: {features['exclamations']}")
    print(f"Money-related words: {features['money_words']}")
    print(f"All-caps words: {features['all_caps']}")
    print(f"Commercial ad terms: {features['commercial_ads']}")
    print(f"Bulk-sending indicators: {features['bulk_sending']}")

    prediction = classify(features, model)
    print(f"This email is likely {prediction.lower()}.")


if __name__ == "__main__":
    main()
