"""Send an image from the dataset to either localhost, the proxy, or a GPU server to be classified."""

import argparse
import io
import logging
import random

import matplotlib.pyplot as plt
import requests
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


API_ENDPOINTS = {
    "local": "http://127.0.0.1:8000/classify/",
    "proxy": "http://159.89.242.75/classify/",
    "direct": "https://dmtzq4wskj5l46-8000.proxy.runpod.net/classify/",
}

CLASS_MAPPING = {
    0: "apple_pie",
    1: "baby_back_ribs",
    2: "baklava",
    3: "beef_carpaccio",
    4: "beef_tartare",
    5: "beet_salad",
    6: "beignets",
    7: "bibimbap",
    8: "bread_pudding",
    9: "breakfast_burrito",
    10: "bruschetta",
    11: "caesar_salad",
    12: "cannoli",
    13: "caprese_salad",
    14: "carrot_cake",
    15: "ceviche",
    16: "cheesecake",
    17: "cheese_plate",
    18: "chicken_curry",
    19: "chicken_quesadilla",
    20: "chicken_wings",
    21: "chocolate_cake",
    22: "chocolate_mousse",
    23: "churros",
    24: "clam_chowder",
    25: "club_sandwich",
    26: "crab_cakes",
    27: "creme_brulee",
    28: "croque_madame",
    29: "cup_cakes",
    30: "deviled_eggs",
    31: "donuts",
    32: "dumplings",
    33: "edamame",
    34: "eggs_benedict",
    35: "escargots",
    36: "falafel",
    37: "filet_mignon",
    38: "fish_and_chips",
    39: "foie_gras",
    40: "french_fries",
    41: "french_onion_soup",
    42: "french_toast",
    43: "fried_calamari",
    44: "fried_rice",
    45: "frozen_yogurt",
    46: "garlic_bread",
    47: "gnocchi",
    48: "greek_salad",
    49: "grilled_cheese_sandwich",
    50: "grilled_salmon",
    51: "guacamole",
    52: "gyoza",
    53: "hamburger",
    54: "hot_and_sour_soup",
    55: "hot_dog",
    56: "huevos_rancheros",
    57: "hummus",
    58: "ice_cream",
    59: "lasagna",
    60: "lobster_bisque",
    61: "lobster_roll_sandwich",
    62: "macaroni_and_cheese",
    63: "macarons",
    64: "miso_soup",
    65: "mussels",
    66: "nachos",
    67: "omelette",
    68: "onion_rings",
    69: "oysters",
    70: "pad_thai",
    71: "paella",
    72: "pancakes",
    73: "panna_cotta",
    74: "peking_duck",
    75: "pho",
    76: "pizza",
    77: "pork_chop",
    78: "poutine",
    79: "prime_rib",
    80: "pulled_pork_sandwich",
    81: "ramen",
    82: "ravioli",
    83: "red_velvet_cake",
    84: "risotto",
    85: "samosa",
    86: "sashimi",
    87: "scallops",
    88: "seaweed_salad",
    89: "shrimp_and_grits",
    90: "spaghetti_bolognese",
    91: "spaghetti_carbonara",
    92: "spring_rolls",
    93: "steak",
    94: "strawberry_shortcake",
    95: "sushi",
    96: "tacos",
    97: "takoyaki",
    98: "tiramisu",
    99: "tuna_tartare",
    100: "waffles",
}


def classify_images(images, mode):
    """Send an image to the API and return results."""
    url = API_ENDPOINTS[mode]
    inference_endpoint = API_ENDPOINTS["direct"] if mode != "direct" else None

    # Prepare multiple images
    files = []
    for i, image in enumerate(images):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)
        files.append(("files", (f"image_{i}.jpg", img_byte_arr, "image/jpeg")))

    headers = {"X-Inference-Endpoint": inference_endpoint} if inference_endpoint else {}

    try:
        response = requests.post(url, files=files, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()
        if mode == "direct" or mode == "local":
            predictions = result.get("predictions")
        elif mode == "proxy":
            predictions = result.get("original_response", {}).get("predictions")
        assert predictions is not None
        return predictions
    except requests.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise


def display_image(image, true_label):
    """Display the image to be sent off to the API so we can get an idea of what we are classifying."""
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(f"True label: {CLASS_MAPPING[true_label]} (ID: {true_label})")
    plt.axis("off")
    plt.show(block=True)


def main(mode, data_path, num_images=1):
    """Main function for handling the request to the API."""
    try:
        dataset = load_dataset("parquet", data_files={"validation": data_path})["validation"]
        random_examples = random.sample(range(len(dataset)), num_images)

        images = []
        true_labels = []
        for idx in random_examples:
            images.append(dataset[idx]["image"])
            true_labels.append(dataset[idx]["label"])

        logger.info(f"Classifying {num_images} image{'s' if num_images > 1 else ''}")

        predictions = classify_images(images, mode)

        if not isinstance(predictions, list) or len(predictions) != num_images:
            raise ValueError(
                f"Expected {num_images} predictions, but got {len(predictions) if isinstance(predictions, list) else type(predictions)}"
            )

        for i, (image, true_label, prediction) in enumerate(zip(images, true_labels, predictions)):
            logger.info(f"\nImage {i+1}/{num_images}")
            logger.info(f"Image dimensions: {image.width}x{image.height}")
            logger.info(f"True label: {CLASS_MAPPING[true_label]} (ID: {true_label})")

            # Uncomment the next line if you want to display each image
            # display_image(image, true_label)

            if isinstance(prediction, list) and len(prediction) > 0:
                prediction = prediction[0]  # Take the first prediction if it's a list

            if (
                not isinstance(prediction, dict)
                or "label" not in prediction
                or "score" not in prediction
            ):
                logger.error(f"Unexpected prediction format: {prediction}")
                continue

            predicted_label = prediction["label"]
            predicted_score = prediction["score"]
            logger.info(f"Top prediction: {predicted_label} (Score: {predicted_score:.4f})")

            is_correct = predicted_label == CLASS_MAPPING[true_label]
            logger.info(f"Classification {'correct' if is_correct else 'incorrect'}")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Food Classification Script")
    parser.add_argument(
        "--mode", choices=list(API_ENDPOINTS.keys()), required=True, help="API endpoint to use"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./food101_data/data/validation-*.parquet",
        help="Path to validation data",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of images to classify (default: 1)",
    )
    args = parser.parse_args()

    main(args.mode, args.data_path, args.num_images)
