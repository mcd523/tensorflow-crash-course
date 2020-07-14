import os

from AnimalsExample import AnimalsExample
from AnimalsExampleWithAugmentation import AnimalsExampleWithAugmentation
from AnimalsExampleWithAugmentationFromTensorflowHub import AnimalsExampleWithAugmentationFromTensorflowHub
from ClothesExample import ClothesExample
from FlowerExercise import FlowerExercise
from HousingPrices import HousingPrices
from Reuters import Reuters
from Reviews import Reviews

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CLOTHES_BATCH_SIZE = 32
ANIMALS_BATCH_SIZE = 100  # Number of training examples to process before updating our models variables
IMG_SHAPE = 150  # Our training data consists of images with width of 150 pixels and height of 150 pixels


def main():
    # animals = AnimalsExample(ANIMALS_BATCH_SIZE, IMG_SHAPE)
    # animals.run()
    # animals_with_aug = AnimalsExampleWithAugmentation(ANIMALS_BATCH_SIZE, IMG_SHAPE)
    # animals_with_aug.run()
    # clothes = ClothesExample(CLOTHES_BATCH_SIZE)
    # clothes.run()
    # flowers = FlowerExercise(CLOTHES_BATCH_SIZE, IMG_SHAPE)
    # flowers.run()
    # hub_animals = AnimalsExampleWithAugmentationFromTensorflowHub(CLOTHES_BATCH_SIZE, 224)
    # hub_animals.run()
    # imdb = Reviews()
    # imdb.run()
    # reuters = Reuters()
    # reuters.run()
    housing = HousingPrices(500)
    housing.run()


if __name__ == "__main__":
    main()
