import os
import io
import warnings
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import random

# Read the contents of the text file
with open("food_type.txt", "r") as file:
    food_content = file.read()

with open("dinnerware.txt", "r") as file:
    tableware = file.read()

with open("drink_type.txt", "r") as file:
    drink_content = file.read()

# Split the contents into individual food items based on the newline character
food_list = food_content.split("\n")
tableware = tableware.split("\n")
drink_list = drink_content.split("\n")

# Create a string for each food item with the desired format
food_strings = [f"{food}" for food in food_list if food.strip()]
tableware = [f"{ware}" for ware in tableware if ware.strip()]
drink_strings = [f"{drink}" for drink in drink_list if drink.strip()]
direction_strings = ['left side', 'right side', 'left front', 'right front', 'front', 'back left', 'back right']

# Our Host URL should not be prepended with "https" nor should it have a trailing slash.
os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'

# Sign up for an account at the following link to get an API Key.
# https://platform.stability.ai/

# Click on the following link once you have created an account to be taken to your API Key.
# https://platform.stability.ai/account/keys

# Paste your API Key below.
os.environ['STABILITY_KEY'] = ''

# Set up our connection to the API.
stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'], # API Key reference.
    verbose=True, # Print debug messages.
    engine="stable-diffusion-512-v2-1", # Set the engine to use for generation.
    # Check out the following link for a list of available engines: https://platform.stability.ai/docs/features/api-parameters#engine
)

for i in range(0,1000):
    food_idx = random.randint(0, len(food_strings) - 1)
    tableware_idx = random.randint(0, len(tableware) - 1)
    drink_idx = random.randint(0, len(drink_strings) - 1)

    chosen_food = food_strings[food_idx]
    chosen_tool1 = tableware[tableware_idx]
    chosen_tool2 = tableware[random.randint(0, len(tableware) - 1)]
    chosen_tool3 = tableware[random.randint(0, len(tableware) - 1)]
    chosen_drink = drink_strings[drink_idx]
    direction1=random.choice(direction_strings)
    direction2=random.choice(direction_strings)

    my_prompt = f'The center of the picture is a plate with {chosen_food} on this plate, a glass of {chosen_drink} on the {direction1} of the plate, a pair of {chosen_tool1} and ' \
                f'{chosen_tool2} on the {direction2} of the plate, and a {chosen_tool3} next to the {chosen_drink}. The picture is photographic.'
    # my_prompt = 'AI is used in middle school classroom for education.'
    print(my_prompt)
    # Set up our initial generation parameters.
    answers = stability_api.generate(
        prompt=my_prompt,
        # seed=4253978046, # If a seed is provided, the resulting generated image will be deterministic.
        # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
        # Note: This isn't quite the case for Clip Guided generations, which we'll tackle in a future example notebook.
        steps=30,  # Amount of inference steps performed on image generation. Defaults to 30.
        cfg_scale=8.0,  # Influences how strongly your generation is guided to match your prompt.
        # Setting this value higher increases the strength in which it tries to match your prompt.
        # Defaults to 7.0 if not specified.
        width=512,  # Generation width, defaults to 512 if not included.
        height=512,  # Generation height, defaults to 512 if not included.
        samples=1,  # Number of images to generate, defaults to 1 if not included.
        sampler=generation.SAMPLER_K_DPMPP_2M  # Choose which sampler we want to denoise our generation with.
        # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
        # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m, k_dpmpp_sde)
    )

    # Set up our warning to print to the console if the adult content classifier is tripped.
    # If adult content classifier is not tripped, save generated images.
    output_path = 'generative_food/'
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                name = chosen_food + '_' + chosen_drink + '_' + direction1+'_'+chosen_tool1 + '_' + chosen_tool2 + '_' + direction2+'_'+chosen_tool3
                img.save(output_path + str(
                    artifact.seed) + name + ".png")  # Save our generated images with their seed number as the filename
