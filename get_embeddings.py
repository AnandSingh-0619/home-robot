import clip
import pickle
import torch
import numpy as np

receptacles = [
    "bathtub", 
    "bed", 
    "bench", 
    "cabinet", 
    "chair", 
    "chest_of_drawers",
    "couch", 
    "counter", 
    "filing_cabinet", 
    "hamper", 
    "serving_cart", 
    "shelves",
    "shoe_rack", 
    "sink", 
    "stand", 
    "stool", 
    "table", 
    "toilet", 
    "trunk", 
    "wardrobe",
    "washer_dryer"
]

objects = ["action_figure", "android_figure", "apple", "backpack", "baseballbat",
    "basket", "basketball", "bath_towel", "battery_charger", "board_game",
    "book", "bottle", "bowl", "box", "bread", "bundt_pan", "butter_dish",
    "c-clamp", "cake_pan", "can", "can_opener", "candle", "candle_holder",
    "candy_bar", "canister", "carrying_case", "casserole", "cellphone", "clock",
    "cloth", "credit_card", "cup", "cushion", "dish", "doll", "dumbbell", "egg",
    "electric_kettle", "electronic_cable", "file_sorter", "folder", "fork",
    "gaming_console", "glass", "hammer", "hand_towel", "handbag", "hard_drive",
    "hat", "helmet", "jar", "jug", "kettle", "keychain", "knife", "ladle", "lamp",
    "laptop", "laptop_cover", "laptop_stand", "lettuce", "lunch_box",
    "milk_frother_cup", "monitor_stand", "mouse_pad", "multiport_hub",
    "newspaper", "pan", "pen", "pencil_case", "phone_stand", "picture_frame",
    "pitcher", "plant_container", "plant_saucer", "plate", "plunger", "pot",
    "potato", "ramekin", "remote", "salt_and_pepper_shaker", "scissors",
    "screwdriver", "shoe", "soap", "soap_dish", "soap_dispenser", "spatula",
    "spectacles", "spicemill", "sponge", "spoon", "spray_bottle", "squeezer",
    "statue", "stuffed_toy", "sushi_mat", "tape", "teapot", "tennis_racquet",
    "tissue_box", "toiletry", "tomato", "toy_airplane", "toy_animal", "toy_bee",
    "toy_cactus", "toy_construction_set", "toy_fire_truck", "toy_food",
    "toy_fruits", "toy_lamp", "toy_pineapple", "toy_rattle", "toy_refrigerator",
    "toy_sink", "toy_sofa", "toy_swing", "toy_table", "toy_vehicle", "tray",
    "utensil_holder_cup", "vase", "video_game_cartridge", "watch", "watering_can",
    "wine_bottle"]

def get_embedding(s):
    inputs = tokenizer(s, return_tensors="pt")
    outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states[0, 0].detach().numpy()


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# Prepare the inputs
text_inputs = torch.cat(
    [clip.tokenize(f"a photo of a {c}") for c in receptacles]
).to(device)
save_path = "clip_vit_recep_embeddings.pickle"

# Get CLIP embeddings
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
embeddings = {
    o: t for o, t in zip(receptacles, text_features.detach().cpu().numpy())
}
pickle.dump(embeddings, open(save_path, "wb"))


# embeddings_file = "clip_embeddings.pickle"
# with open(embeddings_file, "rb") as f:
#             _embeddings = pickle.load(f)
# category_name = objects[8]
# precomputed_embed = _embeddings[category_name]

# # Load the CLIP model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("RN50", device)
# object = objects[8]
# # Tokenize the object name and get its embedding
# text_input = clip.tokenize(f"a photo of a {object}").to(device)
# with torch.no_grad():
#     text_embedding = model.encode_text(text_input).detach().cpu().numpy()

# # Compute the cosine similarity between the precomputed embedding and the new embedding
# similarity = np.dot(precomputed_embed, text_embedding.T) / (
#     np.linalg.norm(precomputed_embed) * np.linalg.norm(text_embedding)
# )

# # Print the similarity score
# print(f"Similarity between the precomputed embedding and the object name '{category_name}': {similarity.item()}")
