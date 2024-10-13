from deepface import DeepFace
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from scipy.spatial.distance import cosine
from mtcnn import MTCNN
import cv2
import numpy as np

# Step 1: Extract Photo from Passport Image using MTCNN
def extract_passport_photo_advanced(passport_image_path):
    """
    Extract the passport photo region from the passport image using MTCNN.
    Args:
        passport_image_path (str): Path to the passport image.
    Returns:
        np.array: Cropped passport photo.
    """
    image = cv2.imread(passport_image_path)
    detector = MTCNN()
    detected_faces = detector.detect_faces(image)

    if len(detected_faces) == 0:
        raise ValueError("No face detected in the passport image.")

    # Assuming the largest detected face is the passport photo
    x, y, w, h = detected_faces[0]['box']
    passport_photo = image[y:y+h, x:x+w]
    return passport_photo

passport_image_path = "images/passport_chaminda.jpeg"
selfie_image_path = "images/selfie/chaminda.jpg"

# Extract passport photo from the passport image
passport_photo = extract_passport_photo_advanced(passport_image_path)

# Save the extracted passport photo for further use
cv2.imwrite("extracted_passport_photo.jpg", passport_photo)

# Step 2: Extract Features from Passport Photo and Selfie
def extract_face_embedding(image_path, model_name='Facenet'):
    """
    Extract facial embeddings using DeepFace with alignment.
    Args:
        image_path (str): Path to the image.
        model_name (str): DeepFace model to use for extraction.
    Returns:
        np.array: Facial embedding vector.
    """
    embedding_result = DeepFace.represent(img_path=image_path, model_name=model_name, enforce_detection=True, align=True)
    if isinstance(embedding_result, list) and isinstance(embedding_result[0], dict):
        embedding = embedding_result[0]['embedding']
    elif isinstance(embedding_result, list):
        embedding = embedding_result[0]
    else:
        embedding = embedding_result
    return np.array(embedding).flatten()

# Extract embeddings from passport photo and selfie
passport_embedding = extract_face_embedding("extracted_passport_photo.jpg", model_name='Facenet')
selfie_embedding = extract_face_embedding(selfie_image_path, model_name='Facenet')

# Ensure that embeddings have the same dimensions
print(f"Passport embedding shape: {passport_embedding.shape}")
print(f"Selfie embedding shape: {selfie_embedding.shape}")

# Step 3: Calculate Similarity Score
def calculate_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings.
    Args:
        embedding1 (np.array): First embedding vector.
        embedding2 (np.array): Second embedding vector.
    Returns:
        float: Similarity score between 0 and 1.
    """
    if embedding1.shape != embedding2.shape:
        raise ValueError(f"Embeddings have different shapes: {embedding1.shape} and {embedding2.shape}")
    return 1 - cosine(embedding1, embedding2)

similarity_score = calculate_similarity(passport_embedding, selfie_embedding)
threshold = 0.7  # Define similarity threshold for matching

# Step 4: Create Match Context
def create_match_context(similarity_score, threshold):
    """
    Create context based on similarity score.
    Args:
        similarity_score (float): Similarity score between passport and selfie.
        threshold (float): Threshold for determining match.
    Returns:
        str: Contextual message for verification.
    """
    if similarity_score >= threshold:
        return "The passport photo matches the selfie. Proceed with onboarding."
    else:
        return "The passport photo does not match the selfie. Request additional verification."

context = create_match_context(similarity_score, threshold)
print(context)

# Step 5: Extract Additional Information from Passport using Vision LLM
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

passport_image = Image.open(passport_image_path)
inputs = processor(images=passport_image, return_tensors="pt")
outputs = model.get_image_features(**inputs)

# Here, outputs can be further processed to generate insights or verify textual data

# Step 6: Evaluation Function for Verification
def evaluate_matching(similarity_score, threshold):
    """
    Evaluate the matching of passport and selfie based on similarity score.
    Args:
        similarity_score (float): Calculated similarity score.
        threshold (float): Similarity threshold.
    Returns:
        str: Evaluation result indicating match or mismatch.
    """
    if similarity_score >= threshold:
        return "Match confirmed. Onboarding can proceed."
    else:
        return "Match not confirmed. Manual verification required."

# Final Evaluation and Print Statements
print("Facial similarity score:", similarity_score)
evaluation_result = evaluate_matching(similarity_score, threshold)
print(evaluation_result)