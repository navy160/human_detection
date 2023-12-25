import face_recognition
from PIL import Image
import os

def detect_and_save_faces(input_image_path):
    # Load the input image
    input_image = face_recognition.load_image_file(input_image_path)

    # Find all face locations in the image
    face_locations = face_recognition.face_locations(input_image)

    # Extract filename without extension
    filename = os.path.splitext(os.path.basename(input_image_path))[0]

    # Create the output folder based on the input image filename
    output_folder = f"{filename}_output_faces"
    os.makedirs(output_folder, exist_ok=True)

    # Extract and save individual faces
    for i, face_location in enumerate(face_locations):
        top, right, bottom, left = face_location
        face_image = input_image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        output_path = os.path.join(output_folder, f"person_{i + 1}.jpg")
        pil_image.save(output_path)

if __name__ == "__main__":
    input_image_path = "input.jpg"  # Replace with your input image path

    detect_and_save_faces(input_image_path)
