import cv2
import os

# Specify the input image path
input_image_path = 'grp.jpg'

# Check if the input image file exists
if not os.path.isfile(input_image_path):
    print(f"Error: The file '{input_image_path}' does not exist.")
else:
    # Create folder for saving face images
    output_folder = 'face_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the input image
    frame = cv2.imread(input_image_path)

    if frame is not None:
        print("Image loaded successfully.")
        print("Input image shape:", frame.shape)
        print("Input image data type:", frame.dtype)

        # Load the Haar Cascade classifier for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using the Haar Cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        face_images = []  # List to store cropped face images

        # Increase the border around the detected faces
        border_size = 20

        for (x, y, w, h) in faces:
            # Define new coordinates with increased border
            x_with_border = max(0, x - border_size)
            y_with_border = max(0, y - border_size)
            w_with_border = min(frame.shape[1] - x_with_border, w + 2 * border_size)
            h_with_border = min(frame.shape[0] - y_with_border, h + 2 * border_size)

            # Crop the face region from the frame with increased border
            face_roi = frame[y_with_border:y_with_border + h_with_border, x_with_border:x_with_border + w_with_border]

            # Append the cropped face image to the list
            face_images.append(face_roi)

            cv2.rectangle(frame, (x_with_border, y_with_border), (x_with_border + w_with_border, y_with_border + h_with_border), (0, 255, 0), 2)

        # Save detected faces as jpg files
        for idx, face_image in enumerate(face_images):
            filename = os.path.join(output_folder, f'face_{idx + 1}.jpg')
            cv2.imwrite(filename, face_image)

        # Display the image with rectangles around detected faces
        cv2.imshow("Detected Faces", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print(f"Error: Unable to read the image file '{input_image_path}'.")
