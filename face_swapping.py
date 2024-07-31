import cv2
import dlib


# Load the images
img1 = cv2.imread("bradley_cooper.jpg")
img2 = cv2.imread("jim_carrey.jpg")

if img1 is None or img2 is None:
    print("Error: One or both images not found or unable to load.")
    exit()

# Ensure images are properly read and converted
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Convert the images to grayscale
img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

# Ensure images are 8-bit grayscale
print(f"img1_gray type: {img1_gray.dtype}, shape: {img1_gray.shape}")
print(f"img2_gray type: {img2_gray.dtype}, shape: {img2_gray.shape}")

# Display the grayscale images for debugging purposes
# cv2.imshow("image1", img1_gray)
# cv2.imshow("image2", img2_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()

# Ensure the shape predictor file path is correct
predictor_path = "shape_predictor_68_face_landmarks.dat"
try:
    predictor = dlib.shape_predictor(predictor_path)
except RuntimeError as e:
    print(f"Error loading shape predictor: {e}")
    exit()

# Detect faces in the first image
faces = detector(img1_gray)

# Process each detected face
for face in faces:
    landmarks = predictor(img1_gray, face)
    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))
        cv2.circle(img1, (x, y), 3, (0, 0, 255), -1)
    
    cv2.imshow("image", img1)

cv2.waitKey(0)
cv2.destroyAllWindows() 
