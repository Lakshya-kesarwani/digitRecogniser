# from flask import Flask, request, jsonify, render_template
# import base64
# from io import BytesIO
# from PIL import Image
# import numpy as np
# import pandas as pd
# import cv2 as cv
# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html',result_text='')

# @app.route('/upload_image', methods=['POST'])
# def upload_image():
#     data = request.get_json()
    
#     if not data or 'imageData' not in data:
#         return jsonify({'error': 'No image data found'}), 400

#     base64_data = data['imageData']
    
#     try:
#         image = decode_base64_image(base64_data)
#         image_np = np.array(image)
#         image_list = image_np.tolist()
        
#         # cv.imshow("image",image_np)
#         cv.waitKey(0)
        

#         return jsonify(result=f'The predicted digit is: {pred(image_np)}')
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# def decode_base64_image(base64_str):
#     img_data = base64.b64decode(base64_str)
#     image = Image.open(BytesIO(img_data)).convert('RGB')
#     # print(image)
#     return image
# # decode_base64_image()
# import joblib
# from sklearn.preprocessing import StandardScaler  # Example if you need to scale your data
# import pickle
# # Example RGB image matrix (replace this with your actual matrix)
# # This should be a 28x28x3 matrix
# def pred(img_data):
#     # Load the model and scaler from file
#     with open('knn_model.pkl', 'rb') as model_file:
#         knn = pickle.load(model_file)

#     with open('scaler.pkl', 'rb') as scaler_file:
#         scaler = pickle.load(scaler_file)
#         image_matrix = img_data
#     print("printed till here: 1")
    
#     print(image_matrix.shape)
#     # Step 1: Convert RGB matrix to grayscale by averaging the RGB values
#     # The resulting matrix will be 28x28
#     grayscale_matrix = np.mean(image_matrix, axis=2)
#     # print("grayscale_matrix: ", grayscale_matrix)
#     # Step 2: Flatten the grayscale matrix to a 1D array (28*28 = 784 elements)
#     flattened_image = grayscale_matrix.flatten()
#     # flattened_image = flattened_image.reshape(1,-1)
    
#     # flattened_image = scaler.transform(flattened_image.reshape(1, -1))
#     print(flattened_image.shape)
#     downsampled_image = downsample_image(grayscale_matrix, block_size=10)
#     print(downsampled_image.shape)
#     downsampled_image = downsampled_image.flatten()
#     # downsampled_image = downsample_image.reshape(1,-1)
#     print((downsampled_image))
#     data = pd.DataFrame([downsampled_image],columns=[f"pixel{i}" for i in range(784)])
#     # print(data)

#     print("printed till here:  2")

#     # If your model expects the input to be standardized or normalized, do that here
#     # For example, using StandardScaler if you have used it during training
#     # scaler = StandardScaler()
#     # flattened_image = scaler.fit_transform(flattened_image.reshape(-1, 1)).flatten()

#     # Step 3: Load your trained model
#     # model = joblib.load("model.joblib")
#     print("model loaded successfully")
#     # Predict using the model
#     predicted_digit = knn.predict(data)[0]

#     retu(f'The predicted digit is: {predicted_digit}')

# def downsample_image(image, block_size):
#     """
#     Downsamples an image by averaging blocks of pixels.
    
#     Parameters:
#     - image: A numpy array of shape (height, width)
#     - block_size: The size of each block to average, e.g., 10 for 100x100 cells
    
#     Returns:
#     - A numpy array of the downsampled image
#     """
#     # Get the shape of the input image
#     height, width = image.shape

#     # Calculate the shape of the downsampled image
#     new_height = height // block_size
#     new_width = width // block_size

#     # Initialize the downsampled image
#     downsampled_image = np.zeros((new_height, new_width), dtype=np.float32)

#     # Reshape and average blocks
#     for i in range(new_height):
#         for j in range(new_width):
#             block = image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
#             downsampled_image[i, j] = np.mean(block)

#     return downsampled_image.astype(np.uint8)
# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', result_text='')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    data = request.get_json()
    print(-1)
    if not data or 'imageData' not in data:
        return jsonify({'error': 'No image data found'}), 400

    base64_data = data['imageData']
    print(0)
    
    try:
        image = decode_base64_image(base64_data)
        image_np = np.array(image)
        print(0.1)
        
        # Process the image
        result_text = pred(image_np)
        print("sent successfully 2")
        
        return jsonify(result=result_text)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def decode_base64_image(base64_str):
    img_data = base64.b64decode(base64_str.split(',')[1])
    image = Image.open(BytesIO(img_data)).convert('RGB')
    return image

def pred(img_data):
    # Load the model and scaler from file
    with open('knn_model.pkl', 'rb') as model_file:
        knn = pickle.load(model_file)
    print(1)
    print(img_data.shape)
    grayscale_matrix = np.mean(img_data, axis=2)
    print(grayscale_matrix.shape)
    # Downsample image if needed
    downsampled_image = downsample_image(grayscale_matrix, block_size=10)
    print(downsampled_image.shape)
    import cv2
# Display the image
    rgb_image = np.stack([downsampled_image] * 3, axis=-1)
    # cv2.imshow('28x28x3 RGB Image (from grayscale)', rgb_image)

# Display the original grayscale image for comparison
    # grayscale_image_resized = cv2.resize(downsampled_image,interpolation=cv2.INTER_NEAREST)
    # cv2.imshow('Original 28x28 Grayscale Image', downsampled_image)
    # cv2.imwrite('rgb_image.png', rgb_image)
    # cv2.imwrite('grayscale_image.png', downsampled_image)
    # Wait for a key press
    cv2.waitKey(0)
    # Convert image to grayscale by averaging RGB values
    # grayscale_matrix = np.mean(downsampled_image, axis=2)
    print(2)
    
    # Flatten the grayscale matrix to a 1D array
    flattened_image = downsampled_image.flatten()
    print(3)
    print(flattened_image.shape)
    # Create DataFrame
    data = pd.DataFrame([flattened_image], columns=[f"pixel{i}" for i in range(784)])
    
    # Predict using the model
    predicted_digit = knn.predict(data)[0]
    print("sent successfully")
    return f'The predicted digit is: {predicted_digit}'

def downsample_image(image, block_size):
    """
    Downsamples an image by averaging blocks of pixels.
    
    Parameters:
    - image: A numpy array of shape (height, width, channels)
    - block_size: The size of each block to average, e.g., 10 for 10x10 blocks
    
    Returns:
    - A numpy array of the downsampled image
    """
    # Convert to grayscale
    # grayscale_image = np.mean(image, axis=2)
    
    # Get the shape of the input image
    height, width = image.shape

    # Calculate the shape of the downsampled image
    new_height = height // block_size
    new_width = width // block_size

    # Initialize the downsampled image
    downsampled_image = np.zeros((new_height, new_width), dtype=np.float32)

    # Reshape and average blocks
    for i in range(new_height):
        for j in range(new_width):
            block = image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            downsampled_image[i, j] = np.mean(block)

    return downsampled_image.astype(np.uint8)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
