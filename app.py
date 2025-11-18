import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from streamlit_drawable_canvas import st_canvas

# Set page config
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🔢 MNIST Handwritten Digit Classifier</div>', unsafe_allow_html=True)

st.write("""
This app uses a Convolutional Neural Network (CNN) trained on the MNIST dataset 
to recognize handwritten digits (0-9). Upload an image or draw a digit to see the prediction!
""")

# Load pre-trained model
@st.cache_resource
def load_model():
    # In a real deployment, you would load your saved model
    # For demo purposes, we'll use a simple approach
    try:
        model = tf.keras.models.load_model('mnist_cnn_model.h5')
    except:
        st.warning("Using demo mode. Train and save your model for full functionality.")
        # Create a simple model for demo
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_model()

# Sidebar
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose Input Method:", 
                         ["Upload Image", "Draw Digit"])

def preprocess_image(image):
    """Preprocess image for MNIST model"""
    # If image is from canvas (RGBA), convert it to grayscale using the alpha channel
    if image.mode == 'RGBA':
        # Create a white background and paste the image on it
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3]) # 3 is the alpha channel
        image = background.convert('L')
    elif image.mode != 'L': # For uploaded images
        image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Convert to array and normalize
    image_array = np.array(image) / 255.0
    
    # Invert colors if needed (MNIST has white digits on black background)
    if np.mean(image_array) > 0.5:  # If background is light
        image_array = 1 - image_array
    
    # Reshape for model
    image_array = image_array.reshape(1, 28, 28, 1)
    
    return image_array

if option == "Upload Image":
    st.subheader("📁 Upload a Digit Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=200)
        
        # Preprocess and predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        # Display results
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.subheader("🎯 Prediction Results")
        st.write(f"**Predicted Digit:** {predicted_digit}")
        st.write(f"**Confidence:** {confidence:.2%}")
        
        # Show probability distribution
        fig, ax = plt.subplots(figsize=(10, 4))
        digits = range(10)
        ax.bar(digits, prediction[0], color='skyblue')
        ax.set_xlabel('Digit')
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probabilities for Each Digit')
        ax.set_xticks(digits)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

else:  # Draw Digit
    st.subheader("✏️ Draw a Digit")
    st.write("Draw a single digit (0-9) in the canvas below. Try to make it large and centered.")
    
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    # If the user has drawn something
    if canvas_result.image_data is not None and st.button('Predict Drawn Digit'):
        # Get the image data from the canvas
        img_data = canvas_result.image_data
        
        # Convert to PIL Image
        image = Image.fromarray(img_data.astype('uint8'), 'RGBA')
        
        # Preprocess and predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        # Display results
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.subheader("🎯 Prediction Results")
        st.write(f"**Predicted Digit:** {predicted_digit}")
        st.write(f"**Confidence:** {confidence:.2%}")
        
        # You can also display the probability chart here if desired
        st.markdown('</div>', unsafe_allow_html=True)


# Model information
st.sidebar.markdown("---")
st.sidebar.subheader("Model Info")
st.sidebar.write("**Architecture:** CNN")
st.sidebar.write("**Training Data:** MNIST")
st.sidebar.write("**Input:** 28x28 grayscale images")
st.sidebar.write("**Output:** Digit 0-9")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using TensorFlow & Streamlit</p>
    <p>AI Tools Assignment - Mastering the AI Toolkit</p>
</div>
""", unsafe_allow_html=True)