# real-time-skin-tone 🎨

## 🔍 Overview  
SkinToneDetection is a real-time skin tone detection system that identifies a user's skin tone from an uploaded image or a live camera capture. The detected skin tone is displayed along with its HEX code, helping users choose fashion styles that best complement their complexion.  

## 📌 Features  
✅ Detects skin tone from an uploaded image or live camera capture  
✅ Predicts and classifies the skin tone into categories: **Fair, Light Brown, Medium Brown, Ebony, Tan, Deep Dark, Dark Brown, Rich Dark**  
✅ Uses a deep learning model for accurate classification  
✅ Fast and efficient real-time detection  
✅ Runs completely offline without requiring a backend server  


## 🛠️ Technologies Used  
🔹 **Python 🐍** – Image processing & ML  
🔹 **OpenCV 🎥** – Camera handling & image capturing  
🔹 **TensorFlow/Keras 🤖** – Deep learning model for classification  
🔹 **NumPy & Scikit-Learn 📊** – For numerical operations & preprocessing  

## ⚙️ Installation & Setup  

### 1️⃣ Clone the Repository  

git clone https://github.com/SwarupaMasanam19/real-time-skin-tone.git 


cd real-time-skin-tone


### 2️⃣ Set Up Virtual Environment  

python -m venv myenv

source myenv/bin/activate  # macOS/Linux

myenv\Scripts\activate     # Windows


### 3️⃣ Install Dependencies  

pip install -r requirements.txt


### 4️⃣ Run the Application  
To start the real-time skin tone detection, run:  
python camera_detect.py

- Press `Space` to capture an image and predict skin tone.  
- The predicted skin tone will be displayed in the terminal.  

## **🖼️ How It Works**  
1️⃣ Capture an image or upload one.  
2️⃣ The model processes the image and extracts the dominant skin tone.  
3️⃣ The detected HEX color code is displayed on the UI.  
4️⃣ Use this information for personalized fashion recommendations!  

## 🚀 Future Improvements  
🔹 Improve accuracy using advanced Deep Learning models (CNNs)  
🔹 Integrate with the **AI-Powered Fashion Revolution** project  
🔹 Add a rule-based suggestion system for outfit recommendations  

---

## 💡 Have Ideas or Suggestions?  
Feel free to contribute or open an issue! 😊  

📌 **Developed by:** Swarupa Masanam  
📧 **Contact:** mlnswarupa05@gmail.com  
