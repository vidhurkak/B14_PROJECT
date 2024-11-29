'''from django.shortcuts import render,redirect,HttpResponseRedirect
from django.contrib.auth.forms import AuthenticationForm,UserCreationForm
from django.contrib.auth import authenticate
import base64
import numpy as np
import cv2
from django.core.files.storage import FileSystemStorage
from joblib import load
# Create your views here.
def home(request):
    return render(request,'home.html')
 
def login(request):
    if(request.user.is_authenticated):
        return render(request,'login.html')
    if(request.method == "POST"):
        un = request.POST['username']
        pw = request.POST['password']
        #authenticate() is used to check for the values present in the database or not
        #if the values are matched, then it will return the username
        #if the values are not matched, then it will return as 'None'
        # use authenticate(), need to import it from auth package
        user = authenticate(request,username=un,password=pw)
        if(user is not None):
            return redirect('/profile')
        else:
            msg = 'Invalid Username/Password'
            form = AuthenticationForm(request.POST)
            return render(request,'login.html',{'form':form,'msg':msg})
    else:
        form = AuthenticationForm()
        #used to create a basic login page with username and password
        return render(request,'login.html',{'form':form})
 
def register(request):
    if(request.user.is_authenticated):
        return redirect('/')
    if(request.method == "POST"):
        form = UserCreationForm(request.POST)
        if(form.is_valid()):
            form.save()
            un = form.cleaned_data.get('username')
            pw = form.cleaned_data.get('password1')
            authenticate(username=un,password=pw)
            return redirect('/login')
    else:
        form = UserCreationForm()
        #UserCreationForm() is used to create a basic registration page with username, password and confirm password
    return render(request,'register.html',{'form':form})
   '''


'''def profile(request):
    if request.method == "POST":
        if request.FILES.get('uploadImage'):
            img_name = request.FILES['uploadImage'].read()
            encode = base64.b64encode(img_name).decode('utf-8')
            img_url = f"data:image/jpeg;base64,{encode}"
            return render(request, 'profile.html', {'img': img_url})
    return render(request, 'profile.html')'''





'''def profile(request):
    if request.method == "POST":
        if request.FILES.get('uploadImage'):
            # Read the uploaded image file
            img_name = request.FILES['uploadImage'].read()
            
            # Convert image to NumPy array
            nparr = np.frombuffer(img_name, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Read image using OpenCV

            if img is None:
                # Handle case where image is not decoded properly
                return render(request, 'profile.html', {'error': "Error processing the image."})

            # Get image dimensions
            height, width, channels = img.shape  # Get height, width, and number of channels

            # Normalize the image if necessary
            img_normalized = img.astype(np.float32) / 255.0  # Normalize the image to 0-1 range
            img_display = (img_normalized * 255).astype(np.uint8)  # Convert back to 0-255 range for display

            # Encode the processed image in base64 to display in HTML
            _, buffer = cv2.imencode('.jpg', img_display)  # Encode image to JPEG format
            encode = base64.b64encode(buffer).decode('utf-8')
            img_url = f"data:image/jpeg;base64,{encode}"

            # Render the template with the image URL and dimensions
            return render(request, 'profile.html', {
                'img': img_url,
                'height': height,
                'width': width,
                 # You can also show the number of channels (e.g., 3 for RGB)
            })

    # If the method is not POST or no image uploaded, render the page without an image
    return render(request, 'profile.html')
'''




'''def profile(request):
    if request.method == "POST":
        if request.FILES.get('uploadImage'):
            img_name = request.FILES['uploadImage'].read()
            npimg = np.frombuffer(img_name, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            dimensions_array = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                dimensions_array.append([float(w) / img.shape[1], float(h) / img.shape[0]])

                
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            
            dimensions_nested_array = np.array(dimensions_array).reshape(-1, 3).tolist()

            
            _, buffer = cv2.imencode('.jpg', img)
            processed_img = base64.b64encode(buffer).decode('utf-8')
            processed_img_url = f"data:image/jpeg;base64,{processed_img}"

            img_url = f"data:image/jpeg;base64,{base64.b64encode(img_name).decode('utf-8')}"

            return render(request, 'profile.html', {'img': img_url, 'processed_img': processed_img_url, 'dimensions': dimensions_nested_array})

    return render(request, 'profile.html')
'''

import os
import base64
from django.shortcuts import render, redirect  # type: ignore
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm  # type: ignore
from django.contrib.auth import authenticate  # type: ignore
from django.core.files.storage import FileSystemStorage  # type: ignore
from django.conf import settings  # Import settings
import tensorflow as tf
from joblib import load
import cv2
import numpy as np

# Load the TensorFlow model
model = tf.keras.models.load_model('skin_disease_model.h5')

# Load the optimized model
rf_model = load('skin_disease_ten_model.joblib')

def home(request):
    return render(request, 'home.html')

def login(request):
    if request.user.is_authenticated:
        return render(request, 'login.html')
    if request.method == "POST":
        un = request.POST['username']
        pw = request.POST['password']
        user = authenticate(request, username=un, password=pw)
        if user is not None:
            return redirect('/profile')
        else:
            msg = 'Invalid Username/Password'
            form = AuthenticationForm(request.POST)
            return render(request, 'login.html', {'form': form, 'msg': msg})
    else:
        form = AuthenticationForm()
        return render(request, 'login.html', {'form': form})

def register(request):
    if request.user.is_authenticated:
        return redirect('/')
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            un = form.cleaned_data.get('username')
            pw = form.cleaned_data.get('password1')
            authenticate(username=un, password=pw)
            return redirect('/login')
    else:
        form = UserCreationForm()
        return render(request, 'register.html', {'form': form})

def profile(request):
    if request.method == "POST":
        if request.FILES.get('uploadImage'):
            img_name = request.FILES['uploadImage']
            print(f"Uploaded file name: {img_name.name}")  # Debugging the uploaded file name
            fs = FileSystemStorage(location=settings.MEDIA_ROOT)  # Save to media root
            filename = fs.save(img_name.name, img_name)
            img_url = fs.url(filename)
            img_path = fs.path(filename)
            print(f"Image URL: {img_url}")  # Debugging the image URL
            print(f"Image path on disk: {img_path}")  # Debugging the path where the image is stored

            # Image processing and prediction code
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print("Error: Could not read image.")
                return render(request, 'profile.html', {'error': 'Error reading image. Please try again.'})

            # Resize to (128, 128) for the TensorFlow model
            img = cv2.resize(img, (128, 128))
            
            # Normalize the image based on how it was trained (check your model training preprocessing)
            img = img / 255.0  # This is an example normalization, adjust if your model uses different scaling
            print(f"Normalized image: {img}")  # Debugging normalized image values

            img = np.expand_dims(img, axis=0)  # Add batch dimension
            print(f"Input shape for prediction: {img.shape}")  # Debugging input shape

            # Predict using the TensorFlow model
            try:
                predict = model.predict(img)  # Get prediction probabilities
                print(f"Prediction result: {predict}")  # Debugging prediction result

                # If the model outputs a probability distribution
                if predict.shape[-1] > 1:
                    predict_proba = predict[0]  # Get the class probabilities
                    predict_class = np.argmax(predict_proba)  # Get the predicted class with highest probability
                    print(f"Prediction probabilities: {predict_proba}, Predicted class: {predict_class}")
                else:
                    predict_class = np.argmax(predict)  # For binary classification

            except Exception as e:
                print(f"Error during prediction: {e}")
                return render(request, 'profile.html', {'error': 'Prediction error. Check logs for details.'})

            # Classes and corresponding diagnoses
            skin_disease_names = ['Cellulitis', 'Impetigo', 'Athlete Foot', 'Nail Fungus', 'Ringworm', 
                                  'Cutaneous Larva Migrans', 'Chickenpox', 'Shingles']
            diagnosis = [
                'Seek immediate medical attention.',
                'Topical antibiotics may help.',
                'Use antifungal creams or powders.',
                'Consult a dermatologist for treatment.',
                'Apply antifungal creams as prescribed.',
                'Avoid walking barefoot; consult a doctor.',
                'Stay hydrated; avoid scratching blisters.',
                'Antiviral medication may be required.'
            ]

            # Map prediction to label and diagnosis
            if 0 <= predict_class < len(skin_disease_names):
                result1 = skin_disease_names[predict_class]
            else:
                result1 = "Unknown Disease"

            if 0 <= predict_class < len(diagnosis):
                result2 = diagnosis[predict_class]
            else:
                result2 = "No diagnosis available"

            return render(request, 'profile.html', {'img_url': img_url, 'obj1': result1, 'obj2': result2})
        else:
            print("No image file uploaded")  # Debugging when no image is uploaded
            return render(request, 'profile.html', {'error': 'No image uploaded. Please try again.'})

    return render(request, 'profile.html')
