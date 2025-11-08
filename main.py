'''
You dont need to pip install anything on your main computerto run the code. Just create a virtual machine on the root repository.
How to do that in terminal:
1) Run: python -m venv myenv
That creates a folder called myenv which is basically your virtual machine
2) Run: source myenv/Scripts/activate
You will seen in your terminal (myenv)
This means you are now techinically running off your virtual machine
3) Run: pip install -r requirements.txt
Nice everything should work now. When you are done, just type deactivate in terminal.
Do not push your virtual machine into github. Just leave it local.
You only need to run pip install -r requirements.txt once ever.
'''

########## Installing necessary libraries
import os
import numpy as np
# For PDF reading and OCR if required
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
import pdf2image
# Embedding vectors
import tensorflow_hub as hub
# Neural Network
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.optimizers import Adam

########## File setup
DirectoryName = 'TrainingData'                                              # Directory name from folder structure
files = os.listdir(DirectoryName)                                           # Grab names of PDF files and places it in list in files variable
embedder = hub.load("https://www.kaggle.com/models/google/nnlm/TensorFlow2/en-dim128/1")                         # Embedding model used to process PDF text into inputs for NN | 768th dimension --> 768 vectors
layerOneNeurons = 4                                                        # Number of neurons in the first layer
# layerTwoNeurons = 5                                                         # Number of neurons in the second layer | There is such little training data, dont use second layer
listOfEmbeddings = []                                                       # Will hold all embedding values for each training data file in this list
labels = []                                                                 # For training data, we have to determine what value to give it. 2, 1, or 0.

def extract_text_from_pdf(path):
    '''
    Input: Path to the file including folder name and file name
    Output: Text contents of the actual PDF file
    '''

    text = ""
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            text += page.extract_text()
        if text.strip():
            return text
    except:
        pass

    # If file is a scanned PDF instead of pure PDF, use this following portion | Not tested | Requires poppler download directly on computer
    images = pdf2image.convert_from_path(path)
    for image in images:
        text += pytesseract.image_to_string(image)

    return text

# Itterating through each file's contents
for item in files:
    text = extract_text_from_pdf(f'{DirectoryName}/{item}')
    embedding = embedder([text]).numpy()[0]

    listOfEmbeddings.append(embedding)                                      # Appends the entirety of a single files' embeddings into the embeddings list for training data
    labels.append(item[0])                                                  # The first character of the files' name is the value we've determined the expected output is for training data


# Neural network build out

model = Sequential([
    Dense(layerOneNeurons, activation='relu', input_shape=(128,)),          # input_shape = 128 as there are that many dimensions in embeddings
    # layers.Dense(layerTwoNeurons, activation='relu'),
    Dense(3)                                                                # 3 potential outputs. 2 = good | 1 = average | 0 = poor
])

# Neural network Keras works off of array datatype.
X = np.array(listOfEmbeddings)
y = to_categorical(labels, num_classes = 3)                                 # to_categorical does one-hot encoding, basically when training 1 output gets a value 1, the rest get 0

# More details on how the model trains 
optimizer = Adam(learning_rate=1e-3)  # 0.0001
model.compile(
    optimizer = optimizer,                                                       # ChatGPT states adam is the best optimizer for small datasets
    loss = 'categorical_crossentropy',                                        # Suitable loss for multi-class classification
    metrics = ['accuracy']
)

history = model.fit( 
    X, y,
    epochs = 10,                                                              
    batch_size = 4,                                                           # Small batch size for tiny dataset
    verbose = 1
)

print("Final training accuracy:", history.history['accuracy'][-1])

# ---------------------------------------------------------------

# Example test PDF path
test_pdf_path = 'TestData/test2.pdf'

test_text = extract_text_from_pdf(test_pdf_path)

test_embedding = embedder([test_text]).numpy()[0]  # shape: (512,)

X_test = np.expand_dims(test_embedding, axis=0)  # shape: (1, 512)

y_pred = model.predict(X_test)

predicted_class = np.argmax(y_pred, axis=1)[0]

print("Predicted class:", predicted_class)