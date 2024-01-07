import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# Carica il modello VGG16 pre-addestrato su ImageNet
model = VGG16(weights='imagenet', include_top=False)


# VGG16 è una rete neurale convoluzionale (CNN) pre-addestrata utilizzata per l'analisi di immagini.
# 'weights='imagenet'' indica che stiamo usando i pesi pre-addestrati sull'insieme di dati ImageNet.
# 'include_top=False' indica che non includeremo il layer completamente connesso finale, adatto a problemi di classificazione a 1000 classi.
# Invece, useremo le feature estratte da uno degli ultimi layer convoluzionali.

# Funzione per estrarre features da un'immagine
def extract_features(img_path):
    # Estrai il nome del file dall'intero percorso
    img_name = os.path.basename(img_path)

    # Carica l'immagine e ridimensiona alle dimensioni richieste
    img = image.load_img(img_path, target_size=(224, 224))

    # Converti l'immagine in un array numpy
    img_array = image.img_to_array(img)

    # Aggiungi una dimensione all'array (batch dimension)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocessa l'immagine per essere compatibile con il modello VGG16
    img_array = preprocess_input(img_array)

    # Estrai le features utilizzando il modello VGG16
    features = model.predict(img_array)

    # Restituisci il nome dell'immagine e le features appiattite
    return img_name, features.flatten()


# Percorso della cartella contenente le immagini
folder_path = 'C:\\Users\\becca\\PycharmProjects\\ImageDescriptionGeneratorNLP\\outputImages'

# Elenco delle immagini nella cartella con estensioni supportate
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Estrai le features per ciascuna immagine nella cartella
all_features = []
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)

    # Chiama la funzione per estrarre le features dall'immagine
    img_name, features = extract_features(image_path)

    # Aggiungi il nome dell'immagine e le features alla lista
    all_features.append((img_name, features))

# Creazione di un array strutturato per contenere le features
dtype = [('image_name', 'U100'), ('features', np.float32, (len(all_features[0][1]),))]
all_features_array = np.array(all_features, dtype=dtype)

# Salva le features in un file numpy (.npy) nella stessa cartella dello script
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'features.npy')
np.save(save_path, all_features_array)

print(f"Features salvate in: {save_path}")
