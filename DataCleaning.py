import pandas as pd
import random
import os

# Cartella delle immagini originali
input_images_folder = 'C:\\Users\\becca\\PycharmProjects\\ProgettoIANLP\\flickr30k_images'

# File CSV con le informazioni sulle immagini
csv_path = 'C:\\Users\\becca\\PycharmProjects\\ProgettoIANLP\\results.csv'

# Numero di immagini da eliminare casualmente
numero_immagini_da_eliminare = 30000  # Modifica questo numero in base alle tue esigenze

# Carica il DataFrame dal file CSV
df = pd.read_csv(csv_path, delimiter='|')

df = df.drop('comment_number', axis=1)

# Elimina le righe nulle
df = df.dropna()

# Rimuovi immagini associate alle righe nulle
image_folder_path = 'C:\\Users\\becca\\PycharmProjects\\ProgettoIANLP\\outputImages'
for index, row in df.iterrows():
    image_name = row['image_name']
    image_path = os.path.join(image_folder_path, image_name)
    if os.path.exists(image_path):
        os.remove(image_path)

# Salva il nuovo DataFrame nel file CSV
df.to_csv('C:\\Users\\becca\\PycharmProjects\\ProgettoIANLP\\results.csv', index=False, sep='|')


# Estrai un elenco casuale di nomi di immagini da eliminare
immagini_da_eliminare = random.sample(os.listdir(input_images_folder), numero_immagini_da_eliminare)

# Elimina le immagini dalla cartella
for img_name in immagini_da_eliminare:
    img_path = os.path.join(input_images_folder, img_name)
    os.remove(img_path)

# Filtra il DataFrame per mantenere solo le righe non associate alle immagini eliminate
df = df[~df['image_name'].isin(immagini_da_eliminare)]
