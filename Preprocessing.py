import pandas as pd
from PIL import Image
import os


# Funzione per ridimensionare le immagini
def resize_images(input_folder, output_folder, target_size=(299, 299)):
    # Crea la cartella di output se non esiste
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Itera attraverso i file nella cartella di input
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)

        # Verifica se il file è un'immagine
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            print(f"Ignorato il file non immagine: {img_path}")
            continue

        try:
            # Apre l'immagine
            img = Image.open(img_path)

            # Ridimensiona l'immagine con il metodo Lanczos
            img = img.resize(target_size, resample=Image.LANCZOS)

            # Crea il percorso di output e salva l'immagine ridimensionata
            output_path = os.path.join(output_folder, filename)
            img.save(output_path)
        except Exception as e:
            # Gestisce gli errori nel caso in cui il ridimensionamento non sia possibile
            print(f"Errore nel ridimensionare l'immagine {img_path}: {str(e)}")


# Percorsi delle cartelle di input e output e dimensioni target
input_folder = 'C:\\Users\\becca\\PycharmProjects\\ImageDescriptionGeneratorNLP\\flickr30k_images'
output_folder = 'C:\\Users\\becca\PycharmProjects\\ImageDescriptionGeneratorNLP\\outputImages'
target_size = (299, 299)

# Chiama la funzione per ridimensionare le immagini
resize_images(input_folder, output_folder, target_size)
