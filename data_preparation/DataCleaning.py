import pandas as pd
import random
import os

def data_cleaning():
    # Cartella delle immagini originali
    input_images_folder = '../input/images'

    # File CSV con le informazioni sulle immagini
    csv_path = '../input/dataset.csv'

    # Numero di immagini da eliminare casualmente
    numero_immagini_da_eliminare = 15000  # Modifica questo numero in base alle tue esigenze

    # Carica il DataFrame dal file CSV
    df = pd.read_csv(csv_path, delimiter='|')

    df = df.drop(' comment_number', axis=1)

    # Rimuovi le righe nulle dal DataFrame
    df = df.dropna()

    # Estrai un elenco casuale di nomi di immagini da eliminare
    elenco_immagini = os.listdir(input_images_folder)
    if len(elenco_immagini) > numero_immagini_da_eliminare:
        immagini_da_eliminare = random.sample(elenco_immagini, numero_immagini_da_eliminare)
    else:
        immagini_da_eliminare = elenco_immagini

    # Elimina le immagini dalla cartella
    for img_name in immagini_da_eliminare:
        img_path = os.path.join(input_images_folder, img_name)
        os.remove(img_path)

    # Filtra il DataFrame per mantenere solo le righe non associate alle immagini eliminate
    df = df[~df['image_name'].isin(immagini_da_eliminare)]

    # Salva il nuovo DataFrame nel file CSV
    df.to_csv('../processed/dataset.csv', index=False, sep='|')

if __name__ == '__main__':
    data_cleaning()
