from PIL import Image
import os


class ImageResizer:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder

        # Crea la cartella di output se non esiste
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)


    def resize_images(self, target_size):
        # Itera attraverso i file nella cartella di input
        print(f"Ridimensionamento immagini in corso...")
        for filename in os.listdir(self.input_folder):
            img_path = os.path.join(self.input_folder, filename)

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
                output_path = os.path.join(self.output_folder, filename)
                img.save(output_path)
            except Exception as e:
                # Gestisce gli errori nel caso in cui il ridimensionamento non sia possibile
                print(f"Errore nel ridimensionare l'immagine {img_path}: {str(e)}")

        print(f"Ridimensionamento immagini terminato")

