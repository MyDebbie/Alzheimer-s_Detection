"""
Membres du groupe:
Mydleyka Dimanche
Rooseline Alisme
Dashka Cine
Owens Jean Paulson Mathurin
"""
import tkinter
from tkinter import Canvas
from tkinter import BOTH, YES
from tkinter import filedialog

import cv2
from PIL import ImageTk, Image
import numpy as np
import tensorflow as tf
from tensorflow_addons.metrics import F1Score


class Application(tkinter.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack(fill=BOTH, expand=YES)
        self.create_widgets()

    def create_widgets(self):
        # Charger l'image
        image = Image.open("image\\brain-1787622__340.webp")
        self.photo = ImageTk.PhotoImage(image)
        image_array = np.array(image)

        # Calculer la couleur moyenne de l'image
        average_color = np.mean(image_array, axis=(0, 1)).astype(int)

        # Créer un Canvas
        self.canvas = Canvas(self, width=1000, height=600)
        self.canvas.pack(fill=BOTH, expand=YES)

        # Dessiner l'image sur le Canvas
        self.canvas_image = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)

        # Fonction pour redimensionner l'image
        def resize_image(event):
            new_width = event.width
            new_height = event.height
            resized_image = image.resize((new_width, new_height))
            self.photo = ImageTk.PhotoImage(resized_image)
            self.canvas.itemconfig(self.canvas_image, image=self.photo)

        # Détecter les modifications de taille de la fenêtre principale
        self.canvas.bind('<Configure>', resize_image)

        self.master.iconphoto(False, ImageTk.PhotoImage(file='image\\8725540_brain_icon.png'))

        message = "Upload une image du cerveau \npour dectecter si elle est atteinte d'Alzheimer"
        self.canvas.create_text(500, 50, anchor='e', text=message, font=("Arial", 18), fill='white',
                                justify=tkinter.LEFT)

        # Ajouter des widgets à la fenêtre principale
        btn = tkinter.Button(self, text="UPLOAD YOUR IMAGE", font='sans 16 bold', command=self.upload_image)
        btn.place(relx=0.05, rely=0.7)

    def upload_image(self):
        # open the file dialog to select an image file
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        IMG_SIZE = 176
        img_array = cv2.imread(file_path)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

        # check if a file is selected
        if file_path:
            # open the image file
            image = Image.open(file_path)
            image = image.resize((400, 400))

            # create a PhotoImage object from the image
            photo = ImageTk.PhotoImage(image)

            # update the label with the new image
            label = tkinter.Label(self)
            label.place(x=570, y=30)

            label.config(image=photo)
            label.image = photo

            # Define the custom metric function
            def f1_score(y_true, y_pred):
                return F1Score(num_classes=4)(y_true, y_pred)

            # Register the custom metric function
            with tf.keras.utils.custom_object_scope({'f1_score': f1_score}):
                # Load the saved model
                model = tf.keras.models.load_model('Alzheimer_Detection.h5')
                data = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

            stage = model.predict(data)
            stage_list = ['MildDemented', 'NonDemented', 'ModerateDemented', 'VeryMildDemented']

            def trouver(p):
                ar = []
                for n in p:
                    ar.append(int(n))
                return ar.index(1, 0)

            stage_name = stage_list[trouver(stage[0])]
            print(stage_name)
            label1 = tkinter.Label(self, text=f"The stage of Alzheimer is {stage_name}", font='sans 14 bold')
            label1.place(x=600, y=500)


# Créer la fenêtre principale
root = tkinter.Tk()
root.title("Application pour detecter le stage de l'Alzheimer")

# Créer une instance de l'application
app = Application(master=root)
# Démarrer la boucle d'événements
app.mainloop()
