# -*- coding: utf-8 -*-
import Tkinter
import tkFileDialog
import tkMessageBox
import librosa, numpy as np, os, sklearn
from keras.models import load_model

from Tkinter import *

n_mfcc = 12
scaler = sklearn.preprocessing.StandardScaler()
model = load_model('my_model.h5')
fs = 22050

def extract_MFCC(song):
    mfcc_s = librosa.feature.mfcc(song, sr=fs, n_mfcc=n_mfcc).T
    mfcc_scaled = scaler.fit_transform(mfcc_s)
    features = mfcc_scaled.reshape(len(mfcc_scaled)*n_mfcc)
    return features

def get_output_filename(input_file_name):
    return input_file_name.rpartition(".")[0] + ".rst"


def gui():
#TODO: conectar desde aqui
    def identify_callback():
        root.update()
        path = entry.get()
        x, fs0 = librosa.load(path, duration=30)
        """ what to do when the "Go" button is pressed """
        features = extract_MFCC(x)
        prediction = model.predict(features.reshape(-1,15504))
        genre = np.argmax(prediction)
        if genre == 0:
            output = 'Vallenato'
        elif genre == 1:
            output = 'Pasillo'
        elif genre == 2:
            output = 'Llanera'
        else:
            output = 'Carranga'
        tkMessageBox.showinfo("Resultado",output)

    def browse_callback():
        root.update()
        """ What to do when the Browse button is pressed """
        file_opt = options = {}
        options['defaultextension'] = '.mp3','.wav'
        options['filetypes'] = [('all files', '.*'), ('sound files', '.mp3')]
        options['title'] = 'This is a title'
        filename = tkFileDialog.askopenfilename(**file_opt)
        entry.delete(0, END)
        entry.insert(0, filename)
        return filename

    #config page
    button_opt = {'padx': 5, 'pady': 5}
    root = Tk()
    root.title("Identificador de generos musicales")
    frame = Frame(root)
    frame.pack()
    label = Label(root, text="Para comenzar seleccione una pista:")
    label.pack()
    entry = Entry(root, width=50)
    entry.pack()
    separator = Frame(root, height=2, bd=1, relief=SUNKEN)
    separator.pack(fill=X, padx=5, pady=5)
    button_browse = Button(root,text="Browse", command=browse_callback)
    button_browse.pack(**button_opt)
    button_identify = Button(root, text="Identify", command=identify_callback)
    button_identify.pack(**button_opt)


    separator = Frame(root, height=2, bd=1, relief=SUNKEN)
    separator.pack(fill=X, padx=5, pady=5)



    mainloop()





if __name__ == "__main__":

    gui()