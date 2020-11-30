import tkinter as tk
from tkinter import ttk
from tkinter import Canvas
from tkinter import messagebox
from sudoku_logic import *
import os
from PIL import Image, ImageTk
import cv2

class GUI:
    def __init__(self, title, WIDTH, HEIGHT, RESIZABLE):
        # Tworzenie okna
        self.window = tk.Tk()
        self.window.title("Sudoku Recognizer")
        self.window.geometry(str(WIDTH) + "x" + str(HEIGHT))
        self.window.resizable(RESIZABLE, RESIZABLE)
        
        # Specjalna zmienna, która przechowuje stan checkbuttona
        self.checkButton_value = tk.IntVar()
        
        self.result_value = tk.StringVar()
        
        self.result_value.set("Percent of digits recognized:")
        
        self.canvasWidth = WIDTH - 200
        self.canvasHeight = HEIGHT - 100
        self.pics = None
        
        # Tworzenie elementow
        self.topFrame = ttk.Frame(self.window)
        self.topFrame.pack()
        self.middleFrame = ttk.Frame(self.window)
        self.middleFrame.pack()
        self.bottomFrame = ttk.Frame(self.window)
        self.bottomFrame.pack()
        self.nameLabel = ttk.Label(self.topFrame, text="Image filename:")
        self.nameLabel.pack(side = tk.LEFT, pady=10)
        self.filenameEntry = ttk.Entry(self.topFrame, width = 30)
        self.filenameEntry.pack(side = tk.LEFT)
        self.startButton = ttk.Button(self.topFrame, text="Start", command=lambda: self.start_clicked())
        self.startButton.pack(side = tk.RIGHT, padx=50)
        self.imgCanvas = Canvas(self.middleFrame, width=self.canvasWidth, height=self.canvasHeight)#, bg="black")
        self.imgCanvas.pack(side = tk.LEFT, padx=30)
        self.checkButton = ttk.Checkbutton(self.middleFrame, text="Show final only", offvalue = 0, onvalue = 1, command=lambda: self.                              checkButton_change(), variable = self.checkButton_value)
        self.checkButton.pack(side = tk.RIGHT)
        self.resultLabel = ttk.Label(self.bottomFrame, textvariable = self.result_value)
        self.resultLabel.pack(side = tk.LEFT, pady=10)
        
        self.newPics = [None, None, None, None]
        
        # Uruchamianie petli zdarzen
        self.window.mainloop()
        
    # Tutaj funkcje obslugujace zdarzenia
    def start_clicked(self):
        filenames = os.listdir()
        if self.filenameEntry.get() in filenames:
            manager = AlgoManager(self.filenameEntry.get())
            self.pics = manager.getPictures()
            result = manager.getResult() * 100
            self.result_value.set(str("Percent of digits recognized: %i" % (result)) + "%")
            if self.checkButton_value.get() == 0:
                self.redrawCanvasFull()
            else:
                self.redrawCanvasFinal()
        else:
            messagebox.showinfo("Błąd", "Brak pliku o podanej nazwie")
        self.window.update_idletasks()
        self.window.update()
        
    def checkButton_change(self):
        if self.pics != None:
            if self.checkButton_value.get() == 0:
                self.redrawCanvasFull()
            else:
                self.redrawCanvasFinal()
        
    def redrawCanvasFull(self):
        dimensions = (self.canvasWidth // 2, self.canvasHeight // 2)
    
    
        pic = self.pics[0]
        resized = cv2.resize(pic, dimensions, interpolation = cv2.INTER_AREA)
        b, g, r = cv2.split(resized)
        img  = cv2.merge((r, g, b))
        im = Image.fromarray(img)
        self.newPics[0] = ImageTk.PhotoImage(image = im)
        self.imgCanvas.create_image(0, 0, anchor=tk.NW, image=self.newPics[0])
        
        
        pic = self.pics[1]
        resized = cv2.resize(pic, dimensions, interpolation = cv2.INTER_AREA)
        b, g, r = cv2.split(resized)
        img  = cv2.merge((r, g, b))
        im = Image.fromarray(img)
        self.newPics[1] = ImageTk.PhotoImage(image = im)
        self.imgCanvas.create_image(dimensions[0], 0, anchor=tk.NW, image=self.newPics[1])
        
        pic = self.pics[2]
        resized = cv2.resize(pic, dimensions, interpolation = cv2.INTER_AREA)
        b, g, r = cv2.split(resized)
        img  = cv2.merge((r, g, b))
        im = Image.fromarray(img)
        self.newPics[2] = ImageTk.PhotoImage(image = im)
        self.imgCanvas.create_image(0, dimensions[1], anchor=tk.NW, image=self.newPics[2])
        
        pic = self.pics[3]
        resized = cv2.resize(pic, dimensions, interpolation = cv2.INTER_AREA)
        b, g, r = cv2.split(resized)
        img  = cv2.merge((r, g, b))
        im = Image.fromarray(img)
        self.newPics[3] = ImageTk.PhotoImage(image = im)
        self.imgCanvas.create_image(dimensions[0], dimensions[1], anchor=tk.NW, image=self.newPics[3])
        
        self.imgCanvas.update()
        
    def redrawCanvasFinal(self):
        dimensions = (self.canvasWidth, self.canvasHeight)
        pic = self.pics[3]
        resized = cv2.resize(pic, dimensions, interpolation = cv2.INTER_AREA)
        b, g, r = cv2.split(resized)
        img  = cv2.merge((r, g, b))
        im = Image.fromarray(img)
        self.newPics[3] = ImageTk.PhotoImage(image = im)
        self.imgCanvas.create_image(0, 0, anchor=tk.NW, image=self.newPics[3])
        
        self.imgCanvas.update()