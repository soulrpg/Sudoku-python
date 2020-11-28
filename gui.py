import tkinter as tk
from tkinter import ttk
from tkinter import Canvas
from tkinter import messagebox

class GUI:
    def __init__(self, title, WIDTH, HEIGHT, RESIZABLE):
        # Tworzenie okna
        self.window = tk.Tk()
        self.window.title("Sudoku Recognizer")
        self.window.geometry(str(WIDTH) + "x" + str(HEIGHT))
        self.window.resizable(RESIZABLE, RESIZABLE)
        
        # Tworzenie elementow
        button1 = ttk.Button(text="Test", command=lambda: self.button_clicked())
        button1.pack()
        
        # Uruchamianie petli zdarzen
        self.window.mainloop()
        
    # Tutaj funkcje obslugujace zdarzenia
    def button_clicked(self):
        print("Button clicked!")
        