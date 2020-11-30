from gui import *
import os

def main():
    path = os.getcwd()
    os.chdir(path + "//sudoku_res//")
    gui = GUI("Sudoku recognizer", 1000, 600, False)
    
if __name__ == '__main__':
    main()