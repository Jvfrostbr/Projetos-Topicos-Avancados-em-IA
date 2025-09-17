from gui_pyqt5 import NeuralGUI
from PyQt5.QtWidgets import QApplication
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NeuralGUI()
    window.show()
    sys.exit(app.exec_())