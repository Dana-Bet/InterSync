import sys
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6 import uic


class SynchronyAnalysisDialog(QtWidgets.QDialog):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("synchronyAnalysisDialog.ui", self)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = SynchronyAnalysisDialog()
    window.show()
    app.exec()
