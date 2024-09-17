# Form implementation generated from reading ui file 'SettingsWindow.ui'
#
# Created by: PyQt6 UI code generator 6.7.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class SettingsWindow_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(406, 340)
        font = QtGui.QFont()
        font.setPointSize(10)
        Dialog.setFont(font)
        self.buttonBox = QtWidgets.QDialogButtonBox(parent=Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(70, 280, 211, 52))
        self.buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.groupBox = QtWidgets.QGroupBox(parent=Dialog)
        self.groupBox.setGeometry(QtCore.QRect(20, 80, 361, 81))
        self.groupBox.setObjectName("groupBox")
        self.checkBox_defaultParams = QtWidgets.QCheckBox(parent=self.groupBox)
        self.checkBox_defaultParams.setGeometry(QtCore.QRect(10, 30, 281, 41))
        self.checkBox_defaultParams.setObjectName("checkBox_defaultParams")
        self.checkBox_exportDirectory = QtWidgets.QCheckBox(parent=Dialog)
        self.checkBox_exportDirectory.setGeometry(QtCore.QRect(30, 29, 141, 31))
        self.checkBox_exportDirectory.setObjectName("checkBox_exportDirectory")
        self.lineEdit = QtWidgets.QLineEdit(parent=Dialog)
        self.lineEdit.setGeometry(QtCore.QRect(160, 31, 221, 31))
        self.lineEdit.setObjectName("lineEdit")
        self.groupBox_2 = QtWidgets.QGroupBox(parent=Dialog)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 170, 361, 81))
        self.groupBox_2.setObjectName("groupBox_2")
        self.checkBox_darkMode = QtWidgets.QCheckBox(parent=self.groupBox_2)
        self.checkBox_darkMode.setGeometry(QtCore.QRect(10, 40, 251, 31))
        self.checkBox_darkMode.setObjectName("checkBox_darkMode")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept) # type: ignore
        self.buttonBox.rejected.connect(Dialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Settings "))
        self.groupBox.setTitle(_translate("Dialog", "Analysis Parameters"))
        self.checkBox_defaultParams.setText(_translate("Dialog", "Use Default Analysis Parameters"))
        self.checkBox_exportDirectory.setText(_translate("Dialog", "Export Directory:"))
        self.lineEdit.setText(_translate("Dialog", "C:/InterSyncReports/"))
        self.groupBox_2.setTitle(_translate("Dialog", "Display settings"))
        self.checkBox_darkMode.setText(_translate("Dialog", "Dark mode"))
