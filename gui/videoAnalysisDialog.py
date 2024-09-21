from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QTime


class VideoAnalysisDialog(object):

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(564, 520)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(parent=Dialog)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setFamily("Segoe UI")  # Or any other font you prefer
        Dialog.setFont(font)

        self.frame.setFont(font)
        self.frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame.setObjectName("frame")
        self.buttonBox = QtWidgets.QDialogButtonBox(parent=self.frame)
        self.buttonBox.setGeometry(QtCore.QRect(180, 440, 151, 25))
        self.buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.buttonBox.setEnabled(True)
        self.buttonBox.setObjectName("buttonBox")
        self.groupBoxFullFrames = QtWidgets.QGroupBox(parent=self.frame)
        self.groupBoxFullFrames.setEnabled(True)
        self.groupBoxFullFrames.setGeometry(QtCore.QRect(10, 10, 241, 141))
        self.groupBoxFullFrames.setMouseTracking(True)
        self.groupBoxFullFrames.setCheckable(True)
        self.groupBoxFullFrames.setChecked(False)
        self.groupBoxFullFrames.setObjectName("groupBoxFullFrames")
        self.label_fullframes_frame_skip_low = QtWidgets.QLabel(parent=self.groupBoxFullFrames)
        self.label_fullframes_frame_skip_low.setGeometry(QtCore.QRect(10, 60, 131, 41))
        self.label_fullframes_frame_skip_low.setObjectName("label_fullframes_frame_skip_low")
        self.label_fullframes_frame_skip_high = QtWidgets.QLabel(parent=self.groupBoxFullFrames)
        self.label_fullframes_frame_skip_high.setGeometry(QtCore.QRect(10, 100, 131, 31))
        self.label_fullframes_frame_skip_high.setObjectName("label_fullframes_frame_skip_high")
        self.lineEdit_frame_skip_high_value = QtWidgets.QLineEdit(parent=self.groupBoxFullFrames)
        self.lineEdit_frame_skip_high_value.setGeometry(QtCore.QRect(160, 70, 61, 22))
        self.lineEdit_frame_skip_high_value.setObjectName("lineEdit_frame_skip_high_value")
        self.lineEdit_frame_skip_low_value = QtWidgets.QLineEdit(parent=self.groupBoxFullFrames)
        self.lineEdit_frame_skip_low_value.setGeometry(QtCore.QRect(160, 100, 61, 22))
        self.lineEdit_frame_skip_low_value.setObjectName("lineEdit_frame_skip_low_value")
        self.groupBoxKeyFrames = QtWidgets.QGroupBox(parent=self.frame)
        self.groupBoxKeyFrames.setEnabled(True)
        self.groupBoxKeyFrames.setGeometry(QtCore.QRect(290, 10, 241, 141))
        font.setKerning(True)
        self.groupBoxKeyFrames.setFont(font)
        self.groupBoxKeyFrames.setMouseTracking(True)
        self.groupBoxKeyFrames.setCheckable(True)
        self.groupBoxKeyFrames.setChecked(True)
        self.groupBoxKeyFrames.setEnabled(True)
        self.groupBoxKeyFrames.setObjectName("groupBoxKeyFrames")

        # self.checkBox_keyframes_save_plots = QtWidgets.QCheckBox(parent=self.groupBoxKeyFrames)
        # self.checkBox_keyframes_save_plots.setEnabled(True)
        # self.checkBox_keyframes_save_plots.setGeometry(QtCore.QRect(10, 60, 131, 20))
        # self.checkBox_keyframes_save_plots.setObjectName("checkBox_keyframes_save_plots")

        self.label_threshhold_value = QtWidgets.QLabel(parent=self.groupBoxKeyFrames)
        self.label_threshhold_value.setEnabled(True)
        self.label_threshhold_value.setGeometry(QtCore.QRect(10, 100, 91, 16))
        self.label_threshhold_value.setObjectName("label_threshhold_value")
        self.lineEdit_threshold_value = QtWidgets.QLineEdit(parent=self.groupBoxKeyFrames)
        self.lineEdit_threshold_value.setEnabled(True)
        self.lineEdit_threshold_value.setGeometry(QtCore.QRect(130, 100, 61, 22))
        self.lineEdit_threshold_value.setObjectName("lineEdit_threshold_value")
        self.full_part_video = QtWidgets.QWidget(parent=self.frame)
        self.full_part_video.setGeometry(QtCore.QRect(10, 160, 541, 191))
        self.full_part_video.setMouseTracking(True)
        self.full_part_video.setObjectName("full_part_video")
        self.checkBox_full_video = QtWidgets.QCheckBox(parent=self.full_part_video)
        self.checkBox_full_video.setGeometry(QtCore.QRect(20, 70, 171, 20))
        self.checkBox_full_video.setChecked(True)
        self.checkBox_full_video.setTristate(False)
        self.checkBox_full_video.setObjectName("checkBox_full_video")
        self.checkBox_save_frames = QtWidgets.QCheckBox(parent=self.full_part_video)
        self.checkBox_save_frames.setGeometry(QtCore.QRect(20, 40, 131, 20))
        self.checkBox_save_frames.setObjectName("checkBox_save_frames")
        self.checkBox_define_analysis_duration = QtWidgets.QCheckBox(parent=self.full_part_video)
        self.checkBox_define_analysis_duration.setEnabled(False)
        self.checkBox_define_analysis_duration.setGeometry(QtCore.QRect(20, 120, 221, 20))
        self.checkBox_define_analysis_duration.setCheckable(True)
        self.checkBox_define_analysis_duration.setObjectName("checkBox_define_analysis_duration")
        self.checkBox_define_analysis_duration.setEnabled(True)
        self.label_analysis_duration_start_time = QtWidgets.QLabel(parent=self.full_part_video)
        self.label_analysis_duration_start_time.setGeometry(QtCore.QRect(290, 120, 40, 25))
        self.label_analysis_duration_start_time.setObjectName("label")
        self.label_analysis_duration_end_time = QtWidgets.QLabel(parent=self.full_part_video)
        self.label_analysis_duration_end_time.setGeometry(QtCore.QRect(290, 150, 40, 25))
        self.label_analysis_duration_end_time.setObjectName("label_analysis_duration_end_time")
        self.timeEdit_start = QtWidgets.QTimeEdit(parent=self.full_part_video)
        self.timeEdit_start.setEnabled(True) #
        self.timeEdit_start.setGeometry(QtCore.QRect(350, 120, 68, 25))
        self.timeEdit_start.setObjectName("timeEdit_start")
        self.timeEdit_start.setDisplayFormat("mm:ss")
        self.timeEdit_end = QtWidgets.QTimeEdit(parent=self.full_part_video)
        self.timeEdit_end.setEnabled(True) #
        self.timeEdit_end.setGeometry(QtCore.QRect(350, 150, 68, 25))
        self.timeEdit_end.setObjectName("timeEdit_end")
        self.timeEdit_end.setDisplayFormat("mm:ss")
        self.line = QtWidgets.QFrame(parent=self.full_part_video)
        self.line.setGeometry(QtCore.QRect(20, 100, 501, 16))
        self.line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line.setObjectName("line")
        self.checkBox_loadPostAnalysisVideo = QtWidgets.QCheckBox(parent=self.full_part_video)
        self.checkBox_loadPostAnalysisVideo.setEnabled(True)
        self.checkBox_loadPostAnalysisVideo.setGeometry(QtCore.QRect(20, 10, 281, 20))
        self.checkBox_loadPostAnalysisVideo.setObjectName("checkBox_loadPostAnalysisVideo")
        self.line_2 = QtWidgets.QFrame(parent=self.frame)
        self.line_2.setGeometry(QtCore.QRect(260, 0, 30, 151))
        self.line_2.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(lambda: self.validate_inputs(Dialog))  # type: ignore
        self.buttonBox.rejected.connect(Dialog.reject)  # type: ignore

        # Warning label for error messages
        self.label_for_warnings = QtWidgets.QLabel(parent=Dialog)
        self.label_for_warnings.setGeometry(QtCore.QRect(38, 350, 450, 100))
        self.label_for_warnings.setObjectName("label_for_warnings")
        self.label_for_warnings.setStyleSheet("background-color: transparent; color: red;")

        int_validator = QtGui.QIntValidator(0, 1000)
        self.lineEdit_frame_skip_high_value.setValidator(int_validator)
        self.lineEdit_frame_skip_low_value.setValidator(int_validator)

        double_validator = QtGui.QDoubleValidator(0.0, 1.0, 3)
        double_validator.setNotation(QtGui.QDoubleValidator.Notation.StandardNotation)
        double_validator.setDecimals(3)
        self.lineEdit_threshold_value.setValidator(double_validator)

        QtCore.QMetaObject.connectSlotsByName(Dialog)



    def show_warning(self, message, duration=30000):
        """
        Display a warning message on label_for_warnings.
        The message will disappear after the specified duration (in milliseconds).
        """
        self.label_for_warnings.setText(message)

        # Clear the warning message after the specified duration
        QtCore.QTimer.singleShot(duration, self.clear_warning)

    def clear_warning(self):
        """Clear the warning message."""
        self.label_for_warnings.setText("")
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Analysis settings"))
        self.groupBoxFullFrames.setTitle(_translate("Dialog", "Full Frames"))
        self.checkBox_save_frames.setText(_translate("Dialog", "Save frames"))
        self.label_fullframes_frame_skip_low.setText(_translate("Dialog", "Frame Skip Low:"))
        self.label_fullframes_frame_skip_high.setText(_translate("Dialog", "Frame Skip high:"))
        self.groupBoxKeyFrames.setTitle(_translate("Dialog", "Keyframes"))

        # self.checkBox_keyframes_save_plots.setText(_translate("Dialog", "Save plots"))

        self.label_threshhold_value.setText(_translate("Dialog", "Threshhold:"))
        self.checkBox_full_video.setText(_translate("Dialog", "Full video "))
        self.checkBox_define_analysis_duration.setText(_translate("Dialog", "Define analysis duration"))
        self.label_analysis_duration_start_time.setText(_translate("Dialog", "Start :"))
        self.label_analysis_duration_end_time.setText(_translate("Dialog", "End:"))
        self.checkBox_loadPostAnalysisVideo.setText(_translate("Dialog", "Load the post-analysis video"))

    def toggleKeyframes(self, checked):
        self.groupBoxFullFrames.setEnabled(not checked)

    def toggleFullFrames(self, checked):
        self.groupBoxKeyFrames.setEnabled(not checked)


    def validate_inputs(self, VDialog):
        if not self.groupBoxKeyFrames.isChecked() and not self.groupBoxFullFrames.isChecked():
            self.show_warning("Selecting either one of the key frames or full frames is required.")
            return
        if self.groupBoxKeyFrames.isChecked() and self.groupBoxFullFrames.isChecked():
            self.show_warning("Selecting only one of the key frames or full frames methods.")
            return

        if self.groupBoxFullFrames.isChecked():
            try:
                # Convert the text input to integers
                frame_skip_low_value = int(self.lineEdit_frame_skip_low_value.text())
                frame_skip_high_value = int(self.lineEdit_frame_skip_high_value.text())

                # Check if either value is zero or missing
                if frame_skip_low_value <= 0:
                    self.show_warning("Frame skip low value is missing or invalid. Please provide a valid value.")
                    return
                if frame_skip_high_value <= 0:
                    self.show_warning("Frame skip high value is missing or invalid. Please provide a valid value.")
                    return

            except ValueError:
                # If the input couldn't be converted to an integer, log an error
                self.show_warning("The full frame analysis method requires predefined parameters."
                                 "\nPlease provide valid numerical values for frame skip low and high values.")
                return
        elif self.groupBoxKeyFrames.isChecked():
                try:
                    # Convert the input to a floating-point number
                    threshold_value = float(self.lineEdit_threshold_value.text())

                    # Check if the threshold value is greater than 0
                    if threshold_value <= 0:
                        self.show_warning("The key frames analysis method requires a threshold value greater than 0.")
                        return

                except ValueError:
                    # If the input couldn't be converted to a float, log an error
                    self.show_warning("Please enter a valid numerical threshold value.")
                    return

        VDialog.accept()

    def set_default_params_state(self, checked):
        if checked:
            self.lineEdit_frame_skip_low_value.setText('10')
            self.lineEdit_frame_skip_high_value.setText('5')
            self.lineEdit_threshold_value.setText('0.35')
            self.checkBox_loadPostAnalysisVideo.setChecked(True)
            self.checkBox_save_frames.setChecked(True)
        else:
            self.lineEdit_frame_skip_low_value.setText('')
            self.lineEdit_frame_skip_high_value.setText('')
            self.lineEdit_threshold_value.setText('')



if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = VideoAnalysisDialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec())
