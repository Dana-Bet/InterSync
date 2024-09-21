import warnings

import analysis

import os

from PyQt6 import QtWidgets
from PyQt6.QtCore import QTime
from PyQt6.QtWidgets import QMainWindow, QMenuBar, QVBoxLayout, QWidget, QFileDialog, QPushButton, QHBoxLayout, \
    QApplication, QSpacerItem, QSizePolicy, QLabel, QDialog, QTabWidget, QStyleFactory, QPlainTextEdit
from PyQt6.QtGui import QIcon, QAction, QPixmap
import sys

from datetime import datetime
import mediapipe as mp
import cv2

import data.vectorframe
import frames
import reports
from utility import utility
from data import Target, limbs_dict, excel_export
from detector import Detector, target_detector

from gui.VideoPlayer import VideoPlayer
from gui.videoAnalysisDialog import VideoAnalysisDialog
from gui.synchronyAnalysisDialog import SynchronyAnalysisDialog
from gui.GenerateReportsDialog import GenerateReportsDialog
from gui.SettingsWindow import SettingsWindow_Dialog

# Suppress the specific UserWarning from google.protobuf.symbol_database
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')


def convert_cv2_to_mp_image(cv2_image):
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    # Create a MediaPipe Image object from the RGB image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    return mp_image


class ThemedDialog(QDialog):
    def __init__(self, darkMode):
        super().__init__()
        self.applyStyle(darkMode)

    def applyStyle(self, darkMode):
        if darkMode:
            self.applyDarkMode()
        else:
            self.applyLightMode()

    def applyDarkMode(self):
        dark_stylesheet = """
        QDialog {
            background-color: #1e1e1e;
            color: #c7c7c7;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        QGroupBox {
            border: 1px solid #444444;
            margin-top: 10px;
            background-color: #2c2c2c;
            color: #c7c7c7;
            border-radius: 8px;
            padding: 10px;
            font-size: 12px; /* Consistent font size */
        }

        QLineEdit {

            background-color: #A9A9A9;
            color: #dcdcdc;
            border: 1px solid #555555;
            padding: 5px;
            border-radius: 4px;
            font-size: 12px;  /* Consistent font size */
        }
        QLineEdit:focus {
            background-color: #000000;
            border: 1px solid #86a3c3;
        }
        QCheckBox {
            background-color: #2c2c2c;
            color: #dcdcdc;
            font-size: 12px;  /* Consistent font size */
        }
        QPushButton {
            background-color: #3c3f41;
            color: #dcdcdc;
            border: 1px solid #555555;
            border-radius: 4px;
            padding: 6px 12px;
            font-size: 12px;  /* Consistent font size */
        }
        QPushButton:hover {
            background-color: #2c2c2c;
        }
        QPushButton:pressed {
            background-color: #2c2c2c;
        }
        QDialogButtonBox {
            background-color: #2c2c2c;
        }
        QLabel {
            background-color: transparent;
            color: #dcdcdc;
            font-size: 14px;  /* Consistent font size */
        }
        QTimeEdit {
            background-color: #3b3b3b;
            color: #dcdcdc;
            border: 1px solid #555555;
            padding: 5px;
            border-radius: 4px;
            font-size: 12px;  /* Consistent font size */
        }
        """
        self.setStyleSheet(dark_stylesheet)

    def applyLightMode(self):
        light_stylesheet = """
        QDialog, QWidget {
            background-color: #ffffff;
            color: #000000;
            font-size: 12px;  /* Set font size */
        }

        QMenuBar, QMenuBar::item {
            background-color: #f0f0f0;
            color: #000000;
            font-size: 12px;
        }

        QMenuBar::item:selected {
            background-color: #cccccc;
        }

        QPushButton {
            background-color: #e0e0e0;
            color: #000000;
            border: 1px solid #cccccc;
            padding: 15px;  /* Adjust padding for larger buttons */
            font-size: 12px;
        }

        QPushButton:hover {
            background-color: #cccccc;
        }


        QTabBar::tab:!selected {
            background-color: #e0e0e0;
            color: #000000;
            font-size: 12px;
        }

        QTabBar::tab:hover {
            background-color: #cccccc;
        }

        QTabWidget {
            background-color: #ffffff;
            color: #000000;
            font-size: 12px;
        }

        QTabWidget::tab-bar {
            alignment: left;
        }
        """
        self.setStyleSheet(light_stylesheet)


class VideoAnalysisWindow(ThemedDialog):
    def __init__(self, darkMode):
        super().__init__(darkMode)  # Pass darkMode to the base class
        # Create an instance of the generated UI class
        self.ui = VideoAnalysisDialog()
        # Set up the UI on this dialog instance (self)
        self.ui.setupUi(self)



class GenerateReportsWindow(ThemedDialog):
    def __init__(self, darkMode):
        super().__init__(darkMode)  # Pass darkMode to the base class
        self.ui = GenerateReportsDialog()  # Create an instance of the Synchrony Analysis UI class
        self.ui.setupUi(self)  # Set up the UI


class SynchronyAnalysisWindow(ThemedDialog):
    def __init__(self, darkMode):
        super().__init__(darkMode)  # Pass darkMode to the base class
        self.ui = SynchronyAnalysisDialog()  # Create an instance of the Synchrony Analysis UI class
        self.ui.setupUi(self)  # Set up the UI


class SettingsWindow(ThemedDialog):
    def __init__(self, darkMode):
        super().__init__(darkMode)  # Pass darkMode to the base class
        self.ui = SettingsWindow_Dialog()  # Create an instance of the Synchrony Analysis UI class
        self.ui.setupUi(self)  # Set up the UI


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # variables

        self.loaded_video_name = ''
        self.analysed_video_name = ''

        # self.destination_path = f'./images/test/gui_test'
        self.destination_path = f'./'
        self.detected_frames = None
        self.analysed_frames = None

        self.similarity_analysis_summary_df = None
        self.dtw_analysis_summary_df = None
        self.smith_waterman_analysis_summary_df = None
        self.tlcc_analysis_summary_unified_df = None
        self.tlcc_analysis_summary_separate_df = None

        self.synchronyWindow = None
        self.analysisWindow = None
        self.last_analysis_dir_path = None
        self.last_date_time_string = None

        self.last_analysis_keyframes_flag = False
        self.last_analysis_keyframes_threshold = 0

        self.enable_synchrony_analysis = False

        self.setWindowTitle("InterSync Application")
        self.setWindowIcon(QIcon.fromTheme("multimedia-video-player"))
        self.setGeometry(100, 100, 800, 600)

        # Central widget and main layout
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.mainLayout = QHBoxLayout(self.centralWidget)  # Horizontal layout to place the buttons and the tab widget

        # Create a vertical layout for the buttons on the left side
        self.createButtonLayout()

        # Create a QTabWidget to hold video player tabs
        self.tabWidget = QTabWidget()
        self.firstTab = None
        self.secondTab = None
        self.createFirstTab()

        # Add the vertical button layout to the left side and the tab widget to the right
        self.mainLayout.addLayout(self.leftButtonLayout)  # Buttons on the left
        self.mainLayout.addWidget(self.tabWidget)  # Tabs with video players on the right

        # Create a menu bar
        self.menuBar = QMenuBar()
        self.setMenuBar(self.menuBar)

        # Add "File" menu
        fileMenu = self.menuBar.addMenu("File")

        # Add "Open" action
        openAction = QAction(QIcon.fromTheme("document-open"), "Open", self)
        openAction.triggered.connect(self.openFile)
        fileMenu.addAction(openAction)

        # Add "Exit" action
        exitAction = QAction(QIcon.fromTheme("application-exit"), "Exit", self)
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)

        # Add "Reports" menu
        reportsMenu = self.menuBar.addMenu("Reports")

        # Add "Keyframes plot" action
        keyframesAction = QAction("Keyframes plot", self)
        keyframesAction.triggered.connect(self.showKeyframesPlot)
        reportsMenu.addAction(keyframesAction)

        # Add "Settings" menu
        settingsMenu = self.menuBar.addMenu("Settings")

        # Create "Open Settings" action
        openSettingsAction = QAction("Open Settings", self)
        openSettingsAction.triggered.connect(self.openSettings)
        settingsMenu.addAction(openSettingsAction)


        self.defaultParams = True
        self.darkMode = True

        self.applyDarkMode()

        # Data Structures and internal variables:
        self.__detector_instance = Detector()
        self.__target1 = Target()
        self.__target2 = Target()

    def createButtonLayout(self):
        self.leftButtonLayout = QVBoxLayout()

        # Create additional buttons
        self.button1 = QPushButton("Video Analysis")
        self.button2 = QPushButton("Synchrony Analysis")
        self.button3 = QPushButton("Generate Report")

        self.leftButtonLayout.addWidget(self.button1)
        self.leftButtonLayout.addWidget(self.button2)
        self.leftButtonLayout.addWidget(self.button3)

        # Add a spacer above to align buttons to the top
        spacerItemBefore = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.leftButtonLayout.addItem(spacerItemBefore)

        # Connect buttons to their respective methods
        self.button1.clicked.connect(self.openVideoAnalysisWindow)
        self.button2.clicked.connect(self.openSynchronyAnalysisWindow)
        self.button3.clicked.connect(self.openGenerateReportsWindow)

    def createFirstTab(self):
        # First tab for first video player
        self.firstTab = QWidget()
        self.firstTabLayout = QVBoxLayout(self.firstTab)  # Use vertical layout for video player and controls
        self.videoPlayer = VideoPlayer('')  # Create first VideoPlayer instance
        self.firstTabLayout.addWidget(self.videoPlayer)
        self.tabWidget.addTab(self.firstTab, "Original Video")
        # Add a QTextEdit as a message terminal (log window) below the video player
        self.message_terminal = QPlainTextEdit()
        self.message_terminal.setReadOnly(True)  # Set to read-only to simulate a terminal
        self.message_terminal.setFixedHeight(120)  # Set height for the terminal
        self.firstTabLayout.addWidget(self.message_terminal)  # Add the terminal below the video player
        # Style the terminal with a black background and white text
        self.message_terminal.setStyleSheet("""
        QPlainTextEdit {
            background-color: black;
            color: white;
            border: none;
        }
        """)

        self.firstTabLayout.addWidget(self.message_terminal)  # Add the terminal below the video player

        # Add the tab to the tab widget
        self.tabWidget.addTab(self.firstTab, "Original Video")

    def createSecondTab(self):
        # Second tab for second video player
        self.secondTab = QWidget()
        self.secondTabLayout = QVBoxLayout(self.secondTab)
        self.videoPlayer2 = VideoPlayer('')  # Create second VideoPlayer instance
        self.secondTabLayout.addWidget(self.videoPlayer2)
        self.tabWidget.addTab(self.secondTab, "Analysed Video")

    def clear_terminal(self):
        """Clear all messages from the terminal."""
        self.message_terminal.clear()

    def log_message(self, message, message_type="log"):
        """
        Log messages to the terminal with different colors based on type.
        message_type can be 'log', 'warning', or 'error'.
        """
        if message_type == "error":
            # Red color for errors
            self.clear_terminal()
            formatted_message = f'<span style="color:red;">[ERROR] {message}</span>'
        elif message_type == "warning":
            # White color for warnings
            formatted_message = f'<span style="color:yellow;">[WARNING] {message}</span>'
        elif message_type == "notification":
            # Green color for notifications
            # self.clear_terminal()
            formatted_message = f'<span style="color:green;">[NOTIFICATION] {message}</span>'
        else:
            # Default log message color (white or any other color you want)
            formatted_message = f'<span style="color:white;">[LOG] {message}</span>'

        # Use insertHtml to append the colored message
        self.message_terminal.appendHtml(formatted_message)

        # Ensure that the message terminal updates in real time
        QApplication.processEvents()

    def applyDarkMode(self):
        QtWidgets.QApplication.setStyle(QStyleFactory.create('Windows'))
        dark_stylesheet = """
        QMainWindow {
            background-color: #2e2e2e;
            font-size: 12px;  /* Set font size */
        }
        QWidget {
            background-color: #2e2e2e;
            color: #f0f0f0;
            font-size: 12px;  /* Set font size */
        }
        QMenuBar {
            background-color: #333333;
            color: #f0f0f0;
            font-size: 12px;  /* Set font size */
        }
        QMenuBar::item {
            background: #333333;
            color: #f0f0f0;
            font-size: 12px;  /* Set font size */
        }
        QMenuBar::item:selected {
            background: #555555;
        }
        QPushButton {
            background-color: #444444;
            color: #f0f0f0;
            border: 1px solid #555555;
            padding: 15px;  /* Adjust padding for larger buttons */
            font-size: 12px;  /* Set font size */
        }
        QPushButton:hover {
            background-color: #555555;
        }
        QSlider::groove:horizontal {
            background: #333333;
            border: 1px solid #444444;
        }
        QSlider::handle:horizontal {
            background: #555555;
            border: 1px solid #666666;
        }
        QLineEdit {
            background-color: #333333;
            color: #f0f0f0;
            border: 1px solid #444444;
            font-size: 12px;  /* Set font size */
        }
        /* Style for QTabWidget and QTabBar */
        QTabWidget::pane {
            border: 1px solid #444444;
            background-color: #2e2e2e;
        }
        QTabBar::tab {
            background-color: #444444;
            color: #f0f0f0;
            padding: 10px;
            border: 1px solid #555555;
            min-width: 80px;  /* Ensure tabs are wide enough */
            font-size: 12px;  /* Set font size */
        }
        QTabBar::tab:selected {
            background-color: #555555;
            color: #f0f0f0;
            border-bottom: 2px solid #ff6600;  /* Orange border for the selected tab */
        }
        QTabBar::tab:!selected {
            background-color: #444444;
            color: #f0f0f0;
            font-size: 12px;  /* Set font size */
        }
        QTabBar::tab:hover {
            background-color: #666666;
        }
        QTabBar::tab:focus {
            outline: none;
        }
        /* Force styling of the QTabWidget labels */
        QTabWidget {
            background-color: #2e2e2e;  /* Set a dark background for the entire tab widget */
            color: #f0f0f0;  /* Ensure all text is light */
            font-size: 12px;  /* Set font size */
        }
        QTabWidget::tab-bar {
            alignment: left
        }
        """
        QtWidgets.QApplication.instance().setStyleSheet(dark_stylesheet)

    def applyLightMode(self):
        QtWidgets.QApplication.setStyle(QStyleFactory.create('Windows'))
        light_stylesheet = """
        QMainWindow {
            background-color: #ffffff;
            font-size: 12px;  /* Set font size */
        }
        QWidget {
            background-color: #ffffff;
            color: #000000;
            font-size: 12px;  /* Set font size */
        }
        QMenuBar {
            background-color: #f0f0f0;
            color: #000000;
            font-size: 12px;  /* Set font size */
        }
        QMenuBar::item {
            background: #f0f0f0;
            color: #000000;
            font-size: 12px;  /* Set font size */
        }
        QMenuBar::item:selected {
            background: #cccccc;
        }
        QPushButton {
            background-color: #e0e0e0;
            color: #000000;
            border: 1px solid #cccccc;
            padding: 15px;  /* Adjust padding for larger buttons */
            font-size: 12px;  /* Set font size */
        }
        QPushButton:hover {
            background-color: #cccccc;
        }
        QSlider::groove:horizontal {
            background: #f0f0f0;
            border: 1px solid #cccccc;
        }
        QSlider::handle:horizontal {
            background: #cccccc;
            border: 1px solid #b0b0b0;
        }
        QLineEdit {
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #cccccc;
            font-size: 12px;  /* Set font size */
        }
        QTabWidget::pane {
            border: 1px solid #cccccc;
            background-color: #ffffff;
        }
        QTabBar::tab {
            background-color: #e0e0e0;
            color: #000000;
            padding: 10px;
            border: 1px solid #cccccc;
            min-width: 80px;  /* Ensure tabs are wide enough */
            font-size: 12px;  /* Set font size */
        }
        QTabBar::tab:selected {
            background-color: #cccccc;
            color: #000000;
            border-bottom: 2px solid #ff6600;  /* Orange border for the selected tab */
        }
        QTabBar::tab:!selected {
            background-color: #e0e0e0;
            color: #000000;
            font-size: 12px;  /* Set font size */
        }
        QTabBar::tab:hover {
            background-color: #cccccc;
        }
        QTabWidget {
            background-color: #ffffff;
            color: #000000;
            font-size: 12px;  
        }
        QTabWidget::tab-bar {
            alignment: left;
        }
        """
        QtWidgets.QApplication.instance().setStyleSheet(light_stylesheet)

    def openFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie")
        if fileName:
            self.videoPlayer.loadFilm(fileName)
        video_file_path = self.videoPlayer.get_loaded_file_path()
        if video_file_path:
            self.loaded_video_name = os.path.basename(video_file_path)
            self.log_message('The video has been loaded', "notification")
            # reset the data state
            self.enable_synchrony_analysis = False
            self.__target1 = Target()
            self.__target2 = Target()
            data.vectorframe.reset_global_vectorframe_id()
            return

    # def defineDefaultParams(self):
    #     pass

    def openSettings(self):
        self.SettingsWindow = SettingsWindow(self.darkMode)
        self.SettingsWindow.ui.lineEdit.setText(self.destination_path)
        self.SettingsWindow.ui.checkBox_defaultParams.setChecked(self.defaultParams)
        self.SettingsWindow.ui.checkBox_darkMode.setChecked(self.darkMode)
        if self.SettingsWindow.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            if self.SettingsWindow.ui.checkBox_exportDirectory.isChecked():
                self.destination_path = self.SettingsWindow.ui.lineEdit.text()
            if self.SettingsWindow.ui.checkBox_defaultParams.isChecked():
                self.defaultParams = True
                # self.defineDefaultParams()
            else:
                self.defaultParams = False
                # self.defineDefaultParams()
            self.darkMode = True if self.SettingsWindow.ui.checkBox_darkMode.isChecked() else False
            if self.darkMode:
                self.applyDarkMode()
            else:
                self.applyLightMode()
                self.darkMode = False
        # print(self.darkMode)

    def showKeyframesPlot(self):
        if not self.last_analysis_keyframes_flag:
            return
        if self.last_analysis_dir_path is None:
            return
        image_name = 'frame_pixel_difference.png'
        image_path = os.path.join(self.last_analysis_dir_path, image_name)
        if not os.path.exists(image_path):
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Keyframes Plot")
        layout = QVBoxLayout(dialog)
        label = QLabel(dialog)
        pixmap = QPixmap(image_path)
        label.setPixmap(pixmap)
        layout.addWidget(label)
        dialog.setLayout(layout)
        dialog.resize(pixmap.width(), pixmap.height())
        dialog.show()

    def openVideoAnalysisWindow(self):
        if self.videoPlayer is None:
            self.log_message('No video player initialized', "error")
            return

        video_file_path = self.videoPlayer.get_loaded_file_path()  # Make sure this method exists
        if not video_file_path:
            self.log_message('No video is loaded', "error")
            return

        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
        self.last_date_time_string = date_time_str
        dest_path = os.path.join(self.destination_path, date_time_str)
        utility.ensure_folder_exists(dest_path)
        self.analysisWindow = VideoAnalysisWindow(self.darkMode)
        self.analysisWindow.ui.timeEdit_end.setTime(
            QTime(0, 0, 0).addMSecs(self.videoPlayer.get_loaded_video_duration()))

        if self.defaultParams:
            self.analysisWindow.ui.set_default_params_state(True)
        else:
            self.analysisWindow.ui.set_default_params_state(False)

        if self.analysisWindow.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            save_images_flag = self.analysisWindow.ui.checkBox_save_frames.isChecked()
            analysis_start_time = None
            analysis_end_time = None
            if self.analysisWindow.ui.checkBox_define_analysis_duration.isChecked():
                analysis_start_time = self.analysisWindow.ui.timeEdit_start.time().msecsSinceStartOfDay()
                analysis_end_time = self.analysisWindow.ui.timeEdit_end.time().msecsSinceStartOfDay()
            if self.analysisWindow.ui.groupBoxFullFrames.isChecked():
                try:
                    # Convert the text input to integers
                    frame_skip_low_value = int(self.analysisWindow.ui.lineEdit_frame_skip_low_value.text())
                    frame_skip_high_value = int(self.analysisWindow.ui.lineEdit_frame_skip_high_value.text())

                    # Check if either value is zero or missing
                    if frame_skip_low_value <= 0:
                        self.log_message("Frame skip low value is missing or invalid. Please provide a valid value.",
                                         "warning")
                        return
                    if frame_skip_high_value <= 0:
                        self.log_message("Frame skip high value is missing or invalid. Please provide a valid value.",
                                         "warning")
                        return

                    self.log_message(
                        f"Full frame analysis will proceed with low: {frame_skip_low_value} and high: {frame_skip_high_value} frame skip values.",
                        "notification")

                except ValueError:
                    # If the input couldn't be converted to an integer, log an error
                    self.log_message("The full frame analysis method requires predefined parameters."
                                     "\nPlease provide valid numerical values for frame skip low and high values.",
                                     "error")
                    return
                self.detected_frames = frames.frame_detection(video_source=video_file_path,
                                                              destination=dest_path,
                                                              save_images=save_images_flag, verbose=False,
                                                              frame_skip_low=frame_skip_low_value,
                                                              frame_skip_high=frame_skip_high_value,
                                                              start_time_ms=analysis_start_time,
                                                              end_time_ms=analysis_end_time)
                self.last_analysis_keyframes_flag = False
                self.last_analysis_keyframes_threshold = 0

            elif self.analysisWindow.ui.groupBoxKeyFrames.isChecked():
                try:
                    # Convert the input to a floating-point number
                    threshold_value = float(self.analysisWindow.ui.lineEdit_threshold_value.text())

                    # Check if the threshold value is greater than 0
                    if threshold_value <= 0:
                        self.log_message("The key frames analysis method requires a threshold value greater than 0.",
                                         "error")
                        return

                except ValueError:
                    # If the input couldn't be converted to a float, log an error
                    self.log_message("Please enter a valid numerical threshold value.", "error")
                    return

                self.last_analysis_keyframes_threshold = threshold_value

                self.detected_frames = frames.keyframe_detection(detection_threshold=threshold_value,
                                                                 video_source=video_file_path,
                                                                 destination=dest_path,
                                                                 save_images=save_images_flag, verbose=False,
                                                                 save_plots=True,
                                                                 start_time_ms=analysis_start_time,
                                                                 end_time_ms=analysis_end_time)
                self.last_analysis_keyframes_flag = True
            else:
                print("???")
        else:
            utility.remove_folder(dest_path)
            return
        self.video_analysis_run_from_gui(self.detected_frames)
        if self.analysisWindow.ui.checkBox_loadPostAnalysisVideo.isChecked():
            analysis_preview_video_path = os.path.join(dest_path, 'analysis_preview_video.mp4')
            self.create_analysis_video(analysis_preview_video_path)
            if not os.path.exists(analysis_preview_video_path):
                print("secondTab: no analysis video found to load")
                return
            if self.secondTab is None:
                self.createSecondTab()
            self.videoPlayer2.loadFilm(analysis_preview_video_path)
        # path for the last folder analysed
        self.last_analysis_dir_path = dest_path
        self.analysed_video_name = self.loaded_video_name

    def openSynchronyAnalysisWindow(self):
        # This method will open the synchrony analysis dialog	        # This method will open the synchrony analysis dialog
        if not self.enable_synchrony_analysis:
            self.log_message(f'To perform synchronization analysis, video analysis must be performed first', "warning")
            return
        self.synchronyWindow = SynchronyAnalysisWindow(self.darkMode)

        if self.defaultParams:
            self.synchronyWindow.ui.set_default_params_state(True)
        else:
            self.synchronyWindow.ui.set_default_params_state(False)

        if self.synchronyWindow.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            # build the analysis body parts list:
            analysis_body_parts = []
            if self.synchronyWindow.ui.checkBox_FullBodyAnalysis.isChecked():
                analysis_body_parts.extend(limbs_dict.limbs_connect_dict_arr_index_description.values())
            else:
                if not self.synchronyWindow.ui.groupBox_BodyPartAnalysis.isChecked():
                    return
                else:
                    if self.synchronyWindow.ui.cb_torso_top.isChecked():
                        analysis_body_parts.append(
                            limbs_dict.limbs_connect_dict_arr_index_description[limbs_dict.TORSO_TOP])
                    if self.synchronyWindow.ui.cb_torso_l.isChecked():
                        analysis_body_parts.append(
                            limbs_dict.limbs_connect_dict_arr_index_description[limbs_dict.TORSO_LEFT])
                    if self.synchronyWindow.ui.cb_torso_r.isChecked():
                        analysis_body_parts.append(
                            limbs_dict.limbs_connect_dict_arr_index_description[limbs_dict.TORSO_RIGHT])
                    if self.synchronyWindow.ui.cb_torso_bottom.isChecked():
                        analysis_body_parts.append(
                            limbs_dict.limbs_connect_dict_arr_index_description[limbs_dict.TORSO_BOTTOM])
                    if self.synchronyWindow.ui.cb_arm_l.isChecked():
                        analysis_body_parts.append(
                            limbs_dict.limbs_connect_dict_arr_index_description[limbs_dict.ARM_LEFT])
                    if self.synchronyWindow.ui.cb_arm_r.isChecked():
                        analysis_body_parts.append(
                            limbs_dict.limbs_connect_dict_arr_index_description[limbs_dict.ARM_RIGHT])
                    if self.synchronyWindow.ui.cb_forearm_l.isChecked():
                        analysis_body_parts.append(
                            limbs_dict.limbs_connect_dict_arr_index_description[limbs_dict.FOREARM_LEFT])
                    if self.synchronyWindow.ui.cb_forearm_r.isChecked():
                        analysis_body_parts.append(
                            limbs_dict.limbs_connect_dict_arr_index_description[limbs_dict.FOREARM_RIGHT])
                    if self.synchronyWindow.ui.cb_upper_leg_l.isChecked():
                        analysis_body_parts.append(
                            limbs_dict.limbs_connect_dict_arr_index_description[limbs_dict.UPPER_LEG_LEFT])
                    if self.synchronyWindow.ui.cb_upper_leg_r.isChecked():
                        analysis_body_parts.append(
                            limbs_dict.limbs_connect_dict_arr_index_description[limbs_dict.UPPER_LEG_RIGHT])
                    if self.synchronyWindow.ui.cb_lower_leg_l.isChecked():
                        analysis_body_parts.append(
                            limbs_dict.limbs_connect_dict_arr_index_description[limbs_dict.LOWER_LEG_LEFT])
                    if self.synchronyWindow.ui.cb_lower_leg_r.isChecked():
                        analysis_body_parts.append(
                            limbs_dict.limbs_connect_dict_arr_index_description[limbs_dict.LOWER_LEG_RIGHT])

            if self.synchronyWindow.ui.groupBox_movement_vectors_sim.isChecked():
                self.log_message('Starting Synchrony analysis')
                analysis_similarity_metric = analysis.METRIC_EUCLIDEAN if self.synchronyWindow.ui.checkBox_mov_vectors_distance_2.currentText() == 'Euclidean distance' else analysis.METRIC_COSINE
                analysis_similarity_threshold = float(self.synchronyWindow.ui.doubleSpinBox_mov_vec_threshold.text())
                analysis_similarity_normalize = self.synchronyWindow.ui.checkBox_mov_vec_normalize.isChecked()

                image_save_path = os.path.join(self.last_analysis_dir_path, 'similarity')
                utility.ensure_folder_exists(image_save_path)

                for body_part in analysis_body_parts:
                    similar_movements = analysis.similarity_analyze_similar_movements(
                        df_target1=self.__target1.get_vector_data_as_dataframe(),
                        df_target2=self.__target2.get_vector_data_as_dataframe(),
                        body_part=body_part,
                        threshold=analysis_similarity_threshold,
                        normalize=analysis_similarity_normalize,
                        metric=analysis_similarity_metric)
                    sim_an_res = analysis.dataprep.dataprep_summarize_similarity_analysis_short(
                        results=similar_movements,
                        body_part=body_part,
                        threshold=analysis_similarity_threshold,
                        normalize=analysis_similarity_normalize,
                        metric=analysis_similarity_metric)
                    self.similarity_analysis_summary_df = analysis.dataprep.dataprep_append_summaries(
                        self.similarity_analysis_summary_df,
                        [sim_an_res])

                    analysis.similarity_visualize_similarities(similar_movements,
                                                               title=f'Movement vector similarity {body_part}, thr:{analysis_similarity_threshold:.4f} \n metric: {analysis_similarity_metric}',
                                                               save_visualization=True,
                                                               path_to_save=image_save_path,
                                                               prefix=f'{int(analysis_similarity_threshold * 1000)}_{body_part}')
                self.log_message('Synchrony analysis ended')

            if self.synchronyWindow.ui.groupBox_dtw_method.isChecked():
                self.log_message('Starting DTW analysis')
                analysis_dtw_metric = analysis.METRIC_EUCLIDEAN if self.synchronyWindow.ui.comboBox_dtw_distance.currentText() == 'Euclidean distance' else analysis.METRIC_COSINE
                # analysis_dtw_threshold = float(self.synchronyWindow.ui.doubleSpinBox_dtw_threshold.text())
                analysis_dtw_radius = int(self.synchronyWindow.ui.spinBox_dtw_radius.text())
                analysis_dtw_smoothing = self.synchronyWindow.ui.checkBox_dtw_vec_smoothing.isChecked()
                analysis_dtw_smoothing_window_size = int(self.synchronyWindow.ui.spinBox_dtw_window_size.text())
                analysis_dtw_normalize = self.synchronyWindow.ui.checkBox_dtw_normalize_vectors.isChecked()

                image_save_path = os.path.join(self.last_analysis_dir_path, 'dtw')
                utility.ensure_folder_exists(image_save_path)

                for body_part in analysis_body_parts:
                    path, distance, target1_data, target2_data = analysis.dtw_analyze_fastdtw(
                        df_target1=self.__target1.get_vector_data_as_dataframe(),
                        df_target2=self.__target2.get_vector_data_as_dataframe(),
                        body_part=body_part,
                        normalize=analysis_dtw_normalize,
                        smooth=analysis_dtw_smoothing,
                        smooth_window_size=analysis_dtw_smoothing_window_size,
                        radius=analysis_dtw_radius,
                        metric=analysis_dtw_metric)

                    analysis.dtw_visualize_alignment_separate_axis(target1_data=target1_data, target2_data=target2_data,
                                                                   path=path,
                                                                   distance=distance, body_part=body_part,
                                                                   save_visualization=True,
                                                                   smooth=analysis_dtw_smoothing,
                                                                   title=f'DTW Path Visualization {body_part}, metric {analysis_dtw_metric}',
                                                                   path_to_save=image_save_path,
                                                                   prefix=body_part)

                    analysis.dtw_visualize_alignment_unified(target1_data=target1_data, target2_data=target2_data,
                                                             path=path, distance=distance,
                                                             title=f'DTW Path Visualization {body_part}, metric {analysis_dtw_metric}',
                                                             body_part=body_part, save_visualization=True,
                                                             path_to_save=image_save_path, prefix=body_part)

                    acc_cost_matrix = analysis.dtw_calculate_acc_cost_matrix(target1_data_from_analyze=target1_data,
                                                                             target2_data_from_analyze=target2_data,
                                                                             body_part=body_part,
                                                                             metric=analysis_dtw_metric)

                    analysis.dtw_visualize_acc_cost_matrix_with_path(path=path, acc_cost_matrix=acc_cost_matrix,
                                                                     distance=distance,
                                                                     body_part=body_part,
                                                                     save_visualization=True,
                                                                     path_to_save=image_save_path,
                                                                     prefix=body_part)

                    dtw_ans = analysis.dataprep.dataprep_summarize_dtw_analysis(path=path, distance=distance,
                                                                                body_part=body_part,
                                                                                radius=analysis_dtw_radius,
                                                                                normalize=analysis_dtw_normalize,
                                                                                smooth=analysis_dtw_smoothing,
                                                                                smooth_window_size=analysis_dtw_smoothing_window_size,
                                                                                metric=analysis_dtw_metric,
                                                                                acc_cost_matrix=acc_cost_matrix)
                    self.dtw_analysis_summary_df = analysis.dataprep.dataprep_append_summaries(
                        self.dtw_analysis_summary_df,
                        [dtw_ans])
                self.log_message('Starting DTW analysis ended')

            if self.synchronyWindow.ui.groupBox_smith_waterman.isChecked():
                self.log_message('Starting Smith-Waterman analysis')
                analysis_sw_metric = analysis.METRIC_EUCLIDEAN if self.synchronyWindow.ui.comboBox_smithWm_distance.currentText() == 'Euclidean distance' else analysis.METRIC_COSINE
                analysis_sw_threshold = float(self.synchronyWindow.ui.doubleSpinBox_smithWm_threshold.text())
                analysis_sw_match_score = float(self.synchronyWindow.ui.doubleSpinBox_smithWm_Match_s.text())
                analysis_sw_gap_penalty = -1 * float(self.synchronyWindow.ui.doubleSpinBox_smithWm_G_penalty.text())
                analysis_sw_mismatch_penalty = -1 * float(
                    self.synchronyWindow.ui.doubleSpinBox_smithWm_M_penalty.text())
                analysis_sw_normalize = self.synchronyWindow.ui.checkBox_smithWm_normalize_vectors.isChecked()

                image_save_path = os.path.join(self.last_analysis_dir_path, f'smith_waterman')
                utility.ensure_folder_exists(image_save_path)
                for body_part in analysis_body_parts:
                    aligned_movements_t1, aligned_movements_t2, aligned_region_length, score = analysis.smith_waterman_analyze(
                        df_target1=self.__target1.get_vector_data_as_dataframe(),
                        df_target2=self.__target2.get_vector_data_as_dataframe(),
                        body_part=body_part,
                        verbose_results=False, match_score=analysis_sw_match_score,
                        mismatch_penalty=analysis_sw_mismatch_penalty,
                        gap_penalty=analysis_sw_gap_penalty,
                        threshold=analysis_sw_threshold,
                        metric=analysis_sw_metric, normalize=analysis_sw_normalize)
                    analysis.smith_waterman_visualize_alignment(aligned_data1=aligned_movements_t1,
                                                                aligned_data2=aligned_movements_t2,
                                                                save_visualization=True,
                                                                title=f'Aligned Timeframes with Gaps {body_part}',
                                                                path_to_save=image_save_path,
                                                                prefix=body_part)
                    smith_w_an_res = analysis.dataprep.dataprep_summarize_smith_waterman_analysis(body_part=body_part,
                                                                                                  metric=analysis_sw_metric,
                                                                                                  match_score=analysis_sw_match_score,
                                                                                                  mismatch_penalty=analysis_sw_mismatch_penalty,
                                                                                                  gap_penalty=analysis_sw_gap_penalty,
                                                                                                  threshold=analysis_sw_threshold,
                                                                                                  alignment_score=score,
                                                                                                  aligned_region_length=aligned_region_length)
                    self.smith_waterman_analysis_summary_df = analysis.dataprep.dataprep_append_summaries(
                        self.smith_waterman_analysis_summary_df,
                        [smith_w_an_res])
                self.log_message('Smith-Waterman analysis ended')

            if self.synchronyWindow.ui.groupBox_tlcc_method.isChecked():
                self.log_message('Starting TLCC analysis')
                analysis_tlcc_max_lag = int(self.synchronyWindow.ui.spinBox_tlcc_max_leg.text())
                analysis_tlcc_normalize = self.synchronyWindow.ui.checkBox_smithWm_normalize_vectors_2.isChecked()

                image_save_path = os.path.join(self.last_analysis_dir_path, f'tlcc')
                utility.ensure_folder_exists(image_save_path)

                for body_part in analysis_body_parts:
                    lags, correlations, target1_data, target2_data = analysis.tlcc_analyze(
                        df_target1=self.__target1.get_vector_data_as_dataframe(),
                        df_target2=self.__target2.get_vector_data_as_dataframe(),
                        body_part=body_part, max_lag=analysis_tlcc_max_lag,
                        normalize=analysis_tlcc_normalize, verbose_results=False)

                    analysis.tlcc_visualize(lags=lags, correlations=correlations, body_part=body_part,
                                            separate_dimensions=False, save_visualization=True,
                                            path_to_save=image_save_path, prefix=f'unified_dim_{body_part}')
                    tlcc_ans_unified = analysis.dataprep.dataprep_summarize_tlcc_analysis(body_part=body_part,
                                                                                          max_lag=analysis_tlcc_max_lag,
                                                                                          separate_dimensions=False,
                                                                                          lags=lags,
                                                                                          correlations=correlations)
                    self.tlcc_analysis_summary_unified_df = analysis.dataprep.dataprep_append_summaries(
                        self.tlcc_analysis_summary_unified_df,
                        [tlcc_ans_unified])

                    lags, correlations, target1_data, target2_data = analysis.tlcc_analyze(
                        df_target1=self.__target1.get_vector_data_as_dataframe(),
                        df_target2=self.__target2.get_vector_data_as_dataframe(),
                        body_part=body_part, max_lag=analysis_tlcc_max_lag, normalize=analysis_tlcc_normalize,
                        separate_dimensions=True
                    )
                    analysis.tlcc_visualize(lags, correlations, analysis_tlcc_normalize, separate_dimensions=True,
                                            save_visualization=True,
                                            path_to_save=image_save_path, prefix=f'separate_dim_{body_part}')
                    tlcc_ans_separate = analysis.dataprep.dataprep_summarize_tlcc_analysis(body_part=body_part,
                                                                                           max_lag=analysis_tlcc_max_lag,
                                                                                           separate_dimensions=True,
                                                                                           lags=lags,
                                                                                           correlations=correlations)
                    self.tlcc_analysis_summary_separate_df = analysis.dataprep.dataprep_append_summaries(
                        self.tlcc_analysis_summary_separate_df,
                        [tlcc_ans_separate])
                self.log_message('TLCC analysis ended')
                self.log_message("Synchrony analysis is finished successfully", 'notification')

    def openGenerateReportsWindow(self):
        self.GenerateReportsWindow = GenerateReportsWindow(self.darkMode)
        if self.similarity_analysis_summary_df is None and self.dtw_analysis_summary_df is None \
                and self.smith_waterman_analysis_summary_df is None \
                and self.tlcc_analysis_summary_unified_df is None \
                and self.tlcc_analysis_summary_separate_df is None:
            self.log_message('No data to export as a report!', 'warning')
        if self.GenerateReportsWindow.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            if self.GenerateReportsWindow.ui.groupBox_generate_pdf.isChecked():
                self.log_message('PDF report export started')
                generate_pdf_exclude_explanations = self.GenerateReportsWindow.ui.checkBox_explanation.isChecked()
                generate_pdf_exclude_plots = self.GenerateReportsWindow.ui.checkBox_2.isChecked()
                reports.generate_intersync_report_pdf(path=self.last_analysis_dir_path,
                                                      similarity_report_df=self.similarity_analysis_summary_df,
                                                      dtw_report_df=self.dtw_analysis_summary_df,
                                                      smith_waterman_report_df=self.smith_waterman_analysis_summary_df,
                                                      tlcc_unified_report_df=self.tlcc_analysis_summary_unified_df,
                                                      tlcc_separate_report_df=self.tlcc_analysis_summary_separate_df,
                                                      keyframe_page_present=self.last_analysis_keyframes_flag,
                                                      keyframe_page_threshold=self.last_analysis_keyframes_threshold,
                                                      frames_analysed=len(self.detected_frames),
                                                      video_filename=self.analysed_video_name,
                                                      run_timestamp=self.last_date_time_string,
                                                      explanation_text=(not generate_pdf_exclude_explanations),
                                                      attach_plot_images=(not generate_pdf_exclude_plots))
                self.log_message('PDF report file saved')
            if self.GenerateReportsWindow.ui.checkBox.isChecked():
                self.log_message('RAW data as Excel export started')
                excel_export.target_vectorframe_data_export_to_excel(target_for_export=self.__target1,
                                                                     export_path=self.last_analysis_dir_path)
                excel_export.target_vectorframe_data_export_to_excel(target_for_export=self.__target2,
                                                                     export_path=self.last_analysis_dir_path)
                output_file = os.path.join(self.last_analysis_dir_path, 'similarity_report_summary.xlsx')
                self.similarity_analysis_summary_df.to_excel(output_file, index=False)
                output_file = os.path.join(self.last_analysis_dir_path, 'smith_waterman_report_summary.xlsx')
                self.smith_waterman_analysis_summary_df.to_excel(output_file, index=False)
                output_file = os.path.join(self.last_analysis_dir_path, 'dtw_report_summary.xlsx')
                self.dtw_analysis_summary_df.to_excel(output_file, index=False)
                output_file = os.path.join(self.last_analysis_dir_path, 'tlcc_unified_report_summary.xlsx')
                self.tlcc_analysis_summary_unified_df.to_excel(output_file, index=False)
                output_file = os.path.join(self.last_analysis_dir_path, 'tlcc_separate_report_summary.xlsx')
                self.tlcc_analysis_summary_separate_df.to_excel(output_file, index=False)
                self.log_message('RAW data as Excel export finied')

            if self.GenerateReportsWindow.ui.checkBox_3.isChecked():
                self.log_message('RAW data as CSV export started')
                excel_export.target_vectorframe_data_export_to_excel(target_for_export=self.__target1,
                                                                     export_path=self.last_analysis_dir_path, csv=True)
                excel_export.target_vectorframe_data_export_to_excel(target_for_export=self.__target2,
                                                                     export_path=self.last_analysis_dir_path, csv=True)
                output_file = os.path.join(self.last_analysis_dir_path, 'similarity_report_summary.xlsx')
                self.similarity_analysis_summary_df.to_csv(output_file, index=False)
                output_file = os.path.join(self.last_analysis_dir_path, 'smith_waterman_report_summary.csv')
                self.smith_waterman_analysis_summary_df.to_csv(output_file, index=False)
                output_file = os.path.join(self.last_analysis_dir_path, 'dtw_report_summary.xlsx')
                self.dtw_analysis_summary_df.to_csv(output_file, index=False)
                output_file = os.path.join(self.last_analysis_dir_path, 'tlcc_unified_report_summary.csv')
                self.tlcc_analysis_summary_unified_df.to_csv(output_file, index=False)
                output_file = os.path.join(self.last_analysis_dir_path, 'tlcc_separate_report_summary.csv')
                self.tlcc_analysis_summary_separate_df.to_csv(output_file, index=False)
                self.log_message('RAW data as CSV export finished')
            self.log_message("Export is finished successfully", 'notification')

    def video_analysis_run_from_gui(self, frame_array):
        for index, kf in enumerate(frame_array):
            image = kf[0]
            timestamp = kf[1]
            self.log_message(f'video_analysis_run: frame {index}, timestamp {timestamp}', "log")
            detect_result = self.__detector_instance.run_detect_on_image_object(
                image=convert_cv2_to_mp_image(image))
            tuo = target_detector.create_target_update_by_center_position(detect_result, self.__target1, self.__target2)
            number_of_targets_for_update = tuo.get_number_of_targets()
            if number_of_targets_for_update == 0:
                self.log_message(
                    f'video_analysis_run: skipped frame {index}, timestamp {timestamp}, 0 targets for update', "log")
                continue
            if number_of_targets_for_update == 2:
                self.__target1.update_target_with_normalized_landmarks_list(
                    tuo.get_update_values_norm_landmarks_by_id(self.__target1.get_id()),
                    tuo.get_update_values_center_by_id(self.__target1.get_id()), timestamp)
                self.__target2.update_target_with_normalized_landmarks_list(
                    tuo.get_update_values_norm_landmarks_by_id(self.__target2.get_id()),
                    tuo.get_update_values_center_by_id(self.__target2.get_id()), timestamp)
            if number_of_targets_for_update == 1:
                if tuo.is_target_id_present(self.__target1.get_id()):
                    self.__target1.update_target_with_normalized_landmarks_list(
                        tuo.get_update_values_norm_landmarks_by_id(self.__target1.get_id()),
                        tuo.get_update_values_center_by_id(self.__target1.get_id()), timestamp)
                else:
                    self.__target2.update_target_with_normalized_landmarks_list(
                        tuo.get_update_values_norm_landmarks_by_id(self.__target2.get_id()),
                        tuo.get_update_values_center_by_id(self.__target2.get_id()), timestamp)

        self.log_message("Video analysis is finished successfully", 'notification')
        self.log_message("You can perform the analysis now", 'notification')
        self.enable_synchrony_analysis = True

    def annotate_original_frames(self, original_frames_array):
        ret_array = []

        t1_fdts = self.__target1.first_detected_timestamp()
        t2_fdts = self.__target2.first_detected_timestamp()

        for index, kf in enumerate(original_frames_array):
            image = kf[0]
            timestamp = kf[1]
            if timestamp < max([t1_fdts, t2_fdts]):
                self.log_message(f'waiting for first detection of both targets, skipping frame at t-{timestamp}', "log")
                continue
            tvf1 = self.__target1.get_vector_frame_by_time(time_included=timestamp)
            tvf2 = self.__target2.get_vector_frame_by_time(time_included=timestamp)
            if tvf1 is None or tvf2 is None:
                continue

            image = utility.draw_landmarks_on_image(image, tvf1.get_movement_vectors_origin_points())
            image = utility.draw_landmarks_on_image(image, tvf2.get_movement_vectors_origin_points())

            image = utility.draw_motion_vectors_on_img(image_object=image,
                                                       motion_vectors=tvf1.get_limb_movement_vectors(),
                                                       initial_position_landmarks=None,
                                                       initial_position_points=tvf1.get_limb_movement_vectors_origin_points(),
                                                       scale=100)
            image = utility.draw_motion_vectors_on_img(image_object=image,
                                                       motion_vectors=tvf2.get_limb_movement_vectors(),
                                                       initial_position_landmarks=None,
                                                       initial_position_points=tvf2.get_limb_movement_vectors_origin_points(),
                                                       scale=100)
            image = utility.add_text_to_top(image, f't-{timestamp}')
            ret_array.append(image)

        return ret_array

    def create_analysis_video(self, analyzed_video_path):
        annotated_frames = self.annotate_original_frames(self.detected_frames)
        height, width, layers = self.detected_frames[0][0].shape

        # define the codec and create VideoWriter object
        fps = 3  # 500 ms - frames per second
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For saving as .mp4 file
        video = cv2.VideoWriter(analyzed_video_path, fourcc, fps, (width, height))

        for frame in annotated_frames:
            video.write(frame)

        # Release the video writer object
        video.release()
        self.log_message(f"Video saved as {analyzed_video_path}", "log")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())
