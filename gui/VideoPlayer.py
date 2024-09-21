from typing import Optional
import mediapipe as mp

from PyQt6.QtCore import Qt, QUrl, QTime
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
                             QPushButton, QSlider, QStyle)
import cv2

import frames
from frames import keyframes
import utility
from detector import Detector, target_detector
from data.target import Target


class VideoPlayer(QWidget):

    def __init__(self, aPath, parent=None):
        super(VideoPlayer, self).__init__(parent)

        self.mediaPlayer = QMediaPlayer()
        self.audioOutput = QAudioOutput()
        self.mediaPlayer.setAudioOutput(self.audioOutput)
        self.videoWidget = QVideoWidget()
        self.is_processing = False

        self.lbl = QLineEdit('00:00:00')
        self.lbl.setReadOnly(True)
        self.lbl.setFixedWidth(70)

        self.playButton = QPushButton()
        self.playButton.setFixedWidth(32)
        self.playButton.setStyleSheet("background-color: grey")
        self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.positionSlider = QSlider(Qt.Orientation.Horizontal)
        self.positionSlider.setRange(0, 100)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        controlLayout = QHBoxLayout()
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.lbl)
        controlLayout.addWidget(self.positionSlider)

        layout = QVBoxLayout()
        layout.addWidget(self.videoWidget)
        layout.addLayout(controlLayout)

        self.setLayout(layout)

        self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)

        # non-PYQT class members
        self.__loaded_file_path: Optional[str] = None
        self.__loaded_video_duration = None


    def loadFilm(self, f):
        self.mediaPlayer.setSource(QUrl.fromLocalFile(f))
        self.mediaPlayer.pause()
        self.__loaded_file_path = f

    def play(self):
        if self.mediaPlayer.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def positionChanged(self, position):
        self.positionSlider.setValue(position)
        mtime = QTime(0, 0, 0, 0).addMSecs(position)
        self.lbl.setText(mtime.toString())

    def durationChanged(self, duration):
        self.__loaded_video_duration = duration
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def pauseVideo(self):
        self.mediaPlayer.pause()

    def get_loaded_file_path(self):
        return self.__loaded_file_path

    def get_loaded_video_duration(self):
        return 0 if self.__loaded_video_duration is None else self.__loaded_video_duration
