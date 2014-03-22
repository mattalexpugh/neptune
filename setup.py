from distutils.core import setup
import sys
from glob import glob

"""
This file builds the application for Windows based systems. Usage:

    python setup.py py2exe

Which builds a dist directory containing the executable, and all its
dependencies.

Assumes:

    MS C++ Redistributeable DLLs:       C:\Libs
    OpenCV DLLs:                        C:\opencv
"""

sys.path.append("C:\\Libs\\Microsoft.VC90.CRT")

data_files = [("Microsoft.VC90.CRT", glob(r'C:\Libs\Microsoft.VC90.CRT\*.*')),
              (".", glob(r'C:\opencv\build\x86\vc11\bin\*.dll'))]

excludes = ["MSVFW32.dll", "MSACM32.dll", "AVICAP32.dll", "AVIFIL32.dll"]

setup(windows=[{"script": "neptune.py"}],
      options={
            "py2exe":{
                "includes": ["sip"],
                "dll_excludes": excludes
             }},
      data_files=data_files)
