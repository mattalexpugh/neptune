# Neptune

Testing and experimentation environment developed as part of MPhil study at Aberystwyth University, in conjunction with the Friends of Cardigan Bay. Combines computer vision and machine learning techniques to video sources.

Under heavy development.

## Prerequisites

Neptune is a demanding application. Minimum specifications for useful performance on >= 720p video:

* Core i5 (2nd Gen or higher, older chips will choke) or Xeon
* 8GB RAM (16GB recommended for parallel processing)
* Lots of disk space (depending on methods used and source video)
* Tested on:
	* Windows
	* OS X (primary)
	* Linux

## Installation

This software has a number of prerequisites, although it is written in Python, a number of packages are machine native and should be installed in order.

### OS-Level

Install the following libraries in this order:

* Qt4 Framework
* PyQt4 (and its prerequisite sip)
* ffmpeg
* OpenCV (compiled with ffmpeg support at least)
* cv2 package in Python libraries path

### Python Packages

It is *highly* recommended that you create a virtual environment to contain the Neptune packages. In most Linux distributions, this package is usually called `python-virtualenv`

Once that's installed, follow these steps to get the system running:

	$ virtualenv venv-neptune
	[makes the virtual environment directories]
	$ source venv-neptune/bin/activate
	$ cd mphil-neptune
	$ pip install -r requirements.txt

This will install the following packages to the virtual environment:

* SciPy
* billiard
* mahotas
* setuptools
* scikit-learn

## Running Neptune

Once the software has been successfully installed, navigate to the source directory and type:

	python neptune.py -h

This will list all the arguments, inputs and outputs of Neptune. If you'd rather just launch the GUI:

	python neptune.py

## To-Do

* Write a bash script to automate installation (including ffmpeg and OpenCV) for Linux / OSX
* Possibly remove billiard dependency, depending on testing with multiprocessing
* Distribute fully-encapsulated binary packages for:
  * OS X
  * Linux
* Parallelise where possible. A lot more can be done here.
* Performance improvements:
  * Group similar metric experiments and share resources
  * Move intensive tasks to C-extensions
  * OR move to PyPy

## Questions

If there are any questions, please email me at *map13@aber.ac.uk*
