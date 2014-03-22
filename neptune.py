import argparse
import config
import sys

from util.log import setup_logging

# Bootstrap the application here
setup_logging()
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--api", help="Prints out the API for experiments", action="store_true")
parser.add_argument("-A", "--apiversion", help="Prints out the API Version", action="store_true")
parser.add_argument("-c", "--nogui", help="Launch without PyQt frontend", action="store_true")
parser.add_argument("-e", "--experiment", help="Execute a specific experiment JSON file (Path required)")
parser.add_argument("-v", "--version", help="Prints Neptune version", action="store_true")


if __name__ == '__main__':
    import logging
    logger = logging.getLogger(__name__)
    args = parser.parse_args()

    if args.version:
        print(config.APP_VERSION)
    elif args.api:
        from algorithms import get_api
        API = get_api()
        for api_element in API.structure:
            print(api_element)
    elif args.apiversion:
        from algorithms import get_api
        API = get_api()
        print(API.version)
    elif args.experiment:
        from experiments.holder import EXPHarnessLoader
        exp = EXPHarnessLoader.get_exp_for(args.experiment)
        exp.run()
    elif args.nogui:
        logger.info("Starting in NoGUI Mode")
        logger.error("This hasn't been made yet!")
    else:
        import platform

        from core.gui.main import WAppSelectAction
        from PyQt4.QtGui import QApplication, QFont
        from PyQt4.QtCore import QSysInfo

        if platform.system() == 'Darwin' and QSysInfo.MacintoshVersion > QSysInfo.MV_10_8:
            # Fix a bug in the font rendering for Qt4 on OS X 10.9
            QFont.insertSubstitution(".Lucida Grande UI", "LucidaGrande")

        app = QApplication(sys.argv)
        app.setApplicationName(config.APP_NAME)
        x = WAppSelectAction()
        x.show()
        app.exec_()

    sys.exit(1)