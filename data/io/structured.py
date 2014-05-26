__author__ = 'matt'

import csv
import logging
log = logging.getLogger(__name__)


class CSVEXPSubstrateWriter(object):

    csv_header = ['Input Frame', 'Expected', 'Received', 'Success']

    def __init__(self, fp, exp_hash, classifier):
        self._fp = fp
        self._exp_hash = exp_hash
        self._csv_file = None
        self._classifier = classifier
        self._writer = None

    def __enter__(self):
        log.debug("Writing to CSV file {} ({})".format(self._fp, self._exp_hash))
        self._csv_file = open(self._fp, 'wb')

        exp_csv_writer = csv.writer(self._csv_file)
        exp_csv_writer.writerow(["Video:", self._classifier.video_path,
                                 "GT:", self._classifier.ground_truth])
        exp_csv_writer.writerow(self.csv_header)

        self._writer = exp_csv_writer

        return self

    def __exit__(self):
        self._csv_file.close()

    def writerow(self, row):
        self._writer.writerow(row)
