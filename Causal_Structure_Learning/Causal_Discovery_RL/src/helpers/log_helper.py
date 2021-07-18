import sys
import logging
from datetime import datetime
from pytz import timezone, utc
from visualdl import LogWriter
import os.path as osp
import time
class VisualLogger():

    def __init__(self, log_dir='./log'):

        self.log_dir = osp.join(log_dir, time.strftime("%Y%m%d%H%M%S", time.localtime()))
        self.writer = LogWriter(logdir=osp.join(self.log_dir))

    def hparams(self, hparams_dict, metrics_list):

        self.writer.add_hparams(hparams_dict=hparams_dict, metrics_list=metrics_list)

    def scalar(self, tag, step, value):

        assert type(step) is int
        assert type(tag) is str
        assert type(value) is float or int
        self.writer.add_scalar(tag=tag, value=value, step=step)

    def close(self):

        self.writer.close()

class LogHelper(object):
    log_format = '%(asctime)s %(levelname)s - %(name)s - %(message)s'

    @staticmethod
    def setup(log_path, level_str='INFO'):
        logging.basicConfig(
             filename=log_path,
             level=logging.getLevelName(level_str),
             format= LogHelper.log_format,
         )

        def customTime(*args):
            utc_dt = utc.localize(datetime.utcnow())
            my_tz = timezone("Asia/Hong_Kong")
            converted = utc_dt.astimezone(my_tz)
            return converted.timetuple()

        logging.Formatter.converter = customTime

        # Set up logging to console
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(logging.Formatter(LogHelper.log_format))
        # Add the console handler to the root logger
        logging.getLogger('').addHandler(console)

        # Log for unhandled exception
        logger = logging.getLogger(__name__)
        sys.excepthook = lambda *ex: logger.critical('Unhandled exception', exc_info=ex)

        logger.info('Completed configuring logger.')
