import sys
import logging
from datetime import datetime

import torch
from pytz import timezone, utc
import neptune.new as neptune

class ResultWriter():

    def __init__(self, project=None, api_token=None, log_dir=None, verbose=False):
        """

        Parameters
        ----------
        project: default None, the project id in Naptune project
        api_token: default None, the api_token in Naptune project
        log_dir: default None, the local log directory
        verbose: default False, if true then Naptune monitors the experiment
        """
        self.verbose = verbose
        if self.verbose == True:

            self.writer = neptune.init(project=project, api_token=api_token)  # your credentials

    def add_scalar(self, tag, value):

        if type(value) is torch.Tensor:
            value = value.detach().numpy()
        if self.verbose == True:
            self.writer[tag].log(value)

    def add_hparam(self, hparam_dict):

        if self.verbose == True:

            self.writer['parameters'] = hparam_dict

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
