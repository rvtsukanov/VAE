import datetime
from typing import List, Union, Tuple, Any
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import re
import torch
import logging


class MetaSaver:
    def __init__(self, base_directory: str, postfix: str, **kwargs):
        '''
        :param base_directory: root directory to proceed
        :param postfix: postfix, used to distinguish different experiment types
        '''
        self.base_directory = base_directory
        self.postfix = postfix

        logging.basicConfig(filename=f'{__name__}.log',
                            level=logging.INFO,
                            # filemode='w+',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')

        self.logger = logging.getLogger(f'{__name__}')

    def run_experiment(self,
                       attributes_to_save: Union[List, Tuple],
                       wrapper: str = 'run', **kwargs) -> Any:
        '''
        Wrapper. Usage: pass method to wrap into wrapper parameter. This function will create directory
        named by current time before executing wrapper method of class. After, will save all attributes of class
        (passed by 'attributes_to_save') to created directory.
        :param attributes_to_save: attributes to save
        :param wrapper: function to execute
        :param kwargs: arguments used to pass to the wrapped method
        :return: result of applying wrapper-function
        '''
        self.name_dir = self._generate_name_dir()
        self.full_path = os.path.join(self.base_directory, self.name_dir)
        if not os.path.exists(self.full_path):
            os.makedirs(os.path.join(self.base_directory, self.name_dir))

        self._init_inner_logger(path=self.full_path)
        self.logger_inner.info('Initializing inner log')

        if hasattr(self, wrapper):
            self._init_summary_writer(os.path.join(self.base_directory, self.name_dir))
            logging.basicConfig(filename=os.path.join(self.base_directory, self.name_dir, f'{__name__}.log'),
                                level=logging.INFO,
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                datefmt='%d-%b-%y %H:%M:%S')

            result = self.__getattribute__(wrapper)(**kwargs)
            for attr in attributes_to_save:
                self.logger.info(f'Proceeding attribute: {attr}')
                if hasattr(self, attr):
                    self._save_attribute(attr)
                else:
                    self.logger.error(f'{attr} is not an attribute of {self}. Not saved')

            return self

        else:
            self.logger.error(f'{attr} is not an attribute of {self}')
            raise Exception(f'{attr} is not an attribute of {self}')


    def _generate_name_dir(self, postfix: Union[None, str] = None,
                           time: Union[datetime.datetime, str] = None) -> str:
        '''
        Name generation from current time
        :return: str: directory name
        '''
        if not time:
            now = datetime.datetime.now()
        else:
            now = time
        if postfix is None:
            postfix = self.postfix  # TODO: add hashing
        return now.strftime('%d-%m-%Y--%H-%M-%S') + f'--{postfix}'


    def _init_inner_logger(self):
        handler = logging.FileHandler(os.path.join(self.base_directory, self.name_dir,
                                                   f'{__name__}.log'))
        formatter = logging.Formatter('"%(asctime)s - %(levelname)s - %(message)s"',
                                      datefmt='%d-%b-%y %H:%M:%S')

        handler.setFormatter(formatter)
        self.logger_inner = logging.getLogger(f'{__name__}.log')
        self.logger_inner.addHandler(handler)

    def _save_attribute(self, attribute: str) -> None:
        '''
        Method to save attribute
        :param attribute: str, attribute to save
        '''

        if isinstance(self.__getattribute__(attribute), np.ndarray) or \
                isinstance(self.__getattribute__(attribute), list):
            self.logger_inner.info(f'Saving attribute {attribute}')
            try:
                np.save(os.path.join(self.full_path, attribute), np.array(attribute))
            except Exception as e:
                self.logger_inner.error(f'Saving attribute {attribute} was not successfull. Error: {e}')

            self.logger_inner.info(f'Attribute {attribute} saved successfuly')

        if hasattr(self, 'state_dict'):
            self.logger.info('Saving model')
            try:
                torch.save(self.state_dict(), os.path.join(self.full_path, 'model'))
            except Exception as e:
                self.logger_inner.error(f'Saving model was not successfull. Error: {e}')
            else:
                self.logger_inner.info('Model saved successfully')

        else:
            self.logger.error(f'Model is not savable by torch. Proceeding pickling')


    def load_model(self, dir: str):
        '''
        Method to load model located in 'dir'. Possible to pass 'last' value in order to load last possible model (by)
        directory name
        :param dir: name of dir to load; possible to pass 'last'
        :return: self-object (in order to use in pipeline)
        '''
        self.logger.info('Attempt to load model')
        self.logger.info(f'Model tag: {dir}')

        if dir == 'last':
            date_names = []
            for name in os.listdir('./'):
                try:
                    parsed_name = re.findall(r'\d{2}-\d{2}-\d{4}--\d{2}-\d{2}-\d{2}', name)
                    if parsed_name:
                        date_names.append(datetime.datetime.strptime(parsed_name[0], '%d-%m-%Y--%H-%M-%S'))

                except ValueError as e:
                    self.logger.warning(f'Name: {name} was not properly parsed. Error: {e}')


            dir = sorted(date_names, reverse=True)[0]
            self.logger.info(f'{dir} is the last model found. Proceeding')

        self.load_state_dict(torch.load(os.path.join('.', self._generate_name_dir(self.postfix, time=dir), 'model')))
        return self

    def _init_summary_writer(self, log_dir, **kwargs):
        self.writer = SummaryWriter(log_dir=log_dir, **kwargs)




