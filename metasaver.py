import datetime
import functools
from typing import List, Callable
import os
import torch
import numpy as np
import re


class MetaSaver:
    def __init__(self, base_directory, postfix, **kwargs):
        self.base_directory = base_directory
        self.postfix = postfix

    # def __call__(self, callable, attributes_to_save, *args, **kwargs):
    #     self.setup_experiment(callable, attributes_to_save)


    # def setup_experiment(self, callable: Callable,
    #                      attributes_to_save: List[str]
    #                      ):
    #     @functools.wraps(callable)
    #     def wrapper(*args, **kwargs):
    #         self.name_dir = self._generate_name_dir()
    #         result = callable(*args, **kwargs)
    #
    #         for attr in attributes_to_save:
    #             if hasattr(self, attr):
    #                 self._save_attribute(attr)
    #             else:
    #                 print('Bad attribute')
    #         return result
    #     return wrapper

    def run_experiment(self, attributes_to_save, wrapper='run'):
        if hasattr(self, wrapper):
            self.name_dir = self._generate_name_dir()
            result = self.__getattribute__(wrapper)()
            print(result)
            for attr in attributes_to_save:
                print(attr, hasattr(self, attr))
                if hasattr(self, attr):
                    self._save_attribute(attr)
                else:
                    print('1Bad attribute')

            return result
        else:
            return


    def _generate_name_dir(self, postfix=None, time=None):
        if not time:
            now = datetime.datetime.now()
        else:
            now = time
        if postfix is None:
            postfix = self.postfix  # TODO: add hashing
        return now.strftime('%d-%m-%Y--%H-%M-%S') + f'--{self.postfix}'


    # def save_model(self):
    #     if not os.path.exists(f'{self.name_dir}'):
    #         os.makedirs(f'./{self.name_dir}')
    #     torch.save(self.state_dict(), f'{self.name_dir}/model')
    #     return self


    def _save_attribute(self, attribute):

        full_path = os.path.join(self.base_directory, self.name_dir)
        print(full_path)
        if not os.path.exists(full_path):
            os.makedirs(os.path.join(self.base_directory, self.name_dir))

        print('a', attribute, 't', type(attribute))

        if isinstance(self.__getattribute__(attribute), np.ndarray) or \
                isinstance(self.__getattribute__(attribute), list):
            print('saving loss')
            np.save(os.path.join(full_path, attribute), np.array(attribute))

        if hasattr(self, 'state_dict'):
            print('saving model')
            torch.save(self.state_dict(), os.path.join(full_path, 'model'))

        else:
            print('2Bad attribute')


    def load_model(self, dir):
        if dir == 'last':
            date_names = []
            for name in os.listdir('./'):
                try:
                    # Дешево но что поделать
                    parsed_name = re.findall(r'\d{2}-\d{2}-\d{4}--\d{2}-\d{2}-\d{2}', name)
                    if parsed_name:
                        date_names.append(datetime.datetime.strptime(parsed_name[0], '%d-%m-%Y--%H-%M-%S'))

                except ValueError as e:
                    print(f'Name: {name} was not properly parsed. Error: {e}')


            dir = sorted(date_names, reverse=True)[0]

            print(f'Found maximum: {dir}, proceeding')

        self.load_state_dict(torch.load(os.path.join('.', self._generate_name_dir(self.postfix, time=dir), 'model')))
        return self



