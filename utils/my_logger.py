import sys
import os
import logging
from logging import handlers


class Logger:
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志关系映射

    def __init__(self, filename, level='info', backCount=10, fmt='%(asctime)s %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        # fmt = '%(asctime)s %(thread)d %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        # fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        # fmt = '%(asctime)s Thread:%(thread)d %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        # fmt = '%(asctime)s %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别

        # 输出到控制台
        # sh = logging.StreamHandler(sys.stdout)  # 往屏幕上输出
        # sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        # self.logger.addHandler(sh)  # 把对象加到logger里

        # 输出到文件
        # 1 按照文件大小分割日志文件,一旦达到指定的大小重新生成文件 10MB
        file_handler = handlers.RotatingFileHandler(filename=filename, maxBytes=1024 * 1024 * 10, backupCount=backCount)
        # 2 日志文件按天进行保存，每天一个日志文件
        # file_handler = handlers.TimedRotatingFileHandler(filename=filename, when='s', backupCount=1, encoding='utf-8')
        file_handler.setLevel(self.level_relations.get(level))
        file_handler.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger


# if not os.path.exists('log'):
#     os.makedirs('log')
# log_file = os.path.join('log', '井漏预警日志.log')  # '井漏预警日志.log'

# log = Logger(log_file, level='debug')
# logger = log.get_logger()
