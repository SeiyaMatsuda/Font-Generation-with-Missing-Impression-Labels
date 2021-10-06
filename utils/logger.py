# coding:utf-8
# ログのライブラリ
import logging
from logging import getLogger, StreamHandler, Formatter
def init_logger(OUTPUT_DIR):
    from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger
    log_file = OUTPUT_DIR + "/train.log"
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger