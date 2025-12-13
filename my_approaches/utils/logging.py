import logging

class AppOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith("utils")

def configure_logging(log_file_name="default.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(log_file_name)
    fh.setLevel(logging.DEBUG)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the filter to the file handler
    fh.addFilter(AppOnlyFilter())

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
