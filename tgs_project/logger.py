import logging

LOG_FORMAT = "%(levelname)s %(asctime)s: %(pathname)s-%(lineno)d. %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT
)

logger = logging.getLogger(__name__)