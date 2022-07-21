import logging

logging.basicConfig(
    format="%(asctime)s; %(levelname)s - %(name)s - %(message)s", level=logging.DEBUG
)
logger = logging.getLogger(__name__)


def calc_plus(num1, num2):
    logger.info("Now the calculation starts. Yieah!")
    return num1 + num2


if __name__ == "__main__":
    calc_plus(1, 2)
