from loguru import logger   # logging tool used throughout pipeline — adds timestamps, color, and log levels automatically
import sys                  # built-in Python module — we use sys.stderr to tell loguru to print logs to the terminal



def setup_logger():

    # remove loguru's default handler so we can replace it with our own custom setup
    logger.remove()
    
    # print logs to the terminal with color-coded formatting
    # format: time | log level | file name | message
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <white>{message}</white>",
        level="DEBUG"   # DEBUG means show all messages, even the most detailed ones
    )

    # write the same logs to a file for reviewing after the pipeline runs
    logger.add(
        "logs/live_chat.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
        level="DEBUG",
        rotation="1 MB"     # start a fresh log file once the current one hits 1MB
    )

    # return the configured logger so other files can use it
    return logger