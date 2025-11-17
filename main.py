import logging
from src.pipeline import Pipeline
from src.config.logging import LogSetup

from src.config.settings import settings

def main() -> None:
    """
    Main entry point for the machine learning pipeline.

    Steps:
    1. Set up logging.
    2. Initialize the pipeline.
    3. Run the pipeline.
    """

    logging_setup = LogSetup(settings = settings)
    logging_setup.setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("Starting pipeline...")

    try:
        logging_setup.write_log_metadata()

        pipeline = Pipeline(settings = settings)
        pipeline.run()
        logger.info(f"Evaluation complete. Results saved to {settings.data.output_dir}")
    
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logging_setup.cleanup()

if __name__ == '__main__':
    main()