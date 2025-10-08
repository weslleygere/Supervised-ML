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

    logging_setup = LogSetup(
        log_level=settings.logging.log_level,
        debug_mode=settings.logging.debug,
        output_dir=settings.data.output_dir
    )
    logging_setup.setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("Starting pipeline...")

    try:
        logging_setup.write_log_metadata()

        pipeline = Pipeline(
            data_path=settings.data.data_path,
            schema_path=settings.data.schema_path,
            output_dir=settings.data.output_dir,
            models=settings.model.models,
            random_state=settings.model.random_state,
            n_splits=settings.validation.n_splits
        )
        pipeline.run()
        logger.info(f"Evaluation complete. Results saved to {settings.data.output_dir}")
    
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logging_setup.cleanup()

if __name__ == '__main__':
    main()