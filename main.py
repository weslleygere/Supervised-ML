import logging

from src.config.logging import LogSetup
from src.config.settings import settings
from src.pipeline import Pipeline


def main() -> None:
    """
    Run the instance-MIR machine learning pipeline.

    Steps
    -----
    1. Configure logging.
    2. Initialize the pipeline.
    3. Run the experiment.
    4. Finalize logging.
    """

    logging_setup = LogSetup(settings=settings)
    logging_setup.setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Starting instance-MIR experiment...")

    try:
        logging_setup.write_log_metadata()

        pipeline = Pipeline(settings=settings)
        pipeline.run()

        logger.info(
            "Evaluation complete. Results saved to %s",
            settings.data.output_dir,
        )

    except Exception:
        logger.exception(
            "An unexpected error occurred during the experiment."
        )

    finally:
        logging_setup.cleanup()


if __name__ == "__main__":
    main()