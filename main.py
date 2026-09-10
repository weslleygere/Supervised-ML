import logging

from src.config.logging import LogSetup
from src.config.settings import settings
from src.pipeline import Pipeline


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """
    Run the supervised machine learning pipeline.

    Steps
    -----
    1. Configure logging.
    2. Write experiment metadata.
    3. Run the pipeline.
    4. Save the experiment log.
    """

    logging_setup = LogSetup(
        settings=settings
    )

    logging_setup.setup_logging()

    logger = logging.getLogger(__name__)

    logger.info(
        "Starting pipeline..."
    )

    try:
        logging_setup.write_log_metadata()

        pipeline = Pipeline(
            settings=settings
        )

        pipeline.run()

        logger.info(
            "Evaluation complete. Results saved to %s",
            settings.data.output_dir,
        )

    except Exception:
        logger.critical(
            "Pipeline failed.",
            exc_info=True,
        )

        raise

    finally:
        logging_setup.cleanup()


# =============================================================================
# ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    main()
