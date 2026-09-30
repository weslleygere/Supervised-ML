import logging

from src.config.logging import LogSetup
from src.config.settings import settings
from src.pipeline import Pipeline


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """
    Run the complete supervised machine-learning experiment.

    Steps
    -----
    1. Configure experiment logging.
    2. Record the experiment configuration.
    3. Run pipeline selection and final independent evaluation.
    4. Save the experiment log in the run output directory.
    """

    # =========================================================================
    # LOGGING
    # =========================================================================

    logging_setup = LogSetup(
        settings=settings
    )

    logging_setup.setup_logging()

    logger = logging.getLogger(
        __name__
    )

    logger.info(
        "Starting supervised ML experiment..."
    )

    # =========================================================================
    # EXPERIMENT
    # =========================================================================

    try:

        logging_setup.write_log_metadata()

        pipeline = Pipeline(
            settings=settings
        )

        pipeline.run()

        logger.info(
            "Experiment complete. Results saved to %s",
            settings.data.output_dir,
        )

    except Exception:

        logger.critical(
            "Experiment failed.",
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
