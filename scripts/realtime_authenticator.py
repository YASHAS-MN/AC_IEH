if __name__ == "__main__":

    import json
    import sys
    from pathlib import Path

    ROOT = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    if str(ROOT) not in sys.path:
        sys.path.insert(
            0,
            str(ROOT)
        )

    from scripts.authentication_pipeline import (
        AuthenticationPipeline
    )


    TEST = (

        ROOT

        /

        "experiment"

        /

        "owner_test.parquet"

    )


    auth = (
        AuthenticationPipeline()
    )


    result = (

        auth.authenticate(
            TEST
        )

    )


    print(

        json.dumps(

            result

        )

    )
