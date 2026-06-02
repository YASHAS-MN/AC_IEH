import pandas as pd

from scripts.inference_engine import (
    InferenceEngine
)

from scripts.session_engine import (
    SessionEngine
)


class AuthenticationPipeline:


    def __init__(self):

        self.model = InferenceEngine()

        self.session = SessionEngine()


    def authenticate(

        self,

        parquet

    ):

        df = pd.read_parquet(
            parquet
        )

        result = self.model.predict(
            df
        )

        state = self.session.process(

            result["verified"],

            result["score"]

        )

        return state


    def authenticate_window(

        self,

        df

    ):

        result = self.model.predict(
            df
        )

        state = self.session.process(

            result["verified"],

            result["score"]

        )

        return state


if __name__=="__main__":

    auth = AuthenticationPipeline()

    result = auth.authenticate(

        "experiment/owner_test.parquet"

    )

    print()

    print("===== SESSION =====")

    print(result)
