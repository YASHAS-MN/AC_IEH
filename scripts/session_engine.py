from scripts.trust_engine_v2 import TrustEngine


class SessionEngine:

    def __init__(self):

        self.trust = TrustEngine()

        self.history = []


    def process(

        self,

        verified,

        score

    ):

        if score < 0.10:

            delta = 3

        elif score < 0.20:

            delta = 1

        elif score < 0.50:

            delta = -4

        elif score < 1.0:

            delta = -8

        else:

            delta = -15

        self.trust.trust = max(
            self.trust.min,
            min(
                self.trust.max,
                self.trust.trust + delta
            )
        )

        state = self.trust.state(
        )

        state["score"] = float(
            score
        )

        self.history.append(
            state.copy()
        )

        return state


if __name__=="__main__":

    session = SessionEngine()


    simulation = [

        (True,0.03),

        (True,0.05),

        (False,0.41),

        (False,0.81),

        (False,1.50),

        (True,0.04)

    ]


    for verified,score in simulation:

        print(

            session.process(

                verified,

                score

            )

        )
