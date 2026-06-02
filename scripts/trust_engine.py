class TrustEngine:

    def __init__(

        self,

        initial=90,

        gain=2,

        decay=6,

        min_trust=0,

        max_trust=100

    ):

        self.trust=initial

        self.gain=gain

        self.decay=decay

        self.min=min_trust

        self.max=max_trust


    def update(

        self,

        verified

    ):

        if verified:

            self.trust=min(

                self.max,

                self.trust+self.gain

            )

        else:

            self.trust=max(

                self.min,

                self.trust-self.decay

            )

        return self.state()


    def state(

        self

    ):

        t=self.trust


        if t>=80:

            return {

                "trust":t,

                "state":"verified",

                "action":"allow"

            }


        elif t>=60:

            return {

                "trust":t,

                "state":"observe",

                "action":"monitor"

            }


        elif t>=40:

            return {

                "trust":t,

                "state":"challenge",

                "action":"warn"

            }


        elif t>=20:

            return {

                "trust":t,

                "state":"restrict",

                "action":"restrict"

            }


        else:

            return {

                "trust":t,

                "state":"recovery",

                "action":"recover"

            }



if __name__=="__main__":

    engine=TrustEngine()

    sequence=[

        True,

        True,

        False,

        False,

        False,

        False,

        True

    ]

    for s in sequence:

        print(

            engine.update(

                s

            )

        )