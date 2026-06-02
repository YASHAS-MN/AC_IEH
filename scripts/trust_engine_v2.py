class TrustEngine:

    def __init__(
        self,
        initial=90,
        min_trust=0,
        max_trust=100
    ):

        self.trust = initial
        self.min = min_trust
        self.max = max_trust


    def update(self, verified):

        if verified:

            if self.trust < 80:
                delta = 4
            else:
                delta = 1

            self.trust = min(
                self.max,
                self.trust + delta
            )

        else:

            if self.trust > 80:

                delta = 5

            elif self.trust > 60:

                delta = 8

            else:

                delta = 12

            self.trust = max(
                self.min,
                self.trust - delta
            )

        return self.state()


    def state(self):

        t = self.trust

        if t >= 80:
            return {
                "trust": t,
                "state": "verified",
                "action": "allow"
            }

        elif t >= 60:
            return {
                "trust": t,
                "state": "observe",
                "action": "monitor"
            }

        elif t >= 40:
            return {
                "trust": t,
                "state": "challenge",
                "action": "warn"
            }

        elif t >= 20:
            return {
                "trust": t,
                "state": "restrict",
                "action": "restrict"
            }

        else:
            return {
                "trust": t,
                "state": "recovery",
                "action": "recover"
            }


if __name__ == "__main__":

    engine = TrustEngine()

    sequence = [
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
            engine.update(s)
        )
