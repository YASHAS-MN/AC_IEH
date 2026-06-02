import json
from pathlib import Path

clean=json.load(
open(
"evaluation/statistics_clean.json"
)
)

owner=clean["owner"]
imp=clean["impostor"]

sep=(
imp["median"]
/
max(
owner["median"],
1e-8
)
)

print("\n===== IEH STATUS =====\n")

print(
f"Owner median : {owner['median']:.4f}"
)

print(
f"Impostor median : {imp['median']:.4f}"
)

print(
f"Separation : {sep:.2f}x"
)

if sep>=5:
    verdict="Excellent"

elif sep>=3:
    verdict="Good"

elif sep>=2:
    verdict="Usable"

else:
    verdict="Weak"

print(
f"\nVerdict: {verdict}"
)

print()

if sep<2:
    print(
"Next: collect more owner data"
    )

elif sep<4:
    print(
"Next: improve inference logic"
    )

else:
    print(
"Next: deploy realtime auth"
    )