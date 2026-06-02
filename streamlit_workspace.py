import csv
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from scripts.authentication_pipeline import (
    AuthenticationPipeline
)


RECOVERY_SECRET = "krishna108"
AUDIT_FILE = Path("session_audit.csv")
AUDIT_COLUMNS = [
    "timestamp",
    "mode",
    "score",
    "trust",
    "state",
    "action",
    "page"
]


st.set_page_config(
    page_title="IEH Protected Workspace",
    layout="wide"
)


# -------------------------
# DEMO STATE
# -------------------------

if "pipeline" not in st.session_state:

    st.session_state.pipeline = (
        AuthenticationPipeline()
    )


if "result" not in st.session_state:

    st.session_state.result = {
        "trust": 91,
        "state": "verified",
        "action": "allow",
        "score": 0.0
    }


if "recovery_attempts" not in st.session_state:

    st.session_state.recovery_attempts = 0


if "auto_observe" not in st.session_state:

    st.session_state.auto_observe = False


if "demo_running" not in st.session_state:

    st.session_state.demo_running = False


if "demo_pointer" not in st.session_state:

    st.session_state.demo_pointer = 0


if "demo_sequence" not in st.session_state:

    st.session_state.demo_sequence = []


if "demo_timeline" not in st.session_state:

    st.session_state.demo_timeline = []


if "window_pointer" not in st.session_state:

    st.session_state.window_pointer = {
        "Owner": 0,
        "Impostor": 0
    }


if "behavior_windows" not in st.session_state:

    st.session_state.behavior_windows = {
        "Owner": pd.read_parquet(
            "experiment/owner_test.parquet"
        ),
        "Impostor": pd.read_parquet(
            "experiment/impostor_test.parquet"
        )
    }


def allowed(feature):

    state = st.session_state.result["state"]

    rules = {
        "verified": [
            "all"
        ],
        "observe": [
            "dashboard",
            "profile",
            "transactions"
        ],
        "challenge": [
            "dashboard",
            "profile"
        ],
        "restrict": [
            "dashboard"
        ],
        "recovery": []
    }

    return (
        "all"
        in rules[state]
        or
        feature in rules[state]
    )


def append_audit(mode, page):

    result = st.session_state.result

    row = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "mode": mode,
        "score": result["score"],
        "trust": result["trust"],
        "state": result["state"],
        "action": result["action"],
        "page": page
    }

    should_write_header = not AUDIT_FILE.exists()

    with AUDIT_FILE.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=AUDIT_COLUMNS
        )

        if should_write_header:

            writer.writeheader()

        writer.writerow(
            row
        )


def read_audit_tail(limit=10):

    if (
        not AUDIT_FILE.exists()
        or AUDIT_FILE.stat().st_size == 0
    ):

        return pd.DataFrame(
            columns=AUDIT_COLUMNS
        )

    try:

        audit = pd.read_csv(
            AUDIT_FILE
        )

        missing_columns = [
            column
            for column in AUDIT_COLUMNS
            if column not in audit.columns
        ]

        if missing_columns:

            return pd.DataFrame(
                columns=AUDIT_COLUMNS
            )

        return (
            audit[
                AUDIT_COLUMNS
            ]
            .tail(limit)
        )

    except pd.errors.EmptyDataError:

        return pd.DataFrame(
            columns=AUDIT_COLUMNS
        )


def observe(mode, page):

    data = (
        st.session_state
        .behavior_windows[
            mode
        ]
    )

    pointer = (
        st.session_state
        .window_pointer[
            mode
        ]
    )

    if pointer + 50 > len(data):

        pointer = 0

    window = (
        data
        .iloc[
            pointer:pointer + 50
        ]
        .copy()
    )

    st.session_state.window_pointer[
        mode
    ] = (
        pointer
        +
        50
    )

    result = (
        st.session_state.pipeline
        .authenticate_window(
            window
        )
    )

    st.session_state.result = (
        result
    )

    append_audit(
        mode,
        page
    )


def recover(secret):

    if secret == RECOVERY_SECRET:

        st.session_state.result.update({
            "trust": 90,
            "state": "verified",
            "action": "allow",
            "score": 0
        })

        st.session_state.recovery_attempts = 0

        return True

    else:

        st.session_state.recovery_attempts += 1

        return False


if st.session_state.result["state"] == "recovery":

    st.title(
        "Recovery Required"
    )

    secret = st.text_input(
        "Recovery Key",
        type="password"
    )

    if st.button(
        "Unlock Workspace"
    ):

        if recover(
            secret
        ):

            st.success(
                "Workspace unlocked"
            )

            st.rerun()

        else:

            st.error(
                "Invalid recovery key"
            )

    st.metric(
        "Attempts",
        st.session_state.recovery_attempts
    )

    if st.session_state.recovery_attempts >= 5:

        st.warning(
            "Workspace cooling period active"
        )

    st.stop()


# -------------------------
# SIDEBAR
# -------------------------

st.sidebar.title(
    "Protected Workspace"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Transactions",
        "Profile",
        "Settings",
        "Audit"
    ]
)

st.sidebar.divider()

st.sidebar.subheader(
    "Behavior Simulation"
)


mode = st.sidebar.radio(
    "Session",
    [
        "Owner",
        "Impostor"
    ]
)


st.sidebar.divider()

st.sidebar.subheader(
    "Demo Scenario"
)

demo_scenarios = {
    "Manual": [],
    "Owner Session": [
        "Owner"
    ] * 5,
    "Session Hijack": [
        "Owner",
        "Owner",
        "Impostor",
        "Impostor",
        "Impostor"
    ],
    "Persistent Attack": [
        "Impostor"
    ] * 10,
    "Recovery Flow": [
        "Impostor"
    ] * 8
}

demo_name = st.sidebar.selectbox(
    "Scenario",
    list(
        demo_scenarios.keys()
    )
)

if st.sidebar.button(
    "Run Demo"
):

    st.session_state.demo_sequence = demo_scenarios[
        demo_name
    ]

    st.session_state.demo_pointer = 0
    st.session_state.demo_timeline = []
    st.session_state.demo_running = (
        demo_name != "Manual"
    )
    st.session_state.auto_observe = False


st.sidebar.toggle(
    "Auto Observe",
    key="auto_observe"
)


if st.sidebar.button(
    "Observe Behaviour"
):

    observe(
        mode,
        page
    )


@st.fragment(
    run_every=2
)
def auto_observe():

    if st.session_state.demo_running:

        return

    if not st.session_state.auto_observe:

        return

    if st.session_state.result["state"] == "recovery":

        return

    observe(
        mode,
        page
    )

    st.rerun()


auto_observe()


@st.fragment(
    run_every=2
)
def run_demo_step():

    if not st.session_state.demo_running:

        return

    if st.session_state.result["state"] == "recovery":

        st.session_state.demo_running = False

        return

    if st.session_state.demo_pointer >= len(
        st.session_state.demo_sequence
    ):

        st.session_state.demo_running = False

        return

    current_mode = st.session_state.demo_sequence[
        st.session_state.demo_pointer
    ]

    observe(
        current_mode,
        page
    )

    result = st.session_state.result

    st.session_state.demo_timeline.append({
        "Step": st.session_state.demo_pointer + 1,
        "Mode": current_mode,
        "Score": result["score"],
        "Trust": result["trust"],
        "State": result["state"]
    })

    st.session_state.demo_pointer += 1

    if st.session_state.demo_pointer >= len(
        st.session_state.demo_sequence
    ):

        st.session_state.demo_running = False

    st.rerun()


run_demo_step()


st.sidebar.divider()

st.sidebar.subheader(
    "Session Audit"
)

audit_preview = read_audit_tail(
    10
)

if not audit_preview.empty:

    st.sidebar.dataframe(
        audit_preview,
        use_container_width=True,
        hide_index=True
    )

else:

    st.sidebar.caption(
        "No observations yet"
    )


st.sidebar.divider()

st.sidebar.subheader(
    "Demo Status"
)

demo_total = len(
    st.session_state.demo_sequence
)

if demo_total:

    current_step = min(
        st.session_state.demo_pointer,
        demo_total
    )

    if st.session_state.demo_running:

        current_mode = st.session_state.demo_sequence[
            st.session_state.demo_pointer
        ]

    else:

        current_mode = "Complete"

    st.sidebar.caption(
        f"Current Step: {current_step}/{demo_total}"
    )

    st.sidebar.caption(
        f"Current Mode: {current_mode}"
    )

else:

    st.sidebar.caption(
        "Current Step: Manual"
    )

    st.sidebar.caption(
        f"Current Mode: {mode}"
    )

st.sidebar.caption(
    f"Current Trust: {st.session_state.result['trust']}"
)

st.sidebar.caption(
    f"Current State: {st.session_state.result['state']}"
)


# -------------------------
# HEADER
# -------------------------

left, right = st.columns([3, 1])


with left:

    st.title(
        "Implicit Human Authentication"
    )


with right:

    r = (
        st.session_state.result
    )

    st.metric(
        "Trust",
        f"{r['trust']}/100"
    )

    st.metric(
        "Score",
        f"{r['score']:.4f}"
    )

    st.caption(
        r["state"]
    )

    st.caption(
        r["action"]
    )


# -------------------------
# STATE BANNER
# -------------------------

state = (
    st.session_state
    .result["state"]
)


if state == "verified":

    st.success(
        "Verified Session"
    )


elif state == "observe":

    st.warning(
        "Unusual behaviour detected"
    )


elif state == "challenge":

    st.warning(
        "Sensitive actions may be limited"
    )


elif state == "restrict":

    st.error(
        "Restricted mode"
    )


else:

    st.error(
        "Recovery Required"
    )


st.subheader(
    "Demo Timeline"
)

if st.session_state.demo_timeline:

    timeline = pd.DataFrame(
        st.session_state.demo_timeline
    )

elif AUDIT_FILE.exists():

    timeline = (
        read_audit_tail(
            10
        )
        .tail(10)
        .reset_index(
            drop=True
        )
    )

    timeline = pd.DataFrame({
        "Step": timeline.index + 1,
        "Mode": timeline["mode"],
        "Score": timeline["score"],
        "Trust": timeline["trust"],
        "State": timeline["state"]
    })

else:

    timeline = pd.DataFrame(
        columns=[
            "Step",
            "Mode",
            "Score",
            "Trust",
            "State"
        ]
    )

st.dataframe(
    timeline,
    use_container_width=True,
    hide_index=True
)


# -------------------------
# DASHBOARD
# -------------------------

if page == "Dashboard":

    if not allowed(
        "dashboard"
    ):

        st.error(
            "Dashboard unavailable"
        )

    else:

        st.subheader(
            "Account Overview"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Balance",
                "INR 1,28,450"
            )

        with c2:

            st.metric(
                "Transactions",
                "142"
            )

        with c3:

            st.metric(
                "Devices",
                "3"
            )

        st.divider()

        activity = pd.DataFrame(
            [
                [
                    "Salary",
                    "+INR 50,000"
                ],
                [
                    "Transfer",
                    "-INR 1,200"
                ],
                [
                    "Shopping",
                    "-INR 8,900"
                ]
            ],
            columns=[
                "Activity",
                "Amount"
            ]
        )

        st.dataframe(
            activity,
            use_container_width=True
        )


# -------------------------
# TRANSACTIONS
# -------------------------

elif page == "Transactions":

    if not allowed(
        "transactions"
    ):

        st.warning(
            "Transactions temporarily unavailable"
        )

    else:

        st.subheader(
            "Send Money"
        )

        with st.form(
            "txn"
        ):

            to = st.text_input(
                "Recipient"
            )

            amount = st.number_input(
                "Amount",
                min_value=0
            )

            submit = st.form_submit_button(
                "Transfer"
            )

            if submit:

                st.success(
                    f"INR {amount} transferred to {to}"
                )


# -------------------------
# PROFILE
# -------------------------

elif page == "Profile":

    if not allowed(
        "profile"
    ):

        st.error(
            "Profile locked"
        )

    else:

        st.subheader(
            "Personal Profile"
        )

        st.text_input(
            "Name",
            value="Yashas"
        )

        st.text_input(
            "Email",
            value="yashas@example.com"
        )

        st.text_area(
            "Bio",
            value="Behavioral authentication prototype"
        )

        st.button(
            "Save"
        )


# -------------------------
# SETTINGS
# -------------------------

elif page == "Settings":

    if not allowed(
        "settings"
    ):

        st.error(
            "Settings locked"
        )

    else:

        st.subheader(
            "Workspace Settings"
        )

        st.toggle(
            "Enable Notifications",
            value=True
        )

        st.toggle(
            "Dark Mode",
            value=True
        )

        st.toggle(
            "Show Trust Indicator",
            value=False
        )


# -------------------------
# AUDIT
# -------------------------

elif page == "Audit":

    if not allowed(
        "audit"
    ):

        st.error(
            "Audit locked"
        )

    else:

        st.subheader(
            "Recent Activity"
        )

        audit = pd.DataFrame(
            [
                [
                    "Login",
                    "Success"
                ],
                [
                    "Transfer",
                    "Completed"
                ],
                [
                    "Profile Edit",
                    "Success"
                ]
            ],
            columns=[
                "Event",
                "Status"
            ]
        )

        st.dataframe(
            audit,
            use_container_width=True
        )


st.divider()

st.caption(
    "Phase 3.4 - Restriction Engine"
)
