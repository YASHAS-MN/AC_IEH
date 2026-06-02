import streamlit as st

from scripts.authentication_pipeline import AuthenticationPipeline


st.set_page_config(
    page_title="IEH Portal",
    layout="wide",
)

st.title(
    "Implicit Human Authentication"
)

st.caption(
    "Continuous Behavioral Trust"
)


if "pipeline" not in st.session_state:
    st.session_state.pipeline = AuthenticationPipeline()

if "last_result" not in st.session_state:
    st.session_state.last_result = None


if st.button("Observe Current Behaviour"):
    try:
        result = (
            st.session_state.pipeline
            .authenticate(
                "experiment/owner_test.parquet"
            )
        )

        st.session_state.last_result = result
    except Exception as error:
        st.error(str(error))


if st.session_state.last_result:
    data = st.session_state.last_result

    trust = data.get("trust", 0)
    score = data.get("score", 0)
    state = data.get("state", "unknown")
    action = data.get("action", "unknown")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Trust",
            f"{trust}/100",
        )

    with c2:
        st.metric(
            "Score",
            f"{score:.4f}",
        )

    with c3:
        st.metric(
            "State",
            str(state),
        )

    with c4:
        st.metric(
            "Action",
            str(action),
        )

    if state == "verified":
        st.success("VERIFIED")
    elif state == "observe":
        st.warning("OBSERVE")
    elif state == "challenge":
        st.warning("CHALLENGE")
    elif state == "restrict":
        st.error("RESTRICTED")
    else:
        st.error("RECOVERY REQUIRED")

    st.json(data)
