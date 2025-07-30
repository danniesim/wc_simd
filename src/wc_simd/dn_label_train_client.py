import logging
import streamlit as st
import requests
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask service configuration
SERVICE_URL = "http://localhost:5051"


def call_service(endpoint: str, method: str = "GET",
                 data: Optional[Dict[Any, Any]] = None) -> Dict[Any, Any]:
    """Make a request to the Dedupe service"""
    url = f"{SERVICE_URL}/{endpoint}"

    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("🔌 Cannot connect to Dedupe service. Please start the service first with: python src/wc_simd/dedupe_service.py")
        st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"Service request failed: {e}")
        st.stop()


def streamlit_label() -> None:
    """
    Train a matcher instance using Streamlit interface with Flask backend.
    """
    st.title("Dedupe Training Interface")

    # Check service health
    health = call_service("health")
    if not health.get("initialized", False):
        st.error("Dedupe service is not properly initialized")
        return

    # Initialize session state
    if 'is_finished' not in st.session_state:
        st.session_state.is_finished = False
    if 'current_pair' not in st.session_state:
        st.session_state.current_pair = None
    if 'pair_index' not in st.session_state:
        st.session_state.pair_index = 0
    if 'labeling_mode' not in st.session_state:
        st.session_state.labeling_mode = "new"  # "new" or "relabel"
    if 'current_pair_original_label' not in st.session_state:
        st.session_state.current_pair_original_label = None
    if 'existing_pairs' not in st.session_state:
        st.session_state.existing_pairs = []
    if 'existing_pairs_index' not in st.session_state:
        st.session_state.existing_pairs_index = 0

    # Get current stats
    stats = call_service("stats")

    # Mode selection in sidebar
    st.sidebar.write("**Labeling Mode:**")
    mode = st.sidebar.radio(
        "Choose labeling mode:",
        ["Label new pairs", "Re-label existing pairs"],
        index=0 if st.session_state.labeling_mode == "new" else 1,
        key="mode_selector"
    )

    # Update session state based on mode selection
    new_mode = "new" if mode == "Label new pairs" else "relabel"
    if new_mode != st.session_state.labeling_mode:
        st.session_state.labeling_mode = new_mode
        st.session_state.current_pair = None
        st.session_state.current_pair_original_label = None
        st.session_state.pair_index = 0
        st.session_state.existing_pairs = []
        st.session_state.existing_pairs_index = 0

    # Progress display
    st.sidebar.write(f"**Progress:**")
    if st.session_state.labeling_mode == "new":
        st.sidebar.write(f"✅ Matches: {stats['matches']}/10")
        st.sidebar.write(f"❌ Distinct: {stats['distinct']}/10")
    else:
        st.sidebar.write(f"📝 Total existing pairs: {stats['total']}")
        st.sidebar.write(f"✅ Current matches: {stats['matches']}")
        st.sidebar.write(f"❌ Current distinct: {stats['distinct']}")

    # Check if we're finished
    if st.session_state.is_finished:
        st.success("🎉 Labeling completed!")
        if st.button("Restart Labeling"):
            st.session_state.is_finished = False
            st.session_state.current_pair = None
            st.session_state.current_pair_original_label = None
            st.session_state.pair_index = 0
            st.session_state.existing_pairs = []
            st.session_state.existing_pairs_index = 0
            st.rerun()
        return

    # Get next pair if needed
    if st.session_state.current_pair is None:
        if st.session_state.labeling_mode == "new":
            # Get uncertain pair from service
            pair_response = call_service("uncertain_pair")
            if pair_response.get("has_pair", False):
                st.session_state.current_pair = {
                    "record1": pair_response["record1"],
                    "record2": pair_response["record2"]
                }
                st.session_state.pair_index = stats['total'] + 1
                st.session_state.current_pair_original_label = None
            else:
                st.warning("No more new uncertain pairs available!")
                st.session_state.is_finished = True
                st.rerun()
        else:
            # Handle existing labeled pairs for re-labeling
            if not st.session_state.existing_pairs:
                existing_response = call_service("existing_pairs")
                st.session_state.existing_pairs = existing_response["pairs"]
                st.session_state.existing_pairs_index = 0

            if st.session_state.existing_pairs_index < len(
                    st.session_state.existing_pairs):
                current_existing = st.session_state.existing_pairs[st.session_state.existing_pairs_index]
                st.session_state.current_pair = {
                    "record1": current_existing["record1"],
                    "record2": current_existing["record2"]
                }
                st.session_state.current_pair_original_label = current_existing["label"]
                st.session_state.pair_index = st.session_state.existing_pairs_index + 1
                st.session_state.existing_pairs_index += 1
            else:
                st.warning("No more existing pairs to re-label!")
                st.session_state.is_finished = True
                st.rerun()

    if st.session_state.current_pair:
        record1 = st.session_state.current_pair["record1"]
        record2 = st.session_state.current_pair["record2"]

        st.write(f"### Pair {st.session_state.pair_index}")

        # Show additional context based on mode
        if st.session_state.labeling_mode == "new":
            st.caption(
                f"Labeling new pair #{st.session_state.pair_index} (currently have {stats['total']} labeled pairs)")
        else:
            st.caption(
                f"Re-labeling pair {st.session_state.pair_index} of {stats['total']}")

        # Show mode and original label info
        if st.session_state.labeling_mode == "relabel":
            st.info(
                f"🔄 **Re-labeling mode** - Original label: **{st.session_state.current_pair_original_label}**")

        st.write("**Do these records refer to the same thing?**")

        # Display the two records side by side
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Record 1:**")
            for field, value in record1.items():
                st.write(f"**{field}:** {value}")

        with col2:
            st.write("**Record 2:**")
            for field, value in record2.items():
                st.write(f"**{field}:** {value}")

        # Action buttons
        st.write("---")
        if st.session_state.labeling_mode == "relabel":
            col1, col2, col3, col4, col5 = st.columns(5)
        else:
            col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button(
                "✅ Yes (Match)", key="btn_yes",
                    help="These records refer to the same thing"):
                # Remove from existing training if re-labeling
                if st.session_state.labeling_mode == "relabel":
                    call_service(
                        "remove_pair",
                        "POST",
                        st.session_state.current_pair)

                # Label as match
                call_service("label_pair", "POST", {
                    "record1": record1,
                    "record2": record2,
                    "label": "match"
                })
                st.session_state.current_pair = None
                st.session_state.current_pair_original_label = None
                st.rerun()

        with col2:
            if st.button("❌ No (Distinct)", key="btn_no",
                         help="These records are different"):
                # Remove from existing training if re-labeling
                if st.session_state.labeling_mode == "relabel":
                    call_service(
                        "remove_pair",
                        "POST",
                        st.session_state.current_pair)

                # Label as distinct
                call_service("label_pair", "POST", {
                    "record1": record1,
                    "record2": record2,
                    "label": "distinct"
                })
                st.session_state.current_pair = None
                st.session_state.current_pair_original_label = None
                st.rerun()

        with col3:
            if st.button("❓ Unsure", key="btn_unsure",
                         help="Not sure if they match"):
                # Remove from existing training if re-labeling
                if st.session_state.labeling_mode == "relabel":
                    call_service(
                        "remove_pair",
                        "POST",
                        st.session_state.current_pair)

                # Label as unsure
                call_service("label_pair", "POST", {
                    "record1": record1,
                    "record2": record2,
                    "label": "unsure"
                })
                st.session_state.current_pair = None
                st.session_state.current_pair_original_label = None
                st.rerun()

        with col4:
            if st.button("🏁 Finished", key="btn_finished",
                         help="Complete the labeling process"):
                st.session_state.is_finished = True
                st.rerun()

        # Add skip button for re-labeling mode
        if st.session_state.labeling_mode == "relabel":
            with col5:
                if st.button(
                    "⏭️ Skip", key="btn_skip",
                        help="Keep original label and move to next pair"):
                    st.session_state.current_pair = None
                    st.session_state.current_pair_original_label = None
                    st.rerun()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Dedupe Training",
        page_icon="🔍",
        layout="wide"
    )

    st.header("Dedupe Label Training")
    st.write(
        "Use this interface to train the deduplication model by labeling record pairs.")

    # Show service connection status
    try:
        stats = call_service("stats")
        st.info(
            f"📂 Connected to Dedupe service: {stats['matches']} matches, {stats['distinct']} distinct pairs")
    except BaseException:
        st.error(
            "🔌 Cannot connect to Dedupe service. Please start it first with: python src/wc_simd/dedupe_service.py")
        st.stop()

    # Training section
    st.subheader("Interactive Labeling")
    streamlit_label()

    # Save training data section
    if st.session_state.get('is_finished', False):
        st.subheader("Save Training Data")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Save Training Data",
                         help="Save the labeled training pairs"):
                try:
                    result = call_service("save_training", "POST")
                    if result.get("success"):
                        st.success(result["message"])
                    else:
                        st.error(
                            f"Error saving: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error saving training data: {e}")

        with col2:
            if st.button("📊 Show Training Stats"):
                try:
                    stats = call_service("stats")
                    st.write("### Training Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Matches", stats["matches"])
                    with col2:
                        st.metric("Total Distinct", stats["distinct"])
                    with col3:
                        st.metric("Total Pairs", stats["total"])
                except Exception as e:
                    st.error(f"Error getting stats: {e}")
