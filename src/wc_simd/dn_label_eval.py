import streamlit as st
import pandas as pd
import os
from datetime import datetime

# Configuration
DATA_FILE = "data/dn_labels_filtered_dedup_result.csv"
EVAL_OUTPUT_FILE = "data/dn_label_evaluations.csv"


def load_data():
    """Load the deduplication results CSV file"""
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        st.error(f"Data file not found: {DATA_FILE}")
        return None


def load_evaluated_ids():
    """Load previously evaluated IDs to avoid duplicates"""
    if os.path.exists(EVAL_OUTPUT_FILE):
        eval_df = pd.read_csv(EVAL_OUTPUT_FILE)
        return set(eval_df['id'].tolist()
                   ) if 'id' in eval_df.columns else set()
    return set()


def save_evaluation(row, evaluation):
    """Save evaluation result to CSV file immediately"""
    # Prepare the evaluation record
    eval_record = {
        'timestamp': datetime.now().isoformat(),
        'id': row['id'],
        'canon_id': row['canon_id'],
        'prob': row['prob'],
        'label': row['label'],
        'type': row['type'],
        'label_canon': row['label_canon'],
        'type_canon': row['type_canon'],
        'similarity_score': row['similarity_score'],
        'evaluation': evaluation,
        'evaluator': 'streamlit_app'
    }

    # Convert to DataFrame
    eval_df = pd.DataFrame([eval_record])

    # Append to file (create if doesn't exist)
    if os.path.exists(EVAL_OUTPUT_FILE):
        eval_df.to_csv(EVAL_OUTPUT_FILE, mode='a', header=False, index=False)
    else:
        eval_df.to_csv(EVAL_OUTPUT_FILE, mode='w', header=True, index=False)

    st.success(f"Evaluation saved! Result: {evaluation}")


def main():
    st.set_page_config(
        page_title="DN Label Deduplication Evaluator",
        page_icon="🔍",
        layout="wide"
    )

    # Custom CSS for better readability
    st.markdown("""
    <style>
    .main-question {
        font-size: 20px;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        padding: 15px;
        background-color: #F0F8FF;
        border-radius: 10px;
        margin: 20px 0;
    }
    .stButton > button {
        font-size: 16px;
        font-weight: bold;
        height: 60px;
    }
    .metric-container {
        background-color: #F8F9FA;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🔍 DN Label Deduplication Evaluator")
    st.markdown("---")

    # Load data
    df = load_data()
    if df is None:
        return

    # Load previously evaluated IDs
    evaluated_ids = load_evaluated_ids()

    # Add type filter
    st.subheader("🔍 Filters")
    available_types = sorted(df['type'].unique().tolist())
    selected_types = st.multiselect(
        "Filter by Type:",
        options=available_types,
        default=available_types,
        help="Select which types to include in evaluation"
    )

    # Apply filters
    filtered_df = df[df['type'].isin(selected_types)]

    # Filter out already evaluated rows
    remaining_df = filtered_df[~filtered_df['id'].isin(evaluated_ids)]

    # Display progress
    total_rows = len(filtered_df)
    total_original = len(df)
    evaluated_count = len(evaluated_ids)
    remaining_count = len(remaining_df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Original", total_original)
    with col2:
        st.metric("After Filters", total_rows)
    with col3:
        st.metric("Evaluated", evaluated_count)
    with col4:
        st.metric("Remaining", remaining_count)

    # Progress bar
    if total_rows > 0:
        progress = evaluated_count / total_rows
        st.progress(progress, text=f"Progress: {progress:.1%}")

    st.markdown("---")

    if remaining_count == 0:
        if len(selected_types) < len(available_types):
            st.info(
                "🔍 No remaining pairs for selected types. Try adjusting the type filter.")
        else:
            st.success("🎉 All pairs have been evaluated!")
            st.balloons()
        return

    # Get current row to evaluate
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0

    # Ensure index is within bounds
    if st.session_state.current_index >= len(remaining_df):
        st.session_state.current_index = 0

    current_row = remaining_df.iloc[st.session_state.current_index]

    # Display current pair for evaluation
    st.subheader("📋 Current Pair to Evaluate")
    st.markdown(
        f"<div class='main-question'>Does both refer to the same thing?</div>",
        unsafe_allow_html=True)

    # Create two columns for the labels
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Label A")
        st.markdown(
            f"<div style='font-size: 24px; font-weight: bold; color: #1f77b4; padding: 10px; border: 2px solid #1f77b4; border-radius: 5px; margin: 10px 0;'>{current_row['label']}</div>",
            unsafe_allow_html=True)
        st.markdown(f"**Type:** {current_row['type']}")
        st.markdown(f"**ID:** {current_row['id']}")

    with col2:
        st.markdown("### Label B")
        st.markdown(
            f"<div style='font-size: 24px; font-weight: bold; color: #ff7f0e; padding: 10px; border: 2px solid #ff7f0e; border-radius: 5px; margin: 10px 0;'>{current_row['label_canon']}</div>",
            unsafe_allow_html=True)
        st.markdown(f"**Type:** {current_row['type_canon']}")
        st.markdown(f"**ID:** {current_row['canon_id']}")

    # Additional metadata
    st.markdown("### 📊 Metadata")
    metadata_col1, metadata_col2 = st.columns(2)
    with metadata_col1:
        st.markdown(f"**Probability Score:** {current_row['prob']:.4f}")
    with metadata_col2:
        st.markdown(f"**Similarity Score:** {current_row['similarity_score']}")

    st.markdown("---")

    # Evaluation buttons
    st.subheader("✅ Your Evaluation")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("✅ YES - Same Thing", type="primary",
                     use_container_width=True):
            save_evaluation(current_row, "yes")
            st.session_state.current_index = (
                st.session_state.current_index + 1) % len(remaining_df)
            st.rerun()

    with col2:
        if st.button(
            "❌ NO - Different Things", type="secondary",
                use_container_width=True):
            save_evaluation(current_row, "no")
            st.session_state.current_index = (
                st.session_state.current_index + 1) % len(remaining_df)
            st.rerun()

    with col3:
        if st.button("🤔 UNSURE", use_container_width=True):
            save_evaluation(current_row, "unsure")
            st.session_state.current_index = (
                st.session_state.current_index + 1) % len(remaining_df)
            st.rerun()

    with col4:
        if st.button("⏭️ SKIP", use_container_width=True):
            st.session_state.current_index = (
                st.session_state.current_index + 1) % len(remaining_df)
            st.rerun()

    # Navigation
    st.markdown("---")
    st.subheader("🧭 Navigation")
    nav_col1, nav_col2, nav_col3 = st.columns(3)

    with nav_col1:
        if st.button("⬅️ Previous", use_container_width=True):
            st.session_state.current_index = (
                st.session_state.current_index - 1) % len(remaining_df)
            st.rerun()

    with nav_col2:
        # Jump to specific index
        new_index = st.number_input(
            "Jump to index:",
            min_value=0,
            max_value=len(remaining_df) - 1,
            value=st.session_state.current_index,
            key="jump_index"
        )
        if st.button("Jump", use_container_width=True):
            st.session_state.current_index = new_index
            st.rerun()

    with nav_col3:
        if st.button("➡️ Next", use_container_width=True):
            st.session_state.current_index = (
                st.session_state.current_index + 1) % len(remaining_df)
            st.rerun()

    # Show current position
    st.info(
        f"Currently viewing item {st.session_state.current_index + 1} of {len(remaining_df)} remaining items")

    # Sidebar with recent evaluations
    with st.sidebar:
        st.header("📈 Recent Evaluations")
        if os.path.exists(EVAL_OUTPUT_FILE):
            recent_evals = pd.read_csv(EVAL_OUTPUT_FILE).tail(10)
            for _, eval_row in recent_evals.iterrows():
                with st.expander(f"ID: {eval_row['id']} - {eval_row['evaluation'].upper()}"):
                    st.write(f"**Label A:** {eval_row['label']}")
                    st.write(f"**Label B:** {eval_row['label_canon']}")
                    st.write(f"**Time:** {eval_row['timestamp']}")
        else:
            st.info("No evaluations yet")


if __name__ == "__main__":
    main()
