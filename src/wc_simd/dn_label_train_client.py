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


def search_and_label() -> None:
    """Search for specific records and create training pairs"""
    st.subheader("🔍 Search & Cherry-pick Training Examples")

    # Check if we're in relabel mode from the main interface
    is_relabel_mode = st.session_state.get('labeling_mode') == "relabel"

    if is_relabel_mode:
        st.info("🔄 **Re-label mode**: Searching existing labeled pairs")
        search_target = "existing pairs"
        search_endpoint = "search_existing_pairs"
    else:
        st.info("🆕 **New pairs mode**: Searching raw data to create new training pairs")
        search_target = "raw data"
        search_endpoint = "search_records"

    # Help text for search operators
    with st.expander("ℹ️ Search Help", expanded=False):
        st.write(f"""
        **Search operators (searching {search_target}):**
        - Use `AND` to find records that contain all terms: `children AND criminal`
        - Use `OR` to find records that contain any terms: `law OR legal`
        - Combine operators: `children AND (law OR legal)`
        - Simple search (no operators): searches for the exact phrase in label or type

        **Examples:**
        - `children AND law` - finds records with both "children" and "law"
        - `criminal OR police` - finds records with either "criminal" or "police"
        - `meeting` - finds records containing "meeting" in label or type
        """)

        if is_relabel_mode:
            st.write("""
            **Re-label mode specifics:**
            - Search will look through existing labeled pairs (both match and distinct)
            - You can search for terms in either record of the pair
            - Results will show the current label and allow you to select pairs for re-labeling
            """)

    # Search interface
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        search_query = st.text_input(
            f"Search {search_target} by label or type:",
            placeholder="e.g., 'children AND law', 'criminal OR police', 'meeting'")

    with col2:
        search_limit = st.number_input(
            "Max results:", min_value=10, max_value=500, value=50
        )

    with col3:
        if is_relabel_mode:
            label_filter = st.selectbox(
                "Filter by label:",
                options=["all", "match", "distinct"],
                format_func=lambda x: {
                    "all": "🔍 All pairs",
                    "match": "🟢 Matches only",
                    "distinct": "🔴 Distinct only"
                }[x],
                key="label_filter"
            )
        else:
            # Placeholder for consistent layout
            st.write("")

    if search_query and len(search_query.strip()) >= 2:
        # Perform search
        try:
            # Prepare search parameters
            search_params = {
                "query": search_query.strip(),
                "limit": search_limit
            }

            # Add filter parameter for existing pairs search
            if is_relabel_mode:
                search_params["label_filter"] = label_filter

            search_results = call_service(
                search_endpoint, "POST", search_params)

            if search_results.get("results"):
                results = search_results["results"]

                # Show query parsing info
                if "query_parsed" in search_results:
                    parsed_query = search_results["query_parsed"]
                    if len(parsed_query) > 1:  # Has operators
                        st.info(f"🔍 Query parsed as: {' '.join(parsed_query)}")

                st.write(
                    f"Found {len(results)} results" +
                    (" (limited)" if search_results.get("limited") else ""))

                if is_relabel_mode:
                    # Handle existing pairs for re-labeling
                    display_existing_pairs_results(results, search_query)
                else:
                    # Handle raw data for new pairs
                    display_raw_data_results(results, search_query)
            else:
                st.info(f"No {search_target} found matching your search query")

        except Exception as e:
            st.error(f"Search error: {e}")

    # Display cherry-picked pairs for labeling (same for both modes)
    display_cherry_picked_pairs()


def display_raw_data_results(results, search_query):
    """Display search results from raw data for creating new pairs"""
    # Initialize session state for selected records
    if 'selected_records' not in st.session_state:
        st.session_state.selected_records = []

    # Display results with selection checkboxes
    st.write("**Select records to create training pairs:**")

    # Clear selection button
    if st.button("Clear Selection"):
        st.session_state.selected_records = []
        st.rerun()

    for i, result in enumerate(results):
        record = result['record']
        idx = result['idx']
        match_fields = result.get('match_in', [])
        matched_terms = result.get('matched_terms', [])

        # Create a unique key for this checkbox
        checkbox_key = f"record_{idx}_{hash(search_query)}"

        # Check if this record is already selected
        is_selected = idx in st.session_state.selected_records

        col1, col2, col3 = st.columns([1, 6, 2])

        with col1:
            selected = st.checkbox(
                "",
                value=is_selected,
                key=checkbox_key
            )

            if selected and idx not in st.session_state.selected_records:
                st.session_state.selected_records.append(idx)
            elif not selected and idx in st.session_state.selected_records:
                st.session_state.selected_records.remove(idx)

        with col2:
            # Highlight matching terms in the text
            label_text = record['label']
            type_text = record['type']

            # Highlight matched terms
            if matched_terms:
                for term in matched_terms:
                    label_text = highlight_term(label_text, term)
                    type_text = highlight_term(type_text, term)

            # Show which fields matched
            field_indicators = []
            if isinstance(match_fields, list):
                if 'label' in match_fields:
                    field_indicators.append("📝")
                if 'type' in match_fields:
                    field_indicators.append("🏷️")
            else:
                # Handle old format
                if match_fields == 'label':
                    field_indicators.append("📝")
                elif match_fields == 'type':
                    field_indicators.append("🏷️")

            indicator_text = " ".join(
                field_indicators) + " " if field_indicators else ""

            st.markdown(f"{indicator_text}**Label:** {label_text}")
            st.markdown(f"**Type:** {type_text}")

            # Show matched terms if any
            if matched_terms:
                st.caption(f"Matched: {', '.join(matched_terms)}")

        with col3:
            st.caption(f"ID: {idx}")

    # Show selected records and generate pairs
    if st.session_state.selected_records:
        st.write(
            f"**Selected {len(st.session_state.selected_records)} records**")

        if len(st.session_state.selected_records) >= 2:
            if st.button("🔗 Generate Training Pairs"):
                # Generate pairs from selected records
                try:
                    pairs_response = call_service("get_record_pairs", "POST", {
                        "record_indices": st.session_state.selected_records
                    })

                    if pairs_response.get("pairs"):
                        st.session_state.cherry_picked_pairs = pairs_response["pairs"]
                        st.session_state.cherry_picked_index = 0
                        st.success(
                            f"Generated {len(pairs_response['pairs'])} training pairs!")
                    else:
                        st.warning(
                            "No pairs could be generated from selected records")

                except Exception as e:
                    st.error(f"Error generating pairs: {e}")
        else:
            st.info("Select at least 2 records to generate training pairs")


def display_existing_pairs_results(results, search_query):
    """Display search results from existing pairs for re-labeling"""
    # Initialize session state for selected pairs
    if 'selected_existing_pairs' not in st.session_state:
        st.session_state.selected_existing_pairs = []

    # Display results with selection checkboxes
    st.write("**Select existing pairs to re-label:**")

    # Clear selection button
    if st.button("Clear Selection", key="clear_existing"):
        st.session_state.selected_existing_pairs = []
        st.rerun()

    for i, result in enumerate(results):
        record1 = result['record1']
        record2 = result['record2']
        current_label = result['label']
        match_info = result.get('match_info', {})

        # Create a unique key for this checkbox
        pair_key = f"pair_{i}_{hash(search_query + current_label)}"

        # Check if this pair is already selected
        is_selected = i in st.session_state.selected_existing_pairs

        col1, col2, col3, col4 = st.columns([1, 8, 1, 1])

        with col1:
            selected = st.checkbox(
                "",
                value=is_selected,
                key=pair_key
            )

            if selected and i not in st.session_state.selected_existing_pairs:
                st.session_state.selected_existing_pairs.append(i)
            elif not selected and i in st.session_state.selected_existing_pairs:
                st.session_state.selected_existing_pairs.remove(i)

        with col2:
            # Show current label with color coding
            label_color = "🟢" if current_label == "match" else "🔴"
            st.write(
                f"{label_color} **Current label: {current_label.upper()}**")

            # Display both records with highlighting
            col_r1, col_r2 = st.columns(2)

            with col_r1:
                st.write("**Record 1:**")
                label1 = record1['label']
                type1 = record1['type']

                # Highlight matched terms in record 1
                if match_info.get('record1_matched_terms'):
                    for term in match_info['record1_matched_terms']:
                        label1 = highlight_term(label1, term)
                        type1 = highlight_term(type1, term)

                # Show indicators for matched fields
                indicators1 = []
                if 'label' in match_info.get('record1_match_fields', []):
                    indicators1.append("📝")
                if 'type' in match_info.get('record1_match_fields', []):
                    indicators1.append("🏷️")

                indicator_text1 = " ".join(
                    indicators1) + " " if indicators1 else ""
                st.markdown(f"{indicator_text1}**Label:** {label1}")
                st.markdown(f"**Type:** {type1}")

            with col_r2:
                st.write("**Record 2:**")
                label2 = record2['label']
                type2 = record2['type']

                # Highlight matched terms in record 2
                if match_info.get('record2_matched_terms'):
                    for term in match_info['record2_matched_terms']:
                        label2 = highlight_term(label2, term)
                        type2 = highlight_term(type2, term)

                # Show indicators for matched fields
                indicators2 = []
                if 'label' in match_info.get('record2_match_fields', []):
                    indicators2.append("📝")
                if 'type' in match_info.get('record2_match_fields', []):
                    indicators2.append("🏷️")

                indicator_text2 = " ".join(
                    indicators2) + " " if indicators2 else ""
                st.markdown(f"{indicator_text2}**Label:** {label2}")
                st.markdown(f"**Type:** {type2}")

            # Show which record(s) matched
            match_indicators = []
            if match_info.get('record1_matches'):
                match_indicators.append("Record 1")
            if match_info.get('record2_matches'):
                match_indicators.append("Record 2")

            if match_indicators:
                st.caption(f"Matched in: {', '.join(match_indicators)}")

        with col3:
            st.caption(f"#{i+1}")

        with col4:
            delete_key = f"delete_pair_{i}_{hash(search_query + current_label)}"
            if st.button("🗑️", key=delete_key, help="Delete this pair"):
                try:
                    # Remove the pair from training data
                    call_service("remove_pair", "POST", {
                        "record1": record1,
                        "record2": record2
                    })
                    st.success(f"Deleted {current_label} pair")
                    # Refresh the search results
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting pair: {e}")

        st.write("---")

    # Show selected pairs and action buttons
    if st.session_state.selected_existing_pairs:
        st.write(
            f"**Selected {len(st.session_state.selected_existing_pairs)} pairs**")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Load Pairs for Re-labeling"):
                # Convert selected existing pairs to cherry-picked format
                selected_pairs = []
                for idx in st.session_state.selected_existing_pairs:
                    result = results[idx]
                    selected_pairs.append({
                        'record1': result['record1'],
                        'record2': result['record2'],
                        'original_label': result['label']
                    })

                st.session_state.cherry_picked_pairs = selected_pairs
                st.session_state.cherry_picked_index = 0
                st.success(
                    f"Loaded {len(selected_pairs)} pairs for re-labeling!")

        with col2:
            if st.button(
                "🗑️ Delete Selected Pairs",
                    help="Delete all selected pairs from training data"):
                try:
                    deleted_count = 0
                    for idx in st.session_state.selected_existing_pairs:
                        result = results[idx]
                        call_service("remove_pair", "POST", {
                            "record1": result['record1'],
                            "record2": result['record2']
                        })
                        deleted_count += 1

                    st.success(
                        f"Deleted {deleted_count} pairs from training data")
                    # Clear selection and refresh
                    st.session_state.selected_existing_pairs = []
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting pairs: {e}")


def display_cherry_picked_pairs():
    """Display cherry-picked pairs for labeling (common for both modes)"""
    if 'cherry_picked_pairs' in st.session_state and st.session_state.cherry_picked_pairs:
        st.write("---")
        st.subheader("🏷️ Label Cherry-picked Pairs")

        total_pairs = len(st.session_state.cherry_picked_pairs)
        current_index = st.session_state.get('cherry_picked_index', 0)

        if current_index < total_pairs:
            current_pair = st.session_state.cherry_picked_pairs[current_index]

            st.write(f"**Pair {current_index + 1} of {total_pairs}**")

            # Show original label if this is a re-labeling pair
            if 'original_label' in current_pair:
                st.info(
                    f"🔄 **Original label: {current_pair['original_label'].upper()}**")

            # Display the pair
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Record 1:**")
                record1 = current_pair['record1']
                st.write(f"**Label:** {record1['label']}")
                st.write(f"**Type:** {record1['type']}")
                if 'idx' in record1:
                    st.caption(f"Index: {record1['idx']}")

            with col2:
                st.write("**Record 2:**")
                record2 = current_pair['record2']
                st.write(f"**Label:** {record2['label']}")
                st.write(f"**Type:** {record2['type']}")
                if 'idx' in record2:
                    st.caption(f"Index: {record2['idx']}")

            # Labeling buttons
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                if st.button("✅ Match", key="cherry_match"):
                    label_cherry_picked_pair(current_pair, "match")

            with col2:
                if st.button("❌ Distinct", key="cherry_distinct"):
                    label_cherry_picked_pair(current_pair, "distinct")

            with col3:
                if st.button("❓ Unsure", key="cherry_unsure"):
                    label_cherry_picked_pair(current_pair, "unsure")

            with col4:
                if st.button("⏭️ Skip", key="cherry_skip"):
                    st.session_state.cherry_picked_index += 1
                    st.rerun()

            with col5:
                if 'original_label' in current_pair:
                    if st.button(
                        "🗑️ Delete", key="cherry_delete",
                            help="Delete this pair from training data"):
                        try:
                            # Prepare the pair data for the service
                            record1 = {
                                k: v for k, v in current_pair['record1'].items() if k != 'idx'}
                            record2 = {
                                k: v for k, v in current_pair['record2'].items() if k != 'idx'}

                            call_service("remove_pair", "POST", {
                                "record1": record1,
                                "record2": record2
                            })

                            st.success(
                                f"Deleted pair with label: {current_pair['original_label']}")

                            # Move to next pair
                            st.session_state.cherry_picked_index += 1
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting pair: {e}")
                else:
                    # For new pairs, show placeholder
                    st.write("")

        else:
            st.success("✅ All cherry-picked pairs have been labeled!")
            if st.button("🔄 Start Over"):
                del st.session_state.cherry_picked_pairs
                del st.session_state.cherry_picked_index
                if 'selected_records' in st.session_state:
                    st.session_state.selected_records = []
                if 'selected_existing_pairs' in st.session_state:
                    st.session_state.selected_existing_pairs = []
                st.rerun()


def highlight_term(text, term):
    """Highlight search term in text using markdown"""
    import re
    # Case-insensitive replacement with markdown bold
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    return pattern.sub(lambda m: f"**{m.group()}**", text)


def label_cherry_picked_pair(pair, label):
    """Label a cherry-picked pair and move to next"""
    try:
        # Prepare the pair data for the service
        record1 = {k: v for k, v in pair['record1'].items() if k != 'idx'}
        record2 = {k: v for k, v in pair['record2'].items() if k != 'idx'}

        # If this is a re-labeling pair, remove the old label first
        if 'original_label' in pair:
            call_service("remove_pair", "POST", {
                "record1": record1,
                "record2": record2
            })

        # Add the new label
        call_service("label_pair", "POST", {
            "record1": record1,
            "record2": record2,
            "label": label
        })

        # Move to next pair
        st.session_state.cherry_picked_index += 1

        if 'original_label' in pair:
            st.success(
                f"Pair re-labeled from {pair['original_label']} to {label}")
        else:
            st.success(f"Pair labeled as {label}")

        st.rerun()

    except Exception as e:
        st.error(f"Error labeling pair: {e}")


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
        # Start from 1 for user-friendly numbering
        st.session_state.existing_pairs_index = 1
    if 'interactive_label_filter' not in st.session_state:
        st.session_state.interactive_label_filter = "all"
    if 'show_delete_confirmation' not in st.session_state:
        st.session_state.show_delete_confirmation = False

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
        st.session_state.existing_pairs_index = 1  # Reset to 1

    # Add filter selection for re-label mode
    if st.session_state.labeling_mode == "relabel":
        st.sidebar.write("**Filter:**")
        current_filter = st.sidebar.selectbox(
            "Show pairs:",
            options=[
                "all",
                "match",
                "distinct"],
            format_func=lambda x: {
                "all": "🔍 All pairs",
                "match": "🟢 Matches only",
                "distinct": "🔴 Distinct only"}[x],
            index=[
                "all",
                "match",
                "distinct"].index(
                    st.session_state.interactive_label_filter),
            key="interactive_filter_selector")

        # Reset pairs if filter changed
        if current_filter != st.session_state.interactive_label_filter:
            st.session_state.interactive_label_filter = current_filter
            st.session_state.existing_pairs = []
            st.session_state.existing_pairs_index = 1  # Reset to 1
            st.session_state.current_pair = None
            st.session_state.current_pair_original_label = None

    # Progress display
    st.sidebar.write(f"**Progress:**")
    if st.session_state.labeling_mode == "new":
        st.sidebar.write(f"✅ Matches: {stats['matches']}/10")
        st.sidebar.write(f"❌ Distinct: {stats['distinct']}/10")
    else:
        st.sidebar.write(f"📝 Total existing pairs: {stats['total']}")
        st.sidebar.write(f"✅ Current matches: {stats['matches']}")
        st.sidebar.write(f"❌ Current distinct: {stats['distinct']}")

        # Show filter information if pairs have been loaded
        if st.session_state.existing_pairs:
            filter_label = {
                "all": "🔍 All",
                "match": "🟢 Match",
                "distinct": "🔴 Distinct"
            }[st.session_state.interactive_label_filter]
            st.sidebar.write(
                f"{filter_label} pairs loaded: {len(st.session_state.existing_pairs)}")
            if st.session_state.existing_pairs_index > 1:
                st.sidebar.write(
                    f"Progress: {st.session_state.existing_pairs_index-1}/{len(st.session_state.existing_pairs)}")

        # Bulk management options for re-label mode
        if st.session_state.labeling_mode == "relabel":
            st.sidebar.write("---")
            st.sidebar.write("**Bulk Actions:**")

            # Reset current session
            if st.sidebar.button("🔄 Reset Session"):
                st.session_state.existing_pairs = []
                st.session_state.existing_pairs_index = 1
                st.session_state.current_pair = None
                st.session_state.current_pair_original_label = None
                st.rerun()

            # Advanced delete option with confirmation
            if st.sidebar.button("⚠️ Delete All Filtered Pairs"):
                st.session_state.show_delete_confirmation = True

            # Show confirmation dialog if triggered
            if st.session_state.get('show_delete_confirmation', False):
                st.sidebar.write("**⚠️ Confirm Deletion**")
                filter_desc = {
                    "all": "ALL pairs",
                    "match": "ALL match pairs",
                    "distinct": "ALL distinct pairs"
                }[st.session_state.interactive_label_filter]

                st.sidebar.warning(
                    f"This will delete {filter_desc} from training data!")

                col1, col2 = st.sidebar.columns(2)
                with col1:
                    if st.button("✅ Confirm", key="confirm_delete"):
                        try:
                            # Get all pairs with current filter and delete them
                            delete_response = call_service(
                                "existing_pairs", "POST", {
                                    "label_filter": st.session_state.interactive_label_filter})

                            deleted_count = 0
                            for pair in delete_response["pairs"]:
                                call_service("remove_pair", "POST", {
                                    "record1": pair["record1"],
                                    "record2": pair["record2"]
                                })
                                deleted_count += 1

                            st.sidebar.success(
                                f"Deleted {deleted_count} pairs")
                            st.session_state.show_delete_confirmation = False
                            st.session_state.existing_pairs = []
                            st.session_state.existing_pairs_index = 1
                            st.session_state.current_pair = None
                            st.session_state.current_pair_original_label = None
                            st.rerun()
                        except Exception as e:
                            st.sidebar.error(f"Error: {e}")

                with col2:
                    if st.button("❌ Cancel", key="cancel_delete"):
                        st.session_state.show_delete_confirmation = False
                        st.rerun()

    # Check if we're finished
    if st.session_state.is_finished:
        st.success("🎉 Labeling completed!")
        if st.button("Restart Labeling"):
            st.session_state.is_finished = False
            st.session_state.current_pair = None
            st.session_state.current_pair_original_label = None
            st.session_state.pair_index = 0
            st.session_state.existing_pairs = []
            st.session_state.existing_pairs_index = 1
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
                # Get existing pairs with current filter
                existing_response = call_service("existing_pairs", "POST", {
                    "label_filter": st.session_state.interactive_label_filter
                })
                st.session_state.existing_pairs = existing_response["pairs"]
                # Start from 1 for user-friendly numbering
                st.session_state.existing_pairs_index = 1

            # Load the current pair based on index
            if st.session_state.existing_pairs and st.session_state.existing_pairs_index <= len(
                    st.session_state.existing_pairs):
                current_existing = st.session_state.existing_pairs[
                    st.session_state.existing_pairs_index - 1]
                st.session_state.current_pair = {
                    "record1": current_existing["record1"],
                    "record2": current_existing["record2"]
                }
                st.session_state.current_pair_original_label = current_existing["label"]
                st.session_state.pair_index = st.session_state.existing_pairs_index
            else:
                filter_text = {
                    "all": "pairs",
                    "match": "match pairs",
                    "distinct": "distinct pairs"
                }[st.session_state.interactive_label_filter]
                st.warning(f"No more {filter_text} to re-label!")
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
            filter_desc = {
                "all": "all pairs",
                "match": "match pairs only",
                "distinct": "distinct pairs only"
            }[st.session_state.interactive_label_filter]
            st.caption(
                f"Re-labeling pair {st.session_state.pair_index} of {len(st.session_state.existing_pairs)} ({filter_desc})")

        # Show mode and original label info
        if st.session_state.labeling_mode == "relabel":
            filter_indicator = {
                "all": "🔍",
                "match": "🟢",
                "distinct": "🔴"
            }[st.session_state.interactive_label_filter]
            st.info(
                f"🔄 **Re-labeling mode** {filter_indicator} - Original label: **{st.session_state.current_pair_original_label}**")

        # Add navigation controls for re-label mode
        if st.session_state.labeling_mode == "relabel" and st.session_state.existing_pairs:
            st.write("---")
            st.write("**Navigation:**")

            nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([
                                                                          1, 1, 2, 1, 1])

            with nav_col1:
                if st.button("⬅️ Previous", disabled=(
                        st.session_state.existing_pairs_index <= 1)):
                    if st.session_state.existing_pairs_index > 1:
                        st.session_state.existing_pairs_index -= 1
                        # Load the previous pair
                        current_existing = st.session_state.existing_pairs[
                            st.session_state.existing_pairs_index - 1]
                        st.session_state.current_pair = {
                            "record1": current_existing["record1"],
                            "record2": current_existing["record2"]
                        }
                        st.session_state.current_pair_original_label = current_existing["label"]
                        st.session_state.pair_index = st.session_state.existing_pairs_index
                        st.rerun()

            with nav_col2:
                if st.button(
                    "➡️ Next",
                    disabled=(st.session_state.existing_pairs_index >=
                              len(st.session_state.existing_pairs))):
                    if st.session_state.existing_pairs_index < len(
                            st.session_state.existing_pairs):
                        st.session_state.existing_pairs_index += 1
                        # Load the next pair
                        if st.session_state.existing_pairs_index <= len(
                                st.session_state.existing_pairs):
                            current_existing = st.session_state.existing_pairs[
                                st.session_state.existing_pairs_index - 1]
                            st.session_state.current_pair = {
                                "record1": current_existing["record1"],
                                "record2": current_existing["record2"]
                            }
                            st.session_state.current_pair_original_label = current_existing["label"]
                            st.session_state.pair_index = st.session_state.existing_pairs_index
                            st.rerun()

            with nav_col3:
                # Go to specific pair
                col_input, col_go = st.columns([3, 1])
                with col_input:
                    goto_pair = st.number_input(
                        "Go to pair:",
                        min_value=1,
                        max_value=len(st.session_state.existing_pairs),
                        value=st.session_state.existing_pairs_index,
                        key="goto_pair_input"
                    )
                with col_go:
                    if st.button("Go"):
                        if 1 <= goto_pair <= len(
                                st.session_state.existing_pairs):
                            st.session_state.existing_pairs_index = goto_pair
                            # Load the specified pair
                            current_existing = st.session_state.existing_pairs[goto_pair - 1]
                            st.session_state.current_pair = {
                                "record1": current_existing["record1"],
                                "record2": current_existing["record2"]
                            }
                            st.session_state.current_pair_original_label = current_existing["label"]
                            st.session_state.pair_index = goto_pair
                            st.rerun()

            with nav_col4:
                if st.button("⏭️ Jump to End"):
                    st.session_state.existing_pairs_index = len(
                        st.session_state.existing_pairs)
                    # Load the last pair
                    current_existing = st.session_state.existing_pairs[-1]
                    st.session_state.current_pair = {
                        "record1": current_existing["record1"],
                        "record2": current_existing["record2"]
                    }
                    st.session_state.current_pair_original_label = current_existing["label"]
                    st.session_state.pair_index = len(
                        st.session_state.existing_pairs)
                    st.rerun()

            with nav_col5:
                if st.button("⏪ Jump to Start"):
                    st.session_state.existing_pairs_index = 1
                    # Load the first pair
                    current_existing = st.session_state.existing_pairs[0]
                    st.session_state.current_pair = {
                        "record1": current_existing["record1"],
                        "record2": current_existing["record2"]
                    }
                    st.session_state.current_pair_original_label = current_existing["label"]
                    st.session_state.pair_index = 1
                    st.rerun()

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
            col1, col2, col3, col4, col5, col6 = st.columns(6)
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

                # Navigate to next pair in re-label mode, or clear for new mode
                if st.session_state.labeling_mode == "relabel":
                    # Move to next pair or finish if at end
                    if st.session_state.existing_pairs_index < len(
                            st.session_state.existing_pairs):
                        st.session_state.existing_pairs_index += 1
                        st.session_state.pair_index = st.session_state.existing_pairs_index
                        # Load next pair
                        if st.session_state.existing_pairs_index <= len(
                                st.session_state.existing_pairs):
                            current_existing = st.session_state.existing_pairs[
                                st.session_state.existing_pairs_index - 1]
                            st.session_state.current_pair = {
                                "record1": current_existing["record1"],
                                "record2": current_existing["record2"]
                            }
                            st.session_state.current_pair_original_label = current_existing["label"]
                        else:
                            st.session_state.current_pair = None
                            st.session_state.current_pair_original_label = None
                    else:
                        st.session_state.current_pair = None
                        st.session_state.current_pair_original_label = None
                else:
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

                # Navigate to next pair in re-label mode, or clear for new mode
                if st.session_state.labeling_mode == "relabel":
                    # Move to next pair or finish if at end
                    if st.session_state.existing_pairs_index < len(
                            st.session_state.existing_pairs):
                        st.session_state.existing_pairs_index += 1
                        st.session_state.pair_index = st.session_state.existing_pairs_index
                        # Load next pair
                        if st.session_state.existing_pairs_index <= len(
                                st.session_state.existing_pairs):
                            current_existing = st.session_state.existing_pairs[
                                st.session_state.existing_pairs_index - 1]
                            st.session_state.current_pair = {
                                "record1": current_existing["record1"],
                                "record2": current_existing["record2"]
                            }
                            st.session_state.current_pair_original_label = current_existing["label"]
                        else:
                            st.session_state.current_pair = None
                            st.session_state.current_pair_original_label = None
                    else:
                        st.session_state.current_pair = None
                        st.session_state.current_pair_original_label = None
                else:
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

                # Navigate to next pair in re-label mode, or clear for new mode
                if st.session_state.labeling_mode == "relabel":
                    # Move to next pair or finish if at end
                    if st.session_state.existing_pairs_index < len(
                            st.session_state.existing_pairs):
                        st.session_state.existing_pairs_index += 1
                        st.session_state.pair_index = st.session_state.existing_pairs_index
                        # Load next pair
                        if st.session_state.existing_pairs_index <= len(
                                st.session_state.existing_pairs):
                            current_existing = st.session_state.existing_pairs[
                                st.session_state.existing_pairs_index - 1]
                            st.session_state.current_pair = {
                                "record1": current_existing["record1"],
                                "record2": current_existing["record2"]
                            }
                            st.session_state.current_pair_original_label = current_existing["label"]
                        else:
                            st.session_state.current_pair = None
                            st.session_state.current_pair_original_label = None
                    else:
                        st.session_state.current_pair = None
                        st.session_state.current_pair_original_label = None
                else:
                    st.session_state.current_pair = None
                    st.session_state.current_pair_original_label = None
                st.rerun()

        with col4:
            if st.button("🏁 Finished", key="btn_finished",
                         help="Complete the labeling process"):
                st.session_state.is_finished = True
                st.rerun()

        # Add skip button and delete button for re-labeling mode
        if st.session_state.labeling_mode == "relabel":
            with col5:
                if st.button(
                    "⏭️ Skip", key="btn_skip",
                        help="Keep original label and move to next pair"):
                    # Navigate to next pair
                    if st.session_state.existing_pairs_index < len(
                            st.session_state.existing_pairs):
                        st.session_state.existing_pairs_index += 1
                        st.session_state.pair_index = st.session_state.existing_pairs_index
                        # Load next pair
                        if st.session_state.existing_pairs_index <= len(
                                st.session_state.existing_pairs):
                            current_existing = st.session_state.existing_pairs[
                                st.session_state.existing_pairs_index - 1]
                            st.session_state.current_pair = {
                                "record1": current_existing["record1"],
                                "record2": current_existing["record2"]
                            }
                            st.session_state.current_pair_original_label = current_existing["label"]
                        else:
                            st.session_state.current_pair = None
                            st.session_state.current_pair_original_label = None
                    else:
                        st.session_state.current_pair = None
                        st.session_state.current_pair_original_label = None
                    st.rerun()

            with col6:
                if st.button(
                    "🗑️ Delete", key="btn_delete",
                        help="Delete this pair from training data"):
                    try:
                        # Remove the pair from training data
                        call_service(
                            "remove_pair", "POST", st.session_state.current_pair)
                        st.success(
                            f"Deleted pair with original label: {st.session_state.current_pair_original_label}")

                        # Reset the existing pairs to refresh the list
                        st.session_state.existing_pairs = []
                        st.session_state.existing_pairs_index = 1
                        st.session_state.current_pair = None
                        st.session_state.current_pair_original_label = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting pair: {e}")


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

    # Create tabs for different labeling modes
    tab1, tab2 = st.tabs(["🎯 Interactive Labeling", "🔍 Search & Cherry-pick"])

    with tab1:
        streamlit_label()

    with tab2:
        search_and_label()

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
