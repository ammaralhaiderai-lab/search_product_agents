
import asyncio
import json
import os
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv

# ------------------------------------------------------------
# Secrets: local .env OR Streamlit Community Cloud secrets
# ------------------------------------------------------------
load_dotenv()


from agent_backend import run_pipeline  # noqa: E402


st.set_page_config(
    page_title="AI Product Decision Agent",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 1.4rem 0 1rem 0;
        }

        .hero h1 {
            margin-bottom: 0.25rem;
            font-size: 2.6rem;
            letter-spacing: -0.03em;
        }

        .hero p {
            color: #667085;
            font-size: 1.05rem;
            margin-top: 0;
        }

        .result-card {
            border: 1px solid rgba(120, 120, 120, 0.18);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin: 0.65rem 0;
            background: rgba(255,255,255,0.02);
        }

        .price {
            font-size: 1.5rem;
            font-weight: 700;
        }

        .muted {
            color: #667085;
            font-size: 0.92rem;
        }

        .pill {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: rgba(99,102,241,0.10);
            margin-right: 0.3rem;
            margin-bottom: 0.3rem;
            font-size: 0.82rem;
        }

        .footer-note {
            color: #667085;
            font-size: 0.82rem;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "search"

if "result" not in st.session_state:
    st.session_state.result = None

if "search_context" not in st.session_state:
    st.session_state.search_context = None


def parse_json(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def make_query(product_name: str, purpose: str, budget: float) -> str:
    return f"""
I am looking for the product: {product_name.strip()}.

Purpose / use case:
{purpose.strip()}

Maximum budget:
{int(budget)} SAR

Location:
Saudi Arabia.

Please research current products that match these requirements, compare the
available evidence, and recommend the best-supported option.
""".strip()


def reset_search():
    st.session_state.page = "search"
    st.session_state.result = None
    st.session_state.search_context = None


# ------------------------------------------------------------
# Search page
# ------------------------------------------------------------
if st.session_state.page == "search":
    st.markdown(
        """
        <div class="hero">
            <h1>🛍️ AI Product Decision Agent</h1>
            <p>Tell the agent what you need. It searches the live web, compares products, and gives you sources you can visit.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("product_search_form"):
        product_name = st.text_input(
            "What product are you looking for?",
            placeholder="Example: Laptop, iPhone 17 Pro, Monitor, Camera",
        )

        purpose = st.text_area(
            "What will you use it for?",
            placeholder="Example: Python, PyTorch, gaming and university work",
            height=120,
        )

        budget = st.number_input(
            "Maximum budget (SAR)",
            min_value=100.0,
            max_value=100000.0,
            value=5000.0,
            step=100.0,
        )

        submitted = st.form_submit_button(
            "🔎 Find Products",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not product_name.strip():
            st.error("Please enter the product you are looking for.")
        elif not purpose.strip():
            st.error("Please describe what you will use the product for.")
        else:
            query = make_query(product_name, purpose, budget)
            thread_id = f"streamlit-{uuid.uuid4().hex[:12]}"

            with st.spinner(
                "The agent is researching live sources, comparing products, and checking the recommendation..."
            ):
                try:
                    result = asyncio.run(run_pipeline(query, thread_id))
                    st.session_state.result = result
                    st.session_state.search_context = {
                        "product": product_name.strip(),
                        "purpose": purpose.strip(),
                        "budget": budget,
                    }
                    st.session_state.page = "results"
                    st.rerun()
                except Exception as exc:
                    st.error(f"Search failed: {exc}")
                    st.exception(exc)


# ------------------------------------------------------------
# Results page
# ------------------------------------------------------------
else:
    result = st.session_state.result or {}
    context = st.session_state.search_context or {}

    top_left, top_right = st.columns([5, 1])
    with top_left:
        st.markdown(
            f"""
            <div class="hero">
                <h1>Search Results</h1>
                <p>{context.get('product', 'Product')} · up to {context.get('budget', 0):,.0f} SAR</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_right:
        if st.button("← New Search", use_container_width=True):
            reset_search()
            st.rerun()

    verified_data = parse_json(result.get("products_json", "{}"))
    comparison = parse_json(result.get("comparison_json", "{}"))
    critique = parse_json(result.get("critique_json", "{}"))

    products = verified_data.get("products", []) or []
    best_name = str(comparison.get("best_choice") or "").strip()

    # Put the comparison winner first when possible.
    if best_name:
        products = sorted(
            products,
            key=lambda p: 0 if str(p.get("name", "")).strip().lower() == best_name.lower() else 1,
        )

    st.subheader("Agent Recommendation")
    final_answer = result.get("final_answer", "").strip()

    if final_answer:
        st.markdown(final_answer)

    # If the pipeline has no structured products, still show the final answer.
    if not products:
        st.warning(
            "No structured product candidates were returned by the agent. "
            "Use the final explanation above and try a broader search."
        )
    else:
        st.subheader("Products & How to Reach Them")

        for idx, product in enumerate(products):
            name = product.get("name") or "Unknown product"
            store = product.get("store") or "Unknown"
            url = product.get("url") or ""
            price = product.get("price_sar")
            cpu = product.get("cpu") or "Unknown"
            gpu = product.get("gpu") or "Unknown"
            ram = product.get("ram_gb")
            storage = product.get("storage_gb")
            display = product.get("display") or "Unknown"
            availability = product.get("availability") or "Unknown"
            evidence = product.get("evidence") or ""

            is_best = best_name and str(name).strip().lower() == best_name.lower()

            with st.container():
                if is_best:
                    st.markdown("### 🏆 Best Match")

                st.markdown('<div class="result-card">', unsafe_allow_html=True)

                c1, c2 = st.columns([4, 1])

                with c1:
                    st.markdown(f"#### {name}")
                    st.markdown(f"**Store:** {store}")

                    specs = []
                    if price is not None:
                        specs.append(f"💰 {price:,.0f} SAR")
                    if gpu and gpu != "Unknown":
                        specs.append(f"🎮 {gpu}")
                    if cpu and cpu != "Unknown":
                        specs.append(f"⚙️ {cpu}")
                    if ram is not None:
                        specs.append(f"🧠 {ram} GB RAM")
                    if storage is not None:
                        specs.append(f"💾 {storage} GB")
                    if display and display != "Unknown":
                        specs.append(f"🖥️ {display}")

                    if specs:
                        st.markdown(" ".join(f"`{x}`" for x in specs))

                    st.markdown(
                        f"<span class='muted'>Availability: {availability}</span>",
                        unsafe_allow_html=True,
                    )

                with c2:
                    if url:
                        st.link_button(
                            "Open Product ↗",
                            url,
                            use_container_width=True,
                        )

                if evidence:
                    with st.expander("Evidence"):
                        st.write(evidence)

                st.markdown("</div>", unsafe_allow_html=True)

    # Quality signal
    status = str(critique.get("status", "")).upper()
    if status:
        if status == "PASS":
            st.success("Quality check: PASS")
        else:
            st.warning("Quality check: the critic requested caution or another check.")

    st.markdown(
        "<div class='footer-note'>Prices and availability can change. "
        "Always verify the final price and stock status on the linked retailer page.</div>",
        unsafe_allow_html=True,
    )
