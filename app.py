
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import base64

from textwrap import dedent
from pathlib import Path
from src.pdf_generator import generate_valuation_pdf
from src.car_finder import filter_car_listings
# =====================================================
# 1. PATHS AND PAGE CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "autovalue_final_model.joblib"
TRAIN_DATA_PATH = BASE_DIR / "data" / "processed" / "X_train.csv"

st.set_page_config(
    page_title="AutoValue AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. LOAD MODEL AND TRAINING DATA
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_training_data():
    return pd.read_csv(TRAIN_DATA_PATH)



RECENT_DATA_PATH = (
    BASE_DIR / "data" / "raw" / "recent_used_listings.parquet"
)


@st.cache_data
def load_recent_listings():
    return pd.read_parquet(RECENT_DATA_PATH)

# =====================================================
# 3. GLOBAL THEME SYSTEM
# =====================================================

dark_mode = st.sidebar.toggle(
    "Dark Mode",
    value=True,
    key="dark_mode"
)

app_theme = "Dark" if dark_mode else "Light"

if app_theme == "Dark":
    bg_color = "#0B1220"
    sidebar_color = "#111C2E"
    card_color = "#172438"
    text_color = "#F1F5F9"
    muted_color = "#A7B4C8"
    border_color = "#2B405B"
    accent_color = "#38BDF8"

else:
    bg_color = "#F3F6FA"
    sidebar_color = "#FFFFFF"
    card_color = "#FFFFFF"
    text_color = "#172033"
    muted_color = "#64748B"
    border_color = "#DCE3ED"
    accent_color = "#1677C8"


# =====================================================
# 4. GLOBAL CUSTOM CSS
# =====================================================

st.markdown(f"""
<style>

.stApp {{
    background-color: {bg_color};
    color: {text_color};
}}

[data-testid="stAppViewContainer"] {{
    background-color: {bg_color};
}}

[data-testid="stHeader"] {{
    background-color: {bg_color};
}}

[data-testid="stSidebar"] {{
    background-color: {sidebar_color};
    border-right: 1px solid {border_color};
}}

.block-container {{
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}}

h1, h2, h3, h4 {{
    color: {text_color} !important;
}}

p, label, li {{
    color: {text_color};
}}

[data-testid="stMetric"] {{
    background-color: {card_color};
    border: 1px solid {border_color};
    padding: 18px;
    border-radius: 14px;
}}

[data-testid="stMetricValue"] {{
    color: {accent_color};
    font-weight: 700;
}}

.stButton > button,
.stDownloadButton > button {{
    border-radius: 10px;
    font-weight: 600;
    border: 1px solid {accent_color};
}}

[data-testid="stDataFrame"] {{
    border: 1px solid {border_color};
    border-radius: 12px;
}}

[data-testid="stExpander"] {{
    border: 1px solid {border_color};
    border-radius: 12px;
}}

[data-testid="stForm"] {{
    border: 1px solid {border_color};
    border-radius: 14px;
    padding: 20px;
}}

.hero {{
    padding: 35px;
    border-radius: 18px;
    background: linear-gradient(
        120deg,
        #14213D,
        #1D3557,
        #264653
    );
    color: white;
    margin-bottom: 25px;
    border: 1px solid #35577D;
}}

.hero h1 {{
    font-size: 42px;
    color: white !important;
}}

.hero p {{
    font-size: 17px;
    color: #DCE6F1 !important;
}}

hr {{
    border-color: {border_color};
}}

</style>
""", unsafe_allow_html=True)



# =====================================================
# 4. SIDEBAR NAVIGATION
# =====================================================

LOGO_PATH = (
    BASE_DIR / "assets" / "branding" / "autovalue_logo.png"
)

# Brand Logo
if LOGO_PATH.exists():

    st.sidebar.image(
        str(LOGO_PATH),
        use_container_width=True
    )

else:
    st.sidebar.title("🚗 AutoValue AI")

st.sidebar.caption(
    "INTELLIGENT VEHICLE VALUATION & MARKET ANALYTICS"
)

st.sidebar.divider()

# Navigation
st.sidebar.markdown("### NAVIGATION")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home Dashboard",
        "Price Prediction",
        "Market Analytics",
        "Compare Cars",
        "Model Performance",
        "Explainable AI",
        "Valuation Report"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()

# Sidebar Footer
st.sidebar.markdown(
    """
    <div style="
        padding: 15px;
        border: 1px solid #38BDF8;
        border-radius: 12px;
        text-align: center;
        font-size: 14px;
    ">
        <strong>⚡ Powered by Machine Learning</strong>
        <br>
        <span style="font-size: 12px;">
            AutoValue AI | Academic Project
        </span>
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# 5. HOME DASHBOARD
# =====================================================

if page == "Home Dashboard":

   
    
   
    # ---------------------------------------------
    # PREMIUM SPLIT AUTOMOTIVE HERO BANNER
    # ---------------------------------------------

    hero_image_path = (
        BASE_DIR / "assets" / "banners" / "hero_dashboard.jpg"
    )

    if hero_image_path.exists():

        with open(hero_image_path, "rb") as image_file:
            hero_base64 = base64.b64encode(
                image_file.read()
            ).decode()

        hero_html = f"""
        <div style="
            width:100%;
            height:245px;
            display:flex;
            overflow:hidden;
            border-radius:18px;
            border:1px solid #315477;
            background:#101D32;
            box-sizing:border-box;
        ">

            <div style="
                width:55%;
                padding:25px 28px;
                display:flex;
                flex-direction:column;
                justify-content:center;
                box-sizing:border-box;
                background:linear-gradient(
                    135deg,#101D32,#192E4B
                );
            ">

                <div style="
                    color:#38BDF8;
                    font-size:12px;
                    font-weight:700;
                    letter-spacing:2px;
                    margin-bottom:8px;
                ">
                    SMARTER VALUATIONS
                </div>

                <div style="
                    color:#FFFFFF;
                    font-size:32px;
                    font-weight:800;
                    line-height:1.2;
                    margin-bottom:10px;
                ">
                    AutoValue <span style="color:#38BDF8;">AI</span>
                </div>

                <div style="
                    color:#E2E8F0;
                    font-size:16px;
                    font-weight:600;
                    margin-bottom:8px;
                ">
                    Intelligent Vehicle Valuation
                </div>

                <div style="
                    color:#AFC1D8;
                    font-size:13px;
                    line-height:1.5;
                ">
                    Discover smarter insights into the used-car
                    market with Machine Learning.
                </div>

            </div>

            <div style="
                width:45%;
                height:245px;
                overflow:hidden;
            ">
                <img
                    src="data:image/jpeg;base64,{hero_base64}"
                    style="
                        width:100%;
                        height:100%;
                        object-fit:cover;
                        object-position:center;
                        display:block;
                    "
                />
            </div>

        </div>
        """

        st.html(hero_html)

    else:

        st.warning(
            "Hero image not found. Check assets/banners/hero_dashboard.jpg."
        )




    # =====================================================
    # DASHBOARD OVERVIEW
    # =====================================================

    st.markdown("")
    st.subheader("Welcome to AutoValue AI")

    st.write(
        "An intelligent machine learning platform designed "
        "to estimate used-car values and explore automotive "
        "market insights."
    )

    st.markdown("### Platform Overview")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            label="Historical Training Records",
            value="5,524",
            delta="Model training dataset"
        )

    with metric2:
        st.metric(
            label="ML Algorithms",
            value="4",
            delta="Regression models evaluated"
        )

    with metric3:
        st.metric(
            label="Selected Model R²",
            value="0.9197",
            delta="Tuned Random Forest"
        )

    st.divider()

    # =====================================================
    # PLATFORM FEATURES
    # =====================================================

    st.subheader("Explore Platform Features")

    st.write(
        "Discover the tools available in AutoValue AI."
    )

    # First row
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("### 🚘 Price Prediction")
            st.write(
                "Estimate the resale value of a used car "
                "using our trained machine learning model."
            )
            st.caption("Vehicle valuation")

    with col2:
        with st.container(border=True):
            st.markdown("### 📊 Market Analytics")
            st.write(
                "Explore used-car listings, prices, brands, "
                "fuel types and city-wise market insights."
            )
            st.caption("Market intelligence")

    with col3:
        with st.container(border=True):
            st.markdown("### ⚖️ Compare Cars")
            st.write(
                "Compare two vehicles using their specifications "
                "and advertised listing prices."
            )
            st.caption("Vehicle comparison")

    # Second row
    col4, col5, col6 = st.columns(3)

    with col4:
        with st.container(border=True):
            st.markdown("### 🧠 Explainable AI")
            st.write(
                "Understand how vehicle features influence "
                "machine learning predictions using SHAP."
            )
            st.caption("Model explainability")

    with col5:
        with st.container(border=True):
            st.markdown("### 📈 Model Performance")
            st.write(
                "Explore regression metrics and compare "
                "the performance of different algorithms."
            )
            st.caption("Model evaluation")

    with col6:
        with st.container(border=True):
            st.markdown("### 📄 Valuation Report")
            st.write(
                "Generate and download a PDF report "
                "containing your vehicle valuation."
            )
            st.caption("PDF report generation")

    st.divider()

    # =====================================================
    # PROJECT INFORMATION
    # =====================================================

    st.subheader("About AutoValue AI")

    st.write(
        "AutoValue AI combines historical vehicle data, "
        "machine learning regression algorithms and "
        "interactive market analytics to provide "
        "data-driven used-car valuation estimates."
    )

    st.caption(
        "Academic Machine Learning Project | "
        "Powered by Python, Scikit-learn and Streamlit"
    )

# =====================================================
# 6. PRICE PREDICTION
# =====================================================

elif page == "Price Prediction":

    st.title("💰 Used-Car Price Prediction")

    st.write(
        "Enter the vehicle specifications below to "
        "estimate its resale price using our trained "
        "Random Forest model."
    )

    st.info(
        "This model was trained on historical vehicle "
        "records from 1983–2020. Predictions outside "
        "this training period are not supported."
    )

    try:
        model = load_model()
        training_data = load_training_data()

        available_brands = sorted(
            training_data["brand"].dropna().unique().tolist()
        )

        with st.form("vehicle_prediction_form"):

            st.subheader("1. Basic Vehicle Information")

            col1, col2 = st.columns(2)

            with col1:

                brand = st.selectbox(
                    "Vehicle Brand",
                    available_brands
                )

                year = st.number_input(
                    "Manufacturing Year",
                    min_value=1983,
                    max_value=2020,
                    value=2018,
                    step=1
                )

                km_driven = st.number_input(
                    "Kilometres Driven (KM)",
                    min_value=0,
                    max_value=1000000,
                    value=45000,
                    step=1000
                )

                fuel = st.selectbox(
                    "Fuel Type",
                    ["Petrol", "Diesel", "CNG", "LPG"]
                )

            with col2:

                seller_type = st.selectbox(
                    "Seller Type",
                    ["Individual", "Dealer", "Trustmark Dealer"]
                )

                transmission = st.selectbox(
                    "Transmission",
                    ["Manual", "Automatic"]
                )

                owner = st.selectbox(
                    "Ownership",
                    [
                        "First Owner",
                        "Second Owner",
                        "Third Owner",
                        "Fourth & Above Owner",
                        "Test Drive Car"
                    ]
                )

            st.divider()

            st.subheader("2. Technical Specifications")

            st.caption(
                "If a specification is unavailable, select "
                "the corresponding checkbox. The model will "
                "handle missing values through its trained "
                "imputation pipeline."
            )

            col3, col4 = st.columns(2)

            with col3:

                mileage_unknown = st.checkbox(
                    "Mileage not available"
                )

                if mileage_unknown:
                    mileage_value = None
                else:
                    mileage_value = st.number_input(
                        "Mileage (numeric value)",
                        min_value=0.1,
                        max_value=100.0,
                        value=18.9,
                        step=0.1
                    )

                engine_unknown = st.checkbox(
                    "Engine capacity not available"
                )

                if engine_unknown:
                    engine_cc = None
                else:
                    engine_cc = st.number_input(
                        "Engine Capacity (CC)",
                        min_value=100,
                        max_value=10000,
                        value=1197,
                        step=50
                    )

                power_unknown = st.checkbox(
                    "Maximum power not available"
                )

                if power_unknown:
                    max_power_bhp = None
                else:
                    max_power_bhp = st.number_input(
                        "Maximum Power (BHP)",
                        min_value=1.0,
                        max_value=2000.0,
                        value=81.8,
                        step=1.0
                    )

            with col4:

                torque_unknown = st.checkbox(
                    "Torque not available"
                )

                if torque_unknown:
                    torque_nm = None
                else:
                    torque_nm = st.number_input(
                        "Torque (Nm)",
                        min_value=1.0,
                        max_value=3000.0,
                        value=113.0,
                        step=1.0
                    )

                seats_unknown = st.checkbox(
                    "Number of seats not available"
                )

                if seats_unknown:
                    seats = None
                else:
                    seats = st.number_input(
                        "Number of Seats",
                        min_value=2,
                        max_value=20,
                        value=5,
                        step=1
                    )

            st.divider()

            predict_button = st.form_submit_button(
                "🔍 Predict Vehicle Price",
                use_container_width=True
            )

        if predict_button:

            input_data = pd.DataFrame([{
                "year": year,
                "km_driven": km_driven,
                "fuel": fuel,
                "seller_type": seller_type,
                "transmission": transmission,
                "owner": owner,
                "mileage_value": mileage_value,
                "engine_cc": engine_cc,
                "max_power_bhp": max_power_bhp,
                "torque_nm": torque_nm,
                "seats": seats,
                "brand": brand
            }])

            prediction = model.predict(input_data)[0]

            st.session_state["last_prediction"] = float(prediction)
            st.session_state["last_vehicle_data"] = input_data.copy()

            if prediction <= 0:
                st.error(
                    "The model returned an invalid price. "
                    "Please review the vehicle details."
                )
            else:
                st.success("Vehicle valuation completed!")

                st.markdown("### Estimated Resale Price")

                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(
                            120deg, #14213D, #264653
                        );
                        padding: 25px;
                        border-radius: 15px;
                        text-align: center;
                        color: white;
                    ">
                        <h2 style="color:#DCE6F1;">
                            Estimated Vehicle Value
                        </h2>
                        <h1 style="
                            font-size: 42px;
                            color: #FFFFFF;
                        ">
                            ₹{prediction:,.0f}
                        </h1>
                        <p>
                            Predicted using Random Forest Regression
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.caption(
                    "This is a model-generated estimate based "
                    "on historical training data. It is not a "
                    "guaranteed selling or transaction price."
                )

                with st.expander("View submitted vehicle details"):
                    st.dataframe(
                        input_data,
                        use_container_width=True
                    )

    except FileNotFoundError as error:
        st.error(
            f"Required project file was not found: {error}"
        )

    except Exception as error:
        st.error(f"Something went wrong: {error}")


    # =====================================================
    # FIND CARS WITHIN YOUR BUDGET
    # =====================================================

    st.divider()

    st.title("🚗 Find Cars Within Your Budget")

    st.write(
        "Discover available used-car listings based on your "
        "budget, preferred company and vehicle requirements."
    )

    st.info(
        "This feature searches our collected vehicle listings. "
        "Prices are advertised asking prices, not confirmed "
        "transaction prices or live market inventory."
    )

    try:

        car_data = load_recent_listings()

        # Prepare filter options
        companies = ["All Companies"] + sorted(
            car_data["make"].dropna().astype(str).unique().tolist()
        )

        fuel_types = ["All Fuel Types"] + sorted(
            value for value in
            car_data["fuel_type"].dropna().astype(str).str.strip().unique()
            if value
        )

        transmissions = ["All Transmission Types"] + sorted(
            value for value in
            car_data["transmission"].dropna().astype(str).str.strip().unique()
            if value
        )

        cities = ["All Cities"] + sorted(
            car_data["city_name"].dropna().astype(str).unique().tolist()
        )

        # ---------------------------------------------
        # SEARCH FILTER FORM
        # ---------------------------------------------

        with st.form("car_finder_form"):

            st.subheader("1. Set Your Budget")

            budget_col1, budget_col2 = st.columns(2)

            with budget_col1:
                min_budget = st.number_input(
                    "Minimum Budget (₹)",
                    min_value=0,
                    max_value=50000000,
                    value=0,
                    step=50000
                )

            with budget_col2:
                max_budget = st.number_input(
                    "Maximum Budget (₹)",
                    min_value=0,
                    max_value=50000000,
                    value=2000000,
                    step=50000
                )

            st.divider()
            st.subheader("2. Select Your Preferences")

            filter_col1, filter_col2 = st.columns(2)

            with filter_col1:

                selected_make = st.selectbox(
                    "Car Company",
                    companies
                )

                max_km = st.number_input(
                    "Maximum Kilometres Driven",
                    min_value=0,
                    max_value=1000000,
                    value=100000,
                    step=5000
                )

                selected_fuel = st.selectbox(
                    "Fuel Type",
                    fuel_types
                )

            with filter_col2:

                selected_transmission = st.selectbox(
                    "Transmission",
                    transmissions
                )

                selected_city = st.selectbox(
                    "Preferred City",
                    cities
                )

                sort_by = st.selectbox(
                    "Sort Results By",
                    [
                        "Price: Low to High",
                        "Price: High to Low",
                        "Newest First",
                        "Lowest KM"
                    ]
                )

            search_button = st.form_submit_button(
                "🔍 Find Matching Cars",
                use_container_width=True
            )

        # ---------------------------------------------
        # SEARCH VALIDATION AND FILTERING
        # ---------------------------------------------

        if search_button:

            if min_budget > max_budget:

                st.error(
                    "Minimum budget cannot be greater than "
                    "maximum budget."
                )

                st.session_state["car_finder_searched"] = False

            else:

                matching_cars = filter_car_listings(
                    data=car_data,
                    min_budget=min_budget,
                    max_budget=max_budget,
                    selected_make=selected_make,
                    max_km=max_km,
                    selected_fuel=selected_fuel,
                    selected_transmission=selected_transmission,
                    selected_city=selected_city,
                    sort_by=sort_by
                )

                st.session_state["car_finder_results"] = matching_cars
                st.session_state["car_finder_searched"] = True

        # ---------------------------------------------
        # DISPLAY SEARCH RESULTS
        # ---------------------------------------------

        if st.session_state.get("car_finder_searched", False):

            results = st.session_state.get(
                "car_finder_results",
                pd.DataFrame()
            )

            st.divider()
            st.subheader("3. Matching Cars")

            if results.empty:

                st.warning(
                    "No matching cars found. Try increasing "
                    "your budget or changing your filters."
                )

            else:

                st.success(
                    f"Found {len(results)} matching vehicle listings!"
                )

                display_columns = [
                    "make",
                    "model",
                    "variant",
                    "city_name",
                    "model_year",
                    "price_inr",
                    "km",
                    "fuel_type",
                    "transmission",
                    "owners",
                    "listing_url"
                ]

                display_columns = [
                    column for column in display_columns
                    if column in results.columns
                ]

                st.dataframe(
                    results[display_columns],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "price_inr": st.column_config.NumberColumn(
                            "Asking Price",
                            format="₹%d"
                        ),
                        "km": st.column_config.NumberColumn(
                            "Kilometres Driven",
                            format="%d km"
                        ),
                        "listing_url": st.column_config.LinkColumn(
                            "Original Listing",
                            display_text="View Car"
                        )
                    }
                )


                # ---------------------------------------------
                # VISUAL CAR LISTING CARDS
                # ---------------------------------------------

                st.divider()
                st.subheader("Explore Cars Visually")

                st.caption(
                    "Showing up to 12 matching vehicles with "
                    "photographs from the original listings."
                )

                cards_to_show = min(len(results), 12)

                for start in range(0, cards_to_show, 3):

                    card_columns = st.columns(3)

                    for position, column in enumerate(card_columns):

                        index = start + position

                        if index >= cards_to_show:
                            break

                        car = results.iloc[index]

                        with column:

                            with st.container(border=True):

    
                                # Car photograph - fixed dimensions
                                image_url = car.get("image_url")

                                if pd.notna(image_url) and str(image_url).strip():

                                    st.markdown(
                                        f"""
                                        <div style="
                                            width:100%;
                                            height:190px;
                                            overflow:hidden;
                                            border-radius:10px;
                                            background:#172438;
                                        ">
                                            <img
                                                src="{image_url}"
                                                style="
                                                    width:100%;
                                                    height:190px;
                                                    object-fit:cover;
                                                    display:block;
                                                "
                                            />
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

                                else:
                                    st.markdown(
                                        """
                                        <div style="
                                            width:100%;
                                            height:190px;
                                            display:flex;
                                            align-items:center;
                                            justify-content:center;
                                            background:#172438;
                                            border-radius:10px;
                                            color:#A7B4C8;
                                        ">
                                            🚗 Image unavailable
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

                                # Car name
                                car_name = (
                                    f"{car.get('make', 'Unknown')} "
                                    f"{car.get('model', '')}"
                                )

                                st.markdown(f"### {car_name}")

                                variant = car.get("variant", "Not specified")

                                if pd.isna(variant):
                                    variant = "Not specified"

                                st.caption(str(variant))

                                # Asking price
                                st.markdown(
                                    f"## ₹{car['price_inr']:,.0f}"
                                )

                                # Vehicle specifications
                                st.write(
                                    f"📅 Year: {car.get('model_year', 'N/A')}"
                                )

                                st.write(
                                    f"🛣️ KM Driven: {car['km']:,.0f}"
                                )

                                st.write(
                                    f"⛽ Fuel: {car.get('fuel_type', 'Unknown')}"
                                )

                                st.write(
                                    f"⚙️ Transmission: "
                                    f"{car.get('transmission', 'Unknown')}"
                                )

                                st.write(
                                    f"📍 City: {car.get('city_name', 'Unknown')}"
                                )

                                # Original listing link
                                listing_url = car.get("listing_url")

                                if (
                                    pd.notna(listing_url)
                                    and str(listing_url).startswith("https://")
                                ):
                                    st.link_button(
                                        "View Original Listing ↗",
                                        str(listing_url),
                                        use_container_width=True
                                    )

                st.caption(
                    "Results are drawn from the collected dataset "
                    "snapshot dated 24 August 2026."
                )

    except FileNotFoundError:

        st.error(
            "Recent listings dataset not found. "
            "Please check data/raw/recent_used_listings.parquet."
        )

    except Exception as error:

        st.error(f"Car Finder error: {error}")


# =====================================================
# 7. OTHER PAGES - TEMPORARY PLACEHOLDERS
# =====================================================


elif page == "Market Analytics":

    st.title("📊 Used-Car Market Analytics")

    st.write(
        "Explore recent used-car listings, price distributions, "
        "popular brands, fuel preferences and city-wise market activity."
    )

    recent_data_path = (
        BASE_DIR / "data" / "raw" / "recent_used_listings.parquet"
    )

    try:
        @st.cache_data
        def load_recent_listings(path):
            return pd.read_parquet(path)

        market_df = load_recent_listings(str(recent_data_path))

        # ---------------------------------------------
        # DATA PREPARATION
        # ---------------------------------------------

        market_df["price_inr"] = pd.to_numeric(
            market_df["price_inr"], errors="coerce"
        )

        market_df["km"] = pd.to_numeric(
            market_df["km"], errors="coerce"
        )

        market_df["model_year"] = pd.to_numeric(
            market_df["model_year"], errors="coerce"
        )

        market_df = market_df.dropna(
            subset=["price_inr", "make", "city_name"]
        )

        st.info(
            "Dataset snapshot: 24 August 2026. "
            "These are advertised listing prices, not confirmed "
            "vehicle transaction prices. This is not a live feed."
        )

        # ---------------------------------------------
        # FILTERS
        # ---------------------------------------------

        st.subheader("Market Filters")

        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            cities = ["All Cities"] + sorted(
                market_df["city_name"].dropna().unique().tolist()
            )

            selected_city = st.selectbox(
                "Select City",
                cities
            )

        with filter_col2:
            fuels = ["All Fuel Types"] + sorted(
                market_df["fuel_type"].dropna().unique().tolist()
            )

            selected_fuel = st.selectbox(
                "Select Fuel Type",
                fuels
            )

        with filter_col3:
            brands = ["All Brands"] + sorted(
                market_df["make"].dropna().unique().tolist()
            )

            selected_brand = st.selectbox(
                "Select Brand",
                brands
            )

        filtered_df = market_df.copy()

        if selected_city != "All Cities":
            filtered_df = filtered_df[
                filtered_df["city_name"] == selected_city
            ]

        if selected_fuel != "All Fuel Types":
            filtered_df = filtered_df[
                filtered_df["fuel_type"] == selected_fuel
            ]

        if selected_brand != "All Brands":
            filtered_df = filtered_df[
                filtered_df["make"] == selected_brand
            ]

        if filtered_df.empty:
            st.warning("No listings match the selected filters.")

        else:

            # -----------------------------------------
            # KEY MARKET METRICS
            # -----------------------------------------

            st.divider()
            st.subheader("Market Overview")

            total_listings = len(filtered_df)
            average_price = filtered_df["price_inr"].mean()
            median_price = filtered_df["price_inr"].median()
            average_km = filtered_df["km"].mean()

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Listings",
                f"{total_listings:,}"
            )

            col2.metric(
                "Average Asking Price",
                f"₹{average_price:,.0f}"
            )

            col3.metric(
                "Median Asking Price",
                f"₹{median_price:,.0f}"
            )

            col4.metric(
                "Average KM Driven",
                f"{average_km:,.0f}"
            )

            # -----------------------------------------
            # PRICE DISTRIBUTION
            # -----------------------------------------

            st.divider()
            st.subheader("1. Vehicle Price Distribution")

            price_chart = px.histogram(
                filtered_df,
                x="price_inr",
                nbins=30,
                title="Distribution of Used-Car Asking Prices",
                labels={
                    "price_inr": "Asking Price (INR)",
                    "count": "Number of Listings"
                }
            )

            price_chart.update_layout(
                template="plotly_dark"
            )

            st.plotly_chart(
                price_chart,
                use_container_width=True
            )

            # -----------------------------------------
            # BRAND ANALYSIS
            # -----------------------------------------

            st.divider()
            st.subheader("2. Brand-wise Market Analysis")

            brand_analysis = (
                filtered_df.groupby("make")
                .agg(
                    listings=("price_inr", "count"),
                    median_price=("price_inr", "median")
                )
                .reset_index()
                .sort_values(
                    "median_price",
                    ascending=False
                )
            )

            brand_col1, brand_col2 = st.columns(2)

            with brand_col1:

                brand_chart = px.bar(
                    brand_analysis,
                    x="make",
                    y="listings",
                    title="Listings by Brand",
                    labels={
                        "make": "Brand",
                        "listings": "Number of Listings"
                    }
                )

                brand_chart.update_layout(
                    template="plotly_dark",
                    xaxis_tickangle=-35
                )

                st.plotly_chart(
                    brand_chart,
                    use_container_width=True
                )

            with brand_col2:

                median_chart = px.bar(
                    brand_analysis,
                    x="make",
                    y="median_price",
                    title="Median Asking Price by Brand",
                    labels={
                        "make": "Brand",
                        "median_price": "Median Price (INR)"
                    }
                )

                median_chart.update_layout(
                    template="plotly_dark",
                    xaxis_tickangle=-35
                )

                st.plotly_chart(
                    median_chart,
                    use_container_width=True
                )

            # -----------------------------------------
            # FUEL ANALYSIS
            # -----------------------------------------

            st.divider()
            st.subheader("3. Fuel Type Distribution")

            fuel_analysis = (
                filtered_df["fuel_type"]
                .value_counts()
                .reset_index()
            )

            fuel_analysis.columns = [
                "fuel_type",
                "listings"
            ]

            fuel_chart = px.pie(
                fuel_analysis,
                names="fuel_type",
                values="listings",
                title="Listings by Fuel Type",
                hole=0.4
            )

            fuel_chart.update_layout(
                template="plotly_dark"
            )

            st.plotly_chart(
                fuel_chart,
                use_container_width=True
            )

            # -----------------------------------------
            # CITY ANALYSIS
            # -----------------------------------------

            st.divider()
            st.subheader("4. City-wise Market Activity")

            city_analysis = (
                filtered_df["city_name"]
                .value_counts()
                .reset_index()
            )

            city_analysis.columns = [
                "city",
                "listings"
            ]

            city_chart = px.bar(
                city_analysis,
                x="city",
                y="listings",
                title="Used-Car Listings by City",
                labels={
                    "city": "City",
                    "listings": "Number of Listings"
                }
            )

            city_chart.update_layout(
                template="plotly_dark",
                xaxis_tickangle=-35
            )

            st.plotly_chart(
                city_chart,
                use_container_width=True
            )

            # -----------------------------------------
            # DATA TABLE
            # -----------------------------------------

            st.divider()
            st.subheader("5. Recent Vehicle Listings")

            display_columns = [
                "make",
                "model",
                "variant",
                "city_name",
                "model_year",
                "km",
                "fuel_type",
                "price_inr"
            ]

            st.dataframe(
                filtered_df[display_columns].head(100),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Showing up to 100 matching listings. "
                "The displayed prices are asking prices."
            )

    except FileNotFoundError:
        st.error(
            "Recent listings dataset not found. "
            "Check data/raw/recent_used_listings.parquet."
        )

    except Exception as error:
        st.error(f"Market Analytics error: {error}")



elif page == "Compare Cars":

    st.title("🚘 Compare Used Cars")

    st.write(
        "Compare two vehicles using actual specifications "
        "and advertised prices from our collected listings."
    )

    comparison_path = (
        BASE_DIR / "data" / "raw" / "recent_used_listings.parquet"
    )

    try:

        @st.cache_data
        def load_comparison_data():
            return pd.read_parquet(comparison_path)

        compare_df = load_comparison_data().reset_index(drop=True)

        compare_df["price_inr"] = pd.to_numeric(
            compare_df["price_inr"], errors="coerce"
        )

        compare_df["km"] = pd.to_numeric(
            compare_df["km"], errors="coerce"
        )

        compare_df["model_year"] = pd.to_numeric(
            compare_df["model_year"], errors="coerce"
        )

        compare_df = compare_df.dropna(
            subset=["make", "model", "price_inr"]
        ).reset_index(drop=True)

        def car_label(index):
            car = compare_df.iloc[index]

            return (
                f"{car['make']} | {car['model']} | "
                f"{car['variant']} | {car['city_name']} | "
                f"₹{car['price_inr']:,.0f}"
            )

        st.info(
            "Vehicle comparisons use advertised listing information "
            "from the 24 August 2026 dataset snapshot."
        )

        st.subheader("Select Vehicles")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Vehicle A")

            car_a_index = st.selectbox(
                "Select First Vehicle",
                options=list(range(len(compare_df))),
                index=0,
                format_func=car_label,
                key="compare_car_a"
            )

        with col2:
            st.markdown("### Vehicle B")

            default_b = 1 if len(compare_df) > 1 else 0

            car_b_index = st.selectbox(
                "Select Second Vehicle",
                options=list(range(len(compare_df))),
                index=default_b,
                format_func=car_label,
                key="compare_car_b"
            )

        if car_a_index == car_b_index:

            st.warning(
                "Please select two different listings to compare."
            )

        else:

            car_a = compare_df.iloc[car_a_index]
            car_b = compare_df.iloc[car_b_index]

            st.divider()
            st.subheader("Vehicle Comparison")

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("### 🔵 Vehicle A")
                st.markdown(f"## {car_a['make']} {car_a['model']}")
                st.caption(str(car_a["variant"]))

            with col_b:
                st.markdown("### 🟢 Vehicle B")
                st.markdown(f"## {car_b['make']} {car_b['model']}")
                st.caption(str(car_b["variant"]))

            # -----------------------------------------
            # PRICE COMPARISON
            # -----------------------------------------

            st.divider()
            st.subheader("1. Asking Price Comparison")

            price_a = car_a["price_inr"]
            price_b = car_b["price_inr"]

            price_col1, price_col2 = st.columns(2)

            price_col1.metric(
                "Vehicle A Price",
                f"₹{price_a:,.0f}"
            )

            price_col2.metric(
                "Vehicle B Price",
                f"₹{price_b:,.0f}",
                delta=f"₹{price_b - price_a:,.0f}"
            )

            # -----------------------------------------
            # SPECIFICATION COMPARISON
            # -----------------------------------------

            st.divider()
            st.subheader("2. Technical Specifications")

            comparison_table = pd.DataFrame({
                "Specification": [
                    "Brand",
                    "Model",
                    "Variant",
                    "Manufacturing Year",
                    "Kilometres Driven",
                    "Fuel Type",
                    "Transmission",
                    "Number of Owners",
                    "City",
                    "Seller Type",
                    "Asking Price"
                ],

                "Vehicle A": [
                    car_a["make"],
                    car_a["model"],
                    car_a["variant"],
                    car_a["model_year"],
                    car_a["km"],
                    car_a["fuel_type"],
                    car_a["transmission"],
                    car_a["owners"],
                    car_a["city_name"],
                    car_a["seller_name"],
                    f"₹{price_a:,.0f}"
                ],

                "Vehicle B": [
                    car_b["make"],
                    car_b["model"],
                    car_b["variant"],
                    car_b["model_year"],
                    car_b["km"],
                    car_b["fuel_type"],
                    car_b["transmission"],
                    car_b["owners"],
                    car_b["city_name"],
                    car_b["seller_name"],
                    f"₹{price_b:,.0f}"
                ]
            })

            st.dataframe(
                comparison_table,
                use_container_width=True,
                hide_index=True
            )

            # -----------------------------------------
            # VISUAL PRICE COMPARISON
            # -----------------------------------------

            st.divider()
            st.subheader("3. Visual Price Comparison")

            chart_df = pd.DataFrame({
                "Vehicle": ["Vehicle A", "Vehicle B"],
                "Asking Price": [price_a, price_b]
            })

            price_comparison_chart = px.bar(
                chart_df,
                x="Vehicle",
                y="Asking Price",
                color="Vehicle",
                text_auto=".2s",
                title="Comparison of Advertised Prices"
            )

            price_comparison_chart.update_layout(
                template="plotly_dark",
                showlegend=False,
                yaxis_title="Asking Price (INR)"
            )

            st.plotly_chart(
                price_comparison_chart,
                use_container_width=True
            )

            # -----------------------------------------
            # COMPARISON SUMMARY
            # -----------------------------------------

            st.divider()
            st.subheader("4. Comparison Summary")

            if price_a < price_b:
                st.success(
                    f"Vehicle A has the lower advertised price "
                    f"by ₹{price_b - price_a:,.0f}."
                )

            elif price_b < price_a:
                st.success(
                    f"Vehicle B has the lower advertised price "
                    f"by ₹{price_a - price_b:,.0f}."
                )

            else:
                st.info("Both vehicles have the same advertised price.")

            st.caption(
                "A lower asking price does not automatically mean "
                "better value. Consider vehicle condition, service "
                "history, ownership, location and inspection results."
            )

    except FileNotFoundError:
        st.error(
            "Recent vehicle dataset not found. "
            "Check data/raw/recent_used_listings.parquet."
        )

    except Exception as error:
        st.error(f"Comparison error: {error}")


# =====================================================
# 8. MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.title("📈 Model Performance Evaluation")

    st.write(
        "Evaluate and compare the machine learning algorithms "
        "used for predicting used-car resale prices."
    )

    DATA_DIR = BASE_DIR / "data" / "processed"

    comparison_path = DATA_DIR / "final_model_comparison.csv"
    validation_path = DATA_DIR / "model_validation_results.csv"

    try:

        comparison_df = pd.read_csv(comparison_path)
        validation_df = pd.read_csv(validation_path)

        st.success("Model evaluation results loaded successfully.")

        st.info(
            "This is a regression problem. MAE, RMSE and R² "
            "are used to evaluate model performance."
        )

        # 1. MODEL COMPARISON TABLE

        st.divider()
        st.subheader("1. Algorithm Comparison")

        st.dataframe(
            comparison_df.style.format({
                "MAE": "₹{:,.2f}",
                "RMSE": "₹{:,.2f}",
                "R2": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        # 2. SELECTED MODEL

        best_model = comparison_df.sort_values(
            "MAE"
        ).iloc[0]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Selected Model",
            str(best_model["Model"])
        )

        col2.metric(
            "Test MAE",
            f"₹{best_model['MAE']:,.0f}"
        )

        col3.metric(
            "Test R²",
            f"{best_model['R2']:.4f}"
        )

        # 3. MAE COMPARISON

        st.divider()
        st.subheader("2. Mean Absolute Error (MAE)")

        mae_chart = px.bar(
            comparison_df.sort_values("MAE"),
            x="Model",
            y="MAE",
            title="MAE Comparison - Lower is Better",
            text_auto=".3s"
        )

        mae_chart.update_layout(
            template="plotly_dark",
            xaxis_tickangle=-25,
            yaxis_title="MAE (INR)"
        )

        st.plotly_chart(
            mae_chart,
            use_container_width=True
        )

        st.caption(
            "MAE measures the average absolute difference "
            "between actual and predicted prices."
        )

        # 4. RMSE COMPARISON

        st.divider()
        st.subheader("3. Root Mean Squared Error (RMSE)")

        rmse_chart = px.bar(
            comparison_df.sort_values("RMSE"),
            x="Model",
            y="RMSE",
            title="RMSE Comparison - Lower is Better",
            text_auto=".3s"
        )

        rmse_chart.update_layout(
            template="plotly_dark",
            xaxis_tickangle=-25,
            yaxis_title="RMSE (INR)"
        )

        st.plotly_chart(
            rmse_chart,
            use_container_width=True
        )

        st.caption(
            "RMSE penalizes larger prediction errors more heavily."
        )

        # 5. R2 COMPARISON

        st.divider()
        st.subheader("4. R² Score Comparison")

        r2_chart = px.bar(
            comparison_df.sort_values("R2", ascending=False),
            x="Model",
            y="R2",
            title="R² Score Comparison - Higher is Better",
            text_auto=".4f"
        )

        r2_chart.update_layout(
            template="plotly_dark",
            xaxis_tickangle=-25,
            yaxis_title="R² Score"
        )

        st.plotly_chart(
            r2_chart,
            use_container_width=True
        )

        st.caption(
            "R² indicates how much variation in vehicle prices "
            "is explained by the model on the test dataset."
        )

        # 6. CROSS VALIDATION

        st.divider()
        st.subheader("5. Cross-Validation Results")

        st.write(
            "Three-fold cross-validation results on the training data."
        )

        st.dataframe(
            validation_df.style.format({
                "Train MAE": "₹{:,.2f}",
                "Test MAE": "₹{:,.2f}",
                "Train RMSE": "₹{:,.2f}",
                "Test RMSE": "₹{:,.2f}",
                "Train R2": "{:.4f}",
                "Test R2": "{:.4f}",
                "CV MAE": "₹{:,.2f}",
                "CV RMSE": "₹{:,.2f}",
                "CV R2": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        # 7. TRAINING VS TESTING

        st.divider()
        st.subheader("6. Training vs Testing Performance")

        training_test_df = validation_df[
            ["Model", "Train R2", "Test R2"]
        ].melt(
            id_vars="Model",
            var_name="Dataset",
            value_name="R2 Score"
        )

        training_chart = px.bar(
            training_test_df,
            x="Model",
            y="R2 Score",
            color="Dataset",
            barmode="group",
            title="Training vs Testing R² Scores",
            text_auto=".3f"
        )

        training_chart.update_layout(
            template="plotly_dark",
            xaxis_tickangle=-25
        )

        st.plotly_chart(
            training_chart,
            use_container_width=True
        )

        # 8. FINAL MODEL SELECTION

        st.divider()
        st.subheader("7. Final Model Selection")

        st.success(
            "Random Forest (Tuned) was selected as the final model. "
            "It achieved the lowest test MAE and RMSE and the "
            "highest test R² among the six evaluated models."
        )

        st.markdown("""
        **Why Random Forest?**

        - Combines predictions from multiple decision trees.
        - Captures nonlinear relationships in vehicle data.
        - Handles interactions between numerical and categorical features.
        - Achieved the strongest test-set results in our comparison.
        - Its trained pipeline is saved for the prediction system.
        """)

        st.warning(
            "The test set was used during model comparison and tuning. "
            "These results should therefore not be described as a "
            "completely untouched independent benchmark."
        )

    except FileNotFoundError as error:

        st.error(
            f"Required evaluation file not found: {error}"
        )

    except Exception as error:

        st.error(f"Model Performance error: {error}")

# =====================================================
# 9. EXPLAINABLE AI
# =====================================================

elif page == "Explainable AI":

    st.title("🧠 Explainable AI")

    st.write(
        "Understand how different vehicle characteristics "
        "influence the predictions made by AutoValue AI."
    )

    # ---------------------------------------------
    # 1. SHAP SUMMARY PLOT
    # ---------------------------------------------

    st.divider()
    st.subheader("1. SHAP Summary Plot")

    shap_path = BASE_DIR / "assets" / "shap_summary.png"

    if shap_path.exists():

        st.image(
            str(shap_path),
            caption="SHAP Summary Plot - Random Forest",
            use_container_width=True
        )

    else:
        st.warning("SHAP summary image not found.")

    st.info(
        "Each point represents a vehicle. Red indicates a higher "
        "feature value and blue indicates a lower feature value. "
        "Positive SHAP values increase the model prediction, "
        "while negative SHAP values decrease it."
    )

    # ---------------------------------------------
    # 2. SHAP FEATURE IMPORTANCE
    # ---------------------------------------------

    st.divider()
    st.subheader("2. SHAP Feature Importance")

    shap_csv_path = (
        BASE_DIR / "data" / "processed" /
        "shap_feature_importance.csv"
    )

    if shap_csv_path.exists():

        shap_df = pd.read_csv(shap_csv_path)

        shap_df["mean_absolute_shap"] = (
            shap_df["mean_absolute_shap"].round(2)
        )

        st.dataframe(
            shap_df.head(15),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning("SHAP feature importance data not found.")

    # ---------------------------------------------
    # 3. RANDOM FOREST FEATURE IMPORTANCE
    # ---------------------------------------------

    st.divider()
    st.subheader("3. Random Forest Feature Importance")

    importance_path = (
        BASE_DIR / "assets" / "feature_importance.png"
    )

    if importance_path.exists():

        st.image(
            str(importance_path),
            caption="Random Forest Feature Importance",
            use_container_width=True
        )

    else:
        st.warning("Feature importance image not found.")

    st.caption(
        "Note: SHAP values and feature importance describe "
        "model behaviour. They do not establish causal relationships."
    )


# =====================================================
# 10. VALUATION REPORT
# =====================================================

elif page == "Valuation Report":

    st.title("📄 Vehicle Valuation Report")

    st.write(
        "Generate and download a professional PDF report "
        "containing your vehicle details and estimated resale value."
    )

    st.divider()

    if (
        "last_prediction" in st.session_state
        and "last_vehicle_data" in st.session_state
    ):

        prediction = st.session_state["last_prediction"]
        vehicle_data = st.session_state["last_vehicle_data"]

        st.subheader("Latest Vehicle Valuation")

        col1, col2 = st.columns(2)

        col1.metric(
            "Estimated Vehicle Value",
            f"₹{prediction:,.0f}"
        )

        col2.metric(
            "Manufacturing Year",
            int(vehicle_data.iloc[0]["year"])
        )

        st.subheader("Vehicle Details")

        st.dataframe(
            vehicle_data.T.rename(columns={0: "Vehicle Details"}),
            use_container_width=True
        )

        st.divider()

        st.subheader("Download Your Report")

        try:

            pdf_bytes = generate_valuation_pdf(
                vehicle_data,
                prediction
            )

            st.download_button(
                label="📥 Download Valuation Report (PDF)",
                data=pdf_bytes,
                file_name="AutoValue_AI_Valuation_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

            st.success("Your valuation report is ready!")

        except Exception as error:

            st.error(f"PDF generation error: {error}")

    else:

        st.warning(
            "No vehicle prediction is available yet. "
            "Please visit Price Prediction, enter vehicle details "
            "and generate a prediction first."
        )

        st.info(
            "After generating a prediction, return here "
            "to download your valuation report."
        )
