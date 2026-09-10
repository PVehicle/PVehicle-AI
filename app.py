"""Giao dien web cho he thong nhan dien va tu van xe (Streamlit).

Chay bang:
    .\\.venv\\Scripts\\streamlit.exe run app.py
"""

import cv2
import numpy as np
import requests
import streamlit as st

from src.cv.pipeline import (
    CLASS_NAMES_FILE,
    CLASSIFIER_MODEL,
    DETECTOR_MODEL,
    RecognitionPipeline,
    draw_detections,
    load_image,
)
from src.rec.recommender import CarRecommender
from src.utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

PAGE_TITLE = "PVehicle-AI"
REQUEST_TIMEOUT = 15  # giay
MAX_DOWNLOAD_MB = 20


@st.cache_resource
def get_pipeline() -> RecognitionPipeline | None:
    """Nap mo hinh mot lan roi dung lai cho moi lan tuong tac."""
    try:
        return RecognitionPipeline()
    except FileNotFoundError as exc:
        logger.error("Khong nap duoc mo hinh: %s", exc)
        return None


@st.cache_resource
def get_recommender() -> CarRecommender:
    return CarRecommender()


def fetch_image_from_url(url: str) -> bytes:
    """Tai anh tu URL, gioi han dung luong de tranh tai file qua lon."""
    response = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        raise ValueError(
            f"URL khong tro toi anh (Content-Type: {content_type})"
        )

    data = b""
    for chunk in response.iter_content(8192):
        data += chunk
        if len(data) > MAX_DOWNLOAD_MB * 1024 * 1024:
            raise ValueError(f"Anh vuot qua {MAX_DOWNLOAD_MB} MB.")
    return data


def to_rgb(image: np.ndarray) -> np.ndarray:
    """OpenCV dung BGR, Streamlit hien thi RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def format_price(million_vnd: float) -> str:
    """Dinh dang gia cho de doc: 1.250 tr hoac 1,25 ty."""
    if million_vnd >= 1000:
        return f"{million_vnd / 1000:.2f} ty".replace(".", ",")
    return f"{million_vnd:,.0f} tr".replace(",", ".")


def show_recommendations(items, empty_message: str) -> None:
    """Hien thi danh sach xe duoc goi y duoi dang bang."""
    if not items:
        st.info(empty_message)
        return

    st.dataframe(
        [
            {
                "Dong xe": item.class_name,
                "Kieu dang": item.body_style,
                "So cho": item.seats,
                "Gia": format_price(item.price_million_vnd),
                "Tieu hao": f"{item.fuel_l_per_100km:.1f} L/100km",
                "Phu hop": f"{item.score:.0%}",
            }
            for item in items
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_missing_models() -> None:
    """Man hinh huong dan khi chua co mo hinh."""
    st.error("Chua co mo hinh trong thu muc `models/`")

    missing = [
        path.name
        for path in (DETECTOR_MODEL, CLASSIFIER_MODEL, CLASS_NAMES_FILE)
        if not path.exists()
    ]
    st.write("**Cac file con thieu:**")
    for name in missing:
        st.write(f"- `{name}`")

    st.markdown(
        """
        **Cach khac phuc:**

        1. Huan luyen mo hinh bang `notebooks/train_classifier_colab.ipynb`
           tren Google Colab, roi chep `car_classifier.onnx` va
           `class_names.json` vao thu muc `models/`.
        2. Hoac de thu giao dien truoc khi co mo hinh that, sinh mo hinh
           gia (ket qua du doan vo nghia):

        ```powershell
        .\\.venv\\Scripts\\python.exe scripts/make_dummy_models.py
        ```
        """
    )


def render_recognition_tab(pipeline, recommender) -> None:
    """Tab 1: nhan dien xe tu anh."""
    st.subheader("Nhan dien dong xe tu hinh anh")

    source = st.radio(
        "Nguon anh",
        ["Tai anh len", "Dan URL"],
        horizontal=True,
        label_visibility="collapsed",
    )

    image_bytes = None
    if source == "Tai anh len":
        uploaded = st.file_uploader(
            "Chon anh xe", type=["jpg", "jpeg", "png", "bmp", "webp"]
        )
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
    else:
        url = st.text_input("Dan duong dan anh", placeholder="https://...")
        if url:
            try:
                with st.spinner("Dang tai anh..."):
                    image_bytes = fetch_image_from_url(url)
            except (requests.RequestException, ValueError) as exc:
                st.error(f"Khong tai duoc anh: {exc}")

    if image_bytes is None:
        st.info("Chon mot anh de bat dau nhan dien.")
        return

    try:
        image = load_image(image_bytes)
    except ValueError as exc:
        st.error(str(exc))
        return

    with st.spinner("Dang nhan dien..."):
        results = pipeline.recognize(image)

    left, right = st.columns([3, 2])
    with left:
        st.image(
            to_rgb(draw_detections(image, results)),
            caption="Ket qua nhan dien",
            use_container_width=True,
        )

    with right:
        if results and results[0].is_likely_not_a_car:
            st.error(
                "**Nhieu kha nang anh nay khong chua o to.**\\n\\n"
                "Khong phat hien duoc xe nao, va mo hinh phan loai cung "
                "khong chac chan ve ket qua. Hay thu anh khac."
            )
        elif results and results[0].is_whole_image:
            st.warning(
                "Khong tim thay xe trong anh — dang phan loai toan bo "
                "buc anh. Ket qua co the khong chinh xac."
            )
        st.metric("So xe phat hien", len(results))

    for index, result in enumerate(results, start=1):
        if result.is_confident:
            header = f"Xe {index}: **{result.best.class_name}**"
        else:
            header = f"Xe {index}: ket qua khong chac chan"

        with st.expander(header, expanded=(index == 1)):
            render_single_result(result, recommender)


def render_single_result(result, recommender) -> None:
    """Hien thi chi tiet cho mot xe da nhan dien."""
    crop_col, info_col = st.columns([1, 2])

    with crop_col:
        st.image(to_rgb(result.crop), use_container_width=True)

    with info_col:
        # Cho biet ket qua den tu mo hinh nao — hai mo hinh phu hai tap
        # xe khac nhau nen nguoi dung can biet de danh gia.
        if result.source == "vietnam":
            st.caption("🇻🇳 Mo hinh xe thi truong Viet Nam (20 dong)")
        else:
            st.caption("🌍 Mo hinh xe quoc te (196 dong, doi ≤2012)")

        if not result.is_confident:
            st.warning(
                f"Do tin cay cao nhat chi {result.best.confidence:.1%} — "
                "duoi nguong dang tin. Ket qua duoi day chi de tham khao."
            )
        st.write("**Cac kha nang cao nhat:**")
        for prediction in result.predictions:
            label = (
                f"{prediction.class_name} — "
                f"{prediction.confidence:.1%}"
            )
            st.progress(prediction.confidence, text=label)

        if result.alternative:
            other = (
                "xe quoc te" if result.source == "vietnam"
                else "xe Viet Nam"
            )
            with st.expander(f"Mo hinh {other} doan gi?"):
                for prediction in result.alternative[:3]:
                    st.write(
                        f"- {prediction.class_name} — "
                        f"{prediction.confidence:.1%}"
                    )

    specs = recommender.get_car(result.best.class_name)
    if specs is None:
        st.info("Khong co thong so cho dong xe nay.")
        return

    st.divider()
    st.write("**Thong so ky thuat**")
    cols = st.columns(4)
    cols[0].metric("Kieu dang", specs["body_style"])
    cols[1].metric("So cho", int(specs["seats"]))
    cols[2].metric("Gia tham khao",
                   format_price(specs["price_million_vnd"]))
    cols[3].metric("Tieu hao", f"{specs['fuel_l_per_100km']:.1f} L")

    st.caption(
        "Gia ban va muc tieu hao la **so lieu mo phong** — xem "
        "`docs/car_specs_generation.md`."
    )

    st.write("**Cac xe tuong tu:**")
    show_recommendations(
        recommender.recommend_similar(result.best.class_name),
        "Khong tim duoc xe tuong tu.",
    )


def render_advisor_tab(recommender) -> None:
    """Tab 2: tu van xe theo nhu cau."""
    st.subheader("Tim xe phu hop voi nhu cau")

    min_price, max_price = recommender.price_range

    col_left, col_right = st.columns(2)
    with col_left:
        budget = st.slider(
            "Ngan sach toi da (trieu VND)",
            min_value=int(min_price),
            max_value=int(max_price),
            value=1000,
            step=50,
        )
        seats = st.select_slider(
            "So cho toi thieu",
            options=[2, 4, 5, 7, 8],
            value=5,
        )

    with col_right:
        styles = st.multiselect(
            "Kieu dang mong muon",
            options=recommender.body_styles,
            default=[],
            help="De trong de xet tat ca kieu dang",
        )
        fuel_limit = st.slider(
            "Muc tieu hao toi da (L/100km)",
            min_value=5.0,
            max_value=20.0,
            value=12.0,
            step=0.5,
        )

    top_n = st.slider("So goi y", min_value=3, max_value=15, value=5)

    if not st.button("Tim xe phu hop", type="primary"):
        return

    results = recommender.recommend_by_needs(
        max_price=budget,
        min_seats=seats,
        body_styles=styles or None,
        max_fuel=fuel_limit,
        top_n=top_n,
    )
    show_recommendations(
        results,
        "Khong co xe nao thoa man. Hay noi long dieu kien loc.",
    )


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🚗", layout="wide")
    st.title("🚗 PVehicle-AI")
    st.caption("Nhan dien dong xe qua hinh anh va tu van xe phu hop")

    pipeline = get_pipeline()
    recommender = get_recommender()

    tab_recognize, tab_advise = st.tabs(["Nhan dien xe", "Tu van xe"])

    with tab_recognize:
        if pipeline is None:
            render_missing_models()
        else:
            render_recognition_tab(pipeline, recommender)

    with tab_advise:
        # Tab tu van chi can bang thong so, chay duoc ke ca khi thieu model.
        render_advisor_tab(recommender)


if __name__ == "__main__":
    main()
