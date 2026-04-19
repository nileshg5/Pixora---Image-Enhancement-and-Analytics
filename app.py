import io
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter

try:
    import cv2  # type: ignore

    CV2_AVAILABLE = True
    CV2_IMPORT_ERROR = ""
except Exception as import_error:
    cv2 = None
    CV2_AVAILABLE = False
    CV2_IMPORT_ERROR = str(import_error)


st.set_page_config(page_title="Pixora", layout="wide")


# ----------------------------
# Helper functions
# ----------------------------
def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def rgb_array_to_pil(image_array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))


def apply_enhancements(
    image: Image.Image,
    brightness: float,
    contrast: float,
    sharpness: float,
    blur_strength: int,
    denoise_strength: int,
) -> Image.Image:
    enhanced = image.convert("RGB")
    enhanced = ImageEnhance.Brightness(enhanced).enhance(brightness)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(contrast)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(sharpness)

    if blur_strength > 0:
        if CV2_AVAILABLE:
            image_array = pil_to_rgb_array(enhanced)
            k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
            image_array = cv2.GaussianBlur(image_array, (k, k), 0)
            enhanced = rgb_array_to_pil(image_array)
        else:
            enhanced = enhanced.filter(ImageFilter.GaussianBlur(radius=max(1, blur_strength // 2)))

    if denoise_strength > 0 and CV2_AVAILABLE:
        image_array = pil_to_rgb_array(enhanced)
        # Keep API simple: use a single slider for all strength params.
        image_array = cv2.fastNlMeansDenoisingColored(
            image_array,
            None,
            denoise_strength,
            denoise_strength,
            7,
            21,
        )
        enhanced = rgb_array_to_pil(image_array)

    return enhanced


def apply_transformations(
    image: Image.Image,
    angle: int,
    flip_mode: str,
    width: int,
    height: int,
    to_grayscale: bool,
) -> Image.Image:
    transformed = image.convert("RGB")

    if angle % 360 != 0:
        transformed = transformed.rotate(angle, expand=True)

    if flip_mode == "Horizontal":
        transformed = transformed.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip_mode == "Vertical":
        transformed = transformed.transpose(Image.FLIP_TOP_BOTTOM)

    if width > 0 and height > 0:
        transformed = transformed.resize((width, height))

    if to_grayscale:
        transformed = transformed.convert("L")

    return transformed


def apply_filter(image: Image.Image, filter_name: str, threshold_val: int) -> Image.Image:
    rgb = pil_to_rgb_array(image)

    if filter_name == "Edge Detection (Canny)":
        if CV2_AVAILABLE:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            return Image.fromarray(edges)
        return image.convert("L").filter(ImageFilter.FIND_EDGES)

    if filter_name == "Sepia":
        sepia_matrix = np.array(
            [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]
        )
        sepia = rgb @ sepia_matrix.T
        return rgb_array_to_pil(np.clip(sepia, 0, 255))

    if filter_name == "Binary Threshold":
        gray = np.array(image.convert("L"))
        binary = np.where(gray > threshold_val, 255, 0).astype(np.uint8)
        return Image.fromarray(binary)

    return image


def get_image_metadata(image: Image.Image) -> dict:
    image_array = np.array(image)
    rgb_array = pil_to_rgb_array(image)
    mean_vals = rgb_array.reshape(-1, 3).mean(axis=0)

    return {
        "size": image.size,
        "mode": image.mode,
        "dtype": str(image_array.dtype),
        "mean_r": float(mean_vals[0]),
        "mean_g": float(mean_vals[1]),
        "mean_b": float(mean_vals[2]),
    }


def figure_rgb_histogram(image: Image.Image):
    rgb = pil_to_rgb_array(image)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["red", "green", "blue"]
    for idx, color in enumerate(colors):
        hist, _ = np.histogram(rgb[:, :, idx].ravel(), bins=256, range=(0, 256))
        ax.plot(hist, color=color, label=f"{color.upper()}")
    ax.set_title("RGB Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    return fig


def figure_intensity_histogram(image: Image.Image):
    gray = np.array(image.convert("L"))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(gray.ravel(), bins=256, range=(0, 256), color="gray")
    ax.set_title("Intensity Histogram")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def figure_heatmap(image: Image.Image):
    gray = np.array(image.convert("L"))
    fig, ax = plt.subplots(figsize=(6, 5))
    heat = ax.imshow(gray, cmap="hot")
    ax.set_title("2D Heatmap of Pixel Intensities")
    fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def convert_image_to_pdf_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.convert("RGB").save(output, format="PDF")
    output.seek(0)
    return output.read()


def convert_pdf_first_page_to_image(pdf_bytes: bytes) -> Tuple[Optional[Image.Image], str]:
    # Option 1: pdf2image (if installed)
    try:
        from pdf2image import convert_from_bytes  # type: ignore

        pages = convert_from_bytes(pdf_bytes, first_page=1, last_page=1)
        if pages:
            return pages[0].convert("RGB"), "Converted using pdf2image"
    except Exception:
        pass

    # Option 2: PyMuPDF / fitz (if installed)
    try:
        import fitz  # type: ignore

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(doc) > 0:
            page = doc.load_page(0)
            pix = page.get_pixmap()
            mode = "RGBA" if pix.alpha else "RGB"
            pil_image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            doc.close()
            return pil_image.convert("RGB"), "Converted using PyMuPDF (fitz)"
        doc.close()
    except Exception:
        pass

    message = (
        "PDF to image conversion library not available. Install `pdf2image` (and poppler) "
        "or `PyMuPDF` to enable this feature."
    )
    return None, message


def init_enhancement_state():
    defaults = {
        "brightness": 1.0,
        "contrast": 1.0,
        "sharpness": 1.0,
        "blur": 0,
        "denoise": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_enhancement_state():
    st.session_state["brightness"] = 1.0
    st.session_state["contrast"] = 1.0
    st.session_state["sharpness"] = 1.0
    st.session_state["blur"] = 0
    st.session_state["denoise"] = 0


# ----------------------------
# Main app
# ----------------------------
st.title("Pixora")
st.caption("A simple Streamlit app for basic image editing, analysis, and image/PDF conversion.")

if not CV2_AVAILABLE:
    st.warning(
        "OpenCV could not be loaded in this environment. "
        "Some features use fallback behavior (blur, edge detection, thresholding). "
        f"Details: {CV2_IMPORT_ERROR}"
    )

uploaded_image_file = st.file_uploader(
    "Upload an image (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"]
)

base_pil_image: Optional[Image.Image] = None
base_np_image: Optional[np.ndarray] = None

if uploaded_image_file is not None:
    try:
        base_pil_image = Image.open(uploaded_image_file)
        base_np_image = np.array(base_pil_image)
    except Exception as err:
        st.error(f"Could not read uploaded image: {err}")

upload_tab, enhancements_tab, transform_tab, filter_tab, analytics_tab, pdf_tab = st.tabs(
    ["Upload", "Enhancements", "Transformations", "Filters", "Analytics", "PDF Tools"]
)

with upload_tab:
    st.subheader("Upload Preview")
    if base_pil_image is None:
        st.info("Please upload an image to begin.")
    else:
        st.image(base_pil_image, caption="Uploaded Image", use_container_width=True)
        st.write(f"PIL mode: {base_pil_image.mode}")
        st.write(f"NumPy shape: {base_np_image.shape if base_np_image is not None else 'N/A'}")

with enhancements_tab:
    st.subheader("Basic Enhancements")

    if base_pil_image is None:
        st.info("Upload an image first.")
    else:
        init_enhancement_state()

        col_controls, col_preview = st.columns([1, 2])
        with col_controls:
            st.slider("Brightness", 0.0, 3.0, key="brightness", step=0.1)
            st.slider("Contrast", 0.0, 3.0, key="contrast", step=0.1)
            st.slider("Sharpness", 0.0, 3.0, key="sharpness", step=0.1)
            st.slider("Blur (Gaussian)", 0, 31, key="blur", step=1)
            st.slider("Noise Removal", 0, 30, key="denoise", step=1)
            if not CV2_AVAILABLE:
                st.caption("Noise removal requires OpenCV and is disabled in fallback mode.")
            if st.button("Reset to Original", use_container_width=True):
                reset_enhancement_state()

        enhanced_image = apply_enhancements(
            base_pil_image,
            st.session_state["brightness"],
            st.session_state["contrast"],
            st.session_state["sharpness"],
            st.session_state["blur"],
            st.session_state["denoise"],
        )

        with col_preview:
            left, right = st.columns(2)
            with left:
                st.image(base_pil_image, caption="Before", use_container_width=True)
            with right:
                st.image(enhanced_image, caption="After", use_container_width=True)

with transform_tab:
    st.subheader("Basic Transformations")

    if base_pil_image is None:
        st.info("Upload an image first.")
    else:
        col_controls, col_preview = st.columns([1, 2])

        with col_controls:
            angle = st.slider("Rotate (degrees)", 0, 360, 0)
            flip_mode = st.selectbox("Flip", ["None", "Horizontal", "Vertical"])
            current_w, current_h = base_pil_image.size
            resize_w = st.number_input("Resize Width", min_value=1, value=current_w, step=1)
            resize_h = st.number_input("Resize Height", min_value=1, value=current_h, step=1)
            to_gray = st.checkbox("Convert to Grayscale", value=False)

        transformed_image = apply_transformations(
            base_pil_image, angle, flip_mode, int(resize_w), int(resize_h), to_gray
        )

        with col_preview:
            left, right = st.columns(2)
            with left:
                st.image(base_pil_image, caption="Before", use_container_width=True)
            with right:
                st.image(transformed_image, caption="After", use_container_width=True)

with filter_tab:
    st.subheader("Basic Filters")

    if base_pil_image is None:
        st.info("Upload an image first.")
    else:
        col_controls, col_preview = st.columns([1, 2])

        with col_controls:
            filter_name = st.selectbox(
                "Select Filter",
                ["None", "Edge Detection (Canny)", "Sepia", "Binary Threshold"],
            )
            threshold_value = st.slider("Threshold (for Binary)", 0, 255, 127)

        filtered_image = apply_filter(base_pil_image, filter_name, threshold_value)

        with col_preview:
            left, right = st.columns(2)
            with left:
                st.image(base_pil_image, caption="Before", use_container_width=True)
            with right:
                st.image(filtered_image, caption="After", use_container_width=True)

with analytics_tab:
    st.subheader("Image Analytics & Graphs")

    if base_pil_image is None:
        st.info("Upload an image first.")
    else:
        metadata = get_image_metadata(base_pil_image)
        st.write("**Metadata**")
        st.write(f"Size: {metadata['size']}")
        st.write(f"Mode: {metadata['mode']}")
        st.write(f"Data type: {metadata['dtype']}")
        st.write(
            "Mean pixel values (R, G, B): "
            f"({metadata['mean_r']:.2f}, {metadata['mean_g']:.2f}, {metadata['mean_b']:.2f})"
        )

        fig1 = figure_rgb_histogram(base_pil_image)
        st.pyplot(fig1)
        plt.close(fig1)

        fig2 = figure_intensity_histogram(base_pil_image)
        st.pyplot(fig2)
        plt.close(fig2)

        fig3 = figure_heatmap(base_pil_image)
        st.pyplot(fig3)
        plt.close(fig3)

with pdf_tab:
    st.subheader("Simple PDF Tools")

    st.markdown("### 1) Image → PDF")
    if base_pil_image is None:
        st.info("Upload an image to enable Image → PDF conversion.")
    else:
        try:
            pdf_bytes = convert_image_to_pdf_bytes(base_pil_image)
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name="converted_image.pdf",
                mime="application/pdf",
            )
        except Exception as err:
            st.error(f"Image to PDF conversion failed: {err}")

    st.markdown("---")
    st.markdown("### 2) PDF → Image (first page)")

    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader")
    if uploaded_pdf is not None:
        try:
            first_page_image, status_message = convert_pdf_first_page_to_image(uploaded_pdf.read())
            if first_page_image is not None:
                st.success(status_message)
                st.image(first_page_image, caption="First Page as Image", use_container_width=True)

                img_buffer = io.BytesIO()
                first_page_image.save(img_buffer, format="PNG")
                st.download_button(
                    "Download First Page (PNG)",
                    data=img_buffer.getvalue(),
                    file_name="pdf_first_page.png",
                    mime="image/png",
                )
            else:
                st.warning(status_message)
        except Exception as err:
            st.error(f"PDF to image conversion failed: {err}")
