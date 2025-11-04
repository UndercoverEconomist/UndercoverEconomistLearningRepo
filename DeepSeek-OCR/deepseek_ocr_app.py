import streamlit as st
from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image, ImageOps
import os
import tempfile

st.set_page_config(page_title="DeepSeek OCR", layout="wide")

# === Configuration (Fixed Settings) ===
# Use original image size and no cropping/compression for best OCR results
crop_mode = False
test_compress = False

# === Header ===
st.title("🧠 DeepSeek-OCR — Full Document OCR")

# === Model Loading ===
@st.cache_resource(show_spinner=True)
def load_model():
    model_name = "deepseek-ai/DeepSeek-OCR"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True,
        _attn_implementation="flash_attention_2"
    )
    model = model.eval().cuda().to(torch.bfloat16)
    return tokenizer, model

tokenizer, model = load_model()

# Display model info
st.sidebar.write("✅ Model loaded successfully")
st.sidebar.write(f"Model: {model.__class__.__name__}")
st.sidebar.write(f"Device: {next(model.parameters()).device}")

# === File Upload ===
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image and automatically fix orientation based on EXIF data
    image = Image.open(uploaded_file)
    
    # Use ImageOps.exif_transpose to automatically handle EXIF orientation
    image = ImageOps.exif_transpose(image)
    
    # Always make the image vertical (portrait orientation)
    if image.width > image.height:
        image = image.rotate(90, expand=True)
        st.info("📱 Image rotated to vertical orientation for better OCR")
    
    # Convert to RGB while preserving orientation
    image = image.convert("RGB")
    
    st.image(image, caption="Uploaded Image (Vertical Orientation)", use_container_width=True)
    
    # Display image dimensions
    st.write(f"Image dimensions: {image.width} x {image.height} pixels")

    # Try different prompt formats
    prompt_options = {
        "Basic OCR": "<image>\nExtract all text content from this image.",
        "Detailed OCR": "<image>\nPlease read all the text in this image and return it exactly as it appears.",
        "Grounding OCR": "<image>\n<|grounding|>Extract all text from this document.",
        "Custom": ""
    }
    
    prompt_choice = st.selectbox("Choose prompt type:", list(prompt_options.keys()))
    
    if prompt_choice == "Custom":
        prompt = st.text_area("Custom Prompt", "<image>\n", height=80)
    else:
        prompt = st.text_area("Prompt", prompt_options[prompt_choice], height=80)

    if st.button("🔍 Run OCR", use_container_width=True):
        with st.spinner("Running DeepSeek-OCR on entire image..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                # Save image at high quality to preserve text clarity
                image.save(tmp.name, quality=95, optimize=False)
                output_dir = tempfile.mkdtemp()
                print(f"Created output directory: {output_dir}")
                print(f"Saved image to: {tmp.name}")

                # Debug: Print all parameters being passed
                print(f"Prompt: {prompt}")
                print(f"Image file: {tmp.name}")
                print(f"Output path: {output_dir}")
                print(f"Crop mode: {crop_mode}")
                print(f"Test compress: {test_compress}")
                
                try:
                    print("Running DeepSeek OCR inference...")
                    
                    # Let's try to bypass the stupid output_path requirement
                    # First, try without save_results to see if it returns text directly
                    res = model.infer(
                        tokenizer,
                        prompt=prompt,
                        image_file=tmp.name,
                        output_path=output_dir,
                        save_results=False  # Don't save files, just return text
                    )
                    
                    print(f"Inference result (save_results=False): {res}")
                    
                    # If that doesn't work, try the standard way but grab files immediately
                    if not res:
                        print("Trying with save_results=True...")
                        res = model.infer(
                            tokenizer,
                            prompt=prompt,
                            image_file=tmp.name,
                            output_path=output_dir,
                            save_results=True
                        )
                        print(f"Inference result (save_results=True): {res}")
                        
                        # Check for any output files immediately
                        if os.path.exists(output_dir):
                            files = os.listdir(output_dir)
                            print(f"Output files: {files}")
                            
                            for f in files:
                                file_path = os.path.join(output_dir, f)
                                if os.path.isfile(file_path):
                                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                                        content = file.read().strip()
                                        if content:
                                            print(f"Found OCR result in {f}")
                                            res = content
                                            break
                    
                    # Final fallback
                    if not res:
                        res = "DeepSeek OCR completed but returned no readable text. The model may have processed the image but didn't find recognizable text."
                        
                except Exception as e:
                    print(f"DeepSeek OCR Error: {e}")
                    res = f"OCR failed with error: {str(e)}"
                
                # Clean up temporary file
                os.unlink(tmp.name)

            st.success("✅ OCR Complete!")
            
            # Debug: Print the result to console and show in app
            print(f"OCR Result: {res}")
            print(f"Result type: {type(res)}")
            
            # Handle None result
            if res is None:
                st.error("❌ OCR returned no result. Please try again.")
                st.write("Debug info: Model returned None")
            else:
                # Display character and word count
                char_count = len(str(res))
                word_count = len(str(res).split())
                st.write(f"📊 Extracted {char_count} characters, {word_count} words")
                
                st.subheader("📝 Detected Text")
                st.text_area("OCR Output", str(res), height=400)
            
                # Also show as markdown if it contains markdown formatting
                if any(marker in str(res) for marker in ['#', '*', '-', '|', '```']):
                    st.subheader("📋 Formatted View")
                    st.markdown(str(res))

                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "⬇️ Download as Text",
                        data=str(res),
                        file_name=f"ocr_output_{uploaded_file.name}.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    st.download_button(
                        "⬇️ Download as Markdown",
                        data=str(res),
                        file_name=f"ocr_output_{uploaded_file.name}.md",
                        mime="text/markdown"
                    )

            # If the model saves additional files
            try:
                for f in os.listdir(output_dir):
                    if f.endswith(".md") and f != f"ocr_output_{uploaded_file.name}.md":
                        with open(os.path.join(output_dir, f), 'r') as file:
                            content = file.read()
                            st.download_button(
                                f"⬇️ Download {f}",
                                data=content,
                                file_name=f,
                                mime="text/markdown"
                            )
            except Exception as e:
                pass  # Ignore if no additional files
