import streamlit as st
import numpy as np
import tensorflow as tf
import tiktoken
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from transformers import PaddingMask,PositionalEmbedding,TransformerDecoder,ImagePositionalEmbedding,TransformerEncoder
from tensorflow.keras.layers import Reshape
# ============================================
# Load Models
# ============================================

@st.cache_resource
def load_cnn():
    resnet = ResNet50(weights='imagenet', include_top=False)
    x = resnet.get_layer("conv5_block3_out").output
    x = Reshape((49, 2048))(x)
    resnet_model = Model(inputs=resnet.input, outputs=x)
    return resnet_model

@st.cache_resource
def load_caption_model():
    # 👇 change path to your trained model
    return tf.keras.models.load_model(
        "image_caption_40_epochs.keras",
        custom_objects={
            "TransformerDecoder": TransformerDecoder,
            "PositionalEmbedding": PositionalEmbedding,
            "PaddingMask": PaddingMask,
            "ImagePositionalEmbedding":ImagePositionalEmbedding,
            "TransformerEncoder":TransformerEncoder
        }
    )

cnn_model = load_cnn()
model = load_caption_model()

# ============================================
# Load tokenizer (tiktoken)
# ============================================

base_enc = tiktoken.get_encoding("cl100k_base")

special_tokens = {
    "<|pad|>": base_enc.n_vocab,
    "<|startoftext|>": base_enc.n_vocab + 1,
    "<|endoftext|>": base_enc.n_vocab + 2,
}

enc = tiktoken.Encoding(
    name="cl100k_base_custom",
    pat_str=base_enc._pat_str,
    mergeable_ranks=base_enc._mergeable_ranks,
    special_tokens=special_tokens,
)


# Special tokens
BOS_ID = enc.encode("<|startoftext|>", allowed_special={"<|startoftext|>"})[0]
EOS_ID = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
PAD_ID = enc.encode("<|pad|>", allowed_special={"<|pad|>"})[0]

# ============================================
# Feature extraction
# ============================================

def extract_image_feature(img: Image.Image):
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, 0)
    img_array = preprocess_input(img_array)

    features = cnn_model.predict(img_array, verbose=0)  # (1, 2048)
    return features[0]  # (2048,)

# ============================================
# Caption generation
# ============================================

def pad_decoder_sequence(output_ids, max_len):
    seq = np.full((1, max_len), PAD_ID)   # fill with PAD tokens
    seq[0, :len(output_ids)] = output_ids
    return seq


def generate_caption_beam(model, image_feature, max_len, beam_size=3):

    encoder_input = np.expand_dims(image_feature, 0)

    sequences = [([BOS_ID], 0.0)]   # (token_ids, score)

    for step in range(max_len):

        all_candidates = []

        for seq, score in sequences:

            if seq[-1] == EOS_ID:
                all_candidates.append((seq, score))
                continue

            decoder_input = pad_decoder_sequence(seq, max_len)

            with tf.device('/GPU:0'):
                preds = model.predict([encoder_input, decoder_input], verbose=0)

            probs = preds[0, step, :]

            top_ids = np.argsort(probs)[-beam_size:]

            for token_id in top_ids:

                new_seq = seq + [int(token_id)]

                new_score = score - np.log(probs[token_id] + 1e-9)

                all_candidates.append((new_seq, new_score))

        ordered = sorted(all_candidates, key=lambda x: x[1])

        sequences = ordered[:beam_size]

    best_seq = sequences[0][0]

    caption_ids = best_seq[1:]

    caption = enc.decode(caption_ids)

    return caption

# ============================================
# Streamlit UI
# ============================================

st.set_page_config(page_title="Image Caption Generator", layout="centered")
st.title("🖼️ Transformer Image Caption Generator")
st.write("Upload an image and generate a caption using your trained Transformer model.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = ImageOps.exif_transpose(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            # Extract CNN features
            image_feature = extract_image_feature(image)

            # Generate caption
            caption = generate_caption_beam(model, image_feature, max_len=30, beam_size=5)

        st.success("Caption Generated!")
        st.markdown("### 📝 Predicted Caption:")
        st.write(caption)
