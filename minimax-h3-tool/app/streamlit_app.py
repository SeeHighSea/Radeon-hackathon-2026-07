#!/usr/bin/env python3
"""MiniMax H3 · Local Multimodal Video Studio (AMD Radeon) — Streamlit UI.

Runs open-weights MiniMax H3 on a single AMD Radeon GPU (ROCm) through
ComfyUI's native nodes. T2V / I2V / R2V modes, video + stereo audio.

Usage:
    /opt/venv/bin/python -m streamlit run app/streamlit_app.py \
        --server.port 7860 --server.address 127.0.0.1
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import urllib.request

import numpy as np
import streamlit as st

from comfy_client import generate, SAMPLER, SCHEDULER, STEPS, API_BASE

RESOLUTIONS = {
    "Fast preview (864x480)": (864, 480),
    "HD (1152x640)": (1152, 640),
    "Full 768p short edge (1344x768)": (1344, 768),
}

CLIP_NAME = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"

OUT_DIR = "/opt/ComfyUI/output/video"


def _comfy_online() -> bool:
    try:
        with urllib.request.urlopen(f"{API_BASE}/system_stats", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _find_output_mp4(outputs: dict) -> str | None:
    for o in outputs.values():
        for g in o.get("images", []):
            if g.get("type") == "output":
                path = os.path.join("/opt/ComfyUI/output", g.get("subfolder", ""), g["filename"])
                if os.path.exists(path):
                    return path
    return None


def _now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _generate_and_serve(prompt, mode, resolution, duration, seed, images):
    w, h = RESOLUTIONS[resolution]
    kwargs = dict(
        prompt=prompt, mode=mode, width=w, height=h,
        duration_s=duration, seed=int(seed), clip_name=CLIP_NAME,
    )
    if mode == "i2v" and images:
        if len(images) >= 1:
            kwargs["first_frame"] = np.array(images[0])
        if len(images) >= 2:
            kwargs["last_frame"] = np.array(images[1])
    if mode == "r2v" and images:
        kwargs["ref_images"] = [np.array(im) for im in images]

    if not _comfy_online():
        st.error("ComfyUI is not running — start it first (see README).")
        return

    bar = st.progress(0.05, text="submitting to ComfyUI…")
    t0 = time.time()

    def _cb(elapsed, status):
        bar.progress(min(0.98, elapsed / 900.0),
                     text=f"{status} · {elapsed:.0f}s elapsed ({elapsed/60:.1f} min)")

    result = generate(**kwargs, progress_cb=_cb)
    if result.get("status") != "success":
        st.error(f"Generation failed: {result.get('status')}")
        bar.progress(1.0, text="failed")
        return

    mp4 = _find_output_mp4(result.get("outputs", {}))
    if not mp4:
        st.error("output mp4 not found")
        return

    dst = os.path.join(tempfile.gettempdir(), f"minimax_h3_{_now_ts()}.mp4")
    shutil.copy(mp4, dst)
    elapsed = time.time() - t0
    bar.progress(1.0, text=f"done in {elapsed:.0f}s")
    st.success(f"Generated in {elapsed:.0f}s")
    st.video(dst)

    st.subheader("Recent local outputs")
    recent = sorted(
        (f for f in os.listdir(OUT_DIR) if f.endswith(".mp4")),
        reverse=True,
    )[:6]
    for f in recent:
        st.markdown(f"`{f}`")


def _sidebar_status():
    online = _comfy_online()
    st.sidebar.title("MiniMax H3 · Studio")
    st.sidebar.markdown(
        "**Open-weights omni-modal video model** · AMD Radeon (ROCm)."
    )
    st.sidebar.markdown(
        f"ComfyUI: {'🟢 online' if online else '🔴 offline'}"
        f"\n\nSampler `{SAMPLER}` / `{SCHEDULER}` / {STEPS} steps"
    )
    return online


def main():
    st.set_page_config(page_title="MiniMax H3 · Video Studio", layout="centered")
    st.title("MiniMax H3 · Local Multimodal Video Studio")
    st.caption(
        "Text / images → video + **stereo audio** · 100% local on AMD Radeon "
        "(ROCm) · pruned int8 weights"
    )

    _sidebar_status()

    tab_t2v, tab_i2v, tab_r2v = st.tabs(["Text → Video (T2V)", "Image → Video (I2V)", "Reference → Video (R2V)"])

    with tab_t2v:
        st.text_area("Prompt", value=(
            "Cinematic drone shot flying over a neon cyberpunk city at dusk, "
            "rain-slick streets reflecting holographic signs, volumetric fog, "
            "slow sweeping camera, subtle wind and distant traffic sound."
        ), height=140, key="t2v_prompt")
        c1, c2, c3 = st.columns(3)
        t2v_res = c1.selectbox("Resolution", list(RESOLUTIONS), key="t2v_res")
        t2v_dur = c2.slider("Duration (s)", 2.0, 15.0, 5.0, 0.5, key="t2v_dur")
        t2v_seed = c3.number_input("Seed", value=556589502035082, key="t2v_seed")
        if st.button("Generate", type="primary", key="t2v_btn"):
            _generate_and_serve(
                st.session_state.t2v_prompt, "t2v", t2v_res, t2v_dur, t2v_seed, []
            )

    with tab_i2v:
        st.text_area("Prompt", value=(
            "The camera slowly pushes in while the character turns to look at the sky, "
            "wind and a low ambient drone in the background."
        ), height=110, key="i2v_prompt")
        i2v_img = st.file_uploader(
            "First / last keyframe (optional, up to 2)", type=["png", "jpg", "jpeg"],
            accept_multiple_files=True, key="i2v_img",
        )
        c1, c2, c3 = st.columns(3)
        i2v_res = c1.selectbox("Resolution", list(RESOLUTIONS), key="i2v_res")
        i2v_dur = c2.slider("Duration (s)", 2.0, 15.0, 5.0, 0.5, key="i2v_dur")
        i2v_seed = c3.number_input("Seed", value=556589502035082, key="i2v_seed")
        if st.button("Generate", type="primary", key="i2v_btn"):
            imgs = _load_uploads(i2v_img)
            if len(imgs) > 2:
                st.warning("I2V uses at most 2 keyframes (first/last); ignoring extras.")
            _generate_and_serve(
                st.session_state.i2v_prompt, "i2v", i2v_res, i2v_dur, i2v_seed, imgs[:2]
            )

    with tab_r2v:
        st.markdown(
            "Lock identity / style from reference images. Use `<Picture 1>` (and so on) "
            "in the prompt to reference each image in order. Requires the ref2va model."
        )
        st.text_area("Prompt", value=(
            "Using <Picture 1> as the character reference, show the character running "
            "through a rainy alley, camera following behind, footsteps and rain sounds."
        ), height=110, key="r2v_prompt")
        r2v_img = st.file_uploader(
            "Reference images (up to 9)", type=["png", "jpg", "jpeg"],
            accept_multiple_files=True, key="r2v_img",
        )
        c1, c2, c3 = st.columns(3)
        r2v_res = c1.selectbox("Resolution", list(RESOLUTIONS), key="r2v_res")
        r2v_dur = c2.slider("Duration (s)", 2.0, 15.0, 5.0, 0.5, key="r2v_dur")
        r2v_seed = c3.number_input("Seed", value=556589502035082, key="r2v_seed")
        if st.button("Generate", type="primary", key="r2v_btn"):
            imgs = _load_uploads(r2v_img)
            if len(imgs) > 9:
                st.warning("R2V supports up to 9 reference images.")
            _generate_and_serve(
                st.session_state.r2v_prompt, "r2v", r2v_res, r2v_dur, r2v_seed, imgs[:9]
            )

    st.markdown("---")
    st.caption(
        "Hardware: AMD Radeon PRO (gfx1100, RDNA3, 48 GB) · ROCm 7.2.4 · "
        "ComfyUI · pruned int8 diffusion + int8 Qwen3-VL-32B"
    )


def _load_uploads(uploaded) -> list:
    from PIL import Image
    import io
    imgs = []
    for u in (uploaded or []):
        imgs.append(np.array(Image.open(io.BytesIO(u.getvalue())).convert("RGB")))
    return imgs


if __name__ == "__main__":
    main()
