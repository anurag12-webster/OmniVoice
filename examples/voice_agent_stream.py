"""Voice agent streaming example.

Simulates an LLM streaming tokens into OmniVoice in real time.
Audio plays through your speakers as each sentence is ready.

Install deps:
    pip install omnivoice[stream]   # adds sounddevice

Run:
    python examples/voice_agent_stream.py --ref_audio speaker.wav
    python examples/voice_agent_stream.py --ref_audio speaker.wav --ref_text "Hello."
    python examples/voice_agent_stream.py  # auto voice, no ref needed
"""

import argparse
import time

import sounddevice as sd
import torch

from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig


def fake_llm_stream(text: str, delay: float = 0.02):
    """Simulate an LLM streaming tokens one word at a time."""
    for word in text.split():
        yield word + " "
        time.sleep(delay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--ref_audio", default=None)
    parser.add_argument("--ref_text", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # ── Load model ──────────────────────────────────────────────────────────
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Loading model on {device}...")
    model = OmniVoice.from_pretrained(args.model, device_map=device, dtype=torch.float16)

    print("Compiling model (one-time ~30s, faster every run after)...")
    model.compile_llm()  # defaults to max-autotune, best for 5090

    # ── Generation config — 8 steps for low latency ─────────────────────────
    gen_config = OmniVoiceGenerationConfig(
        num_step=8,
        guidance_scale=2.0,
        t_shift=0.1,
        denoise=True,
        postprocess_output=True,
    )

    # ── Simulate LLM response streaming in ──────────────────────────────────
    llm_response = (
        "Hey there! I'm your voice assistant. "
        "I can answer questions, help you brainstorm, or just have a chat. "
        "What's on your mind today?"
    )
    print(f"\nLLM response: {llm_response}\n")

    text_stream = fake_llm_stream(llm_response, delay=0.02)

    # ── Stream TTS and play each chunk as it arrives ─────────────────────────
    print("Streaming audio (chunk | latency):")
    t_start = time.perf_counter()
    first_chunk = True

    for idx, audio in model.generate_from_text_stream(
        text_iter=text_stream,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        generation_config=gen_config,
        min_chunk_chars=20,
        max_chunk_chars=200,
    ):
        latency_ms = (time.perf_counter() - t_start) * 1000
        duration_s = audio.shape[-1] / model.sampling_rate
        print(f"  chunk {idx:02d} | {latency_ms:6.0f} ms latency | {duration_s:.2f}s audio")

        if first_chunk:
            print(f"\n  *** TTFT: {latency_ms:.0f} ms ***\n")
            first_chunk = False

        sd.play(audio.squeeze(0).numpy(), samplerate=model.sampling_rate)
        sd.wait()

        t_start = time.perf_counter()  # reset for next chunk gap

    print("\nDone.")


if __name__ == "__main__":
    main()
