"""Single-item inference CLI for OmniVoice.

Generates audio from a single text input using voice cloning,
voice design, or auto voice.

Usage:
    # Voice cloning
    omnivoice-infer --model k2-fsa/OmniVoice \
        --text "Hello, this is a text for text-to-speech." \
        --ref_audio ref.wav --ref_text "Reference transcript." --output out.wav

    # Voice design
    omnivoice-infer --model k2-fsa/OmniVoice \
        --text "Hello, this is a text for text-to-speech." \
        --instruct "male, British accent" --output out.wav

    # Auto voice
    omnivoice-infer --model k2-fsa/OmniVoice \
        --text "Hello, this is a text for text-to-speech." --output out.wav
"""

import argparse
import logging

import torch
import torchaudio

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.common import str2bool

try:
    import sounddevice as sd

    _HAS_SOUNDDEVICE = True
except ImportError:
    _HAS_SOUNDDEVICE = False


def get_best_device():
    """Auto-detect the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OmniVoice single-item inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output WAV file path.",
    )
    # Voice cloning
    parser.add_argument(
        "--ref_audio",
        type=str,
        default=None,
        help="Reference audio file path for voice cloning.",
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default=None,
        help="Reference text describing the reference audio.",
    )
    # Voice design
    parser.add_argument(
        "--instruct",
        type=str,
        default=None,
        help="Style instruction for voice design mode.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language name (e.g. 'English') or code (e.g. 'en').",
    )
    # Generation parameters
    parser.add_argument("--num_step", type=int, default=32)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Fixed output duration in seconds. If set, overrides the "
        "model's duration estimation. The speed factor is automatically "
        "adjusted to match while preserving language-aware pacing.",
    )
    parser.add_argument("--t_shift", type=float, default=0.1)
    parser.add_argument("--denoise", type=str2bool, default=True)
    parser.add_argument(
        "--postprocess_output",
        type=str2bool,
        default=True,
    )
    parser.add_argument("--layer_penalty_factor", type=float, default=5.0)
    parser.add_argument("--position_temperature", type=float, default=5.0)
    parser.add_argument("--class_temperature", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--stream",
        type=str2bool,
        default=False,
        help="Stream audio chunks as they are generated. Requires sounddevice "
        "(pip install sounddevice) for real-time playback. Each chunk is also "
        "saved as <output>_chunk_NNNN.wav; the final concatenated file is saved "
        "to <output> as usual.",
    )
    return parser


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_parser().parse_args()

    device = args.device or get_best_device()
    logging.info(f"Loading model from {args.model} on {device} ...")
    model = OmniVoice.from_pretrained(
        args.model, device_map=device, dtype=torch.float16
    )

    gen_kwargs = dict(
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        instruct=args.instruct,
        duration=args.duration,
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        speed=args.speed,
        t_shift=args.t_shift,
        denoise=args.denoise,
        postprocess_output=args.postprocess_output,
        layer_penalty_factor=args.layer_penalty_factor,
        position_temperature=args.position_temperature,
        class_temperature=args.class_temperature,
    )

    if args.stream:
        if _HAS_SOUNDDEVICE:
            logging.info("Streaming mode enabled — audio will play as chunks arrive.")
        else:
            logging.warning(
                "sounddevice not found (pip install sounddevice). "
                "Chunks will be saved to disk but not played back in real time."
            )

        logging.info(f"Streaming audio for: {args.text[:80]}...")
        collected = []
        for chunk_idx, total_chunks, chunk_audio in model.generate_stream(
            text=args.text, **gen_kwargs
        ):
            chunk_path = f"{args.output}_chunk_{chunk_idx:04d}.wav"
            torchaudio.save(chunk_path, chunk_audio, model.sampling_rate)
            logging.info(
                f"  Chunk {chunk_idx + 1}"
                + (f"/{total_chunks}" if total_chunks >= 0 else "")
                + f" saved to {chunk_path}"
            )
            if _HAS_SOUNDDEVICE:
                audio_np = chunk_audio.squeeze(0).numpy()
                sd.play(audio_np, samplerate=model.sampling_rate)
                sd.wait()
            collected.append(chunk_audio)

        final_audio = torch.cat(collected, dim=-1)
        torchaudio.save(args.output, final_audio, model.sampling_rate)
        logging.info(f"Final audio saved to {args.output}")
    else:
        logging.info(f"Generating audio for: {args.text[:80]}...")
        audios = model.generate(text=args.text, **gen_kwargs)
        torchaudio.save(args.output, audios[0], model.sampling_rate)
        logging.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
