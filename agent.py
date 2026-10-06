"""
Basic voice agent 
Maps to the architecture diagram:
client mic --WebRTC--> LiveKit Server --> Agent [VAD/turn -> STT -> LLM(+tools) -> TTS] --> client speaker
"""

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    cli,
    inference,
    metrics,
    room_io,
    stt,
    tts,
    llm
)

from livekit.plugins import silero, noise_cancellation


load_dotenv() 
logger = logging.getLogger("voice-agent")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly voice assistant. "
                "Keep replies to 1-2 short sentences, this is a voice call. "
                "No markdown, no lists, no emojis. "
                
            ),
        )



# --- prewarm: load VAD once per process, not once per call ---------------
def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server = AgentServer(setup_fnc=prewarm)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=stt.FallbackAdapter(
            [
                inference.STT.from_model_string("deepgram/nova-3"),
                inference.STT.from_model_string("assemblyai/universal-3-5-pro"),
            ],
        ),
        llm= llm.FallbackAdapter(
            [
                inference.LLM.from_model_string("openai/gpt-4.1-mini"),
                inference.LLM.from_model_string("google/gemini-pro-3.5"),
            ],
        ),
        tts=tts.FallbackAdapter(
            [
                inference.TTS.from_model_string("xai/tts-1:ursa"),
                inference.TTS.from_model_string("coqui/tts-1:alloy"),
            ],
        ),
        turn_handling={
            # semantic turn detector: reads the transcript to decide if the user is done speaking
            "turn_detection": inference.TurnDetector(),
            # endpointing delay
            "endpointing": {"min_delay": 0.4, "max_delay": 3.0},
            # start the LLM on interim transcripts to save time
            "preemptive_generation": {"enabled": True},
            # barge-in or Interrrupt: if the user starts talking while the LLM is generating, stop the LLM and start listening again
            "interruption": {"enabled": True},
        },
    )

    # (EOU delay, LLM TTFT, TTS TTFB) in the logs
    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)

    await session.start(agent=Assistant(),
                         room=ctx.room,
                         room_options=room_io.RoomOptions(
                             audio_input = room_io.AudioInputOptions(
                                 noise_cancellation = noise_cancellation.BVC()
                             ),
                         ),
    )
    await session.generate_reply(instructions="You are a friendly, sarcastic voice assistant for conversations as a friend." \
    " Keep replies under 3 sentences, this is a voice call. No markdown, no lists, no emojis.")


if __name__ == "__main__":
    cli.run_app(server)
