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
)
from livekit.plugins import silero

load_dotenv()  # reads .env in the cwd
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
        stt="deepgram/nova-3",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-3",
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

    await session.start(agent=Assistant(), room=ctx.room)
    await session.generate_reply(instructions="Greet the user briefly and ask how you can help.")


if __name__ == "__main__":
    cli.run_app(server)
