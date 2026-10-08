import logging
from typing import AsyncIterable
from dotenv import load_dotenv
import time
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    AgentStateChangedEvent,
    ModelSettings,
    metrics, 
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

logger = logging.getLogger(__name__)
from typing import AsyncIterable

CONFIDENCE_THRESHOLD = 0.7


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a friendly, sarcastic voice assistant for conversations as a friend." \
             " Keep replies under 3 sentences, this is a voice call. No markdown, no lists, no emojis.")
        self._turn_min_conf: float | None = None

    # wrap the default STT so we can read confidence on final transcripts
    async def stt_node(self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings):
        async for ev in Agent.default.stt_node(self, audio, model_settings):
            if (
                isinstance(ev, stt.SpeechEvent)
                and ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT
                and ev.alternatives
            ):
                conf = ev.alternatives[0].confidence
                if conf > 0:  # 0.0 = provider didn't send one, ignore
                    self._turn_min_conf = conf if self._turn_min_conf is None else min(self._turn_min_conf, conf)
            yield ev

    # runs after the user finishes, before the LLM replies
    async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
        conf, self._turn_min_conf = self._turn_min_conf, None  # reset for next turn
        if conf is not None and conf < CONFIDENCE_THRESHOLD:
            turn_ctx.add_message(
                role="system",
                content=(
                    f"Low speech-recognition confidence ({conf:.2f}) on the last user message. "
                    "Don't act on it yet. Briefly confirm what you heard by repeating it a part of it ('Is that correct...?'). "
                    "If they say no, ask them to repeat."
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

    usage_collector = metrics.UsageCollector()
    last_eou_metrics: metrics.EOUMetrics | None = None


    # (EOU delay, LLM TTFT, TTS TTFB) in the logs
    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent) -> None:
        nonlocal last_eou_metrics
        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    
    async def log_usage():
        # Print per-session summary (tokens, audio duration, costs)
        summary = usage_collector.get_summary()
        logger.info("Usage summary: %s", summary)  

    ctx.add_shutdown_callback(log_usage)    

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        if ev.new_state == "speaking":
            if last_eou_metrics:
                # Calculate time since user finished speaking
                elapsed = time.time() - last_eou_metrics.timestamp
                logger.info(f"Time to first audio: {elapsed:.3f}s")

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
