from typing import Any, Optional, Dict
from agents import tracing

import logging

from opik.api_objects.span import span_data
from opik.api_objects.trace import trace_data
from opik.api_objects import opik_client

from . import span_data_parsers

LOGGER = logging.Logger(__name__)


class OpikTracingProcessor(tracing.TracingProcessor):
    def __init__(
        self,
        project_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._span_data_map: Dict[str, span_data.SpanData] = {}
        """Map from openai span id to opik span data."""

        self._created_traces_data_map: Dict[str, trace_data.TraceData] = {}
        """Map from openai trace id to opik trace data."""

        self._project_name = project_name

        self._opik_client = opik_client.get_client_cached()
        self._trace_id_to_first_span: Dict[str, span_data.SpanData] = {}
        self._trace_id_to_last_llm_span: Dict[str, span_data.SpanData] = {}

    def on_trace_start(self, trace: tracing.Trace) -> None:
        trace_metadata = trace.metadata or {}
        trace_metadata["created_from"] = "openai-agents"
        if trace.trace_id:
            trace_metadata["agents-trace-id"] = trace.trace_id
        try:
            opik_trace_data = trace_data.TraceData(
                name=trace.name,
                project_name=self._project_name,
                thread_id=trace.group_id,
                metadata=trace_metadata,
            )
            self._created_traces_data_map[trace.trace_id] = opik_trace_data
        except Exception:
            LOGGER.debug("on_trace_start failed", exc_info=True)

    def on_trace_end(self, trace: tracing.Trace) -> None:
        try:
            opik_trace_data = self._created_traces_data_map[trace.trace_id]
            opik_trace_data.init_end_time()

            self._collect_trace_input_and_output_from_spans(opik_trace_data)
            self._opik_client.trace(**opik_trace_data.__dict__)
        except Exception:
            LOGGER.debug("on_trace_end failed", exc_info=True)

    def on_span_start(self, span: tracing.Span[Any]) -> None:
        try:
            parent_opik_span = self._span_data_map.get(span.parent_id)
            parent_opik_span_id = parent_opik_span.id if parent_opik_span else None
            opik_trace_id = self._created_traces_data_map[span.trace_id].id

            opik_span_data = span_data.SpanData(
                parent_span_id=parent_opik_span_id,
                trace_id=opik_trace_id,
                project_name=self._project_name,
            )

            if (
                opik_span_data.trace_id not in self._trace_id_to_first_span
            ) and span.span_data.type in ["response", "generation", "custom"]:
                self._trace_id_to_first_span[opik_span_data.trace_id] = opik_span_data

            self._span_data_map[span.span_id] = opik_span_data
        except Exception:
            LOGGER.debug("on_span_start failed", exc_info=True)

    def on_span_end(self, span: tracing.Span[Any]) -> None:
        try:
            parsed_span_data = span_data_parsers.parse_spandata(span.span_data)

            opik_span_data = self._span_data_map[span.span_id]
            opik_span_data.init_end_time().update(**parsed_span_data.__dict__)

            if opik_span_data.type == "llm":
                self._trace_id_to_last_llm_span[opik_span_data.trace_id] = (
                    opik_span_data
                )

            self._opik_client.span(**opik_span_data.__dict__)
        except Exception:
            LOGGER.debug("on_span_end failed", exc_info=True)

    def force_flush(self) -> bool:
        return self._opik_client.flush()

    def shutdown(self) -> None:
        self.force_flush()

    def _collect_trace_input_and_output_from_spans(
        self, opik_trace_data: trace_data.TraceData
    ) -> None:
        if opik_trace_data.id in self._trace_id_to_first_span:
            opik_trace_data.update(
                input=self._trace_id_to_first_span[opik_trace_data.id].input
            )

        if opik_trace_data.id in self._trace_id_to_last_llm_span:
            opik_trace_data.update(
                output=self._trace_id_to_last_llm_span[opik_trace_data.id].output
            )
