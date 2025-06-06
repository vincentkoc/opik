/**
 * This file was auto-generated by Fern from our API Definition.
 */

import * as serializers from "../index";
import * as OpikApi from "../../api/index";
import * as core from "../../core";
import { SpanExperimentItemBulkWriteViewType } from "./SpanExperimentItemBulkWriteViewType";
import { JsonListStringExperimentItemBulkWriteView } from "./JsonListStringExperimentItemBulkWriteView";
import { JsonNodeExperimentItemBulkWriteView } from "./JsonNodeExperimentItemBulkWriteView";
import { ErrorInfoExperimentItemBulkWriteView } from "./ErrorInfoExperimentItemBulkWriteView";

export const SpanExperimentItemBulkWriteView: core.serialization.ObjectSchema<
    serializers.SpanExperimentItemBulkWriteView.Raw,
    OpikApi.SpanExperimentItemBulkWriteView
> = core.serialization.object({
    id: core.serialization.string().optional(),
    parentSpanId: core.serialization.property("parent_span_id", core.serialization.string().optional()),
    name: core.serialization.string().optional(),
    type: SpanExperimentItemBulkWriteViewType.optional(),
    startTime: core.serialization.property("start_time", core.serialization.date()),
    endTime: core.serialization.property("end_time", core.serialization.date().optional()),
    input: JsonListStringExperimentItemBulkWriteView.optional(),
    output: JsonListStringExperimentItemBulkWriteView.optional(),
    metadata: JsonNodeExperimentItemBulkWriteView.optional(),
    model: core.serialization.string().optional(),
    provider: core.serialization.string().optional(),
    tags: core.serialization.list(core.serialization.string()).optional(),
    usage: core.serialization.record(core.serialization.string(), core.serialization.number()).optional(),
    errorInfo: core.serialization.property("error_info", ErrorInfoExperimentItemBulkWriteView.optional()),
    lastUpdatedAt: core.serialization.property("last_updated_at", core.serialization.date().optional()),
    totalEstimatedCost: core.serialization.property("total_estimated_cost", core.serialization.number().optional()),
    totalEstimatedCostVersion: core.serialization.property(
        "total_estimated_cost_version",
        core.serialization.string().optional(),
    ),
});

export declare namespace SpanExperimentItemBulkWriteView {
    export interface Raw {
        id?: string | null;
        parent_span_id?: string | null;
        name?: string | null;
        type?: SpanExperimentItemBulkWriteViewType.Raw | null;
        start_time: string;
        end_time?: string | null;
        input?: JsonListStringExperimentItemBulkWriteView.Raw | null;
        output?: JsonListStringExperimentItemBulkWriteView.Raw | null;
        metadata?: JsonNodeExperimentItemBulkWriteView.Raw | null;
        model?: string | null;
        provider?: string | null;
        tags?: string[] | null;
        usage?: Record<string, number> | null;
        error_info?: ErrorInfoExperimentItemBulkWriteView.Raw | null;
        last_updated_at?: string | null;
        total_estimated_cost?: number | null;
        total_estimated_cost_version?: string | null;
    }
}
