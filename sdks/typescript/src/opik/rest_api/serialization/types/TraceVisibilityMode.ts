/**
 * This file was auto-generated by Fern from our API Definition.
 */

import * as serializers from "../index";
import * as OpikApi from "../../api/index";
import * as core from "../../core";

export const TraceVisibilityMode: core.serialization.Schema<
    serializers.TraceVisibilityMode.Raw,
    OpikApi.TraceVisibilityMode
> = core.serialization.enum_(["default", "hidden"]);

export declare namespace TraceVisibilityMode {
    export type Raw = "default" | "hidden";
}
