// Local fast-path threshold — comprehensive-plan.md §3.4. Deliberately much more
// sensitive than the hub's tuned ALARM threshold (comprehensive-plan.md §5): its
// only job is deciding when an evidence clip is worth capturing, not making the
// alerting decision itself — that's entirely the hub's six-stage pipeline.
#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

bool detector_should_capture_evidence(float err_mean);

#ifdef __cplusplus
}
#endif
