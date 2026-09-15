#include "detector.h"

#include "provisioning.h"

bool detector_should_capture_evidence(float err_mean) {
    return err_mean >= provisioning_get()->anomaly_threshold;
}
