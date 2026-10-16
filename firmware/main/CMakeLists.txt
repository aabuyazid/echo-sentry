# Sources are added incrementally as each module is written and built against the
# real esp-tflite-micro headers (see idf_component.yml) — comprehensive-plan.md §2.3.
idf_component_register(
    SRCS "app_main.c"
         "audio_capture.c"
         "capture_stream.c"
         "debug_led.c"
         "detector.c"
         "evidence.c"
         "frontend_core.cc"
         "inference.cc"
         "net.c"
         "ota.c"
         "provisioning.c"
         "telemetry.c"
    INCLUDE_DIRS "." "gen"
    REQUIRES espressif__esp-tflite-micro
             espressif__led_strip
             driver
             esp_wifi
             esp_netif
             esp_event
             mqtt
             nvs_flash
             esp_https_ota
             esp_http_client
             json
             esp_driver_usb_serial_jtag
             esp_app_format
    EMBED_FILES "models/model.tflite" "models/normalization.bin"
)
