#pragma once

/*********/
/* Wi-Fi */
/*********/
struct WifiNetwork {
	const char* ssid;
	const char* passwd;
};
#define WIFI_CHECK_INTERVAL_SECONDS 10

/**************/
/* Microphone */
/**************/
// Connections to INMP441 I2S microphone
#define I2S_WS 25
#define I2S_SD 33
#define I2S_SCK 32

// Use I2S Processor 0
#define I2S_PORT I2S_NUM_0

// Define input buffer length
#define SAMPLE_BUFFER_SIZE 512
