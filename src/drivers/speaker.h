#pragma once
#if HW_VERSION == 2

#include "utils/typedefs.h"
#include <Arduino.h>

#define MAX_RTTTL_NOTES 256
#define MAX_RTTTL_TEXT_LENGTH 1024

extern bool speakerLoopOnFastCore;
extern bool speakerLoopOnFastCore2;
extern u8 speakerPwmSlice;

extern const char soundNames[6][20];
extern u8 startSoundId;

/// @brief Initializes the speaker
void initSpeaker();

/// @brief Rickrolls the user
bool rickroll(MenuItem *_item);

/// @brief Called periodically to update the speaker with new PWM data
void speakerLoop();

/// @brief Play the startup sound
void playStartupSound();

/**
 * @brief Start a beeping sound
 *
 * @details The sound will be played for tOnMs, then off for tOffMs, and repeated until duration ms have passed
 *
 * @param frequency Frequency in Hz
 * @param duration Duration in ms (65535 for infinite)
 * @param tOnMs Time the sound is on in ms, default = sound always on
 * @param tOffMs Time the sound is off in ms, default = sound always on
 */
void makeSound(u16 frequency, u16 duration, u16 tOnMs = 65535, u16 tOffMs = 0);

/// @brief Stop the current sound
void stopSound();

/**
 * @brief Play a sweep sound
 *
 * @details sweep from startFrequency to endFrequency over tOnMs, then stop for tOffMs, repeat until duration ms have passed
 *
 * @param startFrequency start frequency of the sweep
 * @param endFrequency end frequency of the sweep
 * @param duration Duration in ms (65535 for infinite)
 * @param tOnMs Time the sound is on in ms
 * @param tOffMs Time the sound is off in ms
 */
void makeSweepSound(u16 startFrequency, u16 endFrequency, u16 duration, u16 tOnMs, u16 tOffMs);

/// @brief Play a sound in RTTTL format
/// @param song RTTTL formatted song string
void makeRtttlSound(const char *song);
#endif