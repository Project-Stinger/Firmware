#if HW_VERSION == 2
#include "global.h"

elapsedMillis soundStart;
u16 soundDuration = 0;
u8 soundType = 0; // 0 = stationary, 1 = sweep, 2 = rtttl
u16 sweepStartFrequency = 0;
u16 sweepEndFrequency = 0;
u16 onTime = 0;
u16 offTime = 0;
u16 currentWrap = 400;
bool speakerLoopOnFastCore = false;
bool speakerLoopOnFastCore2 = false;
fix32 noteFrequencies[13] =
	{
		16.35, // C0
		17.32, // C#0
		18.35, // D0
		19.45, // D#0
		20.60, // E0
		21.83, // F0
		23.12, // F#0
		24.50, // G0
		25.96, // G#0
		27.50, // A0
		29.14, // A#0
		30.87, // B0
		0 // error
};
typedef struct rtttlNote {
	u16 frequency; // pause = 0
	u16 duration; // milliseconds
	u8 sweepToNext; // dash after note will sweep to next, e.g. 8e6-,8d6 will sweep from 8e6 to 8d6 within "duration" milliseconds
	u8 quieter; // 0 = normal, 1-3 = quieter, 4 = ring-tone-like e.g. 8e6$4 will play 8e6 like a ring-tone
				// quietness only sweeps statically (with the quietness level of the first note)
} RTTTLNote;
typedef struct rtttlSong {
	RTTTLNote notes[MAX_RTTTL_NOTES];
	u8 numNotes;
} RTTTLSong;
RTTTLSong songToPlay;

#define FREQ_TO_WRAP(freq) (1000000 / freq)

volatile u8 soundState = 1; // 1: playing PWM, 2: a needs to be filled, 4: b needs to be filled
u8 speakerPwmSlice;

const char *soundRickroll = "rickroll:d=4,o=5,b=125:16g,16a,16c6,16a,8e6,16p,8e6,16p,4d6,8p,16g,16a,16c6,16a,8d6,16p,8d6,16p,8c6.,16b,8a";
const char *soundNokia = "NokiaTune:d=4,o=5,b=180:8e6$4,8d6$4,4f#5$4,4g#5$4,8c#6$4,8b5$4,4d5$4,4e5$4,8b5$4,8a5$4,4c#5$4,4e5$4,2a5$4.";
const char *soundDrone = "Drone:o=6,b=800:1c#6,1d#6,1g#6.,1d#6$1,1g#6.$1,1d#6$2,1g#6$2";
const char *soundOneup = "Oneup:d=4,o=5,b=960:8c5,8g4,8c5,8e5,8g5,8c6,8g5,8g#4,8c5,8d#5,8g#5,8d#5,8g#5,8c6,8d#6,8g#6,8d#6,8d5,8f5,8a#5,8f5,8a#5,8d6,8f6,8d6,8f6,8a#6,8f6";
const char soundNames[6][20] = {
	"None",
	"Beep",
	"One Up",
	"Nokia",
	"Drone",
	"Rickroll",
};
u8 startSoundId = 0;

void initSpeaker() {
	gpio_set_function(PIN_SPEAKER, GPIO_FUNC_PWM);
	gpio_set_dir(PIN_SPEAKER, GPIO_IN);
	gpio_set_pulls(PIN_SPEAKER, false, true); // slightly reduces hissing noise
	speakerPwmSlice = pwm_gpio_to_slice_num(PIN_SPEAKER);
	pwm_set_clkdiv_int_frac(speakerPwmSlice, F_CPU / MHZ, 0);
	pwm_set_wrap(speakerPwmSlice, FREQ_TO_WRAP(1000));
	pwm_set_gpio_level(PIN_SPEAKER, 0);
	pwm_set_enabled(speakerPwmSlice, true);
	gpio_set_drive_strength(PIN_SPEAKER, GPIO_DRIVE_STRENGTH_12MA);
}

void playStartupSound() {
	switch (startSoundId) {
	case 1:
		makeSound(1000, 100);
		break;
	case 2:
		makeRtttlSound(soundOneup);
		break;
	case 3:
		makeRtttlSound(soundNokia);
		break;
	case 4:
		makeRtttlSound(soundDrone);
		break;
	case 5:
		makeRtttlSound(soundRickroll);
		break;
	}
}

bool rickroll(MenuItem *_item) {
	makeRtttlSound(soundRickroll);
	return true;
}

u8 beeperOn = 0;
void speakerLoop() {
	if (soundDuration > 0) {
		u32 sinceStart = soundStart;
		if (soundDuration != 65535 && sinceStart > soundDuration) {
			stopSound();
		} else if (soundType == 1) {
			i32 thisCycle = sinceStart % ((u32)onTime + (u32)offTime);
			if (thisCycle > onTime) {
				pwm_set_gpio_level(PIN_SPEAKER, 0);
			} else {
				u32 thisFreq = sweepStartFrequency + ((sweepEndFrequency - sweepStartFrequency) * thisCycle) / onTime;
				currentWrap = FREQ_TO_WRAP(thisFreq);
				pwm_set_wrap(speakerPwmSlice, currentWrap);
				pwm_set_gpio_level(PIN_SPEAKER, currentWrap >> 1);
			}
		} else if (soundType == 2) {
			int thisCycle = sinceStart;
			int noteIndex = 0;
			while (thisCycle > songToPlay.notes[noteIndex].duration && noteIndex < songToPlay.numNotes && noteIndex < MAX_RTTTL_NOTES) {
				thisCycle -= songToPlay.notes[noteIndex].duration;
				noteIndex++;
			}
			if (noteIndex >= songToPlay.numNotes || noteIndex >= MAX_RTTTL_NOTES) {
				stopSound();
				speakerLoopOnFastCore2 = false;
			} else {
				if (songToPlay.notes[noteIndex].frequency == 0) {
					pwm_set_gpio_level(PIN_SPEAKER, 0);
				} else {
					if (songToPlay.notes[noteIndex].sweepToNext) {
						u32 thisFreq = songToPlay.notes[noteIndex].frequency + ((songToPlay.notes[noteIndex + 1].frequency - songToPlay.notes[noteIndex].frequency) * thisCycle) / songToPlay.notes[noteIndex].duration;
						currentWrap = FREQ_TO_WRAP(thisFreq);
					} else {
						currentWrap = songToPlay.notes[noteIndex].frequency ? FREQ_TO_WRAP(songToPlay.notes[noteIndex].frequency) : 0;
					}
					pwm_set_wrap(speakerPwmSlice, currentWrap);
					int level = currentWrap >> 1;
					switch (songToPlay.notes[noteIndex].quieter) {
					case 1:
						level = level >> 4;
						break;
					case 2:
						level = level >> 6;
						break;
					case 3:
						level = level >> 7;
						break;
					case 4:
						level = level >> 2;
					}
					pwm_set_gpio_level(PIN_SPEAKER, level);
				}
			}
		} else if (soundType == 0) {
			u32 thisCycle = sinceStart % ((u32)onTime + (u32)offTime);
			if (thisCycle > onTime) {
				pwm_set_gpio_level(PIN_SPEAKER, 0);
			} else {
				pwm_set_gpio_level(PIN_SPEAKER, currentWrap >> 1);
			}
		}
	}
}

void makeSound(u16 frequency, u16 duration, u16 tOnMs, u16 tOffMs) {
	if (!tOnMs) return;
	gpio_set_dir(PIN_SPEAKER, GPIO_OUT);
	soundType = 0;
	currentWrap = FREQ_TO_WRAP(frequency);
	pwm_set_wrap(speakerPwmSlice, currentWrap);
	onTime = tOnMs;
	offTime = tOffMs;
	soundDuration = duration;
	soundStart = 0;
}

void stopSound() {
	gpio_set_dir(PIN_SPEAKER, GPIO_IN);
	gpio_set_pulls(PIN_SPEAKER, false, true);
	soundDuration = 0;
	pwm_set_gpio_level(PIN_SPEAKER, 0);
}

// sweep from startFrequency to endFrequency over tOnMs, then stop for tOffMs, repeat for duration
void makeSweepSound(u16 startFrequency, u16 endFrequency, u16 duration, u16 tOnMs, u16 tOffMs) {
	if (!tOnMs) return;
	gpio_set_dir(PIN_SPEAKER, GPIO_OUT);
	soundType = 1;
	sweepStartFrequency = startFrequency;
	sweepEndFrequency = endFrequency;
	onTime = tOnMs;
	offTime = tOffMs;
	soundDuration = duration;
	soundStart = 0;
}

int parseInt(const char *str, int *index, int defaultValue = 0) {
	int result = 0;
	int len = 0;
	while (str[len + *index] >= '0' && str[len + *index] <= '9' && len < 9) {
		result *= 10;
		result += str[len + *index] - '0';
		len++;
	}
	if (len == 0)
		return defaultValue;
	*index += len;
	return result;
}

void makeRtttlSound(const char *song) {
	gpio_set_dir(PIN_SPEAKER, GPIO_OUT);
	int colonCount = 0;
	int colon[2] = {-1, -1};
	int totalDuration = 0;
	for (int i = 0; i < MAX_RTTTL_TEXT_LENGTH; i++) {
		if (song[i] == ':') {
			colon[colonCount] = i;
			colonCount++;
		}
		if (colonCount == 2) {
			break;
		}
		if (song[i] == '\0')
			break;
	}
	int d, o, b, notesStart;
	if (colonCount == 0) {
		notesStart = 0;
		d = 4;
		o = 5;
		b = 125;
	} else if (colonCount >= 1) {
		int start = 0, end = 0;
		if (colonCount == 1) {
			start = colon[0];
			end = 0;
		} else if (colonCount == 2) {
			start = colon[1];
			end = colon[0];
		}
		notesStart = start + 1;
		// find index of 'd' in song before colon
		int i = start - 3;
		while (song[i] != 'd' && i >= end) {
			i--;
		}
		if (i < end) {
			d = 4;
		} else {
			i += 2;
			d = parseInt(song, &i);
		}
		// find index of 'o' in song before colon
		i = start - 3;
		while (song[i] != 'o' && i >= end) {
			i--;
		}
		if (i < end) {
			o = 5;
		} else {
			i += 2;
			o = parseInt(song, &i);
		}
		// find index of 'b' in song before colon
		i = start - 3;
		while (song[i] != 'b' && i >= end) {
			i--;
		}
		if (i < end) {
			b = 125;
		} else {
			i += 2;
			b = parseInt(song, &i);
			if (!b) b = 125;
		}
	}
	int noteIndex = 0;
	for (int i = notesStart; i < MAX_RTTTL_TEXT_LENGTH;) {
		if (song[i] == '\0')
			break;
		if (song[i] == ',') {
			i++;
			continue;
		}
		if (noteIndex >= MAX_RTTTL_NOTES)
			break;
		int noteLength = parseInt(song, &i, -1);
		if (noteLength == -1)
			noteLength = d;
		if (noteLength)
			songToPlay.notes[noteIndex].duration = 240000 / b / noteLength;
		else
			songToPlay.notes[noteIndex].duration = 0;
		if (song[i] == 'p') {
			songToPlay.notes[noteIndex].frequency = 0;
			songToPlay.notes[noteIndex].sweepToNext = 0;
			i++;
		} else {
			int note = 0; // 0 = C, 1 = C#...
			char noteChar = song[i++];
			switch (noteChar) {
			case 'c':
			case 'C':
				note = 0;
				break;
			case 'd':
			case 'D':
				note = 2;
				break;
			case 'e':
			case 'E':
				note = 4;
				break;
			case 'f':
			case 'F':
				note = 5;
				break;
			case 'g':
			case 'G':
				note = 7;
				break;
			case 'a':
			case 'A':
				note = 9;
				break;
			case 'b':
			case 'B':
				note = 11;
				break;
			default:
				note = 12;
			}
			if (song[i] == '#') {
				note++;
				i++;
			}
			int octave = parseInt(song, &i);
			if (octave == 0)
				octave = o;
			songToPlay.notes[noteIndex].frequency = (noteFrequencies[note] << octave).getu32();
			if (song[i] == '.') {
				songToPlay.notes[noteIndex].duration = songToPlay.notes[noteIndex].duration * 3 / 2;
				i++;
			}
			if (song[i] == '-') {
				songToPlay.notes[noteIndex].sweepToNext = 1;
				i++;
			} else {
				songToPlay.notes[noteIndex].sweepToNext = 0;
			}
			if (song[i] == '$') {
				i++;
				songToPlay.notes[noteIndex].quieter = parseInt(song, &i, 0);
			} else {
				songToPlay.notes[noteIndex].quieter = 0;
			}
		}
		totalDuration += songToPlay.notes[noteIndex].duration;
		songToPlay.numNotes = noteIndex + 1;
		noteIndex++;
		// go forward to next ,
		while (song[i] != ',' && song[i] != '\0' && i++ < MAX_RTTTL_TEXT_LENGTH) {
		}
	}
	songToPlay.notes[songToPlay.numNotes - 1].sweepToNext = 0;

	soundDuration = totalDuration;
	soundStart = 0;
	soundType = 2;
}

#endif