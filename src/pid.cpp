#include "global.h"

i16 throttles[4] = {0, 0, 0, 0};
i32 targetRpm = 0;
i32 idleRpm = 0;
i32 rearRpm = 0;
i32 targetFrontLowThres = 0, targetFrontHighThres = 0, targetRearLowThres = 0, targetRearHighThres = 0;
u8 frontRearRatio = 0;
fix64 rpmErrorSum[4] = {0, 0, 0, 0};
i16 pGainNice = 50, iGainNice = 50, dGainNice = 50;
PT1 rpmFilter[4] = {PT1(RPM_FILTER_CUTOFF, PID_RATE), PT1(RPM_FILTER_CUTOFF, PID_RATE), PT1(RPM_FILTER_CUTOFF, PID_RATE), PT1(RPM_FILTER_CUTOFF, PID_RATE)};
PT3 dFilter[4] = {PT3(RPM_D_FILTER_CUTOFF, PID_RATE), PT3(RPM_D_FILTER_CUTOFF, PID_RATE), PT3(RPM_D_FILTER_CUTOFF, PID_RATE), PT3(RPM_D_FILTER_CUTOFF, PID_RATE)};
fix32 rpmPGain = 0, rpmIGain = 0, rpmDGain = 0;
fix32 rpmError[4] = {0, 0, 0, 0};
u8 minThrottle = 40;
u8 rpmThresPct = 10;
fix64 rpmErrorSumLimit = 0;
bool iTermSet[4] = {false, false, false, false};
i32 setITermAtRpmFront = 0, setITermAtRpmRear = 0;
u16 motorKv = 3750;
elapsedMicros blackboxTimer = 0;

#ifdef USE_BLACKBOX
#define BLACKBOX_SIZE 1000
u32 rpms[BLACKBOX_SIZE][4] = {0};
u16 recThrottles[BLACKBOX_SIZE][4] = {0};
i32 pids[BLACKBOX_SIZE][3] = {0};
i32 rpmIndex = -1;
u32 blackboxTimes[BLACKBOX_SIZE] = {0};
bool blFired[BLACKBOX_SIZE] = {false};
#if HW_VERSION == 2
u8 solenoidPosition[BLACKBOX_SIZE] = {0};
u16 blSolenoidCurrent[BLACKBOX_SIZE] = {0};
i32 yAccel[BLACKBOX_SIZE] = {0};
#endif
#endif

void resetITermSet() {
	for (u8 i = 0; i < 4; i++) {
		iTermSet[i] = false;
	}
}

void calcTargetRpms(MenuItem *_item) {
	i32 delta = targetRpm * rpmThresPct / 100; // same absolute delta for both front and rear
	rearRpm = targetRpm * 100 / (100 + frontRearRatio);
	targetFrontLowThres = targetRpm - delta;
	targetFrontHighThres = targetRpm + delta;
	targetRearLowThres = rearRpm - delta;
	targetRearHighThres = rearRpm + delta;
}

void calcITermRpms() {
	// This function calculates the rpm, at which the approximate final I term value should be set
	// The closer the target RPM is to the physically possible maximum RPM, the later the I term should be set
	// 67% of maximum is treated as the actual maximum, and lower values are scaled, e.g. 40% -> 60%, 20% -> 30%
	// I.e. if the target RPM is 60%-100% of the maximum RPM, the I term should be only when the target RPM is reached
	// if the target RPM is 45% (=> 75% after scaling) of the maximum RPM, the I term should be set at targetRpm - 4000 * (1 - .75), so at targetRpm - 1000
	// This is because at higher RPM, the throttle acceleration and therefore overshoot are lower. Meaning, if we were to set the I term too early, it would just rise up again.
	i32 maxRpm = ((fix64)batVoltage * motorKv * fix32(0.6f)).geti32();
	if (targetRpm < maxRpm)
		setITermAtRpmFront = targetRpm - 6000 + 6000 * targetRpm / maxRpm;
	else
		setITermAtRpmFront = targetRpm;
	if (rearRpm < maxRpm)
		setITermAtRpmRear = rearRpm - 6000 + 6000 * rearRpm / maxRpm;
	else
		setITermAtRpmRear = rearRpm;
}

void applyPidSettings(MenuItem *_item) {
	rpmPGain.setRaw(pGainNice << PID_P_SHIFT);
	rpmIGain.setRaw(iGainNice << PID_I_SHIFT);
	rpmDGain.setRaw(dGainNice << PID_D_SHIFT);
	rpmErrorSumLimit = 2000 / rpmIGain.getf32();
}

void resetPid() {
	for (u8 i = 0; i < 4; i++) {
		rpmErrorSum[i] = 0;
		dFilter[i].set(0);
	}
}

#ifdef USE_BLACKBOX
void prepareBlackbox() {
	memset(rpms, 0, sizeof(rpms));
	memset(recThrottles, 0, sizeof(recThrottles));
	memset(pids, 0, sizeof(pids));
	blackboxTimer = 0;
	memset(blFired, 0, sizeof(blFired));
}
void blSetFired() {
	if (rpmIndex < BLACKBOX_SIZE && rpmIndex >= 0)
		blFired[rpmIndex] = true;
}
void checkPrintRpm() {
	if (Serial.available()) {
		Serial.read();
		Serial.printf("P: %d; I: %d; D: %d; minThrottle: %d; rpmThresPct: %d\n", pGainNice, iGainNice, dGainNice, minThrottle, rpmThresPct);
		Serial.printf("targetRpm: %d; idleRpm: %d; rearRpm: %d; frontRearRatio: %d\n", targetRpm, idleRpm, rearRpm, frontRearRatio);
		Serial.printf("rpmInRange: %d\n", rpmInRangeTime);
		Serial.printf("voltage: %.2f\n", ((fix32)batVoltage).getf32());
		Serial.println();
		Serial.print(";RPM 1;RPM 2;RPM 3;RPM 4;Throttle 1;Throttle 2;Throttle 3;Throttle 4;P 1;I 1;D 1;Trigger");
#if HW_VERSION == 2
		Serial.print(";SolenoidPos;SolenoidCurr(mA);yAccel");
#endif
		Serial.println();
		for (int i = 0; i < BLACKBOX_SIZE; i++) {
			bool zero = true;
			Serial.printf("%d,%03d;", blackboxTimes[i] / 1000, blackboxTimes[i] % 1000);
			for (int j = 0; j < 4; j++) {
				Serial.print(rpms[i][j]);
				Serial.print(";");
				if (rpms[i][j] != 0) zero = false;
			}
			for (int j = 0; j < 4; j++) {
				Serial.print(recThrottles[i][j] * 10);
				Serial.print(";");
				if (recThrottles[i][j] != 0) zero = false;
			}
			Serial.print(constrain(pids[i][0] * 10, -20000, targetRpm));
			Serial.print(";");
			Serial.print(constrain(pids[i][1] * 10, -20000, targetRpm));
			Serial.print(";");
			Serial.print(constrain(pids[i][2] * 10, -20000, targetRpm));
			Serial.print(";");
			if (blFired[i])
				Serial.print(targetRpm);
			else
				Serial.print(0);
#if HW_VERSION == 2
			Serial.print(";");
			Serial.print(targetRpm / 4 * solenoidPosition[i]);
			Serial.print(";");
			Serial.print(blSolenoidCurrent[i]);
			Serial.print(";");
			Serial.print(yAccel[i]);
#endif
			Serial.println();
			if (zero) break;
			if (i % 5 == 0) sleep_ms(3);
		}
	}
	while (Serial.available())
		Serial.read();
}
#endif

void pidLoop(i32 rpmFront, i32 rpmRear) {
#ifdef USE_BLACKBOX
	rpmIndex++;
#endif
	i32 target[4] = {rpmFront, rpmFront, rpmRear, rpmRear};
	static i32 lastRpm[4] = {0};
	const i32 maxRpm = ((fix64)batVoltage * motorKv).geti32();

	// see maxT calculation below for explanation
	static elapsedMillis lastCall = 0;
	static elapsedMillis spinupTimerM = 0;
	i32 spinupTimer = 0;
	if (lastCall > 200) {
		spinupTimerM = 0;
	}
	lastCall = 0;
	spinupTimer = spinupTimerM;
	i32 timeStrength = (spinupTimer - 300) * 1024 / 700;
	timeStrength = constrain(timeStrength, 0, 1024);
	timeStrength = 1024 - timeStrength;

	for (int i = 0; i < 4; i++) {
		fix32 rpm = rpmFilter[i].update(fix32().setRaw((i32)escRpm[i] << RPM_SHIFT));
		rpmError[i] = fix32().setRaw(target[i] << RPM_SHIFT) - rpm;
		rpmErrorSum[i] = rpmErrorSum[i] + rpmError[i];
		if (rpmErrorSum[i] < 0) {
			rpmErrorSum[i] = 0;
		} else if (rpmErrorSum[i] > rpmErrorSumLimit) {
			rpmErrorSum[i] = rpmErrorSumLimit;
		}
		// Set I term to approximate final value based on KV and battery voltage to avoid overshoot
		if (!iTermSet[i] && ((i < 2 && escRpm[i] > setITermAtRpmFront) || (i >= 2 && escRpm[i] > setITermAtRpmRear))) {
			iTermSet[i] = true;
			i32 maxRpm = ((fix64)batVoltage * motorKv * fix32(0.8f)).geti32();
			if (i < 2)
				rpmErrorSum[i] = fix64(2000 * setITermAtRpmFront / maxRpm) / rpmIGain;
			else
				rpmErrorSum[i] = fix64(2000 * setITermAtRpmRear / maxRpm) / rpmIGain;
		}
		fix32 pTerm = rpmPGain * rpmError[i];
		fix32 iTerm = rpmIGain * rpmErrorSum[i];
		static i32 lastDelta[4] = {0};
		i32 delta = lastRpm[i] - (i32)escRpm[i];
		delta = constrain(delta, lastDelta[i] - 5, lastDelta[i] + 5); // limit jerk
		lastDelta[i] = delta;
		fix32 change = dFilter[i].update((fix32)(constrain(delta, -300, 300))); // limit to sane acceleration to reduce effect of noise
		fix32 dTerm = rpmDGain * change;
		dTerm = constrain(dTerm, -2000, 2000);
		i32 throttle = pTerm.geti32() * 8 + (iTerm + dTerm).geti32();
		// limit throttle to 2000 at startup and to to 1.5*calculatedThrottle at target RPM, with linear interpolation in between
		const i32 calculatedThrottle = 2000 * target[i] / maxRpm;
		// maxT = (2500 * (1 - escRpm / target) + 1.5 * calculatedThrottle * escRpm / target)
		const i32 progress10bit = 1024 * escRpm[i] / target[i]; // scale with 1024 to get more than 1 bit resolution
		i32 maxT = (2500 * (1024 - progress10bit) + calculatedThrottle * progress10bit * 3 / 2) >> 10;
		// this limit shall be only active at targets from 7k up, with full limit from 10k onwards
		// furthermore, the limit shall be full within the first 300ms, and then linearly decrease to 0 at 1s
		i32 rpmStrength = (target[i] - 7000) * 1024 / 3000;
		rpmStrength = constrain(rpmStrength, 0, 1024);
		const i32 limitStrength = (rpmStrength * timeStrength) >> 10;
		maxT = (maxT * limitStrength + 2000 * (1024 - limitStrength)) >> 10;
		throttle = constrain(throttle, minThrottle, maxT);
		throttle = constrain(throttle, minThrottle, 2000);
		throttles[i] = throttle;
		lastRpm[i] = escRpm[i];
#ifdef USE_BLACKBOX
		if (rpmIndex < BLACKBOX_SIZE && rpmIndex >= 0) {
			rpms[rpmIndex][i] = (u32)escRpm[i];
			recThrottles[rpmIndex][i] = (u16)throttles[i];
			if (i == 0) {
				pids[rpmIndex][0] = pTerm.geti32() * 8;
				pids[rpmIndex][1] = iTerm.geti32();
				pids[rpmIndex][2] = dTerm.geti32();
				blackboxTimes[rpmIndex] = blackboxTimer;
#if HW_VERSION == 2
				solenoidPosition[rpmIndex] = pusherFullyRetracted ? 1 : 0 + (pusherFullyExtended ? 2 : 0);
				blSolenoidCurrent[rpmIndex] = (solenoidCurrent * 1000).geti32();
				yAccel[rpmIndex] = accelDataRaw[1];
#endif
			}
		}
#endif
	}
}
