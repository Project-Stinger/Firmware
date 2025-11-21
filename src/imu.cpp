#include "global.h"
#if HW_VERSION == 2
#define IMU_RATE (PID_RATE / 2)

// COORDINATE SYSTEM:
// X: forward / roll right
// Y: right / pitch up
// Z: down / yaw right
// (Tait-Bryan angles)

// on PCB (Bee) (switched to tait bryan in gyro.cpp before reaching this file)
// X: up / yaw left
// Y: rearward / roll left
// Z: right / pitch up

const f32 FRAME_TIME = 1. / IMU_RATE;
const f32 RAW_TO_HALF_ANGLE = RAW_TO_RAD_PER_SEC * FRAME_TIME / 2;
const f32 ANGLE_CHANGE_LIMIT = .0002;
const f32 ANGLE_CHANGE_LIMIT2 = .02; // faster setup time at the start
bool useNormalLimit = false;
PT1 accelDataFiltered[3] = {PT1(100, IMU_RATE), PT1(100, IMU_RATE), PT1(100, IMU_RATE)};

fix32 roll, pitch, yaw;
PT2 vAccel(20, IMU_RATE), fAccel(20, IMU_RATE), rAccel(20, IMU_RATE);
fix32 upVel = 0;
PT2 vAccelSlow(4, IMU_RATE);
fix32 cosRoll, cosPitch, cosYaw, sinRoll, sinPitch, sinYaw;

bool freeFallDetectionEnabled = true;
bool freeFallDetected = false;

Quaternion q;

void imuInit() {
	pitch = 0; // pitch up
	roll = 0; // roll right
	yaw = 0; // yaw right
	q.w = 1;
	q.v[0] = 0;
	q.v[1] = 0;
	q.v[2] = 0;
	initFixTrig();
}

void __not_in_flash_func(updateFromGyro)() {
	// quaternion of all 3 axis rotations combined

	f32 all[] = {-gyroDataRaw[1] * RAW_TO_HALF_ANGLE, -gyroDataRaw[0] * RAW_TO_HALF_ANGLE, gyroDataRaw[2] * RAW_TO_HALF_ANGLE};
	Quaternion buffer = q;
	q.w += (-buffer.v[0] * all[0] - buffer.v[1] * all[1] - buffer.v[2] * all[2]);
	q.v[0] += (+buffer.w * all[0] - buffer.v[1] * all[2] + buffer.v[2] * all[1]);
	q.v[1] += (+buffer.w * all[1] + buffer.v[0] * all[2] - buffer.v[2] * all[0]);
	q.v[2] += (+buffer.w * all[2] - buffer.v[0] * all[1] + buffer.v[1] * all[0]);

	Quaternion_normalize(&q, &q);
}

f32 orientation_vector[3];
void __not_in_flash_func(updateFromAccel)() {
	// filter accel data
	accelDataFiltered[0].update(accelDataRaw[0]);
	accelDataFiltered[1].update(accelDataRaw[1]);
	accelDataFiltered[2].update(accelDataRaw[2]);

	// Formula from http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/transforms/index.htm
	// p2.x = w*w*p1.x + 2*y*w*p1.z - 2*z*w*p1.y + x*x*p1.x + 2*y*x*p1.y + 2*z*x*p1.z - z*z*p1.x - y*y*p1.x;
	// p2.y = 2*x*y*p1.x + y*y*p1.y + 2*z*y*p1.z + 2*w*z*p1.x - z*z*p1.y + w*w*p1.y - 2*x*w*p1.z - x*x*p1.y;
	// p2.z = 2*x*z*p1.x + 2*y*z*p1.y + z*z*p1.z - 2*w*y*p1.x - y*y*p1.z + 2*w*x*p1.y - x*x*p1.z + w*w*p1.z;
	// with p1.x = 0, p1.y = 0, p1.z = -1, things can be simplified

	orientation_vector[0] = q.w * q.v[1] * -2 + q.v[0] * q.v[2] * -2;
	orientation_vector[1] = q.v[1] * q.v[2] * -2 + q.w * q.v[0] * 2;
	orientation_vector[2] = -q.v[2] * q.v[2] + q.v[1] * q.v[1] + q.v[0] * q.v[0] - q.w * q.w;

	f32 accelVectorNorm = sqrtf((i32)accelDataRaw[1] * (i32)accelDataRaw[1] + (i32)accelDataRaw[0] * (i32)accelDataRaw[0] + (i32)accelDataRaw[2] * (i32)accelDataRaw[2]);
	f32 accelVector[3];
	if (accelVectorNorm > 0.01f) {
		f32 invAccelVectorNorm = 1 / accelVectorNorm;
		accelVector[0] = invAccelVectorNorm * accelDataRaw[1];
		accelVector[1] = invAccelVectorNorm * accelDataRaw[0];
		accelVector[2] = invAccelVectorNorm * -accelDataRaw[2];
	} else
		return;
	Quaternion shortest_path;
	Quaternion_from_unit_vecs(orientation_vector, accelVector, &shortest_path);

	f32 axis[3];
	f32 accAngle = Quaternion_toAxisAngle(&shortest_path, axis); // reduces effect of accel noise on attitude

	if (useNormalLimit) {
		if (accAngle > ANGLE_CHANGE_LIMIT) accAngle = ANGLE_CHANGE_LIMIT;
	} else {
		if (accAngle > ANGLE_CHANGE_LIMIT2) accAngle = ANGLE_CHANGE_LIMIT2;
		if (bootTimer > 2000) useNormalLimit = true;
	}

	// Quaternion c;
	// Quaternion_fromAxisAngle(axis, accAngle, &c);
	// Quaternion_multiply(&c, &q, &q);
	f32 c[3]; // correction quaternion, but w is 1
	f32 co = accAngle * 0.5f;
	c[0] = axis[0] * co;
	c[1] = axis[1] * co;
	c[2] = axis[2] * co;

	Quaternion buffer;
	buffer.w = q.w - c[0] * q.v[0] - c[1] * q.v[1] - c[2] * q.v[2];
	buffer.v[0] = c[0] * q.w + q.v[0] + c[1] * q.v[2] - c[2] * q.v[1];
	buffer.v[1] = q.v[1] - c[0] * q.v[2] + c[1] * q.w + c[2] * q.v[0];
	buffer.v[2] = q.v[2] + c[0] * q.v[1] - c[1] * q.v[0] + c[2] * q.w;
	q = buffer;

	Quaternion_normalize(&q, &q);
}

void __not_in_flash_func(updatePitchRollValues)() {
	startFixTrig();
	roll = atan2Fix(2 * (q.w * q.v[0] - q.v[1] * q.v[2]), 1 - 2 * (q.v[0] * q.v[0] + q.v[1] * q.v[1]));
	pitch = asinf(2 * (q.w * q.v[1] + q.v[2] * q.v[0]));
	yaw = atan2Fix(2 * (q.v[0] * q.v[1] - q.w * q.v[2]), 1 - 2 * (q.v[1] * q.v[1] + q.v[2] * q.v[2]));
	cosPitch = cosFix(pitch);
	cosRoll = cosFix(roll);
	sinPitch = sinFix(pitch);
	sinRoll = sinFix(roll);
	cosYaw = cosFix(yaw);
	sinYaw = sinFix(yaw);
}

void __not_in_flash_func(updateAccels)() {
	vAccel.update((cosRoll * cosPitch * accelDataFiltered[2] * RAW_TO_M_PER_SEC2 + sinRoll * cosPitch * accelDataFiltered[0] * RAW_TO_M_PER_SEC2 - sinPitch * accelDataFiltered[1] * RAW_TO_M_PER_SEC2) - fix32(9.81));
	rAccel.update(cosRoll * accelDataFiltered[0] * RAW_TO_M_PER_SEC2 - sinRoll * accelDataFiltered[2] * RAW_TO_M_PER_SEC2);
	fAccel.update(cosPitch * accelDataFiltered[1] * RAW_TO_M_PER_SEC2 + sinPitch * sinRoll * accelDataFiltered[0] * RAW_TO_M_PER_SEC2 + sinPitch * cosRoll * accelDataFiltered[2] * RAW_TO_M_PER_SEC2);
	vAccelSlow.update(vAccel);
	upVel += fix32(vAccel) / IMU_RATE;
	if (upVel < 0)
		upVel += fix32(0.5 / IMU_RATE); // compensate for drift up to 0.5m/s²
	else if (upVel > 0)
		upVel -= fix32(0.5 / IMU_RATE);
}

void __not_in_flash_func(updateAtti1)() {
	updateFromGyro();
}

void __not_in_flash_func(updateAtti2)() {
	updateFromAccel();
	updatePitchRollValues();
	updateAccels();
}

void __not_in_flash_func(freeFallDetection)() {
	if (freeFallDetectionEnabled) {
		static elapsedMillis soundTimer = 0;
		static i32 freeFallFrames = 0;
		if ((fix32(vAccelSlow) + fix32(9.81)).abs() < fix32(0.2 * 9.81)) {
			freeFallFrames += 5;
		}
		if (fix32(vAccelSlow) > fix32(0.3 * 9.81) && freeFallFrames <= 0) {
			freeFallFrames -= 5;
			if (freeFallFrames < -IMU_RATE) { // 1s recovery during standstill, 360ms during freefall
				freeFallFrames = -IMU_RATE;
			}
		}
		if (freeFallFrames > 0) {
			freeFallFrames--;
		} else if (freeFallFrames < 0) {
			freeFallFrames++;
		}
		if (freeFallFrames > IMU_RATE * 2 / 3 && upVel < -3) { // 180ms AND 3m/s down (= 45cm free fall)
			if (bootTimer > 5000) { // allow vAccel to stabilize
				if (soundTimer > 1000) {
					makeRtttlSound("fall:d=4,o=5,b=1000:g,f#,f,e,d#,d");
					soundTimer = 0;
				}
				freeFallDetected = true;
				freeFallFrames = 0;
			} else {
				freeFallFrames = 0;
			}
		}
	}
}
#endif