# Firmware Integration Guide

## Overview
This guide explains how to integrate the ML model into your RP2040 firmware with tunable parameters and UI feedback.

## 1. Tunable Consecutive Predictions Count

### Current Implementation
The model currently uses a hardcoded value of 20 consecutive predictions before activating flywheels.

### Recommended Implementation

#### A. Add to Settings Structure
```cpp
// In src/settings.h or similar
typedef struct {
    // ... existing settings ...
    uint8_t ml_consecutive_required;  // Default: 20, Range: 5-50
    float ml_threshold;                // Default: 0.100, Range: 0.05-0.50
} UserSettings;

// Default values
#define DEFAULT_ML_CONSECUTIVE 20
#define MIN_ML_CONSECUTIVE 5
#define MAX_ML_CONSECUTIVE 50
```

#### B. Update ML Predictor
```cpp
// In src/ml_predictor.cpp or where prediction happens

class MLPredictor {
private:
    uint8_t consecutive_count;
    uint8_t consecutive_required;  // Now configurable!

public:
    void set_consecutive_required(uint8_t count) {
        consecutive_required = constrain(count, MIN_ML_CONSECUTIVE, MAX_ML_CONSECUTIVE);
    }

    bool predict_and_filter(float* features) {
        // Get raw prediction from model
        bool raw_prediction = predict_prefire(features);  // From rf_model.h

        if (raw_prediction) {
            consecutive_count++;
            if (consecutive_count >= consecutive_required) {
                return true;  // Activate flywheels!
            }
        } else {
            consecutive_count = 0;
        }

        return false;
    }

    // For UI feedback
    uint8_t get_consecutive_count() const { return consecutive_count; }
    uint8_t get_consecutive_required() const { return consecutive_required; }
};
```

#### C. Menu System Integration
```cpp
// In your menu system
void render_ml_settings_menu() {
    // Add menu item for consecutive count
    menu_item("Sensitivity", [](){
        adjust_consecutive_count();
    });
}

void adjust_consecutive_count() {
    // Use rotary encoder or buttons to adjust
    uint8_t value = settings.ml_consecutive_required;

    // Display: "Consecutive: 20"
    // Lower = more sensitive (faster trigger, more false alarms)
    // Higher = less sensitive (slower trigger, fewer false alarms)

    if (button_pressed(BTN_UP)) {
        value = min(value + 1, MAX_ML_CONSECUTIVE);
    }
    if (button_pressed(BTN_DOWN)) {
        value = max(value - 1, MIN_ML_CONSECUTIVE);
    }

    settings.ml_consecutive_required = value;
    ml_predictor.set_consecutive_required(value);
    save_settings();  // Persist to flash
}
```

### Sensitivity Profiles (Optional)
```cpp
// Preset sensitivity levels
enum MLSensitivity {
    SENSITIVITY_LOW = 30,      // Very conservative, minimal false alarms
    SENSITIVITY_MEDIUM = 20,   // Balanced (recommended)
    SENSITIVITY_HIGH = 15,     // Quick trigger, more false alarms
    SENSITIVITY_MAX = 10       // Instant trigger, many false alarms
};

// Quick toggle in menu
void set_sensitivity_preset(MLSensitivity level) {
    ml_predictor.set_consecutive_required((uint8_t)level);
}
```

---

## 2. Confidence Bar on Home Screen

### Visual Feedback Implementation

#### A. Simple Progress Bar
```cpp
// In your main display rendering loop
void render_home_screen() {
    // ... existing home screen elements ...

    // Show ML confidence bar
    draw_ml_confidence_bar();
}

void draw_ml_confidence_bar() {
    uint8_t current = ml_predictor.get_consecutive_count();
    uint8_t required = ml_predictor.get_consecutive_required();

    if (current == 0) return;  // Don't show if no predictions

    // Calculate progress (0-100%)
    uint8_t progress = (current * 100) / required;

    // Draw bar at bottom of screen
    const int BAR_X = 10;
    const int BAR_Y = 110;  // Near bottom
    const int BAR_WIDTH = 108;  // Screen width - margins
    const int BAR_HEIGHT = 8;

    // Background
    display.drawRect(BAR_X, BAR_Y, BAR_WIDTH, BAR_HEIGHT, SSD1306_WHITE);

    // Fill based on progress
    int fill_width = (progress * (BAR_WIDTH - 2)) / 100;
    display.fillRect(BAR_X + 1, BAR_Y + 1, fill_width, BAR_HEIGHT - 2, SSD1306_WHITE);

    // Optional: Text label
    display.setTextSize(1);
    display.setCursor(BAR_X, BAR_Y - 10);
    display.printf("%d/%d", current, required);
}
```

#### B. Alternative: Animated "Warming Up" Indicator
```cpp
void draw_ml_indicator() {
    uint8_t current = ml_predictor.get_consecutive_count();
    uint8_t required = ml_predictor.get_consecutive_required();

    if (current == 0) return;

    // Show pulsing icon when building up predictions
    const int ICON_X = 100;
    const int ICON_Y = 5;

    // Animate based on progress
    if (current < required) {
        // "Warming up" - pulse animation
        uint8_t pulse = (millis() / 100) % 4;
        for (int i = 0; i <= pulse; i++) {
            display.drawCircle(ICON_X + i*3, ICON_Y, 2, SSD1306_WHITE);
        }
    } else {
        // Ready! - solid indicator
        display.fillCircle(ICON_X, ICON_Y, 3, SSD1306_WHITE);
        display.setCursor(ICON_X + 5, ICON_Y - 3);
        display.print("READY");
    }
}
```

#### C. Advanced: Heatmap Display
```cpp
void draw_ml_heatmap() {
    // Show prediction strength over last N samples
    const int HISTORY_SIZE = 50;
    static uint8_t prediction_history[HISTORY_SIZE] = {0};
    static int history_index = 0;

    // Update history
    prediction_history[history_index] = ml_predictor.get_raw_prediction() ? 1 : 0;
    history_index = (history_index + 1) % HISTORY_SIZE;

    // Draw mini heatmap
    const int HEATMAP_X = 0;
    const int HEATMAP_Y = 120;
    const int PIXEL_WIDTH = 2;

    for (int i = 0; i < HISTORY_SIZE; i++) {
        int x = HEATMAP_X + (i * PIXEL_WIDTH);
        if (prediction_history[i]) {
            display.drawLine(x, HEATMAP_Y, x, HEATMAP_Y + 8, SSD1306_WHITE);
        }
    }
}
```

---

## 3. Complete Integration Example

```cpp
// Main loop integration
void loop() {
    // Read IMU data
    float accel[3], gyro[3];
    imu.read(accel, gyro);

    // Extract features (using feature window)
    float features[42];
    feature_extractor.add_sample(accel, gyro);
    feature_extractor.extract_features(features);

    // Get prediction with filtering
    bool should_fire = ml_predictor.predict_and_filter(features);

    // Update display
    if (current_screen == SCREEN_HOME) {
        render_home_screen();
        draw_ml_confidence_bar();  // Show confidence
    }

    // Activate flywheels if prediction positive
    if (should_fire) {
        flywheels.spin_up();
    }
}
```

---

## 4. User Experience Recommendations

### Menu Structure
```
Settings
├── ML Prediction
│   ├── Sensitivity: [Low|Med|High|Max]  ← Quick presets
│   ├── Consecutive: [5-50]               ← Fine tune
│   ├── Threshold: [0.05-0.50]            ← Advanced users
│   └── Show Confidence: [On|Off]         ← Toggle bar display
└── ...
```

### Home Screen Layout
```
┌────────────────────────┐
│ Battery: 85%    [WiFi] │  ← Status icons
│                        │
│   MODE: FULL AUTO      │  ← Current mode
│                        │
│   Ammo: 25/30          │  ← Ammo count
│                        │
│   [●●●○○] 15/20        │  ← ML confidence bar
└────────────────────────┘
```

---

## 5. Tuning Guide for Users

### What Each Setting Does

**Consecutive Count** (Default: 20)
- **Lower (5-15)**:
  - ✓ Faster trigger response
  - ✓ Catches more shots
  - ✗ More false activations
  - Best for: Competitive play, quick reactions needed

- **Higher (25-50)**:
  - ✓ Very few false activations
  - ✓ More predictable behavior
  - ✗ Slower to respond
  - Best for: Casual play, battery conservation

**Confidence Bar**
- Green zone (80-100%): About to fire
- Yellow zone (50-79%): Building confidence
- Red zone (0-49%): Early detection

### Finding Your Sweet Spot
1. Start with default (20)
2. Test in typical play scenario
3. If too many false alarms: Increase by 5
4. If missing triggers: Decrease by 5
5. Repeat until comfortable

---

## 6. Debug Features (Optional)

### Serial Monitoring
```cpp
void debug_ml_predictions() {
    Serial.printf("ML: consecutive=%d/%d, threshold=%0.3f, raw_prob=%0.3f\n",
        ml_predictor.get_consecutive_count(),
        ml_predictor.get_consecutive_required(),
        settings.ml_threshold,
        ml_predictor.get_last_probability()
    );
}
```

### Event Logging
```cpp
// Log to SD card or serial
void log_ml_event(bool is_true_positive) {
    ml_event_log.append({
        .timestamp = millis(),
        .was_prediction = true,
        .was_actual_trigger = is_true_positive,
        .consecutive_at_trigger = ml_predictor.get_consecutive_count()
    });
}
```

---

## 7. Performance Tips

### Optimization
```cpp
// Cache feature extraction to avoid recomputation
class OptimizedMLPredictor {
private:
    float feature_cache[42];
    bool cache_valid = false;

public:
    void invalidate_cache() { cache_valid = false; }

    bool predict() {
        if (!cache_valid) {
            feature_extractor.extract_features(feature_cache);
            cache_valid = true;
        }
        return predict_prefire(feature_cache);
    }
};
```

### Memory Management
- Feature extraction: ~200 bytes (50 samples × 6 features)
- Model: ~104 KB (fixed)
- Prediction state: ~10 bytes
- Total: ~105 KB (fits easily in 264KB RP2040 RAM)

---

## 8. Testing Checklist

- [ ] Default settings (consecutive=20) works as expected
- [ ] Can adjust consecutive count in menu
- [ ] Settings persist after reboot
- [ ] Confidence bar shows correctly
- [ ] Bar updates in real-time
- [ ] No performance impact on main loop
- [ ] False alarm rate acceptable at different sensitivity levels
- [ ] UI is responsive and clear

---

## Model Performance Reference

From validation testing:

| Consecutive | Event Recall | False Alarms/s | Use Case |
|-------------|--------------|----------------|----------|
| 10          | 100%         | ~8-10          | Maximum sensitivity |
| 15          | 100%         | ~5-7           | High sensitivity |
| 20          | 100%         | ~4-5           | **Recommended** |
| 25          | 95%          | ~2-3           | Conservative |
| 30          | 90%          | ~1-2           | Very conservative |

Choose based on your preference for speed vs. false alarm tolerance.
