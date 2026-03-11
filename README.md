# AI-Powered Smart Parking System
### CIT-U Campus | YOLOv5n + Raspberry Pi 5 + Firebase + React

---

## Project Structure

```
smart-parking/
├── pi_system/               ← Runs on Raspberry Pi 5
│   ├── main.py              ← Entry point (start here)
│   ├── detector.py          ← YOLO vehicle detection + occupancy logic
│   ├── auto_mapper.py       ← Auto-discovers parking slot layout
│   ├── firebase_sync.py     ← Pushes data to Firebase
│   └── requirements.txt     ← Python dependencies
│
└── webapp/
    └── SmartParkingApp.jsx  ← React web dashboard (real-time map)
```

---

## Raspberry Pi Setup

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt --break-system-packages
```

### Step 2 — Set up Firebase
1. Go to https://console.firebase.google.com
2. Create a new project → Enable Realtime Database
3. Go to Project Settings → Service Accounts → Generate new private key
4. Save the downloaded JSON as `serviceAccountKey.json` in `pi_system/`
5. Copy your database URL (e.g. https://your-project-default-rtdb.firebaseio.com)
6. Update `main.py` with your database URL

### Step 3 — Connect camera
```bash
# Test camera is detected
ls /dev/video*   # Should show /dev/video0

# Test capture
python3 -c "import cv2; cap=cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"
```

### Step 4 — Run
```bash
cd pi_system
python3 main.py
```

The system will:
1. **Auto-map** for the first ~5 minutes (collecting parking slot positions)
2. Switch to **live occupancy detection** automatically
3. Push updates to Firebase every second

---

## Web App Setup

### Option A — Use as React component
```bash
# In your React project
npm install firebase
# Copy SmartParkingApp.jsx into your src/
# Connect real Firebase in useParkingData() hook
```

### Option B — Quick local test
The app works in demo mode (simulated live updates) without Firebase.
Just open it in any React sandbox (CodeSandbox, StackBlitz, etc.)

### Connecting Real Firebase to Web App
Replace the `useParkingData` hook's `useEffect` with:

```js
import { initializeApp } from "firebase/app";
import { getDatabase, ref, onValue } from "firebase/database";

const firebaseConfig = {
  apiKey: "...",
  databaseURL: "https://your-project-default-rtdb.firebaseio.com",
  // ... rest of config
};
const app = initializeApp(firebaseConfig);
const database = getDatabase(app);

// Inside useParkingData():
useEffect(() => {
  const parkingRef = ref(database, "/parking");
  onValue(parkingRef, (snapshot) => {
    const data = snapshot.val();
    if (data?.slots) setSlots(data.slots);
    if (data?.summary) setLastUpdated(data.summary.last_updated);
  });
}, []);
```

---

## How the AI Works

```
Camera (1080p USB)
    ↓ video frame
Raspberry Pi 5
    ↓ resize + normalize
YOLOv5n (PyTorch)
    ↓ detects vehicles → bounding boxes
Auto-Mapper (DBSCAN clustering)
    ↓ maps boxes to slot IDs
Occupancy Logic (IoU overlap check)
    ↓ "Slot A1 = Occupied"
Temporal Smoother (5-frame majority vote)
    ↓ stable status
Firebase Realtime DB
    ↓ live sync
React Web App → shows to user
```

---

## Performance Targets (from your paper)
| Metric | Target |
|--------|--------|
| Accuracy | ≥ 85% |
| Precision | ≥ 88% |
| Recall | ≥ 82% |
| F1-Score | ≥ 0.85 |
| End-to-end latency | < 5 seconds |
| CPU usage on Pi 5 | < 80% |

---

## Files NOT included (you must provide)
- `serviceAccountKey.json` — your Firebase credentials (never share this)
- `yolov5n.pt` — downloads automatically on first run via Ultralytics

---

## Language Stack
| Component | Language/Framework |
|-----------|-------------------|
| AI Detection | Python + PyTorch (YOLO) |
| Image Processing | Python + OpenCV |
| Auto-mapping | Python + scikit-learn (DBSCAN) |
| Edge Device | Python on Raspberry Pi OS |
| Database | Firebase Realtime DB |
| Web App | React (JavaScript) |
