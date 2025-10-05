import cv2
import numpy as np

# -------------------------------
# Configuration
# -------------------------------
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 600
LIGHT_GAP = 80           # gap between rays
OBJECT_WIDTH = 120
OBJECT_HEIGHT = 160
OBJECT_COLOR = (0, 255, 0)
LIGHT_COLOR = (255, 255, 150)
BLOCKED_COLOR = (255, 80, 80)
BG_COLOR = (10, 10, 20)
SPEED = 3                # object movement speed

# -------------------------------
# Initialize Canvas
# -------------------------------
canvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
canvas[:] = BG_COLOR

# Initial object position
object_x = 100
object_y = (WINDOW_HEIGHT - OBJECT_HEIGHT) // 2
direction = 1  # 1 -> right, -1 -> left

# -------------------------------
# Simulation Loop
# -------------------------------
while True:
    frame = canvas.copy()

    # Draw vertical light rays
    for x in range(0, WINDOW_WIDTH, LIGHT_GAP):
        cv2.line(frame, (x, 0), (x, WINDOW_HEIGHT), LIGHT_COLOR, 2)

    # Draw moving object
    obj_top_left = (object_x, object_y)
    obj_bottom_right = (object_x + OBJECT_WIDTH, object_y + OBJECT_HEIGHT)
    cv2.rectangle(frame, obj_top_left, obj_bottom_right, OBJECT_COLOR, -1)

    # Light interaction visualization
    for x in range(0, WINDOW_WIDTH, LIGHT_GAP):
        if object_x < x < object_x + OBJECT_WIDTH:
            # Blocked portion
            cv2.line(frame, (x, 0), (x, object_y), BLOCKED_COLOR, 2)
            cv2.line(frame, (x, object_y + OBJECT_HEIGHT), (x, WINDOW_HEIGHT), BLOCKED_COLOR, 2)

    # Show object dimensions on screen
    text = f"Width: {OBJECT_WIDTH}px   Height: {OBJECT_HEIGHT}px"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    # Update object position
    object_x += SPEED * direction

    # Reverse direction when hitting edges
    if object_x + OBJECT_WIDTH >= WINDOW_WIDTH or object_x <= 0:
        direction *= -1

    # Display the simulation
    cv2.imshow("Light Simulation - A World Away", frame)

    # Exit when ESC key pressed
    key = cv2.waitKey(20)
    if key == 27:
        break

cv2.destroyAllWindows()
