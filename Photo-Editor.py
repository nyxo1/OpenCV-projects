import cv2
import numpy as np
import matplotlib.pyplot as plt


# Global state
history_stack = []
history_log = []
original_image = None


def show_preview(original, current, title="Preview"):
    orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    if len(current.shape) == 2:
        curr_display = current
        cmap = "gray"
    else:
        curr_display = cv2.cvtColor(current, cv2.COLOR_BGR2RGB)
        cmap = None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(curr_display, cmap=cmap)
    axes[1].set_title(f"Preview: {title}")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


def load_image():
    path = input("Enter image file path: ").strip()
    img = cv2.imread(path)
    if img is None:
        print(f"  [Error] Could not load: {path}")
        return None
    print(f"  Loaded: {img.shape[1]}x{img.shape[0]} px")
    return img


def adjust_brightness(image):
    try:
        beta = int(input("  Brightness offset (0 = no change, +50 = brighter, -50 = darker): "))
    except ValueError:
        beta = 0
    result = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
    log_action(f"Brightness: offset={beta}")
    return result


def adjust_contrast(image):
    try:
        alpha = float(input("  Contrast (1.0 = no change, 1.5 = more, 0.5 = less): "))
    except ValueError:
        alpha = 1.0
    result = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    log_action(f"Contrast: alpha={alpha}")
    return result


def convert_to_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    log_action("Converted to Grayscale")
    return gray


def add_padding(image):
    border_types = {
        "1": ("CONSTANT",  cv2.BORDER_CONSTANT),
        "2": ("REFLECT",   cv2.BORDER_REFLECT),
        "3": ("REPLICATE", cv2.BORDER_REPLICATE),
        "4": ("WRAP",      cv2.BORDER_WRAP),
    }
    print("  Border types: 1.CONSTANT  2.REFLECT  3.REPLICATE  4.WRAP")
    btype_key = input("  Choose border type (1-4): ").strip()
    border_name, border_code = border_types.get(btype_key, ("CONSTANT", cv2.BORDER_CONSTANT))

    constant_color = [0, 0, 0]
    if border_code == cv2.BORDER_CONSTANT:
        c = input("  Border color B,G,R (Enter for black): ").strip()
        if c:
            try:
                constant_color = list(map(int, c.split(",")))
            except ValueError:
                pass

    print("  Padding shape: 1.Square  2.Rectangle  3.Custom Ratio")
    shape_key = input("  Choose (1-3): ").strip()
    h, w = image.shape[:2]
    top = bottom = left = right = 0

    if shape_key == "1":
        size = int(input("  Padding size (px, each side): "))
        top = bottom = left = right = size
        log_action(f"Padding: Square {size}px, border={border_name}")

    elif shape_key == "2":
        tb = int(input("  Top/Bottom padding (px): "))
        lr = int(input("  Left/Right padding (px): "))
        top = bottom = tb
        left = right = lr
        log_action(f"Padding: Rectangle T/B={tb}px L/R={lr}px, border={border_name}")

    elif shape_key == "3":
        ratio_str = input("  Enter ratio W:H (e.g. 4:5): ").strip()
        try:
            rw, rh = map(int, ratio_str.split(":"))
        except ValueError:
            rw, rh = 1, 1

        target_ratio = rw / rh
        current_ratio = w / h

        if current_ratio < target_ratio:
            new_w = int(h * target_ratio)
            min_lr = (new_w - w) // 2
            min_tb = 0
        else:
            new_h = int(w / target_ratio)
            min_tb = (new_h - h) // 2
            min_lr = 0

        print(f"  Min padding for {rw}:{rh} -> T/B={min_tb}px, L/R={min_lr}px")
        extra = input("  Extra padding to add (px, Enter for 0): ").strip()
        extra = int(extra) if extra.isdigit() else 0

        extra_lr = int(extra * rw / (rw + rh))
        extra_tb = extra - extra_lr
        top = bottom = min_tb + extra_tb
        left = right = min_lr + extra_lr
        log_action(f"Padding: Ratio {rw}:{rh} T/B={top}px L/R={left}px, border={border_name}")

    else:
        print("  Invalid. No padding applied.")
        return image

    return cv2.copyMakeBorder(image, top, bottom, left, right, border_code, value=constant_color)


def apply_thresholding(image):
    # thresholding requires a grayscale image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    print("  1. THRESH_BINARY   2. THRESH_BINARY_INV")
    choice = input("  Choose (1-2): ").strip()
    thresh_type = cv2.THRESH_BINARY if choice == "1" else cv2.THRESH_BINARY_INV
    thresh_name = "BINARY" if choice == "1" else "BINARY_INV"
    try:
        thresh_val = int(input("  Threshold value (0-255): "))
    except ValueError:
        thresh_val = 127
    _, thresh = cv2.threshold(gray, thresh_val, 255, thresh_type)
    log_action(f"Threshold: {thresh_name} value={thresh_val}")
    return thresh


def blend_images(image):
    # blending requires a BGR image
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    path2 = input("  Second image path: ").strip()
    img2 = cv2.imread(path2)
    if img2 is None:
        print(f"  [Error] Could not load: {path2}")
        return image
    try:
        alpha = float(input("  Alpha for current image (0.0-1.0): "))
        alpha = max(0.0, min(1.0, alpha))
    except ValueError:
        alpha = 0.5
    img2_resized = cv2.resize(img2, (image.shape[1], image.shape[0]))
    result = cv2.addWeighted(image, alpha, img2_resized, 1 - alpha, 0)
    log_action(f"Blend: alpha={alpha}, second={path2}")
    return result


def push_state(image):
    history_stack.append(image.copy())


def undo(current_image):
    if not history_stack:
        print("  Nothing to undo.")
        return current_image
    if history_log:
        print(f"  Undone: {history_log.pop()}")
    return history_stack.pop()


def log_action(description):
    history_log.append(description)


def show_history():
    if not history_log:
        print("  No actions yet.")
    else:
        print("  -- History --")
        for i, action in enumerate(history_log, 1):
            print(f"  {i}. {action}")


def print_menu():
    print("\n==== Mini Photo Editor ====")
    print("  1. Adjust Brightness")
    print("  2. Adjust Contrast")
    print("  3. Convert to Grayscale")
    print("  4. Add Padding (choose border type)")
    print("  5. Apply Thresholding (binary or inverse)")
    print("  6. Blend with Another Image (manual alpha)")
    print("  7. Undo Last Operation")
    print("  8. View History of Operations")
    print("  9. Save and Exit")


def main():
    global original_image
    print("=== Mini Photo Editor ===")
    current = load_image()
    if current is None:
        return
    original_image = current.copy()

    while True:
        print_menu()
        choice = input("Choose (1-9): ").strip()

        if choice == "1":
            push_state(current)
            current = adjust_brightness(current)
            show_preview(original_image, current, "Brightness")

        elif choice == "2":
            push_state(current)
            current = adjust_contrast(current)
            show_preview(original_image, current, "Contrast")

        elif choice == "3":
            push_state(current)
            current = convert_to_grayscale(current)
            show_preview(original_image, current, "Grayscale")

        elif choice == "4":
            push_state(current)
            current = add_padding(current)
            show_preview(original_image, current, "Padding")

        elif choice == "5":
            push_state(current)
            current = apply_thresholding(current)
            show_preview(original_image, current, "Thresholding")

        elif choice == "6":
            push_state(current)
            current = blend_images(current)
            show_preview(original_image, current, "Blending")

        elif choice == "7":
            current = undo(current)
            show_preview(original_image, current, "After Undo")

        elif choice == "8":
            show_history()

        elif choice == "9":
            show_history()
            if input("\nSave final image? (y/n): ").strip().lower() == "y":
                filename = input("Filename (e.g. output.jpg): ").strip()
                cv2.imwrite(filename, current)
                print(f"  Saved as '{filename}'.")
            print("Goodbye!")
            break

        else:
            print("  Invalid option.")


if __name__ == "__main__":
    main()