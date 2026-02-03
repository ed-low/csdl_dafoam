import os
import sys
import contextlib

def is_headless():
    # 1. Common Unix/Linux: no X11 display variable
    if sys.platform != "win32":
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return True

    # 2. Try to create a simple GUI context (Tkinter) to verify display availability.
    try:
        # suppress stderr from Tk if it complains
        with contextlib.redirect_stderr(open(os.devnull, "w")):
            import tkinter
            root = tkinter.Tk()
            root.withdraw()
            root.update_idletasks()
            root.destroy()
    except Exception:
        return True  # failed to create even a minimal window -> likely headless

    # 3. Check common headless backends (e.g., matplotlib using Agg implies no interactive display)
    try:
        import matplotlib
        backend = matplotlib.get_backend().lower()
        if "agg" in backend and not ("tk" in backend or "qt" in backend or "wx" in backend):
            # Agg can be explicitly set even in non-headless cases, so this is a soft signal.
            pass  # don't conclusively declare headless based solely on this
    except ImportError:
        pass  # matplotlib not present, ignore

    return False  # if we reached here, a display seems available

if __name__ == '__main__':
    print(is_headless())