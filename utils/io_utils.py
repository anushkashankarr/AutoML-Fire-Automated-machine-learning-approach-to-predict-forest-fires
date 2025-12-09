from imports import *
from config.config import CFG

def save_fig(fig, fname):
    try:
        fig.tight_layout()
        fig.savefig(fname, dpi=220, bbox_inches='tight')
        plt.close(fig)
        print(f"[i] Saved figure : {fname}")
    except Exception as e:
        print("[!] save_fig error:", e, fname)

def load_data(path):
    print('[i] load_data: path =', path)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    # Print raw column names
    print("[i] Raw columns:", list(df.columns))
    print(f"[i] Data loaded : shape = {df.shape}")

    # Drop unwanted columns
    df = df.drop(columns=[c for c in CFG.drop_cols if c in df.columns], errors='ignore')

    # Print columns after dropping unused
    print("[i] After dropping CFG.drop_cols:", list(df.columns))

    return df

def clean_output_directory(path, delete_subfolders=True):
    """
    Cleans old training results for reproducibility.
    
    - Removes files inside directory
    - Optionally removes subdirectories
    - Creates directory if missing
    """
    import shutil

    # If not exist — create directory
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"[i] Created output directory: {path}")
        return

    print(f"[i] Cleaning previous outputs in: {path}")

    for root, dirs, files in os.walk(path):
        # Delete files
        for f in files:
            try:
                os.remove(os.path.join(root, f))
            except:
                pass

        # Delete folder contents if allowed
        if delete_subfolders:
            for d in dirs:
                try:
                    shutil.rmtree(os.path.join(root, d))
                except:
                    pass

class DualLogger:
    def __init__(self, path):
        # utf-8 so emojis and unicode don't crash
        self.file = open(path, "a", buffering=1, encoding="utf-8", errors="replace")

    def write(self, message):
        # print to original console
        try:
            sys.__stdout__.write(message)
        except UnicodeEncodeError:
            # if console can't show emoji, at least don’t crash
            sys.__stdout__.write(message.encode("utf-8", "replace").decode("utf-8"))

        # write to log file
        self.file.write(message)
        self.file.flush()

    def flush(self):
        try:
            sys.__stdout__.flush()
        except:
            pass
        self.file.flush()
