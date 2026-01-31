from demucs.audio import save_audio
import inspect

# Find where save_audio is defined
source_file = inspect.getsourcefile(save_audio)
print(f"Source file: {source_file}")

# Try to read the source
try:
    source = inspect.getsource(save_audio)
    print("\nSource code:")
    print(source)
except Exception as e:
    print(f"Could not get source: {e}")
