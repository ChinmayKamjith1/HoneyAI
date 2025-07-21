import os

LOG_DIR = "/home/ubuntu/cowrie/var/log/cowrie"
DEST_FILE = os.path.join(LOG_DIR, "Full_list.json")
STATE_FILE = "/home/ubuntu/cowrie/.processed_log_lines"

# Ensure files exist
open(STATE_FILE, 'a').close()
open(DEST_FILE, 'a').close()

# Load existing state into memory
processed = {}
with open(STATE_FILE, 'r') as f:
    for line in f:
        parts = line.strip().split(':')
        if len(parts) == 2:
            filename, linecount = parts
            processed[filename] = int(linecount)

# Get list of all cowrie.json* files (excluding cowrie.log*)
log_files = sorted([
    f for f in os.listdir(LOG_DIR)
    if f.startswith("cowrie.json") and not f.endswith(".log") and os.path.isfile(os.path.join(LOG_DIR, f))
])

# Process each file
for log_file in log_files:
    path = os.path.join(LOG_DIR, log_file)
    last_line = processed.get(log_file, 0)

    with open(path, 'r') as f:
        lines = f.readlines()

    total_lines = len(lines)

    if total_lines > last_line:
        new_lines = lines[last_line:]

        with open(DEST_FILE, 'a') as out:
            out.writelines(new_lines)

        print(f"[+] Appended {len(new_lines)} lines from {log_file}")
        processed[log_file] = total_lines
    else:
        print(f"[-] No new lines in {log_file}")

# Save updated state
with open(STATE_FILE, 'w') as f:
    for log_file, count in processed.items():
        f.write(f"{log_file}:{count}\n")


