import json
import csv
import os
import re
import string

# Paths
LOG_DIR = "/home/ubuntu/cowrie/var/log/cowrie"
TTY_FOLDER = "/home/ubuntu/cowrie/var/lib/cowrie/tty"  # default tty folder
INPUT_FILE = os.path.join(LOG_DIR, "Full_list.json")
OUTPUT_FILE = "/home/ubuntu/cowrie/var/log/cowrie/events_new.csv"

FIELDS = [
    "username", "password", "eventid", "src_ip", "src_port",
    "dst_port", "session", "protocol", "timestamp", "shasum", "tty_contents"
]

# Regex to remove ANSI escape sequences and other control chars
ANSI_ESCAPE_RE = re.compile(r'''
    \x1B  # ESC
    (?:   # 7-bit C1 Fe (except CSI)
        [@-Z\\-_]
    |     # or [ for CSI, followed by parameters and command
        \[
        [0-?]*  # Parameter bytes
        [ -/]*  # Intermediate bytes
        [@-~]   # Final byte
    )
''', re.VERBOSE)

def clean_tty_content(raw):
    cleaned = ANSI_ESCAPE_RE.sub('', raw)
    printable = set(string.printable)
    filtered = ''.join(ch for ch in cleaned if ch in printable)
    filtered = re.sub(r'\s+', ' ', filtered).strip()
    return filtered

def read_tty_file(shasum):
    if not shasum:
        return ""
    tty_path = os.path.join(TTY_FOLDER, shasum)
    if os.path.exists(tty_path):
        try:
            with open(tty_path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
                return clean_tty_content(raw)
        except Exception as e:
            return f"[Error reading TTY file: {e}]"
    else:
        return "[TTY file not found]"

def main():
    used_shasums = set()
    eligible_count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:

        writer = csv.DictWriter(outfile, fieldnames=FIELDS)
        writer.writeheader()

        for line_num, line in enumerate(infile, 1):
            try:
                event = json.loads(line)
                if any(field in event for field in FIELDS):
                    row = {field: event.get(field, "") for field in FIELDS}

                    shasum = event.get("shasum", "")
                    row["shasum"] = shasum

                    if shasum and shasum not in used_shasums:
                        used_shasums.add(shasum)
                        eligible_count += 1

                    row["tty_contents"] = read_tty_file(shasum)

                    writer.writerow(row)

            except json.JSONDecodeError:
                print(f"Skipping invalid JSON at line {line_num}")

    print(f"✅ CSV generated at: {OUTPUT_FILE}")
    print(f"📊 Total unique eligible TTY files referenced: {eligible_count}")

if __name__ == "__main__":
    main()

