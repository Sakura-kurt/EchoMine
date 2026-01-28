"""Helper to add new memories/knowledge to Sakura."""

import os
from datetime import datetime

KNOWLEDGE_DIR = "./knowledge"


def add_memory(content: str, filename: str = "memories.txt"):
    """Append a new memory to a knowledge file."""
    filepath = os.path.join(KNOWLEDGE_DIR, filename)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}]\n{content}\n")

    print(f"Memory added to {filename}")


def add_preference(content: str):
    """Add something Sakura likes or dislikes."""
    add_memory(content, "sakura_preferences.txt")


def add_event(content: str):
    """Add a shared experience/event."""
    add_memory(content, "shared_experiences.txt")


def list_knowledge_files():
    """List all knowledge files."""
    files = os.listdir(KNOWLEDGE_DIR)
    print("Knowledge files:")
    for f in files:
        filepath = os.path.join(KNOWLEDGE_DIR, f)
        size = os.path.getsize(filepath)
        print(f"  - {f} ({size} bytes)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python add_memory.py 'Your memory text here'")
        print("  python add_memory.py --preference 'Sakura loves stargazing'")
        print("  python add_memory.py --event 'We explored the cave today'")
        print("  python add_memory.py --list")
        sys.exit(0)

    if sys.argv[1] == "--list":
        list_knowledge_files()
    elif sys.argv[1] == "--preference" and len(sys.argv) > 2:
        add_preference(sys.argv[2])
    elif sys.argv[1] == "--event" and len(sys.argv) > 2:
        add_event(sys.argv[2])
    else:
        add_memory(sys.argv[1])

    print("\nRemember to type 'rebuild' in chat to reload knowledge!")
